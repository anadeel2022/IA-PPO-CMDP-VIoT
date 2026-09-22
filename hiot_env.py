from __future__ import annotations

from collections import deque
import heapq
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Sequence, Tuple

import numpy as np

from phy_layer import CHANNEL_RATE_BPS, PHYLayer, SLOT_DURATION_S

# Re-export constants used elsewhere in the codebase.
SLOT_DURATION_S = SLOT_DURATION_S
CHANNEL_RATE_BPS = CHANNEL_RATE_BPS

# -----------------------------------------------------------------------------
# Action model
# -----------------------------------------------------------------------------
# A discrete action selects one candidate device and one scheduling mode.
MODE_DEFER = 0
MODE_GRANT = 1
MODE_PROTECT = 2
MODE_COEXIST = 3
MODE_REJECT = 4
MODE_NAMES = {
    MODE_DEFER: "defer",
    MODE_GRANT: "grant",
    MODE_PROTECT: "protect",
    MODE_COEXIST: "coexist",
    MODE_REJECT: "reject",
}


@dataclass
class DeviceProfile:
    """Per-class traffic, radio, energy, and deadline parameters."""

    name: str
    tx_power_dbm: float
    sampling_hz: float
    bits_per_sample: int
    battery_joules: float
    deadline_ms: float
    energy_per_bit: float = 1.2e-8
    drag_penalty: float = 0.0


@dataclass
class ServiceRequest:
    """Request-level state used for all-arrival accounting."""

    request_id: int
    device_idx: int
    class_id: int
    arrival_step: int
    bits_total: float
    bits_remaining: float
    deadline_steps: int
    deadline_counted: bool = False
    admitted: bool = False
    active: bool = True


# The profiles define explicit per-device request rates. The experiment-level
# arrival_scale is a dimensionless load multiplier (default 1.0). The nominal
# 20/10/4 requests/s rates, together with 24 devices and bits_scale=2, produce
# a moderate offered load; stress scenarios can then move the system into
# congestion without making nominal operation irrecoverably overloaded.
DEFAULT_PROFILES: Dict[int, DeviceProfile] = {
    0: DeviceProfile(
        "SafetyEmergency", tx_power_dbm=18.0, sampling_hz=20.0,
        bits_per_sample=4000, battery_joules=50.0, deadline_ms=20.0,
        energy_per_bit=1.0e-8, drag_penalty=0.0,
    ),
    1: DeviceProfile(
        "Telemetry", tx_power_dbm=12.0, sampling_hz=10.0,
        bits_per_sample=2500, battery_joules=30.0, deadline_ms=100.0,
        energy_per_bit=0.9e-8, drag_penalty=0.0,
    ),
    2: DeviceProfile(
        "BestEffort", tx_power_dbm=8.0, sampling_hz=4.0,
        bits_per_sample=1800, battery_joules=20.0, deadline_ms=250.0,
        energy_per_bit=1.4e-8, drag_penalty=0.04,
    ),
}


class HIoTEnv:
    """Request-level RSU-assisted V-IoT scheduling environment.

    The environment exposes a compact dynamic candidate set. An action is encoded as
    ``candidate_slot * n_action_modes + mode``. The five modes have distinct effects:

    * defer: keep the selected request queued without transmission;
    * grant: serve the selected candidate using the full ordinary resource;
    * protect: serve the most urgent candidate with reliability gain and overhead;
    * coexist: share the slot between the selected candidate and a second candidate;
    * reject: remove the selected head-of-line request and count it as blocked.

    This design gives the policy genuine device-selection authority while keeping the
    discrete action space small enough for PPO and the legacy value-based agents.
    """

    def __init__(
        self,
        steps_per_ep: int = 150,
        n_devices: int = 12,
        n_channels: int = 3,
        seed: int = 42,
        arrival_scale: float = 1.0,
        class_mix: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        burst_k: float | None = None,
        device_skew: float = 0.0,
        enable_dynamic_priority: bool = False,
        enable_channel_realism: bool = True,
        enable_device_heterogeneity: bool = True,
        profiles: Dict[int, DeviceProfile] | None = None,
        warm_start_backlog: bool = True,
        enable_tx_power_rate_scaling: bool = True,
        enable_energy_per_bit_cost: bool = True,
        enable_drag_penalty: bool = True,
        device_jitter_cv: float = 0.10,
        bits_scale: float = 1.0,
        delay_penalty_w: float = 0.0025,
        energy_penalty_w: float = 0.0002,
        prio_weights: tuple[float, float, float] = (1.4, 1.0, 0.85),
        candidate_slots: int = 4,
    ) -> None:
        self._seed_init = int(seed)
        self.rng = np.random.default_rng(seed)
        self.steps_per_ep = int(steps_per_ep)
        self.channel_rate_bps = float(CHANNEL_RATE_BPS)
        self.n_devices = int(n_devices)
        self.n_channels = int(n_channels)
        self.n_classes = 3
        self.slot_s = float(SLOT_DURATION_S)

        self._delay_pen_w = float(delay_penalty_w)
        self._energy_pen_w = float(energy_penalty_w)
        self._prio_w = tuple(float(x) for x in prio_weights)
        self._arrival_scale = float(arrival_scale)
        self._bits_scale = float(max(bits_scale, 1e-6))

        self.enable_dynamic_priority = bool(enable_dynamic_priority)
        self.enable_channel_realism = bool(enable_channel_realism)
        self.enable_device_heterogeneity = bool(enable_device_heterogeneity)
        self.enable_tx_power_rate_scaling = bool(enable_tx_power_rate_scaling)
        self.enable_energy_per_bit_cost = bool(enable_energy_per_bit_cost)
        self.enable_drag_penalty = bool(enable_drag_penalty)
        self.device_jitter_cv = float(max(device_jitter_cv, 0.0))
        self.warm_start_backlog = bool(warm_start_backlog)
        self.burst_k = burst_k
        self.device_skew = float(device_skew)

        self.n_candidate_slots = max(2, int(candidate_slots))
        self.n_action_modes = 5
        self.n_actions = self.n_candidate_slots * self.n_action_modes
        self.action_space_n = self.n_actions

        cm = np.asarray(class_mix, dtype=float)
        cm = np.clip(cm, 1e-6, None)
        self.class_mix = cm / cm.sum()

        self.profiles = profiles if profiles is not None else DEFAULT_PROFILES
        self.dev_class = self._assign_classes(self.n_devices, self.class_mix)

        self.tx_power_dbm = np.zeros(self.n_devices, dtype=float)
        self.sampling_hz = np.zeros(self.n_devices, dtype=float)
        self.bits_per_sample = np.zeros(self.n_devices, dtype=int)
        self.deadline_steps = np.zeros(self.n_devices, dtype=int)
        self.battery_cap_j = np.zeros(self.n_devices, dtype=float)
        self.battery_j = np.zeros(self.n_devices, dtype=float)
        self.energy_per_bit = np.zeros(self.n_devices, dtype=float)
        self.drag_penalty = np.zeros(self.n_devices, dtype=float)
        self._tx_ref_dbm = 18.0
        self._init_device_params()

        self.t = 0
        self.ep_done = False
        self.arrivals_enabled = True
        self.request_queues: List[Deque[ServiceRequest]] = [deque() for _ in range(self.n_devices)]
        self._next_request_id = 1
        self._deadline_heap: List[Tuple[int, int, ServiceRequest]] = []
        self.queue_bits = np.zeros(self.n_devices, dtype=float)
        self.queue_age = np.zeros(self.n_devices, dtype=int)
        self.current_sinr_db = np.full(self.n_devices, float(PHYLayer.cfg.sinr_mu_db), dtype=float)
        self.sinr_hist: List[float] = []
        self._candidate_cache = np.full(self.n_candidate_slots, -1, dtype=int)

        # Base observation: 13 global/class features + 8 per candidate.
        # The final candidate feature indicates whether the head request has
        # already been admitted. This prevents the agent from treating a drop of
        # an active request as a valid admission rejection.
        self.obs_dim = 13 + 8 * self.n_candidate_slots
        self.reset_metrics()

    # ------------------------------------------------------------------
    # Device/request initialization
    # ------------------------------------------------------------------
    def _assign_classes(self, n: int, mix: np.ndarray) -> np.ndarray:
        """Assign an exact largest-remainder class mix, then shuffle reproducibly."""
        probs = np.asarray(mix, dtype=float)
        probs = probs / max(float(probs.sum()), 1e-12)
        raw = probs * int(n)
        counts = np.floor(raw).astype(int)
        remainder = int(n) - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1
        classes = np.concatenate([np.full(int(counts[c]), c, dtype=int) for c in range(len(counts))])
        self.rng.shuffle(classes)
        return classes

    def _init_device_params(self) -> None:
        def jitter(x: float) -> float:
            if self.device_jitter_cv <= 0.0:
                return float(x)
            return float(x) * float(self.rng.normal(1.0, self.device_jitter_cv))

        for i in range(self.n_devices):
            c = int(self.dev_class[i])
            prof = self.profiles.get(c, DEFAULT_PROFILES[c])
            self.tx_power_dbm[i] = jitter(prof.tx_power_dbm)
            self.sampling_hz[i] = max(0.0, jitter(prof.sampling_hz))
            self.bits_per_sample[i] = int(max(1, round(jitter(prof.bits_per_sample))))
            deadline_ms = max(1.0, jitter(prof.deadline_ms))
            self.deadline_steps[i] = max(1, int(np.ceil(deadline_ms / (self.slot_s * 1e3))))
            self.battery_cap_j[i] = max(1e-6, jitter(prof.battery_joules))
            self.energy_per_bit[i] = max(1e-10, jitter(prof.energy_per_bit))
            self.drag_penalty[i] = float(np.clip(jitter(prof.drag_penalty), 0.0, 0.9))

        self.sampling_hz *= self._arrival_scale
        if self._bits_scale != 1.0:
            self.bits_per_sample = np.maximum(
                1, np.rint(self.bits_per_sample.astype(float) * self._bits_scale).astype(int)
            )
        self.battery_j[:] = self.battery_cap_j
        if self.enable_tx_power_rate_scaling:
            self._tx_ref_dbm = float(np.percentile(self.tx_power_dbm, 90.0))

    def reset_metrics(self) -> None:
        self.total_bits_tx = 0.0
        self.total_energy_j = 0.0
        self.success_events = 0
        self.block_events = 0
        self.interrupt_events = 0
        self.channel_use_counts = np.zeros(self.n_channels, dtype=int)

        self.arrivals_this_ep = 0
        self.denies_this_ep = 0
        self.preempt_this_ep = 0
        self.depleted_devices_ep = 0

        self.bits_tx_ep = 0.0
        self.energy_j_ep = 0.0
        self.delay_samples_ms_ep: List[float] = []
        self.total_arrivals_ep = 0
        self.blocked_arrivals_ep = 0
        self.completed_requests_ep = 0
        self.deadline_exposed_requests_ep = 0
        self.deadline_completed_late_ep = 0
        self.expired_pending_requests_ep = 0
        self.preemptions_ep = 0
        self.scheduled_tx_ep = 0
        self.slots_used_ep = 0
        self.served_bits_per_class_ep = np.zeros(self.n_classes, dtype=float)
        self.served_bits_per_device_ep = np.zeros(self.n_devices, dtype=float)
        self.arrivals_per_class_ep = np.zeros(self.n_classes, dtype=int)

        self.throughput_bits_ep: List[float] = []
        self.delay_steps_ep: List[float] = []
        self.energy_eff_ep: List[float] = []
        self.fairness_ep: List[float] = []
        self.block_prob_ep: List[float] = []
        self.interrupt_prob_ep: List[float] = []

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = 0
        self.ep_done = False
        # Standard runs keep arrivals enabled throughout. The terminal-drain
        # sensitivity evaluator disables them only after its fixed cohort window.
        self.arrivals_enabled = True
        self.request_queues = [deque() for _ in range(self.n_devices)]
        self._next_request_id = 1
        self._deadline_heap = []
        self.queue_bits[:] = 0.0
        self.queue_age[:] = 0
        self.sinr_hist.clear()
        self.reset_metrics()
        self.battery_j[:] = self.battery_cap_j

        if self.warm_start_backlog:
            # One short pre-horizon interval prevents empty episodes without creating
            # an unaccounted queue. Warm-start requests are included in all arrivals.
            horizon_s = min(self.steps_per_ep * self.slot_s, 0.10)
            expected = np.clip(self.sampling_hz * horizon_s, 0.0, 2.0)
            draws = self.rng.poisson(expected)
            for i, count in enumerate(draws):
                for _ in range(int(min(count, 2))):
                    self._enqueue_request(i, arrival_step=-1, count_arrival=True)

        self._sample_radio_state()
        self._sync_aggregate_state()
        return self._make_obs(), {}

    # ------------------------------------------------------------------
    # Request-level queue accounting
    # ------------------------------------------------------------------
    def _enqueue_request(self, dev_idx: int, arrival_step: int, count_arrival: bool = True) -> ServiceRequest:
        req = ServiceRequest(
            request_id=self._next_request_id,
            device_idx=int(dev_idx),
            class_id=int(self.dev_class[dev_idx]),
            arrival_step=int(arrival_step),
            bits_total=float(self.bits_per_sample[dev_idx]),
            bits_remaining=float(self.bits_per_sample[dev_idx]),
            deadline_steps=int(self.deadline_steps[dev_idx]),
        )
        self._next_request_id += 1
        self.request_queues[dev_idx].append(req)
        expiry_step = int(req.arrival_step + req.deadline_steps)
        heapq.heappush(self._deadline_heap, (expiry_step, req.request_id, req))
        self.queue_bits[dev_idx] += float(req.bits_remaining)
        if count_arrival:
            self.total_arrivals_ep += 1
            self.arrivals_this_ep += 1
            self.arrivals_per_class_ep[req.class_id] += 1
        return req

    def _arrivals_step(self) -> int:
        """Generate independent Poisson request counts for every device.

        ``arrivals_enabled`` is disabled only by the terminal-drain sensitivity
        evaluator after the pre-specified arrival window. This preserves the
        frozen policy and PHY dynamics while giving every request in the fixed
        cohort its full deadline opportunity without admitting new traffic.
        """
        if not bool(getattr(self, "arrivals_enabled", True)):
            return 0
        lam_dt = np.clip(self.sampling_hz * self.slot_s, 0.0, 20.0)
        draws = self.rng.poisson(lam_dt)
        count = int(np.sum(draws))
        for i, n_req in enumerate(draws):
            for _ in range(int(n_req)):
                # Arrivals occur at the transition boundary and are available next epoch.
                self._enqueue_request(i, arrival_step=self.t + 1, count_arrival=True)
        return count

    def _request_age_steps(self, req: ServiceRequest, at_step: int | None = None) -> int:
        now = self.t if at_step is None else int(at_step)
        return max(0, int(now - req.arrival_step))

    def _sync_aggregate_state(self) -> None:
        # queue_bits is maintained incrementally; only head ages need refreshing.
        for i, q in enumerate(self.request_queues):
            if q:
                self.queue_age[i] = self._request_age_steps(q[0])
            else:
                self.queue_age[i] = 0
                self.queue_bits[i] = 0.0
        self._candidate_cache = self.get_candidate_devices()

    def _mark_deadline_exposure(self) -> int:
        """Mark newly expired active requests using an O(log n) deadline heap."""
        newly_exposed = 0
        while self._deadline_heap and self._deadline_heap[0][0] <= self.t:
            _, _, req = heapq.heappop(self._deadline_heap)
            if req.active and not req.deadline_counted:
                req.deadline_counted = True
                self.deadline_exposed_requests_ep += 1
                self.expired_pending_requests_ep += 1
                newly_exposed += 1
        return newly_exposed

    def _reject_head_request(self, dev_idx: int) -> Dict[str, Any]:
        """Reject one not-yet-admitted head request.

        A request becomes admitted when it first receives a grant/protected/coexist
        transmission attempt. Once admitted, a reject action is invalid and is
        treated as a no-service decision rather than silently dropping active
        traffic. This distinction makes the blocking metric a true admission
        rejection rate.
        """
        if dev_idx < 0 or dev_idx >= self.n_devices or not self.request_queues[dev_idx]:
            return {"rejected": 0, "invalid_reject": 0, "deadline_new": 0, "class_id": -1}
        req = self.request_queues[dev_idx][0]
        if req.admitted:
            return {"rejected": 0, "invalid_reject": 1, "deadline_new": 0, "class_id": req.class_id}
        req = self.request_queues[dev_idx].popleft()
        req.active = False
        self.queue_bits[dev_idx] = max(0.0, self.queue_bits[dev_idx] - float(req.bits_remaining))
        deadline_new = 0
        if not req.deadline_counted:
            req.deadline_counted = True
            self.deadline_exposed_requests_ep += 1
            deadline_new = 1
        self.blocked_arrivals_ep += 1
        self.block_events += 1
        self.denies_this_ep += 1
        return {"rejected": 1, "invalid_reject": 0, "deadline_new": deadline_new, "class_id": req.class_id}

    def head_request_admitted(self, dev_idx: int) -> bool:
        if dev_idx < 0 or dev_idx >= self.n_devices or not self.request_queues[dev_idx]:
            return False
        return bool(self.request_queues[dev_idx][0].admitted)

    # ------------------------------------------------------------------
    # Candidate set and action encoding
    # ------------------------------------------------------------------
    def _priority(self, dev_idx: int) -> float:
        c = int(self.dev_class[dev_idx])
        return float(self._prio_w[c] if 0 <= c < len(self._prio_w) else 1.0)

    def _head_age_ratio(self, dev_idx: int) -> float:
        q = self.request_queues[dev_idx]
        if not q:
            return 0.0
        return float(self._request_age_steps(q[0]) / max(q[0].deadline_steps, 1))

    def get_candidate_devices(self) -> np.ndarray:
        active = np.flatnonzero(self.queue_bits > 0.0)
        if active.size == 0:
            return np.full(self.n_candidate_slots, -1, dtype=int)

        served = np.asarray(self.served_bits_per_device_ep, dtype=float)
        served_scale = max(float(np.mean(served[active])) if active.size else 0.0, 1.0)

        def urgency(i: int) -> float:
            return (
                2.2 * self._priority(i)
                + 3.0 * self._head_age_ratio(i)
                + 0.35 * np.log1p(max(self.queue_bits[i], 0.0) / 1000.0)
            )

        def maxweight(i: int) -> float:
            channel = max(0.15, min(2.0, (float(self.current_sinr_db[i]) + 20.0) / 30.0))
            return self._priority(i) * (1.0 + self.queue_age[i]) * np.log1p(self.queue_bits[i]) * channel

        def fair_channel(i: int) -> float:
            deficit = max(0.0, 1.0 - served[i] / served_scale)
            return 0.55 * deficit + 0.45 * ((float(self.current_sinr_db[i]) + 20.0) / 60.0)

        ordered: List[int] = []
        selectors = [
            max(active, key=lambda i: urgency(int(i))),
            max(active, key=lambda i: self._head_age_ratio(int(i))),
            max(active, key=lambda i: maxweight(int(i))),
            max(active, key=lambda i: fair_channel(int(i))),
        ]
        for i in selectors:
            ii = int(i)
            if ii not in ordered:
                ordered.append(ii)
        for i in sorted((int(x) for x in active), key=urgency, reverse=True):
            if i not in ordered:
                ordered.append(i)
            if len(ordered) >= self.n_candidate_slots:
                break
        out = np.full(self.n_candidate_slots, -1, dtype=int)
        out[: min(len(ordered), self.n_candidate_slots)] = ordered[: self.n_candidate_slots]
        return out

    def encode_action(self, candidate_slot: int, mode: int) -> int:
        slot = int(np.clip(candidate_slot, 0, self.n_candidate_slots - 1))
        mode = int(np.clip(mode, 0, self.n_action_modes - 1))
        return slot * self.n_action_modes + mode

    def encode_action_for_device(self, dev_idx: int, mode: int) -> int:
        candidates = self.get_candidate_devices()
        matches = np.flatnonzero(candidates == int(dev_idx))
        slot = int(matches[0]) if matches.size else 0
        return self.encode_action(slot, mode)

    def decode_action(self, action: int, candidates: Sequence[int] | None = None) -> Tuple[int, int, int]:
        a = int(np.clip(int(action), 0, self.n_actions - 1))
        slot = a // self.n_action_modes
        mode = a % self.n_action_modes
        cands = np.asarray(candidates if candidates is not None else self.get_candidate_devices(), dtype=int)
        dev_idx = int(cands[slot]) if slot < cands.size else -1
        return slot, mode, dev_idx

    def mode_from_action(self, action: int) -> int:
        return int(action) % self.n_action_modes

    def get_feasible_action_mask(self) -> np.ndarray:
        """Return the physical/action-semantics feasibility mask.

        This mask is applied to every learning scheme. It removes actions that
        cannot be executed in the current state, independently of the proposed
        admission-safety mechanism. In particular, an already-admitted request
        cannot be rejected, and an empty candidate slot may only be deferred.
        """
        mask = np.ones(int(self.n_actions), dtype=bool)
        candidates = self.get_candidate_devices()
        for slot, dev in enumerate(candidates):
            if dev < 0 or not self.request_queues[int(dev)]:
                for mode in range(self.n_action_modes):
                    mask[self.encode_action(slot, mode)] = mode == MODE_DEFER
                continue
            if self.head_request_admitted(int(dev)):
                mask[self.encode_action(slot, MODE_REJECT)] = False
        if not np.any(mask):
            mask[self.encode_action(0, MODE_DEFER)] = True
        return mask

    def get_action_mask(self) -> np.ndarray:
        """Return the mask used by generic policies.

        The base environment exposes only physical feasibility. Agentic wrappers
        may add safety filtering, but they must never remove this common mask.
        """
        return self.get_feasible_action_mask()

    def action_name(self, action: int) -> str:
        slot, mode, dev = self.decode_action(action)
        return f"candidate{slot}:{MODE_NAMES.get(mode, 'unknown')}@{dev}"

    def _urgent_device(self, preferred: int = -1) -> int:
        active = np.flatnonzero(self.queue_bits > 0.0)
        if active.size == 0:
            return -1

        def score(i: int) -> float:
            return 4.0 * self._priority(i) + 4.5 * self._head_age_ratio(i) + 0.25 * np.log1p(self.queue_bits[i])

        best = int(max(active, key=lambda i: score(int(i))))
        if preferred >= 0 and self.queue_bits[preferred] > 0.0 and score(preferred) >= 0.90 * score(best):
            return int(preferred)
        return best

    def _coexist_secondaries(self, primary: int, limit: int = 2) -> List[int]:
        """Select distinct secondary users for simultaneous multi-channel service."""
        active = [int(i) for i in np.flatnonzero(self.queue_bits > 0.0) if int(i) != int(primary)]
        if not active or limit <= 0:
            return []
        served = np.asarray(self.served_bits_per_device_ep, dtype=float)
        mean_served = max(float(np.mean(served[active])) if active else 0.0, 1.0)

        def score(i: int) -> float:
            fairness_deficit = max(0.0, 1.0 - served[i] / mean_served)
            return 1.2 * self._head_age_ratio(i) + 0.8 * self._priority(i) + 0.5 * fairness_deficit

        return sorted(active, key=score, reverse=True)[: int(limit)]

    # ------------------------------------------------------------------
    # Radio/service model
    # ------------------------------------------------------------------
    def _sample_radio_state(self) -> None:
        self.current_sinr_db = np.asarray([
            PHYLayer.sample_sinr_db(
                self.rng,
                PHYLayer.cfg.sinr_mu_db,
                PHYLayer.cfg.sinr_sigma_db,
                enable_fading=self.enable_channel_realism,
                enable_shadowing=self.enable_channel_realism,
                enable_interference=self.enable_channel_realism,
            )
            for _ in range(self.n_devices)
        ], dtype=float)
        self.sinr_hist.append(float(np.nanmean(self.current_sinr_db)))

    @staticmethod
    def _link_success_probability(sinr_db: float) -> float:
        # Smooth packet-success curve: approximately 0.16 at -5 dB, 0.50 at 0 dB,
        # and 0.84 at 5 dB. Protected service adds an SINR gain before this mapping.
        return float(np.clip(1.0 / (1.0 + np.exp(-(float(sinr_db)) / 3.0)), 0.01, 0.999))

    def _attempt_service(
        self,
        dev_idx: int,
        capacity_scale: float = 1.0,
        sinr_adjust_db: float = 0.0,
        replicas: int = 1,
    ) -> Dict[str, Any]:
        result = {
            "attempted": 0,
            "success": 0,
            "interrupted": 0,
            "bits": 0.0,
            "energy_j": 0.0,
            "completed": 0,
            "completed_late": 0,
            "deadline_new": 0,
            "delays_ms": [],
            "served_bits_per_class": np.zeros(self.n_classes, dtype=float),
        }
        if dev_idx < 0 or dev_idx >= self.n_devices or not self.request_queues[dev_idx]:
            return result

        result["attempted"] = 1
        # Scheduling the request constitutes admission even if the radio attempt
        # subsequently fails. This prevents a later reject action from being
        # counted as admission control after service has already started.
        self.request_queues[dev_idx][0].admitted = True
        if self.battery_j[dev_idx] <= 0.0:
            result["interrupted"] = 1
            return result

        sinr_db = float(self.current_sinr_db[dev_idx]) + float(sinr_adjust_db)
        replicas = max(1, int(replicas))
        p_single = self._link_success_probability(sinr_db)
        # Independent repetitions provide diversity without multiplying the
        # useful payload. This distinguishes protected service from ordinary
        # grant while accounting for its radio and energy cost.
        p_success = float(1.0 - (1.0 - p_single) ** replicas)
        e_pa_single = float(PHYLayer.energy_joules(self.tx_power_dbm[dev_idx], self.slot_s))
        e_pa = e_pa_single * float(replicas) * max(0.25, capacity_scale)
        if self.rng.random() > p_success:
            # A failed scheduled attempt consumes control/pilot energy.
            e_fail = min(self.battery_j[dev_idx], 0.25 * e_pa)
            self.battery_j[dev_idx] = max(0.0, self.battery_j[dev_idx] - e_fail)
            result["energy_j"] = float(e_fail)
            result["interrupted"] = 1
            return result

        bits_cap = float(PHYLayer.slot_bits(sinr_db)) * float(max(capacity_scale, 0.0))
        if self.enable_tx_power_rate_scaling:
            delta_db = float(self.tx_power_dbm[dev_idx] - self._tx_ref_dbm)
            bits_cap *= float(np.clip(10.0 ** (delta_db / 10.0), 0.30, 1.25))
        if self.enable_drag_penalty:
            bits_cap *= float(1.0 - self.drag_penalty[dev_idx])
        if bits_cap <= 0.0:
            result["interrupted"] = 1
            return result

        # Respect the remaining energy budget before modifying requests.
        per_bit = float(self.energy_per_bit[dev_idx]) if self.enable_energy_per_bit_cost else 0.0
        if e_pa >= self.battery_j[dev_idx]:
            result["interrupted"] = 1
            return result
        if per_bit > 0.0:
            bits_cap = min(bits_cap, max(0.0, (self.battery_j[dev_idx] - e_pa) / per_bit))
        bits_remaining = float(bits_cap)
        delivered = 0.0

        while bits_remaining > 1e-9 and self.request_queues[dev_idx]:
            req = self.request_queues[dev_idx][0]
            served = min(bits_remaining, req.bits_remaining)
            req.bits_remaining -= served
            bits_remaining -= served
            delivered += served
            if req.bits_remaining <= 1e-9:
                self.request_queues[dev_idx].popleft()
                req.active = False
                delay_steps = max(1, self.t - req.arrival_step + 1)
                delay_ms = float(delay_steps * self.slot_s * 1e3)
                result["delays_ms"].append(delay_ms)
                result["completed"] += 1
                self.completed_requests_ep += 1
                if delay_steps > req.deadline_steps:
                    result["completed_late"] += 1
                    self.deadline_completed_late_ep += 1
                    if not req.deadline_counted:
                        req.deadline_counted = True
                        self.deadline_exposed_requests_ep += 1
                        result["deadline_new"] += 1

        if delivered <= 0.0:
            result["interrupted"] = 1
            return result

        e_bits = per_bit * delivered
        energy = min(self.battery_j[dev_idx], e_pa + e_bits)
        self.battery_j[dev_idx] = max(0.0, self.battery_j[dev_idx] - energy)
        if self.battery_j[dev_idx] <= 0.0:
            self.depleted_devices_ep += 1

        self.queue_bits[dev_idx] = max(0.0, self.queue_bits[dev_idx] - float(delivered))
        cls = int(self.dev_class[dev_idx])
        result["success"] = 1
        result["bits"] = float(delivered)
        result["energy_j"] = float(energy)
        result["served_bits_per_class"][cls] = float(delivered)
        return result

    # ------------------------------------------------------------------
    # Environment transition
    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.ep_done:
            return self._make_obs(), 0.0, True, {}

        candidates = self.get_candidate_devices()
        candidate_slot, mode, selected_dev = self.decode_action(action, candidates)
        selected_had_backlog = bool(selected_dev >= 0 and self.queue_bits[selected_dev] > 0.0)
        selected_age_before = float(self.queue_age[selected_dev]) if selected_had_backlog else 0.0
        queue_load_before = float(np.count_nonzero(self.queue_bits > 0.0)) / max(float(self.n_devices), 1.0)

        # Radio conditions are observable from the preceding state and used here.
        action_results: List[Tuple[int, Dict[str, Any]]] = []
        rejected_this_step = 0
        invalid_reject_this_step = 0
        deadline_new = 0
        actual_target = selected_dev
        secondary_targets: List[int] = []
        secondary_target = -1
        preempted_this_step = 0
        coexist = False
        channels_used_this_step = 0

        if mode == MODE_DEFER:
            # Deliberately hold the selected request. No radio resource is used.
            pass
        elif mode == MODE_REJECT:
            # True admission rejection: remove one request and retain it in all-arrival accounting.
            rej = self._reject_head_request(selected_dev)
            rejected_this_step = int(rej["rejected"])
            invalid_reject_this_step = int(rej.get("invalid_reject", 0))
            deadline_new += int(rej["deadline_new"])
        elif mode == MODE_GRANT:
            # One dedicated 1-MHz resource channel.
            channels_used_this_step = int(selected_had_backlog)
            action_results.append((selected_dev, self._attempt_service(selected_dev, 1.0, 0.0, replicas=1)))
        elif mode == MODE_PROTECT:
            # Preempt the requested lower-priority target when a more urgent
            # request exists. The urgent request is repeated over two channels;
            # the remaining channel may serve one secondary request.
            actual_target = self._urgent_device(selected_dev)
            if actual_target >= 0:
                channels_used_this_step = min(2, self.n_channels)
                action_results.append((actual_target, self._attempt_service(actual_target, 0.90, 2.0, replicas=2)))
                # The remaining channel is reserved for control/guard signalling;
                # protected service must not silently behave like coexistence.
                secondary_targets = []
                lower_priority_waiting = any(
                    self.queue_bits[i] > 0.0 and self._priority(i) < self._priority(actual_target)
                    for i in range(self.n_devices) if i != actual_target
                )
                preempted_this_step = int(lower_priority_waiting or actual_target != selected_dev)
        elif mode == MODE_COEXIST:
            # Spatial/frequency coexistence schedules up to one request per
            # channel. A modest SINR/overhead penalty represents simultaneous use.
            coexist = True
            if selected_had_backlog:
                secondary_targets = self._coexist_secondaries(selected_dev, limit=max(0, self.n_channels - 1))
                targets = [selected_dev] + secondary_targets
            else:
                targets = self._coexist_secondaries(-1, limit=self.n_channels)
            for dev in targets[: self.n_channels]:
                action_results.append((dev, self._attempt_service(dev, 0.92, -1.5, replicas=1)))
            channels_used_this_step = len(action_results)
            secondary_target = int(secondary_targets[0]) if secondary_targets else -1

        delivered_bits = 0.0
        energy_j_this_step = 0.0
        completed_this_step = 0
        completed_late_this_step = 0
        delays_ms: List[float] = []
        served_bits_per_class_step = np.zeros(self.n_classes, dtype=float)
        scheduled_this_step = 0
        interrupted_this_step = 0
        successful_links = 0

        for dev_idx, res in action_results:
            scheduled_this_step += int(res["attempted"])
            interrupted_this_step += int(res["interrupted"])
            successful_links += int(res["success"])
            delivered_bits += float(res["bits"])
            energy_j_this_step += float(res["energy_j"])
            completed_this_step += int(res["completed"])
            completed_late_this_step += int(res["completed_late"])
            deadline_new += int(res["deadline_new"])
            delays_ms.extend(float(x) for x in res["delays_ms"])
            served_bits_per_class_step += np.asarray(res["served_bits_per_class"], dtype=float)
            if res["bits"] > 0.0 and dev_idx >= 0:
                self.served_bits_per_device_ep[dev_idx] += float(res["bits"])

        self.interrupt_events += int(interrupted_this_step)
        self.preemptions_ep += int(preempted_this_step)
        self.preempt_this_ep += int(preempted_this_step)
        self.scheduled_tx_ep += int(scheduled_this_step)
        self.slots_used_ep += int(channels_used_this_step)
        self.success_events += int(successful_links)
        self.total_bits_tx += float(delivered_bits)
        self.total_energy_j += float(energy_j_this_step)
        self.bits_tx_ep += float(delivered_bits)
        self.energy_j_ep += float(energy_j_this_step)
        self.served_bits_per_class_ep += served_bits_per_class_step
        self.delay_samples_ms_ep.extend(delays_ms)

        if channels_used_this_step > 0:
            # Channels are homogeneous; mark the number occupied in this slot.
            for ch in range(min(int(channels_used_this_step), self.n_channels)):
                self.channel_use_counts[ch] += 1

        # Queue evolution follows Q(t+1)=[Q(t)-mu(t)]^+ + A(t).
        arrivals_this_step = self._arrivals_step()
        self.t += 1
        deadline_new += self._mark_deadline_exposure()
        self._sample_radio_state()
        self._sync_aggregate_state()

        pending = int(sum(len(q) for q in self.request_queues))
        total_arr = max(int(self.total_arrivals_ep), 1)
        blocking_rate = float(self.blocked_arrivals_ep / total_arr)
        service_success_rate = float(self.completed_requests_ep / total_arr)
        deadline_exposure_rate = float(self.deadline_exposed_requests_ep / total_arr)
        unfinished_rate = float(pending / total_arr)
        interruption_rate = float(self.interrupt_events / max(self.scheduled_tx_ep, 1))

        # Action-coupled base reward. Intent-specific penalties are applied by wrappers.
        nominal_cap = max(float(self.n_channels) * float(PHYLayer.slot_bits(float(np.nanmean(self.current_sinr_db)))), 1.0)
        util = float(np.clip(delivered_bits / nominal_cap, 0.0, 1.0))
        served_priority = 0.0
        if delivered_bits > 0.0:
            served_priority = float(sum(self._prio_w[c] * served_bits_per_class_step[c] for c in range(self.n_classes)) / delivered_bits)
        fairness_now = self._jain_fairness()
        age_risk = 0.0
        if selected_had_backlog and selected_dev >= 0:
            age_risk = selected_age_before / max(float(self.deadline_steps[selected_dev]), 1.0)

        reward = served_priority * util
        reward += 0.06 * (fairness_now - 0.6)
        reward -= self._delay_pen_w * max(0.0, selected_age_before)
        reward -= self._energy_pen_w * energy_j_this_step
        if rejected_this_step:
            # Admission control is a genuine overload trade-off in V4.4. A fresh
            # best-effort request may be worth rejecting when many devices are
            # simultaneously backlogged, but reliability/latency compilers still
            # account for that rejection through all-arrival violation. Critical
            # traffic remains strongly protected by the base reward itself.
            reject_priority = self._priority(selected_dev) if selected_dev >= 0 else 1.0
            selected_class = int(self.dev_class[selected_dev]) if selected_dev >= 0 else 0
            pending_pressure = float(np.clip(pending / max(2.0 * self.n_devices, 1.0), 0.0, 1.0))
            overload = max(0.0, queue_load_before - 0.35)
            if selected_class == 2:
                # Generic load-shedding value, common to every scheme. It becomes
                # competitive only under substantial overload; the intent-specific
                # blocking/success objectives decide whether that rejection is
                # acceptable. This creates a real admission trade-off without
                # hard-coding an IA action preference.
                relief = 1.15 * overload + 0.20 * pending_pressure
                lateness_penalty = 0.08 * min(max(age_risk, 0.0), 2.0)
                reward += relief - 0.06 * reject_priority - lateness_penalty
            else:
                reward -= 0.32 * reject_priority + 0.12 * min(max(age_risk, 0.0), 2.0)
        if invalid_reject_this_step:
            # Dropping already admitted traffic is not a valid admission-control
            # operation. Penalize the request and leave the active request queued.
            reward -= 0.20 * (self._priority(selected_dev) if selected_dev >= 0 else 1.0)
        reward -= 0.35 * interrupted_this_step
        reward -= 0.20 * completed_late_this_step
        if mode == MODE_DEFER and selected_had_backlog:
            reward -= 0.10 * (1.0 + age_risk) * self._priority(selected_dev)
        if mode in (MODE_GRANT, MODE_PROTECT, MODE_COEXIST) and channels_used_this_step <= 0:
            reward -= 0.08
        if mode == MODE_PROTECT:
            reward -= 0.03  # signalling/repetition overhead
        reward = float(np.clip(reward, -5.0, 5.0))

        avg_age_now = float(np.mean(self.queue_age[self.queue_age > 0])) if np.any(self.queue_age > 0) else 0.0
        self.throughput_bits_ep.append(float(delivered_bits))
        self.delay_steps_ep.append(avg_age_now)
        self.energy_eff_ep.append(float(delivered_bits / energy_j_this_step) if energy_j_this_step > 0 else 0.0)
        self.fairness_ep.append(fairness_now)
        self.block_prob_ep.append(blocking_rate)
        self.interrupt_prob_ep.append(interruption_rate)

        done = self.t >= self.steps_per_ep
        self.ep_done = done
        if done:
            self._rollup_episode_metrics()

        delay_ms = float(np.mean(delays_ms)) if delays_ms else None
        elapsed_s = max(float(self.t) * self.slot_s, self.slot_s)
        throughput_mean_mbps_so_far = float(self.bits_tx_ep / elapsed_s / 1e6)
        info: Dict[str, Any] = {
            "sinr_db": float(self.current_sinr_db[actual_target]) if actual_target >= 0 else float(np.nanmean(self.current_sinr_db)),
            "candidate_devices": candidates.copy(),
            "candidate_slot": int(candidate_slot),
            "selected_dev_idx": int(selected_dev),
            "actual_target_dev_idx": int(actual_target),
            "secondary_dev_idx": int(secondary_target),
            "secondary_dev_indices": [int(x) for x in secondary_targets],
            "dev_idx": int(actual_target),
            "class": int(self.dev_class[actual_target]) if actual_target >= 0 else -1,
            "action_mode": int(mode),
            "action_mode_name": MODE_NAMES.get(mode, "unknown"),
            "coexist": bool(coexist),
            "delivered_bits": float(delivered_bits),
            "throughput": float(delivered_bits / max(self.slot_s, 1e-12)),
            "throughput_mean_mbps_so_far": throughput_mean_mbps_so_far,
            "energy": float(energy_j_this_step),
            "energy_j": float(energy_j_this_step),
            "delay_ms": delay_ms,
            "delays_ms": delays_ms,
            "arrivals": int(arrivals_this_step),
            "blocked": int(rejected_this_step),
            "rejected": int(rejected_this_step),
            "invalid_reject": int(invalid_reject_this_step),
            "deny": int(rejected_this_step),
            "preempted": int(preempted_this_step),
            "preempt": int(preempted_this_step),
            "interrupted": int(interrupted_this_step),
            "completed_requests": int(completed_this_step),
            "completed_late": int(completed_late_this_step),
            "deadline_exposed_new": int(deadline_new),
            "scheduled": int(scheduled_this_step),
            "slots_used": int(channels_used_this_step),
            "channels_used": int(channels_used_this_step),
            "served_bits_per_class": served_bits_per_class_step.copy(),
            "focus_backlog": int(selected_had_backlog),
            "focus_age_steps": float(selected_age_before),
            "blocking_rate_so_far": blocking_rate,
            "interruption_rate_so_far": interruption_rate,
            "service_success_rate_so_far": service_success_rate,
            "deadline_exposure_rate": deadline_exposure_rate,
            "unfinished_rate_so_far": unfinished_rate,
            "pending_requests": pending,
            "total_arrivals_so_far": int(self.total_arrivals_ep),
            "blocked_arrivals_so_far": int(self.blocked_arrivals_ep),
            "completed_requests_so_far": int(self.completed_requests_ep),
            "deadline_exposed_requests_so_far": int(self.deadline_exposed_requests_ep),
            "scheduled_attempts_so_far": int(self.scheduled_tx_ep),
            "interrupted_attempts_so_far": int(self.interrupt_events),
            "elapsed_steps": int(self.t),
            "battery_frac": float(self.battery_j[actual_target] / max(self.battery_cap_j[actual_target], 1e-12)) if actual_target >= 0 else 1.0,
        }
        return self._make_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Observation and episode metrics
    # ------------------------------------------------------------------
    def _make_obs(self) -> np.ndarray:
        self._sync_aggregate_state()
        active = self.queue_bits > 0.0
        qload = float(np.count_nonzero(active)) / max(self.n_devices, 1)
        total_backlog_norm = float(np.tanh(np.sum(self.queue_bits) / max(np.sum(self.bits_per_sample) * 4.0, 1.0)))
        oldest_ratio = float(max((self._head_age_ratio(i) for i in np.flatnonzero(active)), default=0.0))
        avg_ratio = float(np.mean([self._head_age_ratio(i) for i in np.flatnonzero(active)])) if np.any(active) else 0.0
        mean_sinr = float(np.nanmean(self.current_sinr_db))
        success_link_rate = float(self.success_events / max(self.scheduled_tx_ep, 1))
        total_arr = max(self.total_arrivals_ep, 1)
        block_rate = float(self.blocked_arrivals_ep / total_arr)
        deadline_rate = float(self.deadline_exposed_requests_ep / total_arr)
        completion_rate = float(self.completed_requests_ep / total_arr)
        utilization = float(self.slots_used_ep / max(self.t, 1))
        cls_counts = np.bincount(self.dev_class, minlength=self.n_classes).astype(float)
        cls_frac = cls_counts / max(float(self.n_devices), 1.0)

        global_features = np.asarray([
            qload,
            total_backlog_norm,
            np.clip(oldest_ratio, 0.0, 2.0) / 2.0,
            np.clip(avg_ratio, 0.0, 2.0) / 2.0,
            np.tanh(mean_sinr / 20.0),
            np.clip(success_link_rate, 0.0, 1.0),
            np.clip(block_rate, 0.0, 1.0),
            np.clip(deadline_rate, 0.0, 1.0),
            np.clip(completion_rate, 0.0, 1.0),
            np.clip(utilization, 0.0, 1.0),
            cls_frac[0], cls_frac[1], cls_frac[2],
        ], dtype=np.float32)

        candidates = self.get_candidate_devices()
        served = self.served_bits_per_device_ep
        active_served_mean = max(float(np.mean(served[np.flatnonzero(active)])) if np.any(active) else 0.0, 1.0)
        cfeatures: List[float] = []
        for dev in candidates:
            if dev < 0:
                cfeatures.extend([0.0] * 8)
                continue
            battery_frac = float(self.battery_j[dev] / max(self.battery_cap_j[dev], 1e-12))
            service_deficit = float(np.clip(1.0 - served[dev] / active_served_mean, -1.0, 1.0))
            cfeatures.extend([
                1.0,
                np.clip(self._priority(dev) / max(self._prio_w), 0.0, 1.0),
                float(np.tanh(self.queue_bits[dev] / max(self.bits_per_sample[dev] * 4.0, 1.0))),
                float(np.clip(self._head_age_ratio(dev), 0.0, 2.0) / 2.0),
                float(np.tanh(self.current_sinr_db[dev] / 20.0)),
                np.clip(battery_frac, 0.0, 1.0),
                (service_deficit + 1.0) / 2.0,
                1.0 if self.head_request_admitted(int(dev)) else 0.0,
            ])
        obs = np.concatenate([global_features, np.asarray(cfeatures, dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _jain_fairness(self) -> float:
        x = np.asarray(self.served_bits_per_device_ep, dtype=float)
        if np.sum(x) <= 1e-12:
            return 0.0
        return float(np.clip((x.sum() ** 2) / (self.n_devices * np.square(x).sum() + 1e-12), 0.0, 1.0))

    def request_accounting(self) -> Dict[str, int]:
        pending = int(sum(len(q) for q in self.request_queues))
        pending_admission = int(sum(1 for q in self.request_queues for req in q if not req.admitted))
        admitted_pending = int(pending - pending_admission)
        return {
            "arrivals": int(self.total_arrivals_ep),
            "completed": int(self.completed_requests_ep),
            "rejected": int(self.blocked_arrivals_ep),
            "pending": pending,
            "pending_admission": pending_admission,
            "admitted_pending": admitted_pending,
            "deadline_exposed": int(self.deadline_exposed_requests_ep),
        }

    def _rollup_episode_metrics(self) -> dict:
        ep_seconds = max(self.steps_per_ep * self.slot_s, 1e-12)
        pending = int(sum(len(q) for q in self.request_queues))
        total_arr = max(int(self.total_arrivals_ep), 1)
        throughput_mbps = float((self.bits_tx_ep / ep_seconds) / 1e6)
        avg_delay_ms = float(np.mean(self.delay_samples_ms_ep)) if self.delay_samples_ms_ep else 0.0
        energy_eff_bpj = float(self.bits_tx_ep / self.energy_j_ep) if self.energy_j_ep > 0 else 0.0
        x_cls = np.asarray(self.served_bits_per_class_ep, dtype=float)
        jain_cls = float((x_cls.sum() ** 2) / (self.n_classes * np.square(x_cls).sum() + 1e-12)) if x_cls.sum() > 0 else 0.0
        x_dev = np.asarray(self.served_bits_per_device_ep, dtype=float)
        jain_dev = float((x_dev.sum() ** 2) / (self.n_devices * np.square(x_dev).sum() + 1e-12)) if x_dev.sum() > 0 else 0.0

        self.episode_metrics = {
            "throughput_mbps": throughput_mbps,
            "avg_delay_ms": avg_delay_ms,
            "energy_eff_bpj": energy_eff_bpj,
            "jain_fairness": jain_cls,
            "jain_fairness_devices": jain_dev,
            "blocking_prob": float(self.blocked_arrivals_ep / total_arr),
            "interrupt_prob": float(self.interrupt_events / max(self.scheduled_tx_ep, 1)),
            "deadline_exposure_rate": float(self.deadline_exposed_requests_ep / total_arr),
            "service_success_rate": float(self.completed_requests_ep / total_arr),
            "unfinished_rate": float(pending / total_arr),
            "utilization": float(self.slots_used_ep / max(self.steps_per_ep, 1)),
            "delay_per_episode": avg_delay_ms,
            "fairness_per_episode": jain_dev,
            "request_conservation_error": int(self.total_arrivals_ep - self.completed_requests_ep - self.blocked_arrivals_ep - pending),
        }
        return self.episode_metrics
