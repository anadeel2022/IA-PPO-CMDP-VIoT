"""
viot_agentic_intent.py

Intent-aware agentic evaluation layer for the RSU-assisted V-IoT scheduling
codebase. This module does not remove or simplify the existing simulation
logic. It adds: (i) operator-level intent profiles, (ii) an agentic monitoring
environment wrapper, (iii) robustness/stress scenario support, and (iv)
fine-grained post-hoc metrics for special-issue-grade evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Sequence
import math
import numpy as np

from hiot_env import (
    HIoTEnv, MODE_DEFER, MODE_GRANT, MODE_PROTECT, MODE_COEXIST, MODE_REJECT, MODE_NAMES,
)
from phy_layer import PHYLayer

EPS = 1e-12


@dataclass(frozen=True)
class IntentProfile:
    """Feasibility-calibrated operator intent for V-IoT scheduling.

    ``delay_target_ms`` is retained for reporting compatibility. Training and
    primary compliance use ``max_deadline_exposure_rate`` when provided, which
    measures all-backlog deadline exposure rather than only delivered packets.
    This prevents a policy from appearing low-latency by denying traffic.
    """
    name: str
    description: str
    block_target: float
    intr_target: float
    delay_target_ms: float
    min_fairness: float
    min_energy_eff_bpj: Optional[float] = None
    min_utilization: Optional[float] = None
    throughput_floor_mbps: Optional[float] = None
    weights: Dict[str, float] = None
    deadline_target_ms: Optional[float] = None
    max_deadline_exposure_rate: Optional[float] = None
    min_service_success_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["weights"] is None:
            d["weights"] = default_violation_weights()
        return d


def default_violation_weights() -> Dict[str, float]:
    return {
        "blocking_prob": 1.0,
        "interrupt_prob": 1.4,
        "avg_delay_ms": 0.7,
        "jain_fairness": 0.5,
        "energy_eff_bpj": 0.4,
        "utilization": 0.2,
        "throughput_mbps": 0.3,
    }


INTENT_LIBRARY: Dict[str, IntentProfile] = {
    # V4.4 targets are calibrated against development-only deterministic anchors.
    # Each intent is demonstrated feasible in nominal operation by at least one
    # service-oriented anchor, while stress scenarios are intentionally allowed
    # to violate the declared objectives.
    "reliability_first": IntentProfile(
        name="reliability_first",
        description="Reliability-first intent: prioritize continuity and low interruption while retaining a minimum useful-service floor.",
        block_target=0.030,
        intr_target=0.160,
        delay_target_ms=80.0,
        deadline_target_ms=80.0,
        max_deadline_exposure_rate=0.72,
        min_service_success_rate=0.60,
        min_fairness=0.50,
        min_energy_eff_bpj=None,
        min_utilization=0.10,
        throughput_floor_mbps=2.40,
        weights={
            "blocking_prob": 1.30, "interrupt_prob": 2.20,
            "deadline_exposure": 1.10, "service_success": 1.00,
            "avg_delay_ms": 0.20, "jain_fairness": 0.35,
            "energy_eff_bpj": 0.10, "utilization": 0.20,
            "throughput_mbps": 0.65,
        },
    ),
    "latency_critical": IntentProfile(
        name="latency_critical",
        description="Latency-critical intent: prioritize deadline exposure, completion rate, and throughput while allowing a wider continuity envelope.",
        block_target=0.100,
        intr_target=0.350,
        delay_target_ms=70.0,
        deadline_target_ms=70.0,
        max_deadline_exposure_rate=0.60,
        min_service_success_rate=0.75,
        min_fairness=0.50,
        min_energy_eff_bpj=None,
        min_utilization=0.10,
        throughput_floor_mbps=3.00,
        weights={
            "blocking_prob": 0.55, "interrupt_prob": 0.55,
            "deadline_exposure": 2.40, "service_success": 1.65,
            "avg_delay_ms": 0.35, "jain_fairness": 0.30,
            "energy_eff_bpj": 0.10, "utilization": 0.20,
            "throughput_mbps": 1.35,
        },
    ),
    "energy_aware": IntentProfile(
        name="energy_aware",
        description="Energy-aware intent retained for legacy experiments; not used in the V4.4 manuscript protocol.",
        block_target=0.080,
        intr_target=0.300,
        delay_target_ms=85.0,
        deadline_target_ms=85.0,
        max_deadline_exposure_rate=0.75,
        min_service_success_rate=0.55,
        min_fairness=0.45,
        min_energy_eff_bpj=7.5e7,
        min_utilization=0.08,
        throughput_floor_mbps=2.20,
        weights={
            "blocking_prob": 0.80, "interrupt_prob": 0.90,
            "deadline_exposure": 0.75, "service_success": 0.70,
            "avg_delay_ms": 0.15, "jain_fairness": 0.25,
            "energy_eff_bpj": 1.15, "utilization": 0.20,
            "throughput_mbps": 0.45,
        },
    ),
    "fairness_balanced": IntentProfile(
        name="fairness_balanced",
        description="Fairness-oriented intent retained for legacy experiments; not used in the V4.4 manuscript protocol.",
        block_target=0.080,
        intr_target=0.280,
        delay_target_ms=85.0,
        deadline_target_ms=85.0,
        max_deadline_exposure_rate=0.75,
        min_service_success_rate=0.55,
        min_fairness=0.55,
        min_energy_eff_bpj=None,
        min_utilization=0.10,
        throughput_floor_mbps=2.20,
        weights={
            "blocking_prob": 0.70, "interrupt_prob": 0.85,
            "deadline_exposure": 0.75, "service_success": 0.70,
            "avg_delay_ms": 0.15, "jain_fairness": 1.20,
            "energy_eff_bpj": 0.20, "utilization": 0.30,
            "throughput_mbps": 0.45,
        },
    ),
    "balanced_agentic": IntentProfile(
        name="balanced_agentic",
        description="Balanced intent: trade service continuity, deadline exposure, completion, throughput, and fairness without privileging one objective.",
        block_target=0.060,
        intr_target=0.200,
        delay_target_ms=80.0,
        deadline_target_ms=80.0,
        max_deadline_exposure_rate=0.70,
        min_service_success_rate=0.60,
        min_fairness=0.50,
        min_energy_eff_bpj=None,
        min_utilization=0.10,
        throughput_floor_mbps=2.50,
        weights={
            "blocking_prob": 1.00, "interrupt_prob": 1.15,
            "deadline_exposure": 1.20, "service_success": 1.05,
            "avg_delay_ms": 0.20, "jain_fairness": 0.65,
            "energy_eff_bpj": 0.20, "utilization": 0.20,
            "throughput_mbps": 0.95,
        },
    ),
}


@dataclass(frozen=True)
class StressScenario:
    """Time-local network disturbance injected without changing the base simulator internals."""
    name: str
    description: str
    shock_start_frac: float = 0.35
    shock_end_frac: float = 0.70
    arrival_multiplier: float = 1.0
    emergency_bias: float = 0.0
    best_effort_bias: float = 0.0
    sinr_mu_offset_db: float = 0.0
    sinr_sigma_multiplier: float = 1.0
    interference_offset_db: float = 0.0


SCENARIO_LIBRARY: Dict[str, StressScenario] = {
    "nominal": StressScenario(
        name="nominal",
        description="No external disturbance; native stochastic traffic, queueing, PHY fading, shadowing, and interference remain active.",
    ),
    "congestion_burst": StressScenario(
        name="congestion_burst",
        description="Mid-episode offered-load burst representing dense vehicular arrivals near an RSU.",
        arrival_multiplier=2.5,
    ),
    "admission_pressure": StressScenario(
        name="admission_pressure",
        description="Development stress dominated by fresh best-effort arrivals, used to test whether admission control and the safety shield are actually exercised.",
        arrival_multiplier=2.8,
        best_effort_bias=0.70,
        sinr_mu_offset_db=-2.0,
        sinr_sigma_multiplier=1.10,
    ),
    "sinr_drop": StressScenario(
        name="sinr_drop",
        description="Mid-episode channel impairment with lower mean SINR and stronger random spread.",
        sinr_mu_offset_db=-7.0,
        sinr_sigma_multiplier=1.45,
        interference_offset_db=-3.0,
    ),
    "emergency_surge": StressScenario(
        name="emergency_surge",
        description="Temporary priority-load surge created by biasing devices toward emergency-class behavior.",
        arrival_multiplier=2.0,
        emergency_bias=0.35,
        sinr_mu_offset_db=-2.0,
    ),
    "mixed_stress": StressScenario(
        name="mixed_stress",
        description="Joint congestion, emergency surge, and radio degradation for resilience evaluation.",
        arrival_multiplier=2.2,
        emergency_bias=0.30,
        sinr_mu_offset_db=-6.0,
        sinr_sigma_multiplier=1.35,
        interference_offset_db=-2.0,
    ),
}


def effective_deadline_ms(intent: IntentProfile) -> float:
    return float(intent.deadline_target_ms if intent.deadline_target_ms is not None else intent.delay_target_ms)


def deadline_target_steps(intent: IntentProfile, slot_s: float) -> int:
    return max(1, int(math.ceil(effective_deadline_ms(intent) / max(float(slot_s) * 1e3, EPS))))


def intent_context_vector(intent: IntentProfile) -> np.ndarray:
    """Compact normalized context supplied to all learning policies.

    The same context is available to every RL comparator. The proposed method
    differs through its compiler, risk memory, and admission shield—not through
    hidden information.
    """
    w = intent.weights or default_violation_weights()
    w_keys = ("blocking_prob", "interrupt_prob", "deadline_exposure", "service_success", "jain_fairness", "energy_eff_bpj", "throughput_mbps")
    raw_w = np.asarray([max(0.0, float(w.get(k, 0.0))) for k in w_keys], dtype=float)
    raw_w = raw_w / max(float(raw_w.sum()), EPS)
    deadline = effective_deadline_ms(intent)
    core = np.asarray([
        np.clip(float(intent.block_target) / 0.10, 0.0, 1.0),
        np.clip(float(intent.intr_target) / 0.40, 0.0, 1.0),
        np.clip(deadline / 120.0, 0.0, 1.0),
        np.clip(float(intent.max_deadline_exposure_rate if intent.max_deadline_exposure_rate is not None else 0.20) / 0.85, 0.0, 1.0),
        np.clip(float(intent.min_service_success_rate if intent.min_service_success_rate is not None else 0.0), 0.0, 1.0),
        np.clip(float(intent.min_fairness), 0.0, 1.0),
        np.clip(float(intent.throughput_floor_mbps if intent.throughput_floor_mbps is not None else 0.0) / 5.0, 0.0, 1.0),
        1.0 if intent.min_energy_eff_bpj is not None else 0.0,
    ], dtype=float)
    return np.concatenate([core, raw_w]).astype(np.float32)


def local_step_violation(info: Dict[str, Any], intent: IntentProfile) -> float:
    """Smoothed online intent violation for risk-memory reward compilation.

    Publication metrics remain exact request-level rates. During online control, a
    small target-centred prior avoids extreme penalties from one event during the
    first few slots while converging rapidly to the observed cumulative rates.
    """
    info = info or {}
    w = intent.weights or default_violation_weights()
    prior = 10.0
    arrivals = max(0.0, float(info.get("total_arrivals_so_far", 0.0) or 0.0))
    blocked_n = max(0.0, float(info.get("blocked_arrivals_so_far", 0.0) or 0.0))
    completed_n = max(0.0, float(info.get("completed_requests_so_far", 0.0) or 0.0))
    exposed_n = max(0.0, float(info.get("deadline_exposed_requests_so_far", 0.0) or 0.0))
    scheduled_n = max(0.0, float(info.get("scheduled_attempts_so_far", 0.0) or 0.0))
    interrupted_n = max(0.0, float(info.get("interrupted_attempts_so_far", 0.0) or 0.0))

    deadline_target = float(intent.max_deadline_exposure_rate if intent.max_deadline_exposure_rate is not None else 0.20)
    success_target = float(intent.min_service_success_rate if intent.min_service_success_rate is not None else 0.0)
    block = (blocked_n + prior * float(intent.block_target)) / max(arrivals + prior, EPS)
    exposure = (exposed_n + prior * deadline_target) / max(arrivals + prior, EPS)
    service_success = (completed_n + prior * success_target) / max(arrivals + prior, EPS)
    intr = (interrupted_n + prior * float(intent.intr_target)) / max(scheduled_n + prior, EPS)
    throughput_mbps = float(info.get("throughput_mean_mbps_so_far", 0.0) or 0.0)

    vals = [
        (max(0.0, block - float(intent.block_target)) / max(float(intent.block_target), EPS), w.get("blocking_prob", 1.0)),
        (max(0.0, intr - float(intent.intr_target)) / max(float(intent.intr_target), EPS), w.get("interrupt_prob", 1.0)),
        (max(0.0, exposure - deadline_target) / max(deadline_target, EPS), w.get("deadline_exposure", w.get("avg_delay_ms", 1.0))),
    ]
    if intent.min_service_success_rate is not None:
        vals.append((max(0.0, success_target - service_success) / max(success_target, EPS), w.get("service_success", 0.5)))
    if intent.throughput_floor_mbps is not None:
        floor = float(intent.throughput_floor_mbps)
        vals.append((max(0.0, floor - throughput_mbps) / max(floor, EPS), w.get("throughput_mbps", 0.3)))
    numerator = sum(float(v) * float(weight) for v, weight in vals)
    denominator = sum(float(weight) for _, weight in vals)
    return float(numerator / max(denominator, EPS))


def intent_outcome_pressure(info: Dict[str, Any], intent: IntentProfile) -> float:
    """Intent-specific outcome pressure used only by the reward compiler.

    The pressure is computed from measurable service outcomes and queue state.
    It never assigns a bonus or penalty to an action label. This makes the
    reliability/latency distinction emerge from operational consequences:
    reliability emphasizes interruptions and continuity, whereas latency
    emphasizes deadline exposure, age pressure, completion, and throughput.
    """
    info = info or {}
    prior = 10.0
    arrivals = max(0.0, float(info.get("total_arrivals_so_far", 0.0) or 0.0))
    blocked_n = max(0.0, float(info.get("blocked_arrivals_so_far", 0.0) or 0.0))
    completed_n = max(0.0, float(info.get("completed_requests_so_far", 0.0) or 0.0))
    exposed_n = max(0.0, float(info.get("deadline_exposed_requests_so_far", 0.0) or 0.0))
    scheduled_n = max(0.0, float(info.get("scheduled_attempts_so_far", 0.0) or 0.0))
    interrupted_n = max(0.0, float(info.get("interrupted_attempts_so_far", 0.0) or 0.0))

    block_t = max(float(intent.block_target), 0.02)
    intr_t = max(float(intent.intr_target), 0.05)
    exp_t = max(float(intent.max_deadline_exposure_rate or 0.70), 0.10)
    succ_t = max(float(intent.min_service_success_rate or 0.0), 0.10)
    thr_t = max(float(intent.throughput_floor_mbps or 1.0), 0.25)

    block = (blocked_n + prior * float(intent.block_target)) / max(arrivals + prior, EPS)
    intr = (interrupted_n + prior * float(intent.intr_target)) / max(scheduled_n + prior, EPS)
    exposure = (exposed_n + prior * exp_t) / max(arrivals + prior, EPS)
    success = (completed_n + prior * succ_t) / max(arrivals + prior, EPS)
    throughput = max(0.0, float(info.get("throughput_mean_mbps_so_far", 0.0) or 0.0))

    v_block = max(0.0, block - float(intent.block_target)) / block_t
    v_intr = max(0.0, intr - float(intent.intr_target)) / intr_t
    v_exp = max(0.0, exposure - exp_t) / exp_t
    v_success = max(0.0, succ_t - success) / succ_t
    v_thr = max(0.0, thr_t - throughput) / thr_t

    oldest_age = max(0.0, float(info.get("oldest_age_steps", 0.0) or 0.0))
    # The manuscript protocol uses 1 ms decision epochs, so the deadline in ms
    # is numerically the active deadline in decision steps.
    deadline_steps = max(1.0, effective_deadline_ms(intent))
    age_ratio = oldest_age / deadline_steps
    v_age = max(0.0, age_ratio - 0.65)

    if intent.name == "reliability_first":
        return float(0.38 * v_intr + 0.22 * v_block + 0.18 * v_exp + 0.17 * v_success + 0.05 * v_thr)
    if intent.name == "latency_critical":
        return float(0.38 * v_exp + 0.20 * v_age + 0.20 * v_success + 0.17 * v_thr + 0.05 * v_intr)
    return float(0.20 * v_block + 0.20 * v_intr + 0.22 * v_exp + 0.18 * v_success + 0.15 * v_thr + 0.05 * v_age)


def switch_settling_time_metrics(
    violation: np.ndarray,
    switch_index: int,
    hold_steps: int = 10,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.10,
) -> Dict[str, float]:
    """Return stochastic settling-time diagnostics after an intent switch.

    Adaptation is measured relative to the policy's own post-switch steady-state
    violation, while post-switch violation remains the separate quality metric.
    This avoids declaring every policy unsuccessful merely because the absolute
    violation scale is above a small fixed threshold. A policy is considered
    settled when the mean violation over ``hold_steps`` consecutive epochs first
    enters a tolerance band around the terminal post-switch reference.
    """
    x = np.asarray(violation, dtype=float).ravel()
    first = int(np.clip(int(switch_index), 0, len(x)))
    post = x[first:]
    finite_post = post[np.isfinite(post)]
    if finite_post.size == 0:
        return {
            "adaptation_time_steps": float(len(post)),
            "adaptation_success": 0.0,
            "post_switch_violation_mean": np.nan,
            "pre_switch_reference": np.nan,
            "post_switch_reference": np.nan,
            "settling_band": np.nan,
        }

    post_mean = float(np.nanmean(finite_post))
    hold = max(1, min(int(hold_steps), len(post)))
    final_window = max(hold, int(math.ceil(0.20 * len(post))))
    final_window = min(final_window, len(post))
    final_values = post[-final_window:]
    final_values = final_values[np.isfinite(final_values)]
    final_ref = float(np.nanmedian(final_values)) if final_values.size else post_mean

    pre_values = x[max(0, first - final_window):first]
    pre_values = pre_values[np.isfinite(pre_values)]
    pre_ref = float(np.nanmedian(pre_values)) if pre_values.size else float(finite_post[0])

    abs_tol = max(0.0, float(absolute_tolerance))
    rel_tol = max(0.0, float(relative_tolerance))
    transition_scale = abs(pre_ref - final_ref)
    band = max(
        abs_tol,
        rel_tol * max(abs(final_ref), EPS),
        rel_tol * transition_scale,
    )

    adaptation = float(len(post))
    success = 0.0
    for offset in range(0, len(post) - hold + 1):
        segment = post[offset:offset + hold]
        if not np.all(np.isfinite(segment)):
            continue
        if abs(float(np.mean(segment)) - final_ref) <= band:
            adaptation = float(offset)
            success = 1.0
            break

    return {
        "adaptation_time_steps": adaptation,
        "adaptation_success": success,
        "post_switch_violation_mean": post_mean,
        "pre_switch_reference": pre_ref,
        "post_switch_reference": final_ref,
        "settling_band": float(band),
    }


class AgenticVIoTEnv(HIoTEnv):
    """
    HIoTEnv subclass that preserves base dynamics while adding intent-aware
    monitoring and stress injection. It records action decisions, per-step
    SLA deviations, stress-window behavior, and context-response statistics.
    """

    def __init__(
        self,
        *args,
        intent: IntentProfile,
        scenario: StressScenario,
        intent_switch_to: Optional[IntentProfile] = None,
        intent_switch_frac: float = 0.50,
        intent_switch_hold_steps: int = 10,
        intent_switch_tolerance: float = 0.05,
        append_intent_context: bool = True,
        **kwargs,
    ):
        self.intent = intent
        self.base_intent = intent
        self.active_intent = intent
        self.intent_switch_to = intent_switch_to
        self.intent_switch_frac = float(np.clip(intent_switch_frac, 0.05, 0.95))
        self.intent_switch_hold_steps = max(1, int(intent_switch_hold_steps))
        self.intent_switch_tolerance = float(max(0.0, intent_switch_tolerance))
        self.append_intent_context = bool(append_intent_context)
        self.scenario = scenario
        self._base_sampling_hz = None
        self._base_profile_snapshot = None
        self.agentic_trace: List[Dict[str, Any]] = []
        self.agentic_episode_summaries: List[Dict[str, Any]] = []
        self._action_counts = None
        self._shock_steps = (0, 0)
        self._intent_switch_step: Optional[int] = None
        self._shield_overrides_ep = 0
        self._shield_requests_ep = 0
        self.admission_shield_enabled = False
        self.admission_shield_age_fraction = 0.70
        self.admission_shield_queue_guard = 0.35
        # Legacy flag is retained for compatibility but V9 reward logic lives in
        # the runner wrapper to avoid duplicate and conflicting shaping.
        self.enable_intent_reward_shaping = False
        self.intent_reward_penalty_scale = 0.0
        self.intent_reward_mode = "off"

        # V4.4 universal curriculum. During training only, reset() may sample the
        # starting intent, switch target, switch time, and stress scenario from
        # predeclared pools. Evaluation environments leave this disabled and use
        # their fixed manuscript context. A dedicated RNG keeps curriculum
        # sampling reproducible and independent of packet/channel randomness.
        self._multi_intent_training_enabled = False
        self._training_intent_pool: List[IntentProfile] = []
        self._training_scenario_pool: List[StressScenario] = []
        self._training_switch_fractions: Tuple[float, ...] = (0.35, 0.65)
        self._training_static_probability = 0.20
        self._training_curriculum_rng = np.random.default_rng(0)
        self._training_episode_context: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)
        self.obs_dim = int(super()._make_obs().size + (intent_context_vector(self.get_active_intent()).size if self.append_intent_context else 0))
        self._base_sampling_hz = np.asarray(self.sampling_hz, dtype=float).copy()
        self._phy_base = dict(
            sinr_mu_db=float(PHYLayer.cfg.sinr_mu_db),
            sinr_sigma_db=float(PHYLayer.cfg.sinr_sigma_db),
            interf_db_mu=float(PHYLayer.cfg.interf_db_mu),
        )

    def configure_multi_intent_training(
        self,
        intents: Sequence[IntentProfile],
        scenarios: Sequence[StressScenario],
        curriculum_seed: int,
        switch_fractions: Sequence[float] = (0.35, 0.65),
        static_probability: float = 0.20,
    ) -> None:
        """Enable a reproducible multi-intent/multi-stress training curriculum.

        Each training episode samples a starting intent. With probability
        ``static_probability`` that intent is held for the full episode; otherwise
        a different target intent and one of the predeclared switch fractions are
        sampled. Stress scenarios are also sampled per episode. This exposes one
        policy to static and switching contexts while keeping final evaluation
        frozen and context-specific.
        """
        pool = [x for x in intents if isinstance(x, IntentProfile)]
        stress_pool = [x for x in scenarios if isinstance(x, StressScenario)]
        if not pool:
            raise ValueError("multi-intent training requires at least one intent")
        if not stress_pool:
            raise ValueError("multi-intent training requires at least one scenario")
        fracs = tuple(float(np.clip(x, 0.05, 0.95)) for x in switch_fractions)
        if not fracs:
            raise ValueError("at least one training switch fraction is required")
        self._multi_intent_training_enabled = True
        self._training_intent_pool = list(pool)
        self._training_scenario_pool = list(stress_pool)
        self._training_switch_fractions = fracs
        self._training_static_probability = float(np.clip(static_probability, 0.0, 1.0))
        self._training_curriculum_rng = np.random.default_rng(int(curriculum_seed))
        # Canonical pre-reset context keeps policy input construction independent
        # of whichever evaluation context happened to instantiate this env.
        canonical = next((x for x in pool if x.name == "balanced_agentic"), pool[0])
        self.intent = canonical
        self.base_intent = canonical
        self.active_intent = canonical
        self.intent_switch_to = None
        self.scenario = stress_pool[0]

    def disable_multi_intent_training(self) -> None:
        self._multi_intent_training_enabled = False

    def get_training_episode_context(self) -> Dict[str, Any]:
        return dict(self._training_episode_context)

    def get_active_intent(self) -> IntentProfile:
        return self.active_intent

    def configure_admission_shield(
        self,
        enabled: bool,
        age_fraction: float = 0.70,
        queue_guard: float = 0.35,
    ) -> None:
        self.admission_shield_enabled = bool(enabled)
        self.admission_shield_age_fraction = float(np.clip(age_fraction, 0.05, 1.0))
        self.admission_shield_queue_guard = float(np.clip(queue_guard, 0.0, 1.0))

    def _update_active_intent(self) -> None:
        if self.intent_switch_to is not None and self._intent_switch_step is not None and int(self.t) >= int(self._intent_switch_step):
            self.active_intent = self.intent_switch_to
        else:
            self.active_intent = self.base_intent

    def _make_obs(self) -> np.ndarray:
        base = super()._make_obs()
        if not self.append_intent_context:
            return base
        return np.concatenate([np.asarray(base, dtype=np.float32).ravel(), intent_context_vector(self.get_active_intent())]).astype(np.float32)

    def reset(self, seed=None, **kwargs):
        if self._base_sampling_hz is not None:
            self.sampling_hz[:] = self._base_sampling_hz

        # Sample the V4.4 curriculum before computing shock/switch windows. The
        # evaluation path never enables this block, so manuscript contexts remain
        # deterministic and explicitly controlled by CLI arguments.
        if self._multi_intent_training_enabled:
            rng = self._training_curriculum_rng
            pool = self._training_intent_pool
            self.scenario = self._training_scenario_pool[int(rng.integers(0, len(self._training_scenario_pool)))]
            start = pool[int(rng.integers(0, len(pool)))]
            do_static = bool(rng.random() < self._training_static_probability or len(pool) < 2)
            target = None
            frac = float(self._training_switch_fractions[int(rng.integers(0, len(self._training_switch_fractions)))])
            if not do_static:
                alternatives = [x for x in pool if x.name != start.name]
                target = alternatives[int(rng.integers(0, len(alternatives)))]
            self.intent = start
            self.base_intent = start
            self.active_intent = start
            self.intent_switch_to = target
            self.intent_switch_frac = frac
            self._training_episode_context = {
                "start_intent": start.name,
                "target_intent": target.name if target is not None else start.name,
                "switch_active": int(target is not None),
                "switch_fraction": frac if target is not None else np.nan,
                "scenario": self.scenario.name,
            }

        self.agentic_trace = []
        self._action_counts = np.zeros(int(getattr(self, "n_actions", 20)), dtype=int)
        self._mode_counts = np.zeros(int(getattr(self, "n_action_modes", 5)), dtype=int)
        s0 = int(round(self.steps_per_ep * self.scenario.shock_start_frac))
        s1 = int(round(self.steps_per_ep * self.scenario.shock_end_frac))
        self._shock_steps = (max(0, s0), min(self.steps_per_ep, max(s0 + 1, s1)))
        self._intent_switch_step = int(round(self.steps_per_ep * self.intent_switch_frac)) if self.intent_switch_to is not None else None
        self.active_intent = self.base_intent
        self._shield_overrides_ep = 0
        self._shield_requests_ep = 0
        return super().reset(seed=seed, **kwargs)

    # ------------------------------------------------------------------
    # Compatibility properties for legacy heuristic schedulers. These do
    # not change the simulator state; they expose the current vectorized
    # queue representation in the dictionary/list form expected by older
    # baselines.
    # ------------------------------------------------------------------
    @property
    def queue(self):
        qbits = np.asarray(getattr(self, "queue_bits", []), dtype=float)
        qage = np.asarray(getattr(self, "queue_age", np.zeros_like(qbits)), dtype=float)
        dev_class = np.asarray(getattr(self, "dev_class", np.zeros_like(qbits, dtype=int)), dtype=int)
        out = []
        for i in np.flatnonzero(qbits > 0.0):
            out.append({"dev_idx": int(i), "cls": int(dev_class[i]), "age": int(qage[i]), "bits": float(qbits[i])})
        return out

    @property
    def class_map(self):
        return {"Emergency": 0, "Glucose": 1, "Fitness": 2, "MissionCritical": 1, "NonCritical": 2}

    @property
    def success_hist(self):
        tr = getattr(self, "agentic_trace", []) or []
        return [int(x.get("service_success", 0)) for x in tr]

    @property
    def served_per_device(self):
        return np.asarray(getattr(self, "served_bits_per_device_ep", np.zeros(getattr(self, "n_devices", 1))), dtype=float)

    @property
    def success_bits_hist(self):
        tr = getattr(self, "agentic_trace", []) or []
        return [float(x.get("delivered_bits", 0.0) or 0.0) for x in tr if float(x.get("delivered_bits", 0.0) or 0.0) > 0.0]

    def _in_shock_window(self) -> bool:
        s0, s1 = self._shock_steps
        return s0 <= int(self.t) < s1

    def _apply_stress_controls(self) -> None:
        # Reset to base each step, then apply only inside shock window.
        if self._base_sampling_hz is not None:
            self.sampling_hz[:] = self._base_sampling_hz
        PHYLayer.cfg.sinr_mu_db = self._phy_base["sinr_mu_db"]
        PHYLayer.cfg.sinr_sigma_db = self._phy_base["sinr_sigma_db"]
        PHYLayer.cfg.interf_db_mu = self._phy_base["interf_db_mu"]

        if not self._in_shock_window():
            return
        sc = self.scenario
        if sc.arrival_multiplier != 1.0:
            self.sampling_hz[:] = self.sampling_hz * float(sc.arrival_multiplier)
        if sc.emergency_bias > 0.0:
            # Do not change classes permanently. Temporarily increase emergency device arrival rates.
            emergency = (self.dev_class == 0)
            non_emergency = ~emergency
            self.sampling_hz[emergency] *= (1.0 + 2.0 * float(sc.emergency_bias))
            self.sampling_hz[non_emergency] *= max(0.15, 1.0 - float(sc.emergency_bias))
        if sc.best_effort_bias > 0.0:
            # Admission-pressure development stress: increase fresh best-effort
            # offered load without changing device classes or request accounting.
            best_effort = (self.dev_class == 2)
            other = ~best_effort
            self.sampling_hz[best_effort] *= (1.0 + 3.0 * float(sc.best_effort_bias))
            self.sampling_hz[other] *= max(0.25, 1.0 - 0.60 * float(sc.best_effort_bias))
        PHYLayer.cfg.sinr_mu_db = self._phy_base["sinr_mu_db"] + float(sc.sinr_mu_offset_db)
        PHYLayer.cfg.sinr_sigma_db = self._phy_base["sinr_sigma_db"] * float(sc.sinr_sigma_multiplier)
        PHYLayer.cfg.interf_db_mu = self._phy_base["interf_db_mu"] + float(sc.interference_offset_db)

    def _safe_low_priority_reject(self, dev: int, queue_load: float) -> bool:
        """Return whether one valid best-effort rejection remains inside intent budgets.

        The test uses projected all-arrival blocking and an optimistic upper bound
        on eventual service success. It therefore avoids the previous early-episode
        artifact where the realized success ratio was necessarily near zero and all
        rejection actions were masked at every epoch.
        """
        if dev < 0 or dev >= self.n_devices or not self.request_queues[int(dev)]:
            return False
        dev = int(dev)
        req = self.request_queues[dev][0]
        if req.admitted or int(self.dev_class[dev]) != 2:
            return False
        age = float(self.queue_age[dev])
        age_risk = age >= self.admission_shield_age_fraction * max(int(self.deadline_steps[dev]), 1)
        if age_risk or queue_load < self.admission_shield_queue_guard:
            return False

        intent = self.get_active_intent()
        arrivals = max(int(getattr(self, "total_arrivals_ep", 0)), 1)
        blocked = int(getattr(self, "blocked_arrivals_ep", 0))
        completed = int(getattr(self, "completed_requests_ep", 0))
        pending = int(sum(len(q) for q in self.request_queues))
        predicted_block = float((blocked + 1) / arrivals)
        projected_success_ceiling = float((completed + max(pending - 1, 0)) / arrivals)
        return (
            predicted_block <= float(intent.block_target)
            and projected_success_ceiling >= float(intent.min_service_success_rate or 0.0)
        )

    def get_action_mask(self) -> np.ndarray:
        """Return common feasibility plus the proposed safety filter.

        Physical feasibility is applied to every scheme. Only the additional
        rejection filtering below belongs to the proposed admission-safety shield.
        """
        mask = np.asarray(super().get_feasible_action_mask(), dtype=bool).copy()
        if not self.admission_shield_enabled:
            return mask
        candidates = self.get_candidate_devices()
        queue_load = float(np.count_nonzero(self.queue_bits > 0.0)) / max(float(self.n_devices), 1.0)
        for slot, dev in enumerate(candidates):
            if dev < 0 or not self.request_queues[int(dev)]:
                continue
            reject_action = self.encode_action(slot, MODE_REJECT)
            if not mask[reject_action]:
                continue  # already physically infeasible, not a shield intervention
            dev = int(dev)
            req = self.request_queues[dev][0]
            age = float(self.queue_age[dev])
            age_risk = age >= self.admission_shield_age_fraction * max(int(self.deadline_steps[dev]), 1)
            emergency = int(self.dev_class[dev]) == 0
            if emergency or age_risk or not self._safe_low_priority_reject(dev, queue_load):
                mask[reject_action] = False
        if not np.any(mask):
            mask = np.asarray(super().get_feasible_action_mask(), dtype=bool)
        return mask

    def _apply_admission_safety_shield(
        self,
        requested_action: int,
        candidates: np.ndarray,
        selected_dev: int,
        selected_backlog: bool,
        selected_age: float,
        queue_load: float,
    ) -> Tuple[int, int, str]:
        """Replace unsafe request rejection with protected or ordinary service."""
        action = int(requested_action)
        slot, mode, _ = self.decode_action(action, candidates)
        if not self.admission_shield_enabled or mode != MODE_REJECT or not selected_backlog:
            return action, 0, "none"
        self._shield_requests_ep += 1
        deadline_steps = int(self.deadline_steps[selected_dev]) if selected_dev >= 0 else 1
        admitted = bool(self.head_request_admitted(selected_dev)) if selected_dev >= 0 else False
        age_risk = selected_age >= self.admission_shield_age_fraction * max(deadline_steps, 1)
        emergency = selected_dev >= 0 and int(self.dev_class[selected_dev]) == 0
        intent = self.get_active_intent()
        # Fresh best-effort requests may be rejected only when the projected
        # all-arrival blocking and eventual-success budgets remain feasible.
        safe_low_priority_reject = self._safe_low_priority_reject(selected_dev, queue_load)
        if safe_low_priority_reject:
            return action, 0, "safe-best-effort-reject"
        executed_mode = MODE_PROTECT if (emergency or age_risk or admitted) else MODE_GRANT
        executed = self.encode_action(slot, executed_mode)
        self._shield_overrides_ep += 1
        return executed, 1, MODE_NAMES[executed_mode]

    def step(self, action: int):
        self._update_active_intent()
        intent = self.get_active_intent()
        candidates = self.get_candidate_devices()
        req_slot, req_mode, selected_dev = self.decode_action(int(action), candidates)
        selected_backlog = bool(selected_dev >= 0 and self.queue_bits[selected_dev] > 0.0)
        selected_age = float(self.queue_age[selected_dev]) if selected_backlog else 0.0
        queue_load_before = float(np.count_nonzero(self.queue_bits > 0.0)) / max(float(self.n_devices), 1.0)
        requested_action = int(action)
        feasible_mask = np.asarray(super().get_feasible_action_mask(), dtype=bool)
        action_mask = np.asarray(self.get_action_mask(), dtype=bool)
        feasibility_filtered_fraction = float(1.0 - np.mean(feasible_mask.astype(float)))
        shield_filtered_fraction = float(np.mean(feasible_mask & ~action_mask))
        executed_action, shield_override, shield_reason = self._apply_admission_safety_shield(
            requested_action, candidates, selected_dev, selected_backlog, selected_age, queue_load_before
        )
        _, executed_mode, _ = self.decode_action(executed_action, candidates)

        shock = self._in_shock_window()
        self._apply_stress_controls()
        obs, reward, done, info = super().step(executed_action)
        info = dict(info) if isinstance(info, dict) else {}
        if self._action_counts is not None and 0 <= executed_action < len(self._action_counts):
            self._action_counts[executed_action] += 1
        if getattr(self, "_mode_counts", None) is not None and 0 <= executed_mode < len(self._mode_counts):
            self._mode_counts[executed_mode] += 1

        qload = float(np.count_nonzero(self.queue_bits > 0.0)) / max(float(self.n_devices), 1.0)
        oldest_age = float(np.max(self.queue_age)) if np.any(self.queue_age > 0) else 0.0
        delivered = float(info.get("delivered_bits", 0.0) or 0.0)
        interrupted = int(info.get("interrupted", 0) or 0)
        deadline_exposure = float(info.get("deadline_exposure_rate", 0.0) or 0.0)
        service_success_rate = float(info.get("service_success_rate_so_far", 0.0) or 0.0)
        deadline_miss_proxy = int(info.get("deadline_exposed_new", 0) or 0) > 0

        info.update({
            "action": int(executed_action),
            "action_mode": int(executed_mode),
            "action_mode_name": MODE_NAMES.get(executed_mode, "unknown"),
            "requested_action": int(requested_action),
            "requested_action_mode": int(req_mode),
            "executed_action": int(executed_action),
            "shield_override": int(shield_override),
            "feasibility_filtered_fraction": feasibility_filtered_fraction,
            "shield_filtered_fraction": shield_filtered_fraction,
            "shield_reason": shield_reason,
            "shock_active": int(shock),
            "intent_name": intent.name,
            "queue_load_frac": qload,
            "oldest_age_steps": oldest_age,
            "focus_backlog": int(selected_backlog),
            "focus_age_steps": float(selected_age),
            "service_success": int((int(info.get("completed_requests", 0) or 0)) > 0),
            "service_success_rate_so_far": service_success_rate,
            "interrupted": int(interrupted),
            "deadline_target_ms": float(effective_deadline_ms(intent)),
            "deadline_exposure_rate": float(deadline_exposure),
            "deadline_miss_proxy": int(deadline_miss_proxy),
        })
        step_violation = local_step_violation(info, intent)
        info["agentic_step_violation"] = float(step_violation)

        self.agentic_trace.append({
            "t": int(self.t),
            "action": int(executed_action),
            "action_mode": int(executed_mode),
            "requested_action": int(requested_action),
            "requested_action_mode": int(req_mode),
            "shield_override": int(shield_override),
            "feasibility_filtered_fraction": feasibility_filtered_fraction,
            "shield_filtered_fraction": shield_filtered_fraction,
            "shield_reason": shield_reason,
            "active_intent": intent.name,
            "intent_switched": int(self.intent_switch_to is not None and intent.name == self.intent_switch_to.name),
            "reward": float(reward),
            "shock_active": int(shock),
            "sinr_db": float(info.get("sinr_db", np.nan)),
            "selected_dev_idx": int(info.get("selected_dev_idx", selected_dev)),
            "actual_target_dev_idx": int(info.get("actual_target_dev_idx", -1)),
            "delivered_bits": delivered,
            "energy_j": float(info.get("energy_j", 0.0) or 0.0),
            "delay_ms": float(info.get("delay_ms", np.nan)) if info.get("delay_ms", None) is not None else np.nan,
            "blocked": int(info.get("blocked", 0) or 0),
            "blocking_rate_so_far": float(info.get("blocking_rate_so_far", 0.0) or 0.0),
            "interrupted": int(interrupted),
            "interruption_rate_so_far": float(info.get("interruption_rate_so_far", 0.0) or 0.0),
            "preempted": int(info.get("preempted", 0) or 0),
            "scheduled": int(info.get("scheduled", 0) or 0),
            "service_success": int((int(info.get("completed_requests", 0) or 0)) > 0),
            "service_success_rate_so_far": service_success_rate,
            "deadline_exposure_rate": float(deadline_exposure),
            "deadline_miss_proxy": int(deadline_miss_proxy),
            "step_violation": float(step_violation),
            "queue_load_frac": qload,
            "oldest_age_steps": oldest_age,
        })
        if done:
            self.agentic_episode_summaries.append(self.agentic_episode_summary())
        return obs, float(reward), done, info

    def agentic_episode_summary(self) -> Dict[str, Any]:
        tr = list(self.agentic_trace)
        if not tr:
            return {}
        actions = np.asarray([x["action"] for x in tr], dtype=int)
        modes = np.asarray([x.get("action_mode", self.mode_from_action(x["action"])) for x in tr], dtype=int)
        shock_mask = np.asarray([bool(x["shock_active"]) for x in tr], dtype=bool)
        delivered = np.asarray([x["delivered_bits"] for x in tr], dtype=float)
        delay = np.asarray([x["delay_ms"] for x in tr], dtype=float)
        blocked = np.asarray([x["blocked"] for x in tr], dtype=float)
        interrupted = np.asarray([x["interrupted"] for x in tr], dtype=float)
        scheduled = np.asarray([x["scheduled"] for x in tr], dtype=float)
        energy = np.asarray([x["energy_j"] for x in tr], dtype=float)
        qload = np.asarray([x["queue_load_frac"] for x in tr], dtype=float)
        deadline_exposure = np.asarray([x["deadline_exposure_rate"] for x in tr], dtype=float)
        deadline_proxy = np.asarray([x["deadline_miss_proxy"] for x in tr], dtype=float)
        service_rate = np.asarray([x.get("service_success_rate_so_far", np.nan) for x in tr], dtype=float)
        shield = np.asarray([x.get("shield_override", 0) for x in tr], dtype=float)
        feasibility_filtered = np.asarray([x.get("feasibility_filtered_fraction", 0.0) for x in tr], dtype=float)
        feasibility_filter_active = (feasibility_filtered > 0.0).astype(float)
        shield_filtered = np.asarray([x.get("shield_filtered_fraction", 0.0) for x in tr], dtype=float)
        shield_filter_active = (shield_filtered > 0.0).astype(float)
        violation = np.asarray([x.get("step_violation", np.nan) for x in tr], dtype=float)

        counts = np.bincount(modes, minlength=int(getattr(self, "n_action_modes", 5))).astype(float)
        probs = counts / max(np.sum(counts), 1.0)
        nonzero = probs[probs > 0]
        action_entropy = float(-np.sum(nonzero * np.log(nonzero + EPS)) / max(math.log(len(probs)), EPS))
        grant_like = float(np.mean(np.isin(modes, [MODE_GRANT, MODE_PROTECT, MODE_COEXIST])))
        preempt_action_rate = float(np.mean(modes == MODE_PROTECT))
        deny_action_rate = float(np.mean(modes == MODE_REJECT))
        defer_action_rate = float(np.mean(modes == MODE_DEFER))

        def _mean_mask(x, mask):
            y = x[mask]
            y = y[np.isfinite(y)]
            return float(np.nanmean(y)) if y.size else np.nan

        def _last_finite(x):
            y = x[np.isfinite(x)]
            return float(y[-1]) if y.size else np.nan

        switch_mask = np.asarray([bool(x.get("intent_switched", 0)) for x in tr], dtype=bool)
        intent_action_shift_l1 = np.nan
        intent_mode_shift_l1 = np.nan
        if np.any(switch_mask) and np.any(~switch_mask):
            intent_action_shift_l1 = _action_shift_l1(actions, switch_mask, int(getattr(self, "n_actions", 20)))
            intent_mode_shift_l1 = _action_shift_l1(modes, switch_mask, int(getattr(self, "n_action_modes", 5)))
        switch_adaptation = np.nan
        switch_adaptation_success = np.nan
        post_switch_violation = np.nan
        switch_pre_reference = np.nan
        switch_post_reference = np.nan
        switch_settling_band = np.nan
        if np.any(switch_mask):
            first = int(np.flatnonzero(switch_mask)[0])
            settling = switch_settling_time_metrics(
                violation,
                switch_index=first,
                hold_steps=self.intent_switch_hold_steps,
                absolute_tolerance=self.intent_switch_tolerance,
                relative_tolerance=0.10,
            )
            post_switch_violation = float(settling["post_switch_violation_mean"])
            switch_adaptation = float(settling["adaptation_time_steps"])
            switch_adaptation_success = float(settling["adaptation_success"])
            switch_pre_reference = float(settling["pre_switch_reference"])
            switch_post_reference = float(settling["post_switch_reference"])
            switch_settling_band = float(settling["settling_band"])

        return {
            "action_entropy_norm": action_entropy,
            "dynamic_violation_mean": float(np.nanmean(violation)) if np.isfinite(violation).any() else np.nan,
            "dynamic_violation_step_cvar95": _cvar_upper(violation[np.isfinite(violation)], 0.95) if np.isfinite(violation).any() else np.nan,
            "grant_like_action_rate": grant_like,
            "preempt_action_rate": preempt_action_rate,
            "protect_action_rate": preempt_action_rate,
            "deny_action_rate": deny_action_rate,
            "defer_action_rate": defer_action_rate,
            "shield_override_rate": float(np.mean(shield)) if shield.size else np.nan,
            "feasibility_filter_active_rate": float(np.mean(feasibility_filter_active)) if feasibility_filter_active.size else np.nan,
            "feasibility_filtered_action_fraction": float(np.mean(feasibility_filtered)) if feasibility_filtered.size else np.nan,
            "shield_filter_active_rate": float(np.mean(shield_filter_active)) if shield_filter_active.size else np.nan,
            "shield_filtered_action_fraction": float(np.mean(shield_filtered)) if shield_filtered.size else np.nan,
            "service_success_rate": _last_finite(service_rate),
            "deadline_exposure_rate": _last_finite(deadline_exposure),
            "deadline_exposure_cvar95": _cvar_upper(deadline_exposure, 0.95),
            "deadline_miss_proxy_rate": float(np.nanmean(deadline_proxy)) if deadline_proxy.size else np.nan,
            "shock_action_shift_l1": _action_shift_l1(modes, shock_mask, int(getattr(self, "n_action_modes", 5))),
            "shock_throughput_bits_mean": _mean_mask(delivered, shock_mask),
            "pre_shock_throughput_bits_mean": _mean_mask(delivered, ~shock_mask),
            "shock_delay_ms_mean": _mean_mask(delay, shock_mask),
            "pre_shock_delay_ms_mean": _mean_mask(delay, ~shock_mask),
            "shock_block_rate": _safe_rate(blocked[shock_mask].sum(), max(1.0, shock_mask.sum())),
            "shock_interruption_rate": _safe_rate(interrupted[shock_mask].sum(), max(1.0, scheduled[shock_mask].sum())),
            "queue_load_mean": float(np.nanmean(qload)) if qload.size else np.nan,
            "energy_per_delivered_bit_step": _safe_rate(np.nansum(energy), max(np.nansum(delivered), EPS)),
            "intent_switch_active": int(np.any(switch_mask)),
            "intent_action_shift_l1": intent_action_shift_l1,
            "intent_mode_shift_l1": intent_mode_shift_l1,
            "post_switch_violation_mean": post_switch_violation,
            "switch_adaptation_time_steps": switch_adaptation,
            "switch_adaptation_success": switch_adaptation_success,
            "switch_pre_violation_reference": switch_pre_reference,
            "switch_post_violation_reference": switch_post_reference,
            "switch_settling_band": switch_settling_band,
        }

    def augment_episode_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Merge wrapper diagnostics into a standardized trainer episode row."""
        out = dict(metrics or {})
        if self.agentic_episode_summaries:
            out.update(self.agentic_episode_summaries[-1])
        return out


def _action_shift_l1(actions: np.ndarray, shock_mask: np.ndarray, n_actions: int) -> float:
    if actions.size == 0 or not np.any(shock_mask) or not np.any(~shock_mask):
        return np.nan
    p0 = np.bincount(actions[~shock_mask], minlength=n_actions).astype(float)
    p1 = np.bincount(actions[shock_mask], minlength=n_actions).astype(float)
    p0 = p0 / max(p0.sum(), 1.0)
    p1 = p1 / max(p1.sum(), 1.0)
    return float(np.sum(np.abs(p1 - p0)))


def _safe_rate(n: float, d: float) -> float:
    return float(n / d) if abs(float(d)) > EPS else np.nan


def _finite_array(values: List[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _cvar_upper(arr: np.ndarray, alpha: float = 0.95) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    q = np.nanquantile(arr, alpha)
    tail = arr[arr >= q]
    return float(np.nanmean(tail)) if tail.size else float(q)


def _cvar_lower(arr: np.ndarray, alpha: float = 0.10) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    q = np.nanquantile(arr, alpha)
    tail = arr[arr <= q]
    return float(np.nanmean(tail)) if tail.size else float(q)


def episode_violation(row: Dict[str, Any], intent: IntentProfile) -> Dict[str, float]:
    """Normalized per-episode violation components; zero means objective met."""
    block = float(row.get("blocking_prob", np.nan))
    intr = float(row.get("interrupt_prob", np.nan))
    delay = float(row.get("avg_delay_ms", np.nan))
    deadline_exposure = float(row.get("deadline_exposure_rate", np.nan))
    service_success = float(row.get("service_success_rate", np.nan))
    fair = float(row.get("jain_fairness", np.nan))
    ee = float(row.get("energy_eff_bpj", np.nan))
    util = float(row.get("utilization", np.nan))
    thr = float(row.get("throughput_mbps", np.nan))

    out = {
        "v_block": max(0.0, (block - intent.block_target) / max(intent.block_target, EPS)) if np.isfinite(block) else np.nan,
        "v_intr": max(0.0, (intr - intent.intr_target) / max(intent.intr_target, EPS)) if np.isfinite(intr) else np.nan,
        "v_deadline": np.nan,
        "v_service": 0.0,
        "v_fair": max(0.0, (intent.min_fairness - fair) / max(intent.min_fairness, EPS)) if np.isfinite(fair) else np.nan,
        "v_energy_eff": 0.0,
        "v_util": 0.0,
        "v_throughput": 0.0,
    }
    if intent.max_deadline_exposure_rate is not None and np.isfinite(deadline_exposure):
        out["v_deadline"] = max(0.0, (deadline_exposure - float(intent.max_deadline_exposure_rate)) / max(float(intent.max_deadline_exposure_rate), EPS))
    elif np.isfinite(delay):
        out["v_deadline"] = max(0.0, (delay - effective_deadline_ms(intent)) / max(effective_deadline_ms(intent), EPS))
    if intent.min_service_success_rate is not None:
        out["v_service"] = max(0.0, (float(intent.min_service_success_rate) - service_success) / max(float(intent.min_service_success_rate), EPS)) if np.isfinite(service_success) else np.nan
    if intent.min_energy_eff_bpj is not None:
        out["v_energy_eff"] = max(0.0, (intent.min_energy_eff_bpj - ee) / max(intent.min_energy_eff_bpj, EPS)) if np.isfinite(ee) else np.nan
    if intent.min_utilization is not None:
        out["v_util"] = max(0.0, (intent.min_utilization - util) / max(intent.min_utilization, EPS)) if np.isfinite(util) else np.nan
    if intent.throughput_floor_mbps is not None:
        out["v_throughput"] = max(0.0, (intent.throughput_floor_mbps - thr) / max(intent.throughput_floor_mbps, EPS)) if np.isfinite(thr) else np.nan
    return out


def _violation_pairs(v: Dict[str, float], intent: IntentProfile):
    w = intent.weights or default_violation_weights()
    return [
        (v["v_block"], w.get("blocking_prob", 1.0)),
        (v["v_intr"], w.get("interrupt_prob", 1.0)),
        (v["v_deadline"], w.get("deadline_exposure", w.get("avg_delay_ms", 1.0))),
        (v["v_service"], w.get("service_success", 0.5)),
        (v["v_fair"], w.get("jain_fairness", 1.0)),
        (v["v_energy_eff"], w.get("energy_eff_bpj", 1.0)),
        (v["v_util"], w.get("utilization", 1.0)),
        (v["v_throughput"], w.get("throughput_mbps", 1.0)),
    ]


def weighted_violation_score(row: Dict[str, Any], intent: IntentProfile) -> float:
    pairs = _violation_pairs(episode_violation(row, intent), intent)
    num = sum(float(wt) * float(val) for val, wt in pairs if np.isfinite(val))
    den = sum(float(wt) for val, wt in pairs if np.isfinite(val))
    return float(num / den) if den > EPS else np.nan


def soft_intent_compliance_score(row: Dict[str, Any], intent: IntentProfile) -> float:
    pairs = _violation_pairs(episode_violation(row, intent), intent)
    num = sum(float(wt) * max(0.0, 1.0 - float(val)) for val, wt in pairs if np.isfinite(val))
    den = sum(float(wt) for val, wt in pairs if np.isfinite(val))
    return float(np.clip(num / den, 0.0, 1.0)) if den > EPS else np.nan

def summarize_agentic_metrics(
    metrics: Dict[str, Any],
    intent: IntentProfile,
    env_summary_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute fine-grained evaluation metrics from standardized per-episode logs."""
    eps_rows = metrics.get("episode_metrics_std", []) or []
    if not eps_rows:
        return {}

    keys = [
        "throughput_mbps", "avg_delay_ms", "blocking_prob", "interrupt_prob",
        "deadline_exposure_rate", "deadline_miss_proxy_rate", "service_success_rate", "unfinished_rate",
        "right_censored_rate", "pending_after_deadline_rate", "shield_override_rate",
        "measurement_service_success_rate", "measurement_deadline_exposure_rate",
        "measurement_unfinished_rate", "measurement_right_censored_rate",
        "measurement_blocking_prob", "measurement_throughput_mbps",
        "drain_added_completions", "drain_added_deadline_exposures", "drain_removed_unfinished_requests",
        "terminal_drain_requested_steps", "terminal_drain_steps", "terminal_drain_extension_steps",
        "terminal_total_steps", "max_realized_device_deadline_steps", "measurement_rng_stream_preserved",
        "energy_j", "energy_eff_bpj", "jain_fairness", "utilization", "bits_tx_ep",
        "total_arrivals_ep", "completed_requests_ep", "pending_requests_ep", "request_conservation_error",
        "scheduled_tx_ep", "preemptions_ep", "slots_used_ep",
    ]
    series = {k: _finite_array([r.get(k, np.nan) for r in eps_rows]) for k in keys}

    violation_scores = _finite_array([weighted_violation_score(r, intent) for r in eps_rows])
    soft_scores = _finite_array([soft_intent_compliance_score(r, intent) for r in eps_rows])
    relaxed_scores = _finite_array([(weighted_violation_score(r, intent) <= 0.10) for r in eps_rows])
    satisfied = violation_scores <= 1e-12
    block = series["blocking_prob"]
    intr = series["interrupt_prob"]
    delay = series["avg_delay_ms"]
    thr = series["throughput_mbps"]
    fair = series["jain_fairness"]
    deadline_exposure = series["deadline_exposure_rate"]
    deadline_miss_proxy = series["deadline_miss_proxy_rate"]
    service_success = series["service_success_rate"]
    unfinished = series["unfinished_rate"]
    right_censored = series["right_censored_rate"]
    pending_after_deadline = series["pending_after_deadline_rate"]
    measurement_success = series["measurement_service_success_rate"]
    measurement_deadline = series["measurement_deadline_exposure_rate"]
    measurement_unfinished = series["measurement_unfinished_rate"]
    measurement_censored = series["measurement_right_censored_rate"]
    measurement_blocking = series["measurement_blocking_prob"]
    measurement_thr = series["measurement_throughput_mbps"]
    drain_added_completions = series["drain_added_completions"]
    drain_added_deadline = series["drain_added_deadline_exposures"]
    drain_removed_unfinished = series["drain_removed_unfinished_requests"]
    drain_requested = series["terminal_drain_requested_steps"]
    drain_actual = series["terminal_drain_steps"]
    drain_extension = series["terminal_drain_extension_steps"]
    terminal_total_steps = series["terminal_total_steps"]
    max_realized_deadline = series["max_realized_device_deadline_steps"]
    measurement_rng_preserved = series["measurement_rng_stream_preserved"]
    conservation = series["request_conservation_error"]
    shield_override = series["shield_override_rate"]
    ee = series["energy_eff_bpj"]
    util = series["utilization"]
    bits = series["bits_tx_ep"]
    energy = series["energy_j"]

    def mean(x): return float(np.nanmean(x)) if x.size else np.nan
    def std(x): return float(np.nanstd(x)) if x.size else np.nan
    def q(x, p): return float(np.nanquantile(x, p)) if x.size else np.nan

    reliability_margin = []
    n = min(len(block), len(intr))
    for i in range(n):
        reliability_margin.append(min(intent.block_target - block[i], intent.intr_target - intr[i]))
    reliability_margin = _finite_array(reliability_margin)

    # Oscillation and stability diagnostics.
    reward = _finite_array(metrics.get("rewards_per_episode", []) or [])
    qmax = _finite_array(metrics.get("q_value_max", []) or [])
    lambda_block = _finite_array(metrics.get("lambda_block", []) or [])
    lambda_intr = _finite_array(metrics.get("lambda_intr", []) or [])
    lambda_energy = _finite_array(metrics.get("lambda_energy", []) or [])

    lambdas = [x for x in [lambda_block, lambda_intr, lambda_energy] if x.size]
    lambda_pressure = float(sum(np.nanmean(x) for x in lambdas)) if lambdas else np.nan
    lambda_terminal_pressure = float(sum(x[-1] for x in lambdas if x.size)) if lambdas else np.nan

    intent_regret_auc = float(np.nansum(violation_scores)) if violation_scores.size else np.nan
    positive_viol = violation_scores[violation_scores > 0]

    # Agentic summaries from the environment wrapper.
    env_summary_rows = env_summary_rows or []
    env_keys = [
        "action_entropy_norm", "dynamic_violation_mean", "dynamic_violation_step_cvar95",
        "grant_like_action_rate", "preempt_action_rate", "deny_action_rate",
        "feasibility_filter_active_rate", "feasibility_filtered_action_fraction",
        "shield_filter_active_rate", "shield_filtered_action_fraction",
        "shock_action_shift_l1", "shock_throughput_bits_mean", "pre_shock_throughput_bits_mean",
        "shock_delay_ms_mean", "pre_shock_delay_ms_mean", "shock_block_rate", "shock_interruption_rate",
        "queue_load_mean", "energy_per_delivered_bit_step", "intent_switch_active",
        "intent_action_shift_l1", "intent_mode_shift_l1",
        "counterfactual_intent_policy_l1", "counterfactual_mode_policy_l1",
        "counterfactual_argmax_change_rate", "frozen_policy_entropy_norm",
        "pre_mask_reject_probability", "feasible_reject_probability", "physical_infeasible_probability_mass",
        "pre_mask_unsafe_probability_mass", "shield_filtered_probability_mass",
        "shield_filtered_raw_probability_mass", "actual_shield_prevention_rate",
        "stochastic_action_entropy", "stochastic_action_entropy_nats",
        "policy_mode_prob_defer", "policy_mode_prob_grant", "policy_mode_prob_protect",
        "policy_mode_prob_coexist", "policy_mode_prob_reject",
        "intent_mode_prob__balanced_agentic__defer", "intent_mode_prob__balanced_agentic__grant",
        "intent_mode_prob__balanced_agentic__protect", "intent_mode_prob__balanced_agentic__coexist",
        "intent_mode_prob__balanced_agentic__reject",
        "intent_mode_prob__reliability_first__defer", "intent_mode_prob__reliability_first__grant",
        "intent_mode_prob__reliability_first__protect", "intent_mode_prob__reliability_first__coexist",
        "intent_mode_prob__reliability_first__reject",
        "intent_mode_prob__latency_critical__defer", "intent_mode_prob__latency_critical__grant",
        "intent_mode_prob__latency_critical__protect", "intent_mode_prob__latency_critical__coexist",
        "intent_mode_prob__latency_critical__reject",
        "post_switch_violation_mean", "switch_adaptation_time_steps", "switch_adaptation_success",
        "switch_pre_violation_reference", "switch_post_violation_reference", "switch_settling_band",
    ]
    env_mean = {}
    for k in env_keys:
        vals = _finite_array([r.get(k, np.nan) for r in env_summary_rows])
        env_mean[k] = mean(vals)

    # Static verification uses exact final request-level episode metrics. During
    # an intent-switch experiment, a single terminal intent cannot represent the
    # whole episode; use the per-episode dynamic violation generated under the
    # active intent at each epoch. CVaR95 is still taken across held-out episodes.
    switch_flags = _finite_array([r.get("intent_switch_active", 0.0) for r in env_summary_rows])
    dynamic_scores = _finite_array([r.get("dynamic_violation_mean", np.nan) for r in env_summary_rows])
    if switch_flags.size and np.nanmax(switch_flags) > 0.0 and dynamic_scores.size == len(eps_rows):
        violation_scores = dynamic_scores
        soft_scores = np.clip(1.0 - violation_scores, 0.0, 1.0)
        relaxed_scores = (violation_scores <= 0.10).astype(float)
        satisfied = violation_scores <= 1e-12
        intent_regret_auc = float(np.nansum(violation_scores))
        positive_viol = violation_scores[violation_scores > 0]

    return {
        "intent_name": intent.name,
        "episodes": int(len(eps_rows)),
        "intent_satisfaction_rate": mean(satisfied.astype(float)) if satisfied.size else np.nan,
        "soft_intent_compliance_score": mean(soft_scores),
        "soft_intent_compliance_p05": q(soft_scores, 0.05),
        "relaxed_intent_satisfaction_rate": mean(relaxed_scores.astype(float)) if relaxed_scores.size else np.nan,
        "intent_violation_rate": 1.0 - mean(satisfied.astype(float)) if satisfied.size else np.nan,
        "weighted_violation_mean": mean(violation_scores),
        "weighted_violation_p95": q(violation_scores, 0.95),
        "weighted_violation_cvar95": _cvar_upper(violation_scores, 0.95),
        "intent_regret_auc": intent_regret_auc,
        "positive_violation_mean": mean(positive_viol),
        "reliability_safety_margin_mean": mean(reliability_margin),
        "reliability_safety_margin_p05": q(reliability_margin, 0.05),
        "blocking_mean": mean(block),
        "blocking_p95": q(block, 0.95),
        "blocking_cvar95": _cvar_upper(block, 0.95),
        "interruption_mean": mean(intr),
        "interruption_p95": q(intr, 0.95),
        "interruption_cvar95": _cvar_upper(intr, 0.95),
        "deadline_exposure_mean": mean(deadline_exposure),
        "deadline_exposure_cvar95": _cvar_upper(deadline_exposure, 0.95),
        "deadline_miss_proxy_mean": mean(deadline_miss_proxy),
        "service_success_rate_mean": mean(service_success),
        "unfinished_rate_mean": mean(unfinished),
        "right_censored_rate_mean": mean(right_censored),
        "pending_after_deadline_rate_mean": mean(pending_after_deadline),
        "measurement_service_success_rate_mean": mean(measurement_success),
        "measurement_deadline_exposure_mean": mean(measurement_deadline),
        "measurement_unfinished_rate_mean": mean(measurement_unfinished),
        "measurement_right_censored_rate_mean": mean(measurement_censored),
        "measurement_blocking_mean": mean(measurement_blocking),
        "measurement_throughput_mean_mbps": mean(measurement_thr),
        "drain_added_completions_mean": mean(drain_added_completions),
        "drain_added_deadline_exposures_mean": mean(drain_added_deadline),
        "drain_removed_unfinished_requests_mean": mean(drain_removed_unfinished),
        "terminal_drain_requested_steps_mean": mean(drain_requested),
        "terminal_drain_actual_steps_mean": mean(drain_actual),
        "terminal_drain_actual_steps_max": float(np.nanmax(drain_actual)) if drain_actual.size else np.nan,
        "terminal_drain_extension_steps_mean": mean(drain_extension),
        "terminal_drain_extension_steps_max": float(np.nanmax(drain_extension)) if drain_extension.size else np.nan,
        "terminal_total_steps_mean": mean(terminal_total_steps),
        "max_realized_device_deadline_steps_mean": mean(max_realized_deadline),
        "max_realized_device_deadline_steps_max": float(np.nanmax(max_realized_deadline)) if max_realized_deadline.size else np.nan,
        "measurement_rng_stream_preserved": float(np.nanmin(measurement_rng_preserved)) if measurement_rng_preserved.size else np.nan,
        "request_conservation_error_max_abs": float(np.nanmax(np.abs(conservation))) if conservation.size else np.nan,
        "shield_override_rate_mean": mean(shield_override),
        "delay_mean_ms": mean(delay),
        "delay_p95_ms": q(delay, 0.95),
        "delay_p99_ms": q(delay, 0.99),
        "delay_cvar95_ms": _cvar_upper(delay, 0.95),
        "throughput_mean_mbps": mean(thr),
        "throughput_p05_mbps": q(thr, 0.05),
        "throughput_cvar_low10_mbps": _cvar_lower(thr, 0.10),
        "throughput_stability_cv": float(std(thr) / max(mean(thr), EPS)) if thr.size else np.nan,
        "energy_eff_mean_bpj": mean(ee),
        "energy_eff_p05_bpj": q(ee, 0.05),
        "energy_per_mbit_j": float(np.nansum(energy) / max(np.nansum(bits) / 1e6, EPS)) if energy.size and bits.size else np.nan,
        "energy_delay_product": float(mean(energy) * mean(delay)) if energy.size and delay.size else np.nan,
        "fairness_mean": mean(fair),
        "fairness_p05": q(fair, 0.05),
        "fairness_deficit_mean": mean(np.maximum(0.0, intent.min_fairness - fair)) if fair.size else np.nan,
        "utilization_mean": mean(util),
        "utilization_p95": q(util, 0.95),
        "reward_mean": mean(reward),
        "reward_stability_cv": float(std(reward) / max(abs(mean(reward)), EPS)) if reward.size else np.nan,
        "qmax_terminal": float(qmax[-1]) if qmax.size else np.nan,
        "qmax_oscillation_std": std(np.diff(qmax)) if qmax.size > 1 else np.nan,
        "lambda_pressure_mean": lambda_pressure,
        "lambda_terminal_pressure": lambda_terminal_pressure,
        **env_mean,
    }


def make_agentic_env(
    intent: IntentProfile,
    scenario: StressScenario,
    steps_per_ep: int,
    n_devices: int,
    seed: int,
    arrival_scale: float,
    class_mix: Tuple[float, float, float],
    burst_k: Optional[float],
    device_skew: float,
    bits_scale: float,
    delay_penalty_w: float,
    energy_penalty_w: float,
    prio_w: Tuple[float, float, float],
    intent_switch_to: Optional[IntentProfile] = None,
    intent_switch_frac: float = 0.50,
    intent_switch_hold_steps: int = 10,
    intent_switch_tolerance: float = 0.05,
) -> AgenticVIoTEnv:
    return AgenticVIoTEnv(
        steps_per_ep=steps_per_ep,
        n_devices=n_devices,
        seed=seed,
        arrival_scale=arrival_scale,
        class_mix=class_mix,
        burst_k=burst_k,
        device_skew=device_skew,
        bits_scale=bits_scale,
        delay_penalty_w=delay_penalty_w,
        energy_penalty_w=energy_penalty_w,
        prio_weights=prio_w,
        enable_dynamic_priority=True,
        enable_channel_realism=True,
        enable_device_heterogeneity=True,
        intent=intent,
        scenario=scenario,
        intent_switch_to=intent_switch_to,
        intent_switch_frac=intent_switch_frac,
        intent_switch_hold_steps=intent_switch_hold_steps,
        intent_switch_tolerance=intent_switch_tolerance,
    )
