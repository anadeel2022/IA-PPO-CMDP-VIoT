# hiot_qlearning.py — tabular Q-learning with robust observation discretization
from __future__ import annotations
import numpy as np
from collections import defaultdict
from hiot_utils import (
    new_episode_acc, accumulate_from_info,
    finalize_episode_metrics, MetricKeys
)

def hiot_qlearning_train(
    env,
    device=None,
    episodes: int = 300,
    steps_per_ep: int = 150,
    gamma: float = 0.99,
    alpha: float = 0.2,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
    eps_decay: float = 0.995,
    n_bins: int = 8,
    rng: np.random.Generator | None = None,
):
    """
    Tabular Q-learning for HIoT envs with continuous observations.
    - Discretizes obs → integer tuple keys
    - Epsilon-greedy with random tie-breaks
    - Logs per-episode metrics for downstream plots
    - Additionally records standardized per-episode metrics with hiot_utils
    """
    rng = rng or np.random.default_rng(12345)
    n_actions = int(getattr(env, "n_actions", getattr(env, "action_space_n", 5)))

    # ---------- helpers ----------
    def _unwrap_reset(ret):
        # Accept obs or (obs, info)
        if isinstance(ret, tuple) and len(ret) >= 1:
            return ret[0]
        return ret

    def _obs_to_key(obs):
        """
        Discretize features to n_bins over [0,1], return hashable integer tuple.
        """
        x = np.asarray(obs, dtype=float).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)
        k = np.minimum((x * n_bins).astype(int), n_bins - 1)
        return tuple(int(v) for v in k)

    def _greedy_with_tie_break(qrow: np.ndarray) -> int:
        vmax = np.max(qrow)
        best = np.flatnonzero(qrow == vmax)
        return int(rng.choice(best))  # random tie-break

    # Q-table: dict[state_key] -> np.array(n_actions)
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    # ---------- logging buffers (existing behavior) ----------
    rewards_per_episode = []
    throughput_per_episode = []
    delay_per_episode = []
    energy_usage_per_episode = []
    energy_efficiency_per_episode = []
    fairness_per_episode = []
    denies_per_episode = []
    arrivals_per_episode = []
    q_value_max = []

    # ---------- standardized per-episode metrics ----------
    episode_metrics_std: list[dict] = []

    eps = float(eps_start)

    for ep in range(episodes):
        obs0 = _unwrap_reset(env.reset())
        sk = _obs_to_key(obs0)

        # standardized accumulator and per-ep Q max
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        qmax_this_ep = -1e9

        ep_reward = 0.0
        ep_thr_bits = 0.0
        ep_delay_ms = 0.0
        ep_energy_j = 0.0
        ep_eff = 0.0
        ep_fair = np.nan  # take last value in episode if provided
        ep_denies = 0.0
        ep_arrivals = 0.0

        done = False
        for t in range(steps_per_ep):
            # epsilon-greedy
            if rng.random() < eps or np.all(Q[sk] == 0.0):
                a = int(rng.integers(n_actions))
            else:
                a = _greedy_with_tie_break(Q[sk])

            obs1, r, done, info = env.step(a)
            sk1 = _obs_to_key(obs1)

            # track per-state-row Q max for this episode (optional)
            try:
                qmax_this_ep = max(qmax_this_ep, float(np.max(Q[sk])))
            except Exception:
                pass

            # TD update
            target = r + (0.0 if done else gamma * float(np.max(Q[sk1])))
            td = target - Q[sk][a]
            Q[sk][a] += alpha * td

            # accumulate standardized info
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))

            # accumulate legacy episode logs (robust to missing/None keys)
            ep_reward   += float(r)
            ep_thr_bits += float(info.get("throughput_bits", 0.0))
            dval = info.get("delay_ms", 0.0)
            ep_delay_ms += float(dval) if dval is not None else 0.0
            ep_energy_j += float(info.get("energy_j", 0.0))
            ep_eff      += float(info.get("energy_efficiency", 0.0))
            if "fairness" in info:
                ep_fair = float(info["fairness"])
            ep_denies   += float(info.get("denies", 0.0))
            ep_arrivals += float(info.get("arrivals", 0.0))

            sk = sk1
            if done:
                break

        # finalize standardized per-episode metrics
        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc
        )
        if qmax_this_ep > -1e8:
            m[MetricKeys.QMAX] = qmax_this_ep
        episode_metrics_std.append(m)

        # If the env exposes end-of-episode rollups, prefer those for legacy logs
        if hasattr(env, "episode_metrics") and isinstance(env.episode_metrics, dict):
            em = env.episode_metrics
            ep_thr_bits = float(em.get("throughput_bits", ep_thr_bits))
            ep_delay_ms = float(em.get("delay_ms", ep_delay_ms))
            ep_energy_j = float(em.get("energy_j", ep_energy_j))
            ep_eff      = float(em.get("energy_efficiency", ep_eff))
            ep_fair     = float(em.get("fairness", np.nan if np.isnan(ep_fair) else ep_fair))
            ep_denies   = float(em.get("denies", ep_denies))
            ep_arrivals = float(em.get("arrivals", ep_arrivals))

        # legacy episode logs
        rewards_per_episode.append(ep_reward)
        throughput_per_episode.append(ep_thr_bits)
        delay_per_episode.append(ep_delay_ms)
        energy_usage_per_episode.append(ep_energy_j)
        energy_efficiency_per_episode.append(ep_eff)
        fairness_per_episode.append(float(np.nan_to_num(ep_fair, nan=0.0)))
        denies_per_episode.append(ep_denies)
        arrivals_per_episode.append(ep_arrivals)

        # track max Q across table for convergence plot
        qmax = max((float(np.max(v)) for v in Q.values()), default=0.0)
        q_value_max.append(qmax)

        # decay epsilon
        eps = max(eps_end, eps * eps_decay)

    class _QPolicy:
        def act(self, obs):
            key = _obs_to_key(obs)
            return int(np.argmax(Q[key]))

    # summary scalars (some plots/tables read these)
    metrics = dict(
        rewards_per_episode=rewards_per_episode,
        throughput_per_episode=throughput_per_episode,
        delay_per_episode=delay_per_episode,
        energy_usage_per_episode=energy_usage_per_episode,
        energy_efficiency_per_episode=energy_efficiency_per_episode,
        fairness_per_episode=fairness_per_episode,
        denies_per_episode=denies_per_episode,
        arrivals_per_episode=arrivals_per_episode,
        q_value_max=q_value_max,
        avg_throughput=float(np.nanmean(throughput_per_episode)) if throughput_per_episode else 0.0,
        avg_delay_ms=float(np.nanmean(delay_per_episode)) if delay_per_episode else 0.0,
        avg_fairness=float(np.nanmean(fairness_per_episode)) if fairness_per_episode else 0.0,
        avg_energy_efficiency=float(np.nanmean(energy_efficiency_per_episode)) if energy_efficiency_per_episode else 0.0,
        episode_metrics_std=episode_metrics_std,
        policy=_QPolicy(),
    )

    # ---- mirror standardized metrics into legacy series for plotting ----
    try:
        eps_std = metrics.get("episode_metrics_std", []) or []
        if eps_std:
            def _to_series_std(key):
                return [float(m.get(key, float('nan'))) for m in eps_std]
            metrics["throughput_per_episode"] = _to_series_std("throughput_mbps")
            metrics["delay_per_episode"] = _to_series_std("avg_delay_ms")
            metrics["energy_usage_per_episode"] = _to_series_std("energy_j")
            metrics["energy_efficiency_per_episode"] = _to_series_std("energy_eff_bpj")
            metrics["fairness_per_episode"] = _to_series_std("jain_fairness")
            metrics["denies_per_episode"] = _to_series_std("blocked_arrivals_ep")
            metrics["arrivals_per_episode"] = _to_series_std("total_arrivals_ep")
            metrics["preempt_counts_per_episode"] = _to_series_std("preemptions_ep")
    except Exception:
        pass

    return metrics
