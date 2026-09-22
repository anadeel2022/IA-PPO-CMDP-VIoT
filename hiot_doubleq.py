# hiot_doubleq.py — Double Q-Learning with stable state aggregation and exploration
from __future__ import annotations
import time
import numpy as np
from typing import Any, Tuple, Dict, List

from hiot_utils import (
    new_episode_acc, accumulate_from_info,
    finalize_episode_metrics, MetricKeys
)

# -------------------------
# Utilities
# -------------------------

def _unwrap_obs(x: Any):
    if isinstance(x, (tuple, list)) and len(x) >= 1:
        return x[0]
    return x

def _to_flat_array(obs: Any) -> np.ndarray:
    obs = _unwrap_obs(obs)
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=np.float32).ravel()
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32, copy=False).ravel()
    try:
        return np.array([float(obs)], dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)

def _state_key(obs: Any, binsize: float = 0.2, clip: float = 5.0) -> Tuple:
    """
    Quantize each feature by 'binsize' after clipping to [-clip, clip].
    This yields a compact, well-visited tabular state space, reducing variance.
    """
    arr = _to_flat_array(obs)
    if arr.size == 0:
        return tuple()
    if clip is not None and clip > 0:
        np.clip(arr, -clip, clip, out=arr)
    if binsize is None or binsize <= 0:
        # fallback to coarse rounding if needed
        arr = np.round(arr, 1)
    else:
        arr = np.floor(arr / float(binsize)) * float(binsize)
    return tuple(arr.tolist())

# -------------------------
# Trainer
# -------------------------

def hiot_doubleq_train(env,
                       episodes: int = 300,
                       device=None,
                       steps_per_ep: int = 150,
                       alpha: float = 0.2,
                       alpha_min: float = 0.01,
                       gamma: float = 0.95,
                       eps_start: float = 1.0,
                       eps_end: float = 0.02,
                       eps_decay_episodes: int = 300,
                       key_binsize: float = 0.25,
                       key_clip: float = 5.0,
                       seed: int | None = None,
                       # simple CMDP-style extras (match PPO defaults)
                       constr_block_target: float | None = 0.02,
                       constr_intr_target: float | None = 0.01,
                       constr_energy_j_target: float | None = None,
                       constr_lr: float = 1e-3) -> Dict[str, List[float]]:
    """
    Double Q-Learning on HIoTEnv with:
      • State aggregation by feature-wise binning (binsize, clip) for stability.
      • Linear epsilon decay over 'eps_decay_episodes' episodes.
      • Terminal masking of the bootstrap term.
      • Per-(s,a) decaying step size: alpha_t = max(alpha_min, alpha / sqrt(1+N(s,a))).
      • Convergence diagnostic from Q̄ = (Q1 + Q2) / 2.
      • Optional CMDP-style shaping using per-step constraint signals and episode metrics.
    """
    rng = np.random.default_rng(seed)
    nA = int(getattr(env, "n_actions", getattr(env, "action_space_n", 5)))

    # Two Q-tables and visit counts
    Q1: Dict[Tuple, np.ndarray] = {}
    Q2: Dict[Tuple, np.ndarray] = {}
    Nsa: Dict[Tuple[Tuple, int], int] = {}

    def _stepsize(sk: Tuple, a: int) -> float:
        key = (sk, a)
        Nsa[key] = Nsa.get(key, 0) + 1
        return float(max(alpha_min, alpha / np.sqrt(Nsa[key])))

    def _eps_for_ep(ep: int) -> float:
        if eps_decay_episodes <= 0:
            return eps_end
        frac = min(1.0, ep / float(eps_decay_episodes))
        return float(eps_start + (eps_end - eps_start) * frac)

    # primal–dual multipliers and histories
    lam_block = 0.0
    lam_intr = 0.0
    lam_energy = 0.0
    lambda_block_hist: List[float] = []
    lambda_intr_hist: List[float] = []
    lambda_energy_hist: List[float] = []

    def _update_lambda(lam: float, metric_val: float, target: float | None) -> float:
        """Episode-level projected dual ascent on constraint E[g(x)] <= target."""
        if target is None:
            return lam
        try:
            v = float(metric_val)
        except Exception:
            return lam
        if not np.isfinite(v):
            return lam
        if constr_lr <= 0.0:
            return lam
        g = v - float(target)
        lam_new = lam + constr_lr * g
        return max(0.0, float(lam_new))

    # Metrics
    rewards_per_episode: List[float] = []
    throughput_per_episode: List[float] = []
    delay_per_episode: List[float] = []
    energy_efficiency_per_episode: List[float] = []
    fairness_per_episode: List[float] = []
    denies_per_episode: List[float] = []
    arrivals_per_episode: List[float] = []
    q_value_max: List[float] = []
    episode_metrics_std: List[dict] = []

    t0 = time.time()
    converged_ep = None

    for ep in range(episodes):
        eps = _eps_for_ep(ep)

        obs0 = env.reset()
        obs = _unwrap_obs(obs0)
        sk = _state_key(obs, key_binsize, key_clip)

        if sk not in Q1:
            Q1[sk] = np.zeros(nA, dtype=np.float32)
            Q2[sk] = np.zeros(nA, dtype=np.float32)

        ep_reward_env = 0.0   # sum of raw env rewards (for logging / convergence)
        ep_qmax = -np.inf
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        qmax_this_ep = -1e9

        for t in range(steps_per_ep):
            # epsilon-greedy over Q̄
            if rng.random() < eps:
                a = int(rng.integers(nA))
            else:
                qbar = (Q1[sk] + Q2[sk]) * 0.5
                maxq = float(np.max(qbar))
                # break ties uniformly to avoid bias
                idx = np.flatnonzero(np.abs(qbar - maxq) < 1e-12)
                a = int(rng.choice(idx))

            # environment step
            obs2, env_r, done, info = env.step(a)
            ep_reward_env += float(env_r)

            # CMDP-style shaping using per-step constraint signals
            blocked_step = float((info.get("blocked", info.get("deny", 0)) or 0))
            preempt_step = float((info.get("preempt", 0) or 0))
            energy_step = float((info.get("energy_j", info.get("energy", 0.0)) or 0.0))
            penalty = lam_block * blocked_step + lam_intr * preempt_step + lam_energy * energy_step
            r = float(np.clip(env_r - penalty, -2.0, 2.0))

            sk2 = _state_key(obs2, key_binsize, key_clip)
            if sk2 not in Q1:
                Q1[sk2] = np.zeros(nA, dtype=np.float32)
                Q2[sk2] = np.zeros(nA, dtype=np.float32)

            # Update exactly one table with terminal masking
            if rng.random() < 0.5:
                a_star = int(np.argmax(Q1[sk2]))
                target = float(r) + (0.0 if done else gamma * float(Q2[sk2][a_star]))
                eta = _stepsize(sk, a)
                Q1[sk][a] += eta * (target - float(Q1[sk][a]))
            else:
                a_star = int(np.argmax(Q2[sk2]))
                target = float(r) + (0.0 if done else gamma * float(Q1[sk2][a_star]))
                eta = _stepsize(sk, a)
                Q2[sk][a] += eta * (target - float(Q2[sk][a]))

            # Convergence diagnostic from the averaged table
            qbar_now = (Q1[sk] + Q2[sk]) * 0.5
            maxq_now = float(np.max(qbar_now))
            if np.isfinite(maxq_now):
                ep_qmax = max(ep_qmax, maxq_now)
                qmax_this_ep = max(qmax_this_ep, maxq_now)

            # accumulate standardized env stats
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))

            sk = sk2
            obs = _unwrap_obs(obs2)
            if done:
                break

        rewards_per_episode.append(ep_reward_env)
        q_value_max.append(ep_qmax if np.isfinite(ep_qmax) else 0.0)

        # Copy env-driven episode summaries without changing your schema
        total_bits = float(getattr(env, "total_bits_tx", 0.0))
        total_energy = float(getattr(env, "total_energy_j", 0.0))
        throughput_per_episode.append(total_bits)
        delay_list = getattr(env, "delay_steps_ep", []) or [0.0]
        delay_per_episode.append(float(np.nan_to_num(np.mean(delay_list), nan=0.0)))
        ee = total_bits / (total_energy if total_energy > 0 else 1e-12)
        energy_efficiency_per_episode.append(float(ee))

        fairness_list = getattr(env, "fairness_ep", [])
        if isinstance(fairness_list, (list, tuple)) and len(fairness_list) > 0:
            fairness_per_episode.append(float(np.nan_to_num(np.mean(fairness_list), nan=0.0)))
        else:
            fairness_per_episode.append(float(np.nan_to_num(getattr(env, "fairness", 0.0), nan=0.0)))

        denies_per_episode.append(float(getattr(env, "denies_this_ep", 0.0)))
        arrivals_per_episode.append(float(getattr(env, "arrivals_this_ep", 0.0)))

        # standardized episode metrics (through hiot_utils)
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

        # primal–dual λ updates based on episode metrics
        blk_val = m.get(MetricKeys.BLOCK, np.nan)
        intr_val = m.get(MetricKeys.INTR, np.nan)
        energy_val = m.get(MetricKeys.ENERGY_J, np.nan)
        lam_block = _update_lambda(lam_block, blk_val, constr_block_target)
        lam_intr = _update_lambda(lam_intr, intr_val, constr_intr_target)
        lam_energy = _update_lambda(lam_energy, energy_val, constr_energy_j_target)
        lambda_block_hist.append(float(lam_block))
        lambda_intr_hist.append(float(lam_intr))
        lambda_energy_hist.append(float(lam_energy))

    class _DoubleQPolicy:
        def act(self, obs):
            sk = _state_key(obs, key_binsize, key_clip)
            q1 = Q1.get(sk)
            q2 = Q2.get(sk)
            if q1 is None or q2 is None:
                return 0
            return int(np.argmax((q1 + q2) * 0.5))

    train_time_s = time.time() - t0
    out = dict(
        rewards_per_episode=rewards_per_episode,
        throughput_per_episode=throughput_per_episode,
        delay_per_episode=delay_per_episode,
        energy_efficiency_per_episode=energy_efficiency_per_episode,
        fairness_per_episode=fairness_per_episode,
        denies_per_episode=denies_per_episode,
        arrivals_per_episode=arrivals_per_episode,
        q_value_max=q_value_max,
        episode_metrics_std=episode_metrics_std,
        train_time_s=train_time_s,
        converged_ep=converged_ep,
        policy=_DoubleQPolicy(),
        # CMDP diagnostics
        lambda_block=lambda_block_hist,
        lambda_intr=lambda_intr_hist,
        lambda_energy=lambda_energy_hist,
    )
    # ---- mirror standardized metrics into legacy series for plotting ----
    try:
        eps_std = out.get("episode_metrics_std", []) or []
        if eps_std:
            def _to_series_std(key):
                return [float(m.get(key, float("nan"))) for m in eps_std]
            out["throughput_per_episode"] = _to_series_std("throughput_mbps")
            out["delay_per_episode"] = _to_series_std("avg_delay_ms")
            out["energy_usage_per_episode"] = _to_series_std("energy_j")
            out["energy_efficiency_per_episode"] = _to_series_std("energy_eff_bpj")
            out["fairness_per_episode"] = _to_series_std("jain_fairness")
            out["denies_per_episode"] = _to_series_std("blocked_arrivals_ep")
            out["arrivals_per_episode"] = _to_series_std("total_arrivals_ep")
            out["preempt_counts_per_episode"] = _to_series_std("preemptions_ep")
    except Exception:
        pass

    return out
