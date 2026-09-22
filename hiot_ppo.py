# hiot_ppo.py — PPO (discrete) stabilized + proper logging
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from hiot_utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics

# ---------------- helpers ----------------
def _to1d(obs):
    o = obs
    # unwrap (obs, info) or nested lists/tuples
    while isinstance(o, (list, tuple)) and len(o) > 0:
        o = o[0]
    return np.asarray(o, dtype=np.float32).ravel()

def _sanitize(x: List[float]) -> List[float]:
    return [float(v) if np.isfinite(v) else 0.0 for v in x]


def _initial_observation(env):
    """Read the observation shape without consuming an episode reset.

    The previous implementation called ``env.reset()`` twice while checking the
    return type. That advanced the simulator RNG before every PPO episode and
    broke common-random-number comparisons across schemes.
    """
    if hasattr(env, "_make_obs"):
        return env._make_obs()
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _reset_observation(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def _action_mask(env, n_actions: int) -> np.ndarray:
    getter = getattr(env, "get_action_mask", None)
    if not callable(getter):
        return np.ones(int(n_actions), dtype=bool)
    mask = np.asarray(getter(), dtype=bool).ravel()
    if mask.size != int(n_actions) or not np.any(mask):
        return np.ones(int(n_actions), dtype=bool)
    return mask

def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

@dataclass
class PPOCfg:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.15
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    train_epochs: int = 4
    minibatch_frac: float = 0.5
    lr: float = 2e-4
    # stability extras
    value_clip: float = 0.25
    entropy_coef_end: float = 0.005
    entropy_decay_episodes: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):
    def __init__(self, n_in:int, n_actions:int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.pi = nn.Linear(128, n_actions)
        self.v  = nn.Linear(128, 1)
    def forward(self, x: torch.Tensor):
        h = self.body(x)
        logits = self.pi(h)
        v = self.v(h)
        return logits, v

class PPOPolicy:
    """Frozen greedy deployment wrapper with portable checkpoint support."""
    def __init__(self, net: ActorCritic, device):
        self.net = net
        self.device = torch.device(device)

    def action_probabilities(self, obs, action_mask=None):
        """Return the frozen actor distribution for counterfactual intent audits."""
        x = torch.from_numpy(_to1d(obs)).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            if action_mask is not None:
                mask = np.asarray(action_mask, dtype=bool).ravel()
                if mask.size == logits.shape[-1] and np.any(mask):
                    mt = torch.from_numpy(mask).to(device=self.device).unsqueeze(0)
                    logits = _masked_logits(logits, mt)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        return np.asarray(probs, dtype=float)

    def act(self, obs, action_mask=None):
        probs = self.action_probabilities(obs, action_mask=action_mask)
        return int(np.argmax(probs))


def save_ppo_policy(policy: PPOPolicy, path: str | Path, metadata: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = policy.net
    payload = {
        "n_in": int(net.body[0].in_features),
        "n_actions": int(net.pi.out_features),
        "state_dict": net.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_ppo_policy(path: str | Path, device="cpu") -> tuple[PPOPolicy, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    net = ActorCritic(int(payload["n_in"]), int(payload["n_actions"])).to(torch.device(device))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return PPOPolicy(net, device), dict(payload.get("metadata", {}))

def _gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimator; values is 1D array length T."""
    values = np.asarray(values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T = len(rewards)
    val_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * val_ext[t+1] * (1.0 - dones[t]) - val_ext[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    ret = adv + val_ext[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret.astype(np.float32), adv.astype(np.float32)

# ---------------- main train ----------------
def hiot_ppo_train(env, episodes=500, steps_per_ep=100, cfg: PPOCfg | None = None, device=None) -> Dict[str, List[float]]:
    cfg = cfg or PPOCfg()
    if device is None or device == "auto":
        device = torch.device(cfg.device)
    elif isinstance(device, str):
        device = torch.device(device)

    s0 = _to1d(_initial_observation(env))
    n_in = int(s0.size)
    n_actions = int(getattr(env, "n_actions", getattr(env, "action_space_n", 5)))

    net = ActorCritic(n_in, n_actions).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    rewards_per_episode: List[float] = []
    q_value_max: List[float] = []
    episode_metrics_std: List[dict] = []
    ppo_policy_loss: List[float] = []
    ppo_value_loss: List[float]  = []
    ppo_entropy: List[float]     = []
    ppo_kl: List[float]          = []
    ppo_total_loss: List[float]  = []

    for ep in range(episodes):
        s = _to1d(_reset_observation(env))
        ep_r = 0.0
        ep_vmax = -np.inf
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))

        traj_s, traj_a, traj_logp, traj_r, traj_v, traj_done, traj_mask = [], [], [], [], [], [], []

        # --- rollout ---
        for t in range(steps_per_ep):
            st = torch.from_numpy(s).float().to(device).unsqueeze(0)  # [1, n_in]
            with torch.no_grad():
                logits, v = net(st)
                mask_np = _action_mask(env, n_actions)
                mask_t = torch.from_numpy(mask_np).to(device=device).unsqueeze(0)
                dist = Categorical(logits=_masked_logits(logits, mask_t))
                a = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(a, device=device)).item())
                v_scalar = float(v.squeeze(0).squeeze(0).item())
                if np.isfinite(v_scalar):
                    ep_vmax = max(ep_vmax, v_scalar)

            s2, env_r, done, info = env.step(a)
            r = float(np.clip(env_r, -2.0, 2.0))

            traj_s.append(s.copy())        # 1D
            traj_a.append(a)
            traj_logp.append(logp)
            traj_r.append(r)
            traj_v.append(v_scalar)
            traj_done.append(float(done))
            traj_mask.append(mask_np.copy())

            ep_r += float(env_r)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))

            s = _to1d(s2)
            if done:
                break

        # --- compute returns/advantages ---
        ret, adv = _gae(traj_r, traj_v, traj_done, cfg.gamma, cfg.lam)

        obs_t = torch.from_numpy(np.stack(traj_s).astype(np.float32)).to(device)      # [T, n_in]
        act_t = torch.tensor(traj_a, dtype=torch.int64, device=device)                 # [T]
        old_logp_t = torch.tensor(traj_logp, dtype=torch.float32, device=device)       # [T]
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)                  # [T]
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)                  # [T]
        v_old_t = torch.tensor(traj_v, dtype=torch.float32, device=device)             # [T]
        mask_t_all = torch.from_numpy(np.stack(traj_mask).astype(bool)).to(device)       # [T,A]

        N = obs_t.size(0)
        mb = max(32, int(cfg.minibatch_frac * N))

        # diagnostics accumulators
        ep_pl, ep_vl, ep_en, ep_kl, ep_tot = [], [], [], [], []

        # --- optimize policy/value ---
        for _ in range(cfg.train_epochs):
            idx = torch.randperm(N, device=device)
            for i in range(0, N, mb):
                j = idx[i:i+mb]
                logits, v_pred = net(obs_t[j])               # v_pred: [mb,1]
                dist = Categorical(logits=_masked_logits(logits, mask_t_all[j]))
                logp = dist.log_prob(act_t[j])               # [mb]
                ratio = torch.exp(logp - old_logp_t[j])      # [mb]
                clip_adv = torch.clamp(ratio, 1.0-cfg.clip_ratio, 1.0+cfg.clip_ratio) * adv_t[j]
                policy_loss = -(torch.min(ratio * adv_t[j], clip_adv)).mean()

                # value loss with clipping
                v_pred_flat = v_pred.view(-1)                # [mb]
                v_old = v_old_t[j]                           # [mb]
                v_clip = v_old + (v_pred_flat - v_old).clamp(-cfg.value_clip, cfg.value_clip)
                v_loss1 = (v_pred_flat - ret_t[j]).pow(2)
                v_loss2 = (v_clip - ret_t[j]).pow(2)
                value_loss = torch.max(v_loss1, v_loss2).mean()

                # entropy with gentle decay
                ent_coef = cfg.ent_coef + (cfg.entropy_coef_end - cfg.ent_coef) * min(1.0, ep / max(1, cfg.entropy_decay_episodes))
                entropy = dist.entropy().mean()

                loss = policy_loss + cfg.vf_coef * value_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

                # simple KL estimate for monitoring
                with torch.no_grad():
                    new_logp = dist.log_prob(act_t[j])
                    kl = (old_logp_t[j] - new_logp).mean().item()

                ep_pl.append(float(policy_loss.item()))
                ep_vl.append(float(value_loss.item()))
                ep_en.append(float(entropy.item()))
                ep_kl.append(float(kl))
                ep_tot.append(float(loss.item()))

        # --- episode summaries ---
        rewards_per_episode.append(float(ep_r))
        # Monotone-smoothed value proxy for the Q/Value plot
        ALPHA = 0.15
        vq = float(ep_vmax if np.isfinite(ep_vmax) else 0.0)
        if q_value_max:
            vq = ALPHA * vq + (1.0 - ALPHA) * float(q_value_max[-1])
        q_value_max.append(vq)

        # Diagnostic means per episode
        def _mean(lst):
            return float(np.nan if not lst else sum(lst)/len(lst))
        ppo_policy_loss.append(_mean(ep_pl))
        ppo_value_loss.append(_mean(ep_vl))
        ppo_entropy.append(_mean(ep_en))
        ppo_kl.append(_mean(ep_kl))
        ppo_total_loss.append(_mean(ep_tot))

        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc,
        )
        # include value-head proxy so CSV has q_value_max
        try:
            m["q_value_max"] = float(ep_vmax) if np.isfinite(ep_vmax) else 0.0
        except Exception:
            m["q_value_max"] = 0.0

        episode_metrics_std.append(m)

    out = {
        "rewards_per_episode": _sanitize(rewards_per_episode),
        "q_value_max": _sanitize(q_value_max),
        "episode_metrics_std": episode_metrics_std,
        "policy": PPOPolicy(net, device),
        # diagnostics
        "ppo_policy_loss": _sanitize(ppo_policy_loss),
        "ppo_value_loss": _sanitize(ppo_value_loss),
        "ppo_entropy": _sanitize(ppo_entropy),
        "ppo_kl": _sanitize(ppo_kl),
        "ppo_total_loss": _sanitize(ppo_total_loss),
    }

    # ---- mirror standardized metrics into legacy series for plotting ----
    try:
        eps_std = out.get("episode_metrics_std", []) or []
        if eps_std:
            def _to_series_std(key):
                return [float(m.get(key, float('nan'))) for m in eps_std]

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
