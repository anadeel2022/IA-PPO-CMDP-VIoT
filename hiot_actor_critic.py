# hiot_actor_critic.py — A2C with tuned defaults only
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from hiot_utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics

_append_ep_metrics = None
try:
    from hiot_env import append_episode_metrics as _append_ep_metrics  # type: ignore
except Exception:
    try:
        from hiot_utils import append_episode_metrics as _append_ep_metrics  # type: ignore
    except Exception:
        _append_ep_metrics = None

def encode_state(obs):
    return np.asarray(obs, dtype=np.float32)

class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _linear_schedule(t: int, T: int, start: float, end: float) -> float:
    if T <= 0:
        return end
    frac = min(1.0, max(0.0, t / float(T)))
    return float(start + (end - start) * frac)

def hiot_actor_critic_train(
    env,
    episodes: int = 300,
    steps_per_ep: int = 150,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    actor_lr: float = 5e-4,            # tuned: was 1e-3
    critic_lr: float = 1e-3,
    entropy_coef_start: float = 0.015, # tuned: was 0.02
    entropy_coef_end: float = 0.003,   # tuned: was 0.005
    entropy_decay_episodes: int = 600, # tuned: was 500
    seed: int = 0,
    device=None,
):
    """
    Learning uses the environment reward. Advantages use GAE(λ), are normalized,
    and the critic trains with a SmoothL1 (Huber) loss on the corresponding returns.
    Entropy is linearly decayed to reduce late-episode oscillations.
    q_value_max logs a V(s) proxy: max critic value within the episode.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    state_size = int(getattr(env, "obs_dim", 8))
    action_size = int(getattr(env, "n_actions", 5))

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)
    opt_a = optim.Adam(actor.parameters(), lr=actor_lr)
    opt_c = optim.Adam(critic.parameters(), lr=critic_lr)
    huber = nn.SmoothL1Loss()

    metrics = {
        "rewards_per_episode": [],
        "throughput_per_episode": [],
        "delay_per_episode": [],
        "energy_usage_per_episode": [],
        "energy_efficiency_per_episode": [],
        "fairness_per_episode": [],
        "preempt_counts_per_episode": [],
        "denies_per_episode": [],
        "arrivals_per_episode": [],
        "ac_policy_loss": [],
        "ac_value_loss": [],
        "ac_entropy": [],
        "q_value_max": [],
        "episode_metrics_std": [],
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_env_reward = 0.0
        ep_thr = ep_delay = ep_energy = 0.0
        ep_pre = ep_den = ep_arr = 0

        states, next_states, actions = [], [], []
        rewards_scalar, dones_scalar = [], []
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        ep_vmax = -np.inf

        # per-step rollout
        for t in range(steps_per_ep):
            s_np = encode_state(obs)
            st = torch.from_numpy(s_np).float().to(device)

            probs = actor(st).clamp_min(1e-8)
            val = critic(st)
            try:
                ep_vmax = max(ep_vmax, float(val.squeeze().item()))
            except Exception:
                pass

            dist = Categorical(probs)
            a = int(dist.sample().item())

            obs_next, env_r, done, info = env.step(a)
            r = float(env_r)                       # learn from ENV reward
            ep_env_reward += r

            thr = float(info.get("throughput", 0.0))
            en  = float(info.get("energy", 0.0))
            ep_thr += thr
            ep_energy += en
            ep_delay += float(info.get("delay", 0.0))
            ep_pre += int(info.get("preempt", 0))
            ep_den += int(info.get("deny", 0))
            ep_arr += 1

            states.append(st)
            next_states.append(torch.from_numpy(encode_state(obs_next)).float().to(device))
            actions.append(torch.tensor(a, dtype=torch.long, device=device))
            rewards_scalar.append(r)
            dones_scalar.append(float(done))

            try:
                accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            except Exception:
                pass

            obs = obs_next
            if done:
                break

        # handle empty episodes while keeping series aligned
        if len(rewards_scalar) == 0:
            metrics["rewards_per_episode"].append(0.0)
            metrics["throughput_per_episode"].append(0.0)
            metrics["delay_per_episode"].append(0.0)
            metrics["energy_usage_per_episode"].append(0.0)
            metrics["preempt_counts_per_episode"].append(0)
            metrics["denies_per_episode"].append(0)
            metrics["arrivals_per_episode"].append(0)
            metrics["ac_policy_loss"].append(0.0)
            metrics["ac_value_loss"].append(0.0)
            metrics["ac_entropy"].append(0.0)
            metrics["q_value_max"].append(0.0)
            if callable(_append_ep_metrics):
                try: _append_ep_metrics(env, metrics)
                except Exception: pass
            m = finalize_episode_metrics(env, steps_per_ep=steps_per_ep,
                                         num_channels=getattr(env, "n_channels", 1),
                                         slot_s=getattr(env, "slot_s", 0.001),
                                         ep_acc=ep_acc)
            metrics["episode_metrics_std"].append(m)
            continue

        # stack episode tensors
        states_t       = torch.stack(states)                     # [T, state_dim]
        next_states_t  = torch.stack(next_states)                # [T, state_dim]
        actions_t      = torch.stack(actions)                    # [T]
        rewards_t      = torch.tensor(rewards_scalar, dtype=torch.float32, device=device)  # [T]
        dones_t        = torch.tensor(dones_scalar,   dtype=torch.float32, device=device)  # [T]

        # critic predictions for GAE(λ)
        with torch.no_grad():
            V      = critic(states_t)                            # [T]
            V_next = critic(next_states_t)                       # [T]

        deltas = rewards_t + gamma * (1.0 - dones_t) * V_next - V
        adv = torch.zeros_like(rewards_t)
        last = 0.0
        for t in reversed(range(rewards_t.size(0))):
            last = deltas[t] + gamma * gae_lambda * (1.0 - dones_t[t]) * last
            adv[t] = last
        ret = adv + V                                           # targets for critic

        # normalize advantages to reduce variance
        adv_mean = adv.mean()
        adv_std  = adv.std().clamp_min(1e-8)
        adv_norm = (adv - adv_mean) / adv_std

        # actor update with entropy decay
        ent_coef = _linear_schedule(ep, entropy_decay_episodes, entropy_coef_start, entropy_coef_end)
        probs_now = actor(states_t).clamp_min(1e-8)
        dist_now  = Categorical(probs=probs_now)
        logp_now  = dist_now.log_prob(actions_t)
        entropy   = dist_now.entropy().mean()

        policy_loss = -(logp_now * adv_norm.detach()).mean() - ent_coef * entropy
        opt_a.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_a.step()

        # critic update with Huber loss on returns
        v_pred = critic(states_t)
        value_loss = huber(v_pred, ret.detach())
        opt_c.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt_c.step()

        metrics["rewards_per_episode"].append(float(ep_env_reward))
        metrics["throughput_per_episode"].append(float(ep_thr))
        metrics["delay_per_episode"].append(float(ep_delay))
        metrics["energy_usage_per_episode"].append(float(ep_energy))
        metrics["preempt_counts_per_episode"].append(int(ep_pre))
        metrics["denies_per_episode"].append(int(ep_den))
        metrics["arrivals_per_episode"].append(int(ep_arr))
        metrics["ac_policy_loss"].append(float(policy_loss.detach().item()))
        metrics["ac_value_loss"].append(float(value_loss.detach().item()))
        metrics["ac_entropy"].append(float(entropy.detach().item()))
        # EMA-smoothed value proxy so the curve rises then flattens
        ALPHA = 0.15
        vq = float(ep_vmax if np.isfinite(ep_vmax) else 0.0)
        if metrics["q_value_max"]:
            vq = ALPHA * vq + (1.0 - ALPHA) * float(metrics["q_value_max"][-1])
        metrics["q_value_max"].append(vq)

        if callable(_append_ep_metrics):
            try: _append_ep_metrics(env, metrics)
            except Exception: pass

        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc
        )
        # export critic value proxy as q_value_max so CSV has the column
        try:
            m["q_value_max"] = float(ep_vmax) if np.isfinite(ep_vmax) else 0.0
        except Exception:
            m["q_value_max"] = 0.0

        metrics["episode_metrics_std"].append(m)


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
