"""Matched value-based SOTA baselines for the frozen V-IoT benchmark.

Implements two publication baselines under the exact request-level action model:
  * SOTA-DQN: standard DQN target-network update.
  * SOTA-DuelingDDQN: dueling network + Double-DQN target.

Both baselines:
  * receive the same observation (including intent context) as the PPO family;
  * use the common physical-feasibility action mask during exploration,
    bootstrap target construction, and deployment;
  * use the same multi-intent/multi-stress curriculum when enabled by the runner;
  * optimize the unmodified base environment reward (no CMDP compiler, tail term,
    or admission shield);
  * deploy greedily, as is standard for value-based DQN policies.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Any, Dict, List
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hiot_utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics


def _obs_vec(obs: Any) -> np.ndarray:
    o = obs[0] if isinstance(obs, tuple) else obs
    return np.asarray(o, dtype=np.float32).reshape(-1)


def _mask_vec(env, n_actions: int) -> np.ndarray:
    getter = getattr(env, "get_feasible_action_mask", None)
    if not callable(getter):
        getter = getattr(env, "get_action_mask", None)
    if callable(getter):
        m = np.asarray(getter(), dtype=bool).reshape(-1)
        if m.size != int(n_actions):
            raise ValueError(f"Action mask size {m.size} != n_actions {n_actions}.")
        if np.any(m):
            return m
    return np.ones(int(n_actions), dtype=bool)


class StandardQNet(nn.Module):
    def __init__(self, n_in: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    def __init__(self, n_in: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


@dataclass(frozen=True)
class ValueDQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    train_every: int = 4
    target_sync_steps: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.70
    hidden_dim: int = 128
    grad_clip_norm: float = 10.0


class ReplayBuffer:
    def __init__(self, capacity: int, n_actions: int):
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self.buf = deque(maxlen=self.capacity)

    def push(self, s, a, r, s2, done, next_mask):
        self.buf.append((
            np.asarray(s, dtype=np.float32).reshape(-1),
            int(a), float(r),
            np.asarray(s2, dtype=np.float32).reshape(-1),
            float(done),
            np.asarray(next_mask, dtype=bool).reshape(-1),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = min(int(batch_size), len(self.buf))
        idx = rng.choice(len(self.buf), size=n, replace=False)
        rows = [self.buf[int(i)] for i in idx]
        s = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.float32))
        a = torch.tensor([r[1] for r in rows], dtype=torch.long)
        rew = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        s2 = torch.from_numpy(np.stack([r[3] for r in rows]).astype(np.float32))
        done = torch.tensor([r[4] for r in rows], dtype=torch.float32)
        mask2 = torch.from_numpy(np.stack([r[5] for r in rows]).astype(bool))
        return s, a, rew, s2, done, mask2

    def __len__(self):
        return len(self.buf)


class GreedyQPolicy:
    """Frozen greedy value policy with common physical-feasibility masking."""
    def __init__(self, net: nn.Module, device: torch.device, architecture: str,
                 n_in: int, n_actions: int, hidden_dim: int = 128):
        self.net = net
        self.device = torch.device(device)
        self.architecture = str(architecture)
        self.n_in = int(n_in)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)

    def act(self, obs, action_mask=None):
        x = torch.from_numpy(_obs_vec(obs)).float().reshape(1, -1).to(self.device)
        with torch.no_grad():
            q = self.net(x).detach().cpu().numpy().reshape(-1)
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=bool).reshape(-1)
            if m.size != q.size:
                raise ValueError(f"Deployment mask size {m.size} != Q size {q.size}.")
            if np.any(m):
                q = q.copy()
                q[~m] = -np.inf
        return int(np.argmax(q))


def _make_net(architecture: str, n_in: int, n_actions: int, hidden_dim: int) -> nn.Module:
    architecture = str(architecture).lower()
    if architecture == "standard":
        return StandardQNet(n_in, n_actions, hidden_dim)
    if architecture == "dueling":
        return DuelingQNet(n_in, n_actions, hidden_dim)
    raise ValueError(f"Unknown value-network architecture: {architecture}")


def train_value_dqn(
    env,
    episodes: int,
    steps_per_ep: int,
    seed: int,
    device,
    *,
    architecture: str = "standard",
    double_dqn: bool = False,
    cfg: ValueDQNConfig | None = None,
) -> Dict[str, Any]:
    """Train a masked DQN-family baseline on the runner-supplied environment."""
    cfg = cfg or ValueDQNConfig()
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    probe = env._make_obs() if hasattr(env, "_make_obs") else env.reset()
    n_in = int(_obs_vec(probe).size)
    n_actions = int(getattr(env, "n_actions"))

    online = _make_net(architecture, n_in, n_actions, cfg.hidden_dim).to(device)
    target = _make_net(architecture, n_in, n_actions, cfg.hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = optim.Adam(online.parameters(), lr=float(cfg.lr))
    replay = ReplayBuffer(cfg.replay_capacity, n_actions)

    total_steps_planned = max(1, int(episodes) * int(steps_per_ep))
    eps_decay_steps = max(1, int(round(cfg.eps_fraction * total_steps_planned)))

    def epsilon_at(step: int) -> float:
        frac = min(1.0, max(0.0, float(step) / float(eps_decay_steps)))
        return float(cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start))

    rewards_per_episode: List[float] = []
    episode_metrics_std: List[dict] = []
    q_value_max: List[float] = []
    loss_per_episode: List[float] = []
    eps_per_episode: List[float] = []
    global_step = 0

    for ep in range(int(episodes)):
        obs = env.reset()
        s = _obs_vec(obs)
        ep_reward = 0.0
        ep_qmax = []
        ep_losses = []
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))

        for _ in range(int(steps_per_ep)):
            mask = _mask_vec(env, n_actions)
            eps = epsilon_at(global_step)
            feasible = np.flatnonzero(mask)
            if rng.random() < eps:
                action = int(feasible[int(rng.integers(0, feasible.size))])
            else:
                with torch.no_grad():
                    q_np = online(torch.from_numpy(s).float().reshape(1, -1).to(device)).detach().cpu().numpy().reshape(-1)
                q_np = q_np.copy(); q_np[~mask] = -np.inf
                action = int(np.argmax(q_np))
                if np.any(np.isfinite(q_np)):
                    ep_qmax.append(float(np.nanmax(q_np[np.isfinite(q_np)])))

            s2_raw, reward, done, info = env.step(action)
            s2 = _obs_vec(s2_raw)
            next_mask = _mask_vec(env, n_actions)
            # Same base scalar reward as unconstrained PPO. No reward compiler,
            # no risk memory, no dual penalty, and no safety shield are applied.
            replay.push(s, action, float(reward), s2, float(done), next_mask)
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))

            if (global_step >= int(cfg.warmup_steps)
                    and len(replay) >= int(cfg.batch_size)
                    and global_step % int(cfg.train_every) == 0):
                bs, ba, br, bs2, bd, bm2 = replay.sample(cfg.batch_size, rng)
                bs = bs.to(device); ba = ba.to(device); br = br.to(device)
                bs2 = bs2.to(device); bd = bd.to(device); bm2 = bm2.to(device)

                with torch.no_grad():
                    if bool(double_dqn):
                        q_online_next = online(bs2).masked_fill(~bm2, -1e9)
                        next_a = q_online_next.argmax(dim=1, keepdim=True)
                        q_boot = target(bs2).gather(1, next_a).squeeze(1)
                    else:
                        q_target_next = target(bs2).masked_fill(~bm2, -1e9)
                        q_boot = q_target_next.max(dim=1).values
                    y = br + float(cfg.gamma) * (1.0 - bd) * q_boot

                q_pred = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), float(cfg.grad_clip_norm))
                opt.step()
                ep_losses.append(float(loss.item()))

            global_step += 1
            if global_step % int(cfg.target_sync_steps) == 0:
                target.load_state_dict(online.state_dict())
            s = s2
            if done:
                break

        m = finalize_episode_metrics(
            env,
            steps_per_ep=int(steps_per_ep),
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc,
        )
        rewards_per_episode.append(float(ep_reward))
        q_value_max.append(float(np.mean(ep_qmax)) if ep_qmax else 0.0)
        loss_per_episode.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        eps_per_episode.append(float(epsilon_at(global_step)))
        episode_metrics_std.append(m)

    policy = GreedyQPolicy(
        online.eval(), device, architecture=architecture,
        n_in=n_in, n_actions=n_actions, hidden_dim=cfg.hidden_dim,
    )
    return {
        "policy": policy,
        "rewards_per_episode": rewards_per_episode,
        "q_value_max": q_value_max,
        "loss_per_episode": loss_per_episode,
        "epsilon_per_episode": eps_per_episode,
        "episode_metrics_std": episode_metrics_std,
        "value_dqn_config": asdict(cfg),
        "value_dqn_architecture": str(architecture),
        "value_dqn_double_target": bool(double_dqn),
    }


def save_value_policy(policy: GreedyQPolicy, path: str | Path, metadata: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "viot_sota_value_dqn_v1",
        "architecture": policy.architecture,
        "n_in": policy.n_in,
        "n_actions": policy.n_actions,
        "hidden_dim": policy.hidden_dim,
        "state_dict": policy.net.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_value_policy(path: str | Path, device=None):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") != "viot_sota_value_dqn_v1":
        raise ValueError(f"Unsupported value-policy checkpoint format in {path}.")
    net = _make_net(
        payload["architecture"], int(payload["n_in"]), int(payload["n_actions"]), int(payload["hidden_dim"])
    ).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    policy = GreedyQPolicy(
        net, device, payload["architecture"], int(payload["n_in"]), int(payload["n_actions"]), int(payload["hidden_dim"])
    )
    return policy, dict(payload.get("metadata", {}))


FIXED_SOTA_DQN_PROTOCOL = {
    "implementation": "viot_sota_value_dqn_v1",
    "SOTA-DQN": {"architecture": "standard", "double_dqn": False},
    "SOTA-DuelingDDQN": {"architecture": "dueling", "double_dqn": True},
    "config": asdict(ValueDQNConfig()),
    "training_reward": "unmodified base environment reward",
    "training_action_mask": "common physical feasibility mask",
    "deployment": "greedy masked argmax",
}
