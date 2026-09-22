# hiot_dueling_dqn.py — Dueling DQN with Double-DQN target, CMDP shaping, and safe logging
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, namedtuple
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hiot_utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics, MetricKeys

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])


def _to_vec(obs):
    o = obs
    while isinstance(o, (list, tuple)) and len(o) > 0:
        o = o[0]
    return np.asarray(o, dtype=np.float32).reshape(1, -1)


class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *t):
        self.buf.append(Transition(*t))

    def sample(self, n: int):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s = torch.from_numpy(np.vstack([b.s for b in batch]).astype(np.float32))
        a = torch.tensor([b.a for b in batch], dtype=torch.int64)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        s2 = torch.from_numpy(np.vstack([b.s2 for b in batch]).astype(np.float32))
        d = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class DuelingQ(nn.Module):
    def __init__(self, n_in: int, n_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(n_in=128, out_features=128) if False else nn.Linear(128, 128),  # keep shape explicit
            nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.val = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        f = self.feature(x)
        a = self.adv(f)
        v = self.val(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


@dataclass
class DuelingCfg:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 50000
    target_sync: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # CMDP-style extras
    constr_block_target: float | None = 0.02
    constr_intr_target: float | None = 0.01
    constr_energy_j_target: float | None = None
    constr_lr: float = 1e-3


def hiot_dueling_dqn_train(
    env,
    episodes: int = 500,
    steps_per_ep: int = 1000,
    cfg: DuelingCfg | None = None,
    device=None,
) -> Dict[str, List[float]]:
    cfg = cfg or DuelingCfg()
    if device is None or device == "auto":
        device = torch.device(cfg.device)
    elif isinstance(device, str):
        device = torch.device(device)

    # Infer sizes without consuming a stochastic episode reset. This keeps the
    # exogenous environment sequence aligned with other schemes.
    probe = env._make_obs() if hasattr(env, "_make_obs") else env.reset()
    s0 = _to_vec(probe)
    n_in = s0.shape[1]
    n_actions = int(getattr(env, "n_actions", 3))

    q_net = DuelingQ(n_in, n_actions).to(device)
    tgt_net = DuelingQ(n_in, n_actions).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=cfg.lr)

    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    buf = Replay(cfg.buffer_size)

    rewards_per_episode: List[float] = []
    q_value_max: List[float] = []
    episode_metrics_std: List[dict] = []

    eps = cfg.eps_start
    global_step = 0

    # primal–dual multipliers and histories
    lam_block = 0.0
    lam_intr = 0.0
    lam_energy = 0.0
    lambda_block_hist: List[float] = []
    lambda_intr_hist: List[float] = []
    lambda_energy_hist: List[float] = []

    def _update_lambda(lam: float, metric_val: float, target: float | None) -> float:
        if target is None:
            return lam
        try:
            v = float(metric_val)
        except Exception:
            return lam
        if not np.isfinite(v):
            return lam
        if cfg.constr_lr <= 0.0:
            return lam
        g = v - float(target)
        lam_new = lam + cfg.constr_lr * g
        return max(0.0, float(lam_new))

    for ep in range(episodes):
        s = _to_vec(env.reset())
        ep_r = 0.0
        ep_qmax = -1e9
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        did_opt_step_this_ep = False

        for t in range(steps_per_ep):
            st = torch.from_numpy(s).float().to(device)
            with torch.no_grad():
                qvals = q_net(st)

            a = int(np.random.randint(n_actions)) if np.random.rand() < eps else int(
                torch.argmax(qvals, dim=1).item()
            )

            qmax = float(qvals.max().item())
            if np.isfinite(qmax) and qmax > ep_qmax:
                ep_qmax = qmax

            step_out = env.step(a)
            if (
                isinstance(step_out, (tuple, list))
                and len(step_out) >= 4
                and not isinstance(step_out[0], (tuple, list))
            ):
                s2_raw, env_r, done, info = step_out
            else:
                s2_raw, env_r, done, info = step_out[0]

            s2 = _to_vec(s2_raw)

            # CMDP-style shaping
            blocked_step = float((info.get("blocked", info.get("deny", 0)) or 0))
            preempt_step = float((info.get("preempt", 0) or 0))
            energy_step = float((info.get("energy_j", info.get("energy", 0.0)) or 0.0))
            penalty = lam_block * blocked_step + lam_intr * preempt_step + lam_energy * energy_step

            r = float(np.clip(env_r - penalty, -2.0, 2.0))
            ep_r += float(env_r)

            buf.push(s, a, r, s2, float(done))

            if len(buf) >= cfg.batch_size:
                bs, ba, br, bs2, bd = buf.sample(cfg.batch_size)
                bs = bs.to(device)
                ba = ba.to(device)
                br = br.to(device)
                bs2 = bs2.to(device)
                bd = bd.to(device)

                with torch.no_grad():
                    online_next = q_net(bs2)
                    next_actions = torch.argmax(online_next, dim=1)
                    target_q = tgt_net(bs2).gather(1, next_actions.view(-1, 1)).squeeze(1)
                    target = br + (1.0 - bd) * cfg.gamma * target_q

                q_pred = q_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                did_opt_step_this_ep = True

                if global_step % cfg.target_sync == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))

            s = s2
            global_step += 1
            if done:
                break

        eps = max(cfg.eps_end, eps * cfg.eps_decay)

        rewards_per_episode.append(float(ep_r))

        # EMA-smoothing for Q-value max
        ALPHA = 0.15
        epq = float(ep_qmax if np.isfinite(ep_qmax) else 0.0)
        if q_value_max:
            epq = ALPHA * epq + (1.0 - ALPHA) * float(q_value_max[-1])
        q_value_max.append(epq)

        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc,
        )
        if did_opt_step_this_ep:
            sched.step()

        try:
            m["q_value_max"] = float(ep_qmax) if np.isfinite(ep_qmax) else 0.0
        except Exception:
            m["q_value_max"] = 0.0

        # λ updates
        blk_val = m.get(MetricKeys.BLOCK, np.nan)
        intr_val = m.get(MetricKeys.INTR, np.nan)
        energy_val = m.get(MetricKeys.ENERGY_J, np.nan)
        lam_block = _update_lambda(lam_block, blk_val, cfg.constr_block_target)
        lam_intr = _update_lambda(lam_intr, intr_val, cfg.constr_intr_target)
        lam_energy = _update_lambda(lam_energy, energy_val, cfg.constr_energy_j_target)
        lambda_block_hist.append(float(lam_block))
        lambda_intr_hist.append(float(lam_intr))
        lambda_energy_hist.append(float(lam_energy))

        episode_metrics_std.append(m)

    class _TorchPolicy:
        def __init__(self, net, device):
            self.net = net
            self.device = device

        def act(self, obs):
            x = torch.from_numpy(_to_vec(obs)).float().to(self.device)
            with torch.no_grad():
                q = self.net(x)
            return int(torch.argmax(q, dim=1).item())

    out = {
        "rewards_per_episode": rewards_per_episode,
        "q_value_max": q_value_max,
        "episode_metrics_std": episode_metrics_std,
        "policy": _TorchPolicy(q_net, device),
        # CMDP diagnostics
        "lambda_block": lambda_block_hist,
        "lambda_intr": lambda_intr_hist,
        "lambda_energy": lambda_energy_hist,
    }

    # mirror standardized metrics into legacy series for plotting
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
