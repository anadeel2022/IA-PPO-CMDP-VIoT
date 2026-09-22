"""
viot_agentic_runner.py

Intent-aware agentic AI runner for RSU-assisted V-IoT scheduling.

Default behavior is deliberately two-stage:
  1) a fast smoke test is executed first to catch interface/runtime errors;
  2) if the smoke test completes, a standard detailed run is launched.

Full manuscript-scale experiments remain available with --profile manuscript.
"""
from __future__ import annotations

import argparse
import copy
import sys
import csv
import json
import math
import time
import tempfile
import random
import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable
from collections import deque
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    # Single-threaded CPU inference/training is usually faster and more stable for
    # these small MLP policies, especially on laptops where PyTorch thread startup
    # overhead can dominate short experiments.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    torch = None

from hiot_doubleq import hiot_doubleq_train
from hiot_qlearning import hiot_qlearning_train
from hiot_dqn import hiot_dqn_train, DQNCfg
from hiot_dueling_dqn import hiot_dueling_dqn_train, DuelingCfg
from hiot_ppo import hiot_ppo_train, PPOCfg, save_ppo_policy, load_ppo_policy
from hiot_actor_critic import hiot_actor_critic_train
from hiot_sota_dqn import train_value_dqn, ValueDQNConfig, save_value_policy, load_value_policy
from hiot_utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics, MetricKeys
from hiot_env import MODE_DEFER, MODE_GRANT, MODE_PROTECT, MODE_COEXIST, MODE_REJECT

from viot_agentic_intent import (
    INTENT_LIBRARY, SCENARIO_LIBRARY, IntentProfile, StressScenario,
    make_agentic_env, summarize_agentic_metrics, effective_deadline_ms,
    local_step_violation, intent_outcome_pressure, intent_context_vector,
)

OUT_ROOT = Path("./outputs_agentic")

# =============================================================================
# PYCHARM GREEN-RUN SETTINGS
# =============================================================================
# Edit only this block when you run the file using PyCharm's green Run button.
# These settings are applied automatically when no command-line arguments are
# provided. Command-line arguments still override these values when used.
#
# Recommended daily setting:
#   profile="auto" runs a smoke test first, then continues the standard run.
#   resume=True and a stable run_id prevent loss of progress after shutdown.
#   make_figures=False keeps long runs focused on one CSV summary. Figures can
#   be generated later from the completed CSV, or by setting make_figures=True.
# =============================================================================
PYCHARM_GREEN_RUN_DEFAULTS: Dict[str, Any] = {
    "profile": "auto",
    "auto_detail_profile": "intent_switch_latency",
    "run_id": "agentic_v9_4_switch_latency_v1",
    "resume": True,
    "make_figures": False,
    "output_mode": "analysis",
    "scheme_set": "intent_switch",
    "plot_style": "composite",
    "skip_smoke": True,
    "device": "auto",
}

# Current default is V9.3 verification: five matched seeds with all mechanism
# ablations retained. Do not set profile="manuscript" until verification and
# online intent-switch validation have both passed and the protocol is frozen.


# For manuscript-scale execution from PyCharm, change the block above to:
#   "profile": "manuscript"
#   "run_id": "agentic_manuscript_v1"
#   "resume": True
#   "make_figures": False
# Then press the green Run button again whenever the PC restarts.


# Final paper order and ablation policy.
# PAPER_MAIN_SCHEMES contains published algorithmic families plus one clearly
# labeled internal ablation and the proposed intent-aware method. Internally,
# PPO_CMDP is retained as the scheme key, but tables/figures label it as
# Non-Agentic PPO-CMDP to avoid presenting it as an external published baseline.
# V9 comparison sets. Development is deliberately narrow: it tests the
# causal mechanism against the strongest constrained baseline before any broad run.
PAPER_MAIN_SCHEMES = (
    "MaxWeight,LyapunovDPP,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP"
)
METRIC_SANITY_SCHEMES = "AllDeny,AllGrant,MaxWeight"
FEASIBILITY_AUDIT_SCHEMES = "MaxWeight,CapacityFirst,AllGrant"
DIAGNOSTIC_CALIBRATION_SCHEMES = "MaxWeight,PPO,PPO-Lagrangian"
FOCUSED_DEVELOPMENT_SCHEMES = "MaxWeight,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP-NoRisk,Intent-PPO-CMDP-NoShield,Intent-PPO-CMDP"
V9_ABLATION_SCHEMES = "MaxWeight,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP-NoRisk,Intent-PPO-CMDP-NoShield,Intent-PPO-CMDP"
INTENT_SWITCH_SCHEMES = "MaxWeight,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP-NoRisk,Intent-PPO-CMDP-NoShield,Intent-PPO-CMDP"

# Retained for historical diagnostics only.
ABLATION_FULL_SCHEMES = (
    "DQN,DQN_CMDP,Intent-DQN-CMDP,"
    "DuelingDQN,DuelingDQN_CMDP,Intent-DuelingDQN-CMDP,"
    "PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP"
)
RL_HEAVY_PAPER_SCHEMES = PAPER_MAIN_SCHEMES
FULL_LEGACY_SCHEMES = "MaxWeight,LyapunovDPP,SLA-Aware,DQN,DuelingDQN,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP"
DEFAULT_PAPER_SCHEMES = PAPER_MAIN_SCHEMES
FAST_SMOKE_SCHEMES = "MaxWeight,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP"
STANDARD_AUTO_SCHEMES = FOCUSED_DEVELOPMENT_SCHEMES


SCHEME_COLORS = {
    "AllDeny": "#b22222",
    "AllGrant": "#228b22",
    "CapacityFirst": "#1b9e77",
    "PriorityAging": "#7f7f7f",
    "ProportionalFair": "#bcbd22",
    "TDMA-Preempt": "#17becf",
    "MaxWeight": "#4d4d4d",
    "LyapunovDPP": "#1f77b4",
    "SLA-Aware": "#2ca02c",
    "MPC-Lite": "#9467bd",
    "DoubleQ": "#ff7f0e",
    "DoubleQ_CMDP": "#ffbb78",
    "DQN": "#2ca02c",
    "DQN_CMDP": "#98df8a",
    "DuelingDQN": "#d62728",
    "DuelingDQN_CMDP": "#ff9896",
    "SOTA-DQN": "#2ca02c",
    "SOTA-DuelingDDQN": "#d62728",
    "Intent-DQN-CMDP": "#006400",
    "Intent-DuelingDQN-CMDP": "#8b0000",
    "ActorCritic": "#9467bd",
    "PPO": "#8c564b",
    "PPO-Lagrangian": "#6a5acd",
    "PPO_CMDP": "#c49c94",
    "Intent-PPO-CMDP-NoRisk": "#9c755f",
    "Intent-PPO-CMDP-NoShield": "#8c564b",
    "Intent-PPO-CMDP": "#6b3e26",
    "Intent-PPO-CMDP-FullPolicy-NoShield": "#7f7f7f",
}

ACTION_DEFER, ACTION_GRANT, ACTION_PREEMPT, ACTION_COEXIST, ACTION_REJECT = (
    MODE_DEFER, MODE_GRANT, MODE_PROTECT, MODE_COEXIST, MODE_REJECT
)
# Backward-compatible name used by the metric-sanity diagnostic. It now means
# true request rejection rather than a no-service defer action.
ACTION_DENY = ACTION_REJECT
EPS = 1e-12


def stable_int_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    """Return a process-independent integer seed from structured identifiers."""
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible policy training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

SCHEME_DISPLAY_NAMES = {
    "AllDeny": "All-Deny Diagnostic",
    "AllGrant": "All-Grant Diagnostic",
    "CapacityFirst": "Capacity-First Feasibility Anchor",
    "PPO-Lagrangian": "PPO-Lagrangian",
    "PPO_CMDP": "Non-Agentic PPO-CMDP",
    "Intent-PPO-CMDP-NoRisk": "IA-PPO-CMDP w/o Tail Risk",
    "Intent-PPO-CMDP-NoShield": "IA-PPO-CMDP w/o Shield",
    "Intent-PPO-CMDP": "IA-PPO-CMDP (Proposed)",
    "Intent-PPO-CMDP-FullPolicy-NoShield": "IA-PPO-CMDP, same policy w/o deployment shield",
    "Intent-DQN-CMDP": "IA-DQN-CMDP",
    "Intent-DuelingDQN-CMDP": "IA-DuelingDQN-CMDP",
    "SOTA-DQN": "DQN",
    "SOTA-DuelingDDQN": "Dueling Double DQN",
    "LyapunovDPP": "Lyapunov-DPP",
    "SLA-Aware": "SLA-Aware Rule",
}

def scheme_label(name: str) -> str:
    return SCHEME_DISPLAY_NAMES.get(str(name), str(name))


def scheme_comparison_role(name: str) -> str:
    name = str(name)
    if name in {"AllDeny", "AllGrant"}:
        return "metric-sanity diagnostic"
    if name == "CapacityFirst":
        return "feasibility anchor"
    if name in {"MaxWeight", "LyapunovDPP"}:
        return "published scheduling baseline"
    if name == "SLA-Aware":
        return "rule-based SLA baseline"
    if name in {"DQN", "DuelingDQN", "SOTA-DQN", "SOTA-DuelingDDQN", "PPO", "DoubleQ", "Q-Learning", "ActorCritic"}:
        return "published RL family"
    if name == "PPO-Lagrangian":
        return "constrained-RL baseline"
    if name in {"DQN_CMDP", "DuelingDQN_CMDP", "DoubleQ_CMDP", "PPO_CMDP"}:
        return "internal CMDP ablation"
    if name.startswith("Intent-"):
        return "proposed agentic variant"
    return "diagnostic baseline"


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def split_float_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def get_device(arg: str):
    if torch is None:
        return None
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Atomically write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, dir=str(path.parent), suffix=".tmp") as f:
        tmp_path = Path(f.name)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float) and not np.isfinite(v):
                    v = ""
                out[k] = v
            w.writerow(out)
    tmp_path.replace(path)


def read_rows_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _task_key(intent_name: str, scenario_name: str, seed: int, scheme: str) -> str:
    return f"{intent_name}::{scenario_name}::seed{int(seed)}::{scheme}"


def _row_task_key(row: Dict[str, Any]) -> str:
    return _task_key(str(row.get("intent", "")), str(row.get("scenario", "")), int(float(row.get("seed", 0))), str(row.get("scheme", "")))


def analysis_metric_columns() -> List[str]:
    """Columns retained in the compact main analysis CSV.

    The runner internally computes many diagnostics, but this list keeps the
    final output manageable and immediately usable for manuscript analysis.
    """
    return [
        "intent", "scenario", "scheme", "scheme_label", "method_family", "comparison_role", "num_seeds",
        "soft_intent_compliance_score_mean", "soft_intent_compliance_score_std",
        "weighted_violation_mean_mean", "weighted_violation_mean_std",
        "weighted_violation_cvar95_mean",
        "interruption_mean_mean", "interruption_mean_std",
        "blocking_mean_mean", "blocking_mean_std",
        "deadline_exposure_mean_mean", "deadline_exposure_cvar95_mean",
        "service_success_rate_mean_mean", "shield_override_rate_mean_mean",
        "reliability_safety_margin_mean_mean",
        "throughput_mean_mbps_mean", "throughput_mean_mbps_std",
        "throughput_cvar_low10_mbps_mean",
        "delay_mean_ms_mean", "delay_cvar95_ms_mean",
        "energy_per_mbit_j_mean", "energy_eff_mean_bpj_mean",
        "fairness_mean_mean", "fairness_deficit_mean_mean",
        "utilization_mean_mean",
        "action_entropy_norm_mean", "shock_action_shift_l1_mean",
        "intent_action_shift_l1_mean", "intent_mode_shift_l1_mean",
        "counterfactual_intent_policy_l1_mean", "counterfactual_mode_policy_l1_mean",
        "counterfactual_argmax_change_rate_mean", "frozen_policy_entropy_norm_mean",
        "deny_action_rate_mean", "shield_filter_active_rate_mean", "shield_filtered_action_fraction_mean",
        "post_switch_violation_mean_mean", "switch_adaptation_time_steps_mean",
        "switch_adaptation_success_mean", "queue_load_mean_mean",
        "train_time_s_mean", "train_time_s_std",
        "rank_by_violation", "rank_by_compliance",
        "is_best_violation", "is_best_compliance",
    ]


def write_compact_main_summary(agg_csv: Path, out_csv: Path) -> None:
    """Create the single recommended CSV for post-run analysis.

    It keeps one row per intent/scenario/scheme, adds method-family labels and
    rankings, and drops long raw/diagnostic columns that make the output folder
    hard to navigate.
    """
    import pandas as pd
    if not agg_csv.exists():
        return
    df = pd.read_csv(agg_csv)
    if df.empty:
        return
    df["scheme_label"] = df["scheme"].map(scheme_label)
    df["method_family"] = df["scheme"].map(_scheme_category)
    df["comparison_role"] = df["scheme"].map(scheme_comparison_role)
    # Ranking: lower violation is better; higher compliance is better.
    df["rank_by_violation"] = df.groupby(["intent", "scenario"])["weighted_violation_mean_mean"].rank(method="min", ascending=True)
    df["rank_by_compliance"] = df.groupby(["intent", "scenario"])["soft_intent_compliance_score_mean"].rank(method="min", ascending=False)
    df["is_best_violation"] = (df["rank_by_violation"] == 1).astype(int)
    df["is_best_compliance"] = (df["rank_by_compliance"] == 1).astype(int)
    keep = [c for c in analysis_metric_columns() if c in df.columns]
    compact = df[keep].copy()
    # Stable ordering for quick inspection.
    compact = compact.sort_values(["intent", "scenario", "rank_by_violation", "rank_by_compliance", "scheme"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_csv, index=False)


def _candidate_resume_dirs(profile: str | None = None) -> List[Path]:
    if not OUT_ROOT.exists():
        return []
    dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
    if profile and profile not in {"auto", "custom"}:
        dirs = [p for p in dirs if p.name.endswith(f"_{profile}") or p.name == profile or f"_{profile}" in p.name]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def resolve_output_dir(args, tag: str | None = None) -> Path:
    """Resolve output directory with optional resume/run-id support."""
    if getattr(args, "resume_dir", None):
        return Path(args.resume_dir)
    if getattr(args, "run_id", None):
        return OUT_ROOT / str(args.run_id)
    if getattr(args, "resume", False):
        cands = _candidate_resume_dirs(getattr(args, "profile", None))
        if cands:
            return cands[0]
    ts = now_ts() if tag is None else f"{now_ts()}_{tag}"
    return OUT_ROOT / ts


def write_manifest(outdir: Path, manifest: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "manifest.json"
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except Exception:
            previous = {}
    if previous:
        manifest["created_from_existing_manifest"] = True
        manifest["previous_timestamp"] = previous.get("timestamp", "")
    manifest["last_runner_update"] = now_ts()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def refresh_incremental_outputs(outdir: Path, checkpoint_rows: List[Dict[str, Any]], keep_seed_summary: bool = False, keep_full_aggregated: bool = False) -> None:
    """Update seed checkpoint, aggregate CSV, and compact main CSV after each finished task."""
    checkpoint_path = outdir / "_resume_checkpoint.csv"
    write_rows_csv(checkpoint_path, checkpoint_rows)
    seed_path = outdir / "agentic_seed_summary.csv"
    agg_path = outdir / "agentic_summary_aggregated.csv"
    write_rows_csv(seed_path, checkpoint_rows)
    aggregate_summary(seed_path, agg_path)
    write_compact_main_summary(agg_path, outdir / "agentic_main_summary.csv")
    if not keep_seed_summary and seed_path.exists():
        seed_path.unlink()
    if not keep_full_aggregated and agg_path.exists():
        agg_path.unlink()


def append_failure(outdir: Path, msg: str) -> None:
    path = outdir / "failures.txt"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def build_env(args, intent: IntentProfile, scenario: StressScenario, seed: int, *, steps_override: int | None = None):
    switch_name = getattr(args, "intent_switch_to", None)
    switch_intent = INTENT_LIBRARY.get(switch_name) if switch_name else None
    steps_per_ep = int(args.steps if steps_override is None else steps_override)
    return make_agentic_env(
        intent=intent,
        scenario=scenario,
        steps_per_ep=steps_per_ep,
        n_devices=args.n_devices,
        seed=seed,
        arrival_scale=args.arrival_scale,
        class_mix=tuple(args.class_mix),
        burst_k=args.burst_k,
        device_skew=args.device_skew,
        bits_scale=args.bits_scale,
        delay_penalty_w=args.delay_penalty_w,
        energy_penalty_w=args.energy_penalty_w,
        prio_w=tuple(args.prio_w),
        intent_switch_to=switch_intent,
        intent_switch_frac=float(getattr(args, "intent_switch_frac", 0.50)),
        intent_switch_hold_steps=int(getattr(args, "intent_switch_hold_steps", 10)),
        intent_switch_tolerance=float(getattr(args, "intent_switch_tolerance", 0.05)),
    )


# -----------------------------------------------------------------------------
# Robust scheduler baselines beyond simple RL
# -----------------------------------------------------------------------------

def _obs_unwrap(reset_out):
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _queue_indices(env) -> np.ndarray:
    q = np.asarray(getattr(env, "queue_bits", []), dtype=float)
    return np.flatnonzero(q > 0.0) if q.size else np.array([], dtype=int)


def _candidate_indices(env) -> np.ndarray:
    getter = getattr(env, "get_candidate_devices", None)
    if callable(getter):
        c = np.asarray(getter(), dtype=int)
        return c[c >= 0]
    return _queue_indices(env)


def _action_for_device(env, dev_idx: int, mode: int) -> int:
    encoder = getattr(env, "encode_action_for_device", None)
    if callable(encoder):
        return int(encoder(int(dev_idx), int(mode)))
    return int(mode)


def _first_candidate_action(env, mode: int) -> int:
    candidates = _candidate_indices(env)
    if candidates.size == 0:
        encoder = getattr(env, "encode_action", None)
        return int(encoder(0, MODE_DEFER)) if callable(encoder) else int(MODE_DEFER)
    return _action_for_device(env, int(candidates[0]), mode)


def _class_priority(env, idx: int) -> float:
    c = int(getattr(env, "dev_class", np.zeros(1, dtype=int))[idx])
    return {0: 3.0, 1: 1.8, 2: 1.0}.get(c, 1.0)


def _recent_success(env, window: int = 25) -> float:
    tr = getattr(env, "agentic_trace", []) or []
    if tr:
        vals = [float(x.get("service_success_rate_so_far", np.nan)) for x in tr[-window:]]
        vals = [x for x in vals if np.isfinite(x)]
        if vals:
            return float(vals[-1])
    completed = float(getattr(env, "completed_requests_ep", 0))
    arrivals = float(getattr(env, "total_arrivals_ep", 0))
    return float(np.clip(completed / max(arrivals, 1.0), 0.0, 1.0))


def _device_sinr(env, idx: int) -> float:
    v = np.asarray(getattr(env, "current_sinr_db", []), dtype=float)
    if 0 <= idx < v.size and np.isfinite(v[idx]):
        return float(v[idx])
    h = getattr(env, "sinr_hist", []) or []
    return float(np.nanmean(h[-20:])) if h else 10.0


def policy_all_deny(env, obs, intent: IntentProfile) -> int:
    """Diagnostic: reject the highest-ranked queued request."""
    return _first_candidate_action(env, MODE_REJECT)


def policy_all_grant(env, obs, intent: IntentProfile) -> int:
    """Diagnostic: grant the highest-ranked queued request."""
    return _first_candidate_action(env, MODE_GRANT)


def policy_capacity_first(env, obs, intent: IntentProfile) -> int:
    """High-service feasibility anchor using all available channels when useful.

    It is not a proposed scheduling baseline. Its purpose is to estimate the
    physically achievable service/deadline envelope before intent targets are
    frozen for manuscript evaluation.
    """
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    candidates = _candidate_indices(env)
    if candidates.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    target = int(candidates[0])
    if qidx.size >= min(2, int(getattr(env, "n_channels", 1))):
        return _action_for_device(env, target, MODE_COEXIST)
    return _action_for_device(env, target, MODE_GRANT)


def policy_priority_aging(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    ages = np.asarray(env.queue_age, dtype=float)
    classes = np.asarray(env.dev_class, dtype=int)
    deadline = np.asarray(getattr(env, "deadline_steps", np.full(env.n_devices, env.steps_per_ep)), dtype=float)
    score = np.array([3.0 * _class_priority(env, int(i)) + 4.0 * ages[i] / max(deadline[i], 1.0) for i in qidx])
    target = int(qidx[int(np.argmax(score))])
    mode = MODE_PROTECT if classes[target] == 0 or ages[target] >= 0.70 * deadline[target] else MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_proportional_fair(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    served = np.asarray(getattr(env, "served_bits_per_device_ep", np.zeros(env.n_devices)), dtype=float)
    mean_served = max(float(np.mean(served[qidx])), 1.0)
    scores = []
    for i in qidx:
        deficit = max(0.0, 1.0 - served[i] / mean_served)
        channel = max(0.1, (_device_sinr(env, int(i)) + 20.0) / 60.0)
        scores.append(deficit + 0.45 * channel)
    target = int(qidx[int(np.argmax(scores))])
    mode = MODE_COEXIST if qidx.size > 1 and _recent_success(env) > 0.70 else MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_tdma_preempt(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    classes = np.asarray(env.dev_class, dtype=int)
    ages = np.asarray(env.queue_age, dtype=float)
    emergency = qidx[classes[qidx] == 0]
    target = int(emergency[np.argmax(ages[emergency])]) if emergency.size else int(qidx[np.argmax(ages[qidx])])
    mode = MODE_PROTECT if emergency.size else MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_max_weight(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    ages = np.asarray(env.queue_age, dtype=float)
    qbits = np.asarray(env.queue_bits, dtype=float)
    scores = []
    for i in qidx:
        channel = max(0.15, min(2.0, (_device_sinr(env, int(i)) + 20.0) / 30.0))
        scores.append(_class_priority(env, int(i)) * (1.0 + ages[i]) * math.log1p(qbits[i]) * channel)
    target = int(qidx[int(np.argmax(scores))])
    deadline = np.asarray(getattr(env, "deadline_steps", np.full(env.n_devices, env.steps_per_ep)), dtype=float)
    age_ratio = float(ages[target] / max(deadline[target], 1.0))
    low_sinr = _device_sinr(env, target) < 0.0
    if age_ratio >= 0.70 or (int(env.dev_class[target]) == 0 and low_sinr):
        mode = MODE_PROTECT
    elif qidx.size >= min(2, int(getattr(env, "n_channels", 1))):
        mode = MODE_COEXIST
    else:
        mode = MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_lyapunov_dpp(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    qbits = np.asarray(env.queue_bits, dtype=float)
    ages = np.asarray(env.queue_age, dtype=float)
    battery = np.asarray(env.battery_j, dtype=float)
    cap = np.asarray(env.battery_cap_j, dtype=float)
    scores = []
    for i in qidx:
        drift = _class_priority(env, int(i)) * (qbits[i] / max(env.bits_per_sample[i], 1.0) + 0.7 * ages[i])
        energy_penalty = (1.0 - battery[i] / max(cap[i], EPS))
        radio_penalty = max(0.0, 2.0 - _device_sinr(env, int(i))) * 0.20
        scores.append(drift - energy_penalty - radio_penalty)
    target = int(qidx[int(np.argmax(scores))])
    mode = MODE_PROTECT if int(env.dev_class[target]) == 0 and scores[int(np.argmax(scores))] > 2.0 else MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_sla_aware(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    ages = np.asarray(env.queue_age, dtype=float)
    deadlines = np.asarray(getattr(env, "deadline_steps", np.full(env.n_devices, env.steps_per_ep)), dtype=float)
    ratios = ages[qidx] / np.maximum(deadlines[qidx], 1.0)
    target = int(qidx[int(np.argmax(ratios + np.array([0.4 * _class_priority(env, int(i)) for i in qidx])) )])
    if int(env.dev_class[target]) == 0 or ages[target] >= 0.70 * deadlines[target]:
        mode = MODE_PROTECT
    elif _recent_success(env) < float(intent.min_service_success_rate or 0.80):
        mode = MODE_GRANT
    else:
        mode = MODE_COEXIST if qidx.size > 1 else MODE_GRANT
    return _action_for_device(env, target, mode)


def policy_mpc_lite(env, obs, intent: IntentProfile) -> int:
    qidx = _queue_indices(env)
    if qidx.size == 0:
        return _first_candidate_action(env, MODE_DEFER)
    ages = np.asarray(env.queue_age, dtype=float)
    deadlines = np.asarray(getattr(env, "deadline_steps", np.full(env.n_devices, env.steps_per_ep)), dtype=float)
    qbits = np.asarray(env.queue_bits, dtype=float)
    best = None
    for i in qidx:
        delay_risk = ages[i] / max(deadlines[i], 1.0)
        outage = max(0.0, (3.0 - _device_sinr(env, int(i))) / 15.0)
        priority = _class_priority(env, int(i))
        backlog = math.log1p(qbits[i] / max(env.bits_per_sample[i], 1.0))
        costs = {
            MODE_DEFER: 1.2 * priority * delay_risk + 0.4 * backlog,
            MODE_GRANT: -priority * (1.0 + delay_risk) + 0.35 * outage,
            MODE_PROTECT: -1.15 * priority * (1.0 + delay_risk) + 0.15 * outage + 0.10,
            MODE_COEXIST: -0.65 * priority + 0.50 * delay_risk + 0.55 * outage,
            MODE_REJECT: 2.5 * priority + 2.0 * delay_risk,
        }
        mode = min(costs, key=costs.get)
        candidate = (costs[mode], int(i), int(mode))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return _action_for_device(env, best[1], best[2])


BASELINE_POLICIES: Dict[str, Callable] = {
    "AllDeny": policy_all_deny,
    "AllGrant": policy_all_grant,
    "CapacityFirst": policy_capacity_first,
    "PriorityAging": policy_priority_aging,
    "ProportionalFair": policy_proportional_fair,
    "TDMA-Preempt": policy_tdma_preempt,
    "MaxWeight": policy_max_weight,
    "LyapunovDPP": policy_lyapunov_dpp,
    "SLA-Aware": policy_sla_aware,
    "MPC-Lite": policy_mpc_lite,
}


def run_policy_baseline(env, policy_fn: Callable, intent: IntentProfile, episodes: int, steps_per_ep: int) -> Dict[str, Any]:
    metrics = {
        "rewards_per_episode": [],
        "q_value_max": [],
        "episode_metrics_std": [],
    }
    for _ in range(episodes):
        obs = _obs_unwrap(env.reset())
        ep_reward = 0.0
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        for _t in range(steps_per_ep):
            action = int(policy_fn(env, obs, intent))
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done:
                break
        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc,
        )
        metrics["episode_metrics_std"].append(m)
        metrics["rewards_per_episode"].append(float(ep_reward))
        metrics["q_value_max"].append(0.0)
    return metrics


def _configure_terminal_drain_reference_windows(env, arrival_window_steps: int) -> None:
    """Keep scenario shock and intent-switch timing referenced to the fixed arrival window.

    The drain phase extends only follow-up time. It must not move the predeclared
    shock or intent-switch locations merely because the environment is allowed to
    continue after arrivals stop.
    """
    window = int(arrival_window_steps)
    scenario = getattr(env, "scenario", None)
    if scenario is not None and hasattr(env, "_shock_steps"):
        s0 = int(round(window * float(getattr(scenario, "shock_start_frac", 0.35))))
        s1 = int(round(window * float(getattr(scenario, "shock_end_frac", 0.70))))
        env._shock_steps = (max(0, s0), min(window, max(s0 + 1, s1)))
    if getattr(env, "intent_switch_to", None) is not None and hasattr(env, "_intent_switch_step"):
        frac = float(getattr(env, "intent_switch_frac", 0.50))
        env._intent_switch_step = int(round(window * frac))


def _pending_right_censored_count(env) -> int:
    """Return pending requests whose deadline opportunity has not yet elapsed."""
    try:
        now_step = int(getattr(env, "t", 0))
        queues = getattr(env, "request_queues", [])
        return int(sum(
            1 for q in queues for req in q
            if bool(getattr(req, "active", True))
            and (int(getattr(req, "arrival_step", 0)) + int(getattr(req, "deadline_steps", 0)) > now_step)
        ))
    except Exception:
        return 0


def _terminal_drain_limit(args) -> int:
    requested = int(max(0, getattr(args, "terminal_drain_steps", 0)))
    if not bool(getattr(args, "terminal_drain_auto_extend", False)):
        return requested
    return int(max(requested, getattr(args, "terminal_drain_max_steps", requested)))


def _attach_terminal_drain_metrics(
    terminal_metrics: Dict[str, Any],
    measurement_metrics: Dict[str, Any],
    arrival_window_steps: int,
    requested_drain_steps: int,
    actual_drain_steps: int,
) -> Dict[str, Any]:
    """Combine fixed-window service-rate metrics with terminal cohort outcomes.

    Throughput and utilization remain fixed-window quantities. Request outcome
    rates (completion, deadline exposure, unfinished, censoring) are taken after
    the no-arrival follow-up. This is a sensitivity analysis, not a replacement
    for the already-frozen primary 300-step results.
    """
    out = dict(terminal_metrics)
    # Preserve the service-rate denominator of the original 300-step experiment.
    for key in ("throughput_mbps", "utilization"):
        if key in measurement_metrics:
            out[key] = measurement_metrics[key]
    for key, value in measurement_metrics.items():
        out[f"measurement_{key}"] = value
    out["terminal_total_elapsed_throughput_mbps"] = terminal_metrics.get("throughput_mbps", np.nan)
    out["terminal_total_elapsed_utilization"] = terminal_metrics.get("utilization", np.nan)
    out["arrival_window_steps"] = int(arrival_window_steps)
    out["terminal_drain_requested_steps"] = int(requested_drain_steps)
    out["terminal_drain_steps"] = int(actual_drain_steps)
    out["terminal_drain_extension_steps"] = int(max(0, actual_drain_steps - requested_drain_steps))
    out["terminal_total_steps"] = int(arrival_window_steps + actual_drain_steps)
    out["drain_added_completions"] = int(
        float(terminal_metrics.get("completed_requests_ep", 0) or 0)
        - float(measurement_metrics.get("completed_requests_ep", 0) or 0)
    )
    out["drain_added_deadline_exposures"] = int(
        float(terminal_metrics.get("deadline_exposed_requests_ep", 0) or 0)
        - float(measurement_metrics.get("deadline_exposed_requests_ep", 0) or 0)
    )
    out["drain_removed_unfinished_requests"] = int(
        float(measurement_metrics.get("pending_requests_ep", 0) or 0)
        - float(terminal_metrics.get("pending_requests_ep", 0) or 0)
    )
    return out


def _snapshot_episode_rng_state(env):
    """Capture the exogenous RNG position at the end of the frozen 300-step window.

    The terminal drain consumes extra radio/arrival-success random draws. Those
    draws must not shift the RNG position used to start the next evaluation
    episode, otherwise the later 300-step measurement windows no longer match
    the already-frozen primary run.
    """
    rng = getattr(env, "rng", None)
    bitgen = getattr(rng, "bit_generator", None)
    if bitgen is None:
        return None
    return copy.deepcopy(bitgen.state)


def _restore_episode_rng_state(env, state) -> None:
    if state is None:
        return
    rng = getattr(env, "rng", None)
    bitgen = getattr(rng, "bit_generator", None)
    if bitgen is None:
        raise RuntimeError("Terminal-drain RNG replay requested but environment RNG is unavailable.")
    bitgen.state = copy.deepcopy(state)


def run_policy_baseline_terminal_drain(
    env, policy_fn: Callable, intent: IntentProfile, episodes: int,
    arrival_window_steps: int, drain_steps: int,
    auto_extend: bool = False, max_drain_steps: int | None = None,
) -> Dict[str, Any]:
    """Evaluate a heuristic on a fixed arrival cohort followed by no-arrival drain."""
    metrics = {"rewards_per_episode": [], "q_value_max": [], "episode_metrics_std": []}
    max_drain_steps = int(max(drain_steps, drain_steps if max_drain_steps is None else max_drain_steps))
    for _ in range(int(episodes)):
        obs = _obs_unwrap(env.reset())
        _configure_terminal_drain_reference_windows(env, arrival_window_steps)
        ep_reward = 0.0
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        for _t in range(int(arrival_window_steps)):
            action = int(policy_fn(env, obs, intent))
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done:
                raise RuntimeError("Terminal-drain environment ended before the arrival window completed.")
        measurement = finalize_episode_metrics(
            env, steps_per_ep=int(arrival_window_steps),
            num_channels=getattr(env, "n_channels", 1), slot_s=getattr(env, "slot_s", 0.001), ep_acc=ep_acc,
        )
        # Preserve the exact RNG position that the primary 300-step run would
        # carry into the next episode. The drain is a branch from this state.
        post_measurement_rng_state = _snapshot_episode_rng_state(env)
        setattr(env, "arrivals_enabled", False)
        actual_drain_steps = 0
        while actual_drain_steps < max_drain_steps:
            if actual_drain_steps >= int(drain_steps):
                if (not auto_extend) or _pending_right_censored_count(env) == 0:
                    break
            action = int(policy_fn(env, obs, intent))
            obs, reward, done, info = env.step(action)
            actual_drain_steps += 1
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done and _pending_right_censored_count(env) > 0:
                raise RuntimeError("Terminal-drain environment ended while right-censored requests remained.")
            if done:
                break
        if auto_extend and _pending_right_censored_count(env) > 0:
            raise RuntimeError(
                f"Adaptive terminal drain hit its {max_drain_steps}-step safety cap while "
                f"{_pending_right_censored_count(env)} right-censored requests remained."
            )
        total_steps = int(arrival_window_steps + actual_drain_steps)
        terminal = finalize_episode_metrics(
            env, steps_per_ep=total_steps, num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001), ep_acc=ep_acc,
        )
        m = _attach_terminal_drain_metrics(
            terminal, measurement, arrival_window_steps, int(drain_steps), actual_drain_steps
        )
        m["max_realized_device_deadline_steps"] = int(np.max(np.asarray(getattr(env, "deadline_steps", [0]), dtype=int)))
        m["measurement_rng_stream_preserved"] = 1
        # Discard only the RNG draws consumed by the no-arrival branch so the
        # next episode starts from exactly the same stochastic stream as the
        # frozen primary evaluation.
        _restore_episode_rng_state(env, post_measurement_rng_state)
        metrics["episode_metrics_std"].append(m)
        metrics["rewards_per_episode"].append(float(ep_reward))
        metrics["q_value_max"].append(0.0)
    return metrics


def _counterfactual_intent_metrics(policy, env, obs) -> Dict[str, float]:
    """Evaluate one frozen actor on the same physical state under three intents.

    The audit uses the common physical-feasibility mask rather than the proposed
    safety shield, so it measures the actor's own dependence on intent context.
    """
    if policy is None or not hasattr(policy, "action_probabilities"):
        return {}
    intents = [
        INTENT_LIBRARY["balanced_agentic"],
        INTENT_LIBRARY["reliability_first"],
        INTENT_LIBRARY["latency_critical"],
    ]
    x = np.asarray(obs, dtype=np.float32).ravel()
    ctx_dim = int(intent_context_vector(intents[0]).size)
    if x.size <= ctx_dim:
        return {}
    physical = x[:-ctx_dim]
    common_getter = getattr(env, "get_feasible_action_mask", None)
    common_mask = common_getter() if callable(common_getter) else None
    probs = []
    for intent in intents:
        cf_obs = np.concatenate([physical, intent_context_vector(intent)]).astype(np.float32)
        prob = np.asarray(policy.action_probabilities(cf_obs, action_mask=common_mask), dtype=float).ravel()
        if prob.size == 0 or not np.isfinite(prob).all():
            return {}
        probs.append(prob / max(float(prob.sum()), 1e-12))
    pairs = [(0, 1), (0, 2), (1, 2)]
    action_l1 = float(np.mean([np.sum(np.abs(probs[i] - probs[j])) for i, j in pairs]))
    n_modes = int(getattr(env, "n_action_modes", 5))
    mode_probs = []
    for prob in probs:
        mp = np.zeros(n_modes, dtype=float)
        for action, value in enumerate(prob):
            mp[int(action) % n_modes] += float(value)
        mode_probs.append(mp)
    mode_l1 = float(np.mean([np.sum(np.abs(mode_probs[i] - mode_probs[j])) for i, j in pairs]))
    argmax_change = float(len({int(np.argmax(prob)) for prob in probs}) > 1)
    active_name = getattr(getattr(env, "get_active_intent", lambda: intents[0])(), "name", "balanced_agentic")
    idx = {"balanced_agentic": 0, "reliability_first": 1, "latency_critical": 2}.get(active_name, 0)
    active = probs[idx]
    nz = active[active > 0]
    entropy = float(-np.sum(nz * np.log(nz + 1e-12)) / max(np.log(max(len(active), 2)), 1e-12))
    return {
        "counterfactual_intent_policy_l1": action_l1,
        "counterfactual_mode_policy_l1": mode_l1,
        "counterfactual_argmax_change_rate": argmax_change,
        "frozen_policy_entropy_norm": entropy,
    }


def _normalize_masked_probabilities(raw_probs, mask=None) -> np.ndarray:
    """Normalize a categorical actor distribution after an optional boolean mask."""
    p = np.asarray(raw_probs, dtype=float).ravel().copy()
    if p.size == 0 or not np.isfinite(p).all():
        raise ValueError("Policy returned an empty or non-finite probability vector.")
    p = np.maximum(p, 0.0)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.size != p.size:
            raise ValueError(f"Action-mask size {m.size} does not match policy size {p.size}.")
        if np.any(m):
            p[~m] = 0.0
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= 0.0:
        if mask is not None and np.any(np.asarray(mask, dtype=bool)):
            p = np.asarray(mask, dtype=bool).astype(float)
        else:
            p = np.ones_like(p, dtype=float)
        total = float(np.sum(p))
    return p / max(total, 1e-12)


def _sample_categorical_common_uniform(probs, u: float) -> int:
    """Inverse-CDF categorical sampling driven by a caller-supplied common uniform."""
    p = _normalize_masked_probabilities(probs)
    uu = float(np.clip(float(u), 0.0, np.nextafter(1.0, 0.0)))
    idx = int(np.searchsorted(np.cumsum(p), uu, side="right"))
    return min(idx, int(p.size) - 1)


def _policy_distribution_diagnostics(policy, env, obs, u: float) -> tuple[int, Dict[str, float]]:
    """Return a stochastic deployment action and V4.4E probability-mass diagnostics.

    Physical feasibility is applied to every method. The proposed shield may then
    remove additional unsafe actions. The same ``u`` is also used to sample a
    counterfactual pre-shield action, so ``actual_shield_prevention_rate`` is a
    reproducible common-random-number estimate of how often the shield prevents
    an action the frozen actor would otherwise execute.
    """
    raw = np.asarray(policy.action_probabilities(obs, action_mask=None), dtype=float).ravel()
    raw = _normalize_masked_probabilities(raw)
    feasible_getter = getattr(env, "get_feasible_action_mask", None)
    feasible_mask = (
        np.asarray(feasible_getter(), dtype=bool).ravel()
        if callable(feasible_getter) else np.ones(raw.size, dtype=bool)
    )
    action_getter = getattr(env, "get_action_mask", None)
    final_mask = (
        np.asarray(action_getter(), dtype=bool).ravel()
        if callable(action_getter) else feasible_mask.copy()
    )
    if feasible_mask.size != raw.size or final_mask.size != raw.size:
        raise ValueError("Environment/action-policy mask dimension mismatch during V4.4E evaluation.")
    # The deployment mask may only further restrict common physical feasibility.
    final_mask = final_mask & feasible_mask
    if not np.any(final_mask):
        final_mask = feasible_mask.copy()

    feasible_probs = _normalize_masked_probabilities(raw, feasible_mask)
    final_probs = _normalize_masked_probabilities(raw, final_mask)
    action = _sample_categorical_common_uniform(final_probs, u)
    pre_shield_action = _sample_categorical_common_uniform(feasible_probs, u)

    shield_removed = feasible_mask & ~final_mask
    n_modes = int(getattr(env, "n_action_modes", 5))
    reject_actions = np.asarray([(a % n_modes) == MODE_REJECT for a in range(raw.size)], dtype=bool)
    nz = final_probs[final_probs > 0.0]
    entropy_nats = float(-np.sum(nz * np.log(nz + 1e-12))) if nz.size else 0.0
    entropy_norm = float(entropy_nats / max(np.log(max(raw.size, 2)), 1e-12))

    diag: Dict[str, float] = {
        "pre_mask_reject_probability": float(np.sum(raw[reject_actions])),
        "feasible_reject_probability": float(np.sum(feasible_probs[reject_actions])),
        "physical_infeasible_probability_mass": float(np.sum(raw[~feasible_mask])),
        "pre_mask_unsafe_probability_mass": float(np.sum(raw[~final_mask])),
        # Conditional on actions that are physically executable. This is the
        # cleanest estimate of actor probability removed specifically by safety.
        "shield_filtered_probability_mass": float(np.sum(feasible_probs[shield_removed])),
        "shield_filtered_raw_probability_mass": float(np.sum(raw[shield_removed])),
        "actual_shield_prevention_rate": float(bool(shield_removed[pre_shield_action])),
        "stochastic_action_entropy": entropy_norm,
        "stochastic_action_entropy_nats": entropy_nats,
    }
    mode_names = ["defer", "grant", "protect", "coexist", "reject"]
    for mode, name in enumerate(mode_names[:n_modes]):
        diag[f"policy_mode_prob_{name}"] = float(
            np.sum([final_probs[a] for a in range(final_probs.size) if (a % n_modes) == mode])
        )
    return int(action), diag


def run_trained_policy_evaluation(
    env, policy, episodes: int, steps_per_ep: int,
    action_mode: str = "greedy", sampling_seed: int | None = None,
) -> Dict[str, Any]:
    """Evaluate one frozen policy with greedy or stochastic deployment.

    V4.4E stochastic evaluation samples the frozen PPO categorical distribution
    using a deterministic common-random-number uniform indexed by episode and
    step. No network parameter is updated and the training checkpoint is not
    modified.
    """
    if policy is None or not hasattr(policy, "act"):
        raise ValueError("The trained scheme did not return an inference policy.")
    action_mode = str(action_mode).strip().lower()
    if action_mode not in {"greedy", "stochastic"}:
        raise ValueError(f"Unsupported evaluation action mode: {action_mode}")
    sampling_seed = int(0 if sampling_seed is None else sampling_seed)
    metrics = {
        "rewards_per_episode": [],
        "q_value_max": [],
        "episode_metrics_std": [],
    }
    known_intents = ["balanced_agentic", "reliability_first", "latency_critical"]
    mode_names = ["defer", "grant", "protect", "coexist", "reject"]

    for ep_idx in range(int(episodes)):
        obs = _obs_unwrap(env.reset())
        ep_reward = 0.0
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        cf_rows = []
        dist_rows: List[Dict[str, float]] = []
        conditioned: Dict[str, List[Dict[str, float]]] = {name: [] for name in known_intents}
        for step_idx in range(int(steps_per_ep)):
            cf = _counterfactual_intent_metrics(policy, env, obs)
            if cf:
                cf_rows.append(cf)

            # One indexed uniform per episode/step gives all learned schemes the
            # same action-sampling random stream even if prior trajectories end
            # at different times.
            u_seed = stable_int_seed("v44e_common_action_uniform", sampling_seed, ep_idx, step_idx)
            u = float(np.random.default_rng(u_seed).random())
            if hasattr(policy, "action_probabilities"):
                stochastic_action, diag = _policy_distribution_diagnostics(policy, env, obs, u)
                dist_rows.append(diag)
                active_getter = getattr(env, "get_active_intent", None)
                active_name = getattr(active_getter(), "name", "balanced_agentic") if callable(active_getter) else "balanced_agentic"
                if active_name in conditioned:
                    conditioned[active_name].append(diag)
            else:
                stochastic_action, diag = None, {}

            if action_mode == "stochastic" and stochastic_action is not None:
                action = int(stochastic_action)
            else:
                mask_getter = getattr(env, "get_action_mask", None)
                action_mask = mask_getter() if callable(mask_getter) else None
                try:
                    action = int(policy.act(obs, action_mask=action_mask))
                except TypeError:
                    action = int(policy.act(obs))

            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done:
                break

        m = finalize_episode_metrics(
            env,
            steps_per_ep=steps_per_ep,
            num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001),
            ep_acc=ep_acc,
        )
        extra_summary: Dict[str, float] = {}
        if cf_rows:
            for key in cf_rows[0].keys():
                vals = [row[key] for row in cf_rows if key in row and np.isfinite(row[key])]
                if vals:
                    extra_summary[key] = float(np.mean(vals))
        if dist_rows:
            for key in dist_rows[0].keys():
                vals = [row[key] for row in dist_rows if key in row and np.isfinite(row[key])]
                if vals:
                    extra_summary[key] = float(np.mean(vals))
            for intent_name, rows in conditioned.items():
                if not rows:
                    continue
                for mode_name in mode_names:
                    key = f"policy_mode_prob_{mode_name}"
                    vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
                    if vals:
                        extra_summary[f"intent_mode_prob__{intent_name}__{mode_name}"] = float(np.mean(vals))

        if extra_summary and getattr(env, "agentic_episode_summaries", None):
            env.agentic_episode_summaries[-1].update(extra_summary)
        m.update(extra_summary)
        metrics["episode_metrics_std"].append(m)
        metrics["rewards_per_episode"].append(float(ep_reward))
        metrics["q_value_max"].append(0.0)
    return metrics


def run_trained_policy_evaluation_terminal_drain(
    env, policy, episodes: int, arrival_window_steps: int, drain_steps: int,
    action_mode: str = "stochastic", sampling_seed: int | None = None,
    auto_extend: bool = False, max_drain_steps: int | None = None,
) -> Dict[str, Any]:
    """Frozen-policy right-censoring sensitivity with a no-arrival deadline drain."""
    if policy is None or not hasattr(policy, "act"):
        raise ValueError("The trained scheme did not return an inference policy.")
    action_mode = str(action_mode).strip().lower()
    if action_mode not in {"greedy", "stochastic"}:
        raise ValueError(f"Unsupported evaluation action mode: {action_mode}")
    sampling_seed = int(0 if sampling_seed is None else sampling_seed)
    metrics = {"rewards_per_episode": [], "q_value_max": [], "episode_metrics_std": []}
    known_intents = ["balanced_agentic", "reliability_first", "latency_critical"]
    mode_names = ["defer", "grant", "protect", "coexist", "reject"]
    max_drain_steps = int(max(drain_steps, drain_steps if max_drain_steps is None else max_drain_steps))

    def choose_action(obs, ep_idx: int, step_idx: int, collect_diag: bool = True):
        u_seed = stable_int_seed("v44e_common_action_uniform", sampling_seed, ep_idx, step_idx)
        u = float(np.random.default_rng(u_seed).random())
        diag = {}
        stochastic_action = None
        if hasattr(policy, "action_probabilities"):
            stochastic_action, diag = _policy_distribution_diagnostics(policy, env, obs, u)
        if action_mode == "stochastic" and stochastic_action is not None:
            return int(stochastic_action), diag
        mask_getter = getattr(env, "get_action_mask", None)
        action_mask = mask_getter() if callable(mask_getter) else None
        try:
            return int(policy.act(obs, action_mask=action_mask)), diag
        except TypeError:
            return int(policy.act(obs)), diag

    for ep_idx in range(int(episodes)):
        obs = _obs_unwrap(env.reset())
        _configure_terminal_drain_reference_windows(env, arrival_window_steps)
        ep_reward = 0.0
        ep_acc = new_episode_acc(getattr(env, "n_classes", 3))
        cf_rows: List[Dict[str, float]] = []
        dist_rows: List[Dict[str, float]] = []
        conditioned: Dict[str, List[Dict[str, float]]] = {name: [] for name in known_intents}

        for step_idx in range(int(arrival_window_steps)):
            cf = _counterfactual_intent_metrics(policy, env, obs)
            if cf:
                cf_rows.append(cf)
            action, diag = choose_action(obs, ep_idx, step_idx)
            if diag:
                dist_rows.append(diag)
                active_getter = getattr(env, "get_active_intent", None)
                active_name = getattr(active_getter(), "name", "balanced_agentic") if callable(active_getter) else "balanced_agentic"
                if active_name in conditioned:
                    conditioned[active_name].append(diag)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done:
                raise RuntimeError("Terminal-drain environment ended before the arrival window completed.")

        measurement = finalize_episode_metrics(
            env, steps_per_ep=int(arrival_window_steps), num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001), ep_acc=ep_acc,
        )
        # Branch the terminal follow-up from the frozen measurement state while
        # preserving the RNG position required for the next primary-style episode.
        post_measurement_rng_state = _snapshot_episode_rng_state(env)
        setattr(env, "arrivals_enabled", False)

        # Follow the frozen policy for the predeclared minimum drain. If realized
        # per-device deadline jitter makes any pending request still right-censored,
        # continue with no new arrivals only until every cohort request has received
        # its complete deadline opportunity. Diagnostics above remain restricted to
        # the original 300-step arrival window.
        actual_drain_steps = 0
        while actual_drain_steps < max_drain_steps:
            if actual_drain_steps >= int(drain_steps):
                if (not auto_extend) or _pending_right_censored_count(env) == 0:
                    break
            step_idx = int(arrival_window_steps + actual_drain_steps)
            action, _diag = choose_action(obs, ep_idx, step_idx, collect_diag=False)
            obs, reward, done, info = env.step(action)
            actual_drain_steps += 1
            ep_reward += float(reward)
            accumulate_from_info(ep_acc, info, n_classes=getattr(env, "n_classes", 3))
            if done and _pending_right_censored_count(env) > 0:
                raise RuntimeError("Terminal-drain environment ended while right-censored requests remained.")
            if done:
                break
        if auto_extend and _pending_right_censored_count(env) > 0:
            raise RuntimeError(
                f"Adaptive terminal drain hit its {max_drain_steps}-step safety cap while "
                f"{_pending_right_censored_count(env)} right-censored requests remained."
            )

        total_steps = int(arrival_window_steps + actual_drain_steps)
        terminal = finalize_episode_metrics(
            env, steps_per_ep=total_steps, num_channels=getattr(env, "n_channels", 1),
            slot_s=getattr(env, "slot_s", 0.001), ep_acc=ep_acc,
        )
        m = _attach_terminal_drain_metrics(
            terminal, measurement, arrival_window_steps, int(drain_steps), actual_drain_steps
        )
        m["max_realized_device_deadline_steps"] = int(np.max(np.asarray(getattr(env, "deadline_steps", [0]), dtype=int)))
        m["measurement_rng_stream_preserved"] = 1
        _restore_episode_rng_state(env, post_measurement_rng_state)

        extra_summary: Dict[str, float] = {}
        if cf_rows:
            for key in cf_rows[0].keys():
                vals = [row[key] for row in cf_rows if key in row and np.isfinite(row[key])]
                if vals:
                    extra_summary[key] = float(np.mean(vals))
        if dist_rows:
            for key in dist_rows[0].keys():
                vals = [row[key] for row in dist_rows if key in row and np.isfinite(row[key])]
                if vals:
                    extra_summary[key] = float(np.mean(vals))
            for intent_name, rows in conditioned.items():
                if not rows:
                    continue
                for mode_name in mode_names:
                    key = f"policy_mode_prob_{mode_name}"
                    vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
                    if vals:
                        extra_summary[f"intent_mode_prob__{intent_name}__{mode_name}"] = float(np.mean(vals))
        if extra_summary and getattr(env, "agentic_episode_summaries", None):
            env.agentic_episode_summaries[-1].update(extra_summary)
        m.update(extra_summary)
        metrics["episode_metrics_std"].append(m)
        metrics["rewards_per_episode"].append(float(ep_reward))
        metrics["q_value_max"].append(0.0)
    return metrics


def configure_evaluation_environment(scheme: str, env, args) -> None:
    """Apply only deployment-time mechanisms to a fresh evaluation environment."""
    if hasattr(env, "configure_admission_shield"):
        env.configure_admission_shield(False)
        if scheme == "Intent-PPO-CMDP":
            env.configure_admission_shield(
                True,
                age_fraction=float(getattr(args, "shield_age_fraction", 0.70)),
                queue_guard=float(getattr(args, "shield_queue_guard", 0.35)),
            )



# -----------------------------------------------------------------------------
# Agentic reward shaping utilities
# -----------------------------------------------------------------------------

def _metric_or_trace(info: Dict[str, Any], env, aliases: Tuple[str, ...], default: float = 0.0) -> float:
    for key in aliases:
        try:
            value = float(info.get(key, np.nan))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    trace = getattr(env, "agentic_trace", []) or []
    if trace:
        for key in aliases:
            try:
                value = float(trace[-1].get(key, np.nan))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return float(default)


def _active_intent(env, fallback: IntentProfile) -> IntentProfile:
    getter = getattr(env, "get_active_intent", None)
    return getter() if callable(getter) else fallback


def _extract_step_violation(env, info: Dict[str, Any], intent: IntentProfile) -> float:
    """Use the same smoothed request-level violation as the environment monitor."""
    return float(local_step_violation(dict(info or {}), _active_intent(env, intent)))


def _extract_step_throughput_mbps(env, info: Dict[str, Any]) -> float:
    info = info or {}
    value = _metric_or_trace(info, env, ("throughput_mbps", "throughput", "service_mbps"), 0.0)
    return max(0.0, float(value) / 1e6 if "throughput" in info and "throughput_mbps" not in info else float(value))


def install_agentic_reward_controller(env, intent: IntentProfile, args, mode: str) -> None:
    """Install non-destructive V9 reward compiler and risk-memory wrapper.

    ``reliability_only``: internal non-agentic ablation.
    ``full_no_tail``: dynamic intent compiler with no tail memory.
    ``full_risk``: compiler + rolling CVaR pressure + guarded recovery.
    """
    if getattr(env, "_agentic_reward_controller_installed", False):
        return
    base_step = env.step
    window = deque(maxlen=int(max(10, getattr(args, "intent_tail_window", 40))))

    def wrapped_step(action):
        if int(getattr(env, "t", 0)) == 0:
            window.clear()
        obs, reward, done, info = base_step(action)
        info = dict(info) if isinstance(info, dict) else {}
        active = _active_intent(env, intent)
        violation = _extract_step_violation(env, info, active)
        outcome_pressure = float(intent_outcome_pressure(info, active))
        # The V4.4 risk signal mixes normalized target violation with an
        # intent-specific outcome pressure. It is action-agnostic: only measured
        # consequences and queue state enter the compiler.
        risk_signal = 0.55 * float(violation) + 0.45 * outcome_pressure
        window.append(float(risk_signal))
        arr = np.asarray(window, dtype=float)
        recent = float(np.nanmean(arr)) if arr.size else float(risk_signal)
        if arr.size:
            q = float(np.nanquantile(arr, float(getattr(args, "intent_tail_cvar_alpha", 0.90))))
            tail = float(np.nanmean(arr[arr >= q])) if np.any(arr >= q) else recent
        else:
            tail = recent

        scale = float(getattr(args, "intent_reward_penalty_scale", 0.30))
        if mode == "reliability_only":
            reward = float(reward) - 0.55 * scale * violation
        else:
            # Immediate outcome pressure creates intent differentiation, while
            # the rolling mean and upper-tail excess regulate transient risk.
            risk_penalty = scale * (0.45 * violation + 0.40 * outcome_pressure + 0.15 * recent)
            if mode == "full_risk":
                tail_excess = max(0.0, tail - recent)
                risk_penalty += float(getattr(args, "intent_tail_penalty_scale", 0.18)) * tail_excess
            reward = float(reward) - risk_penalty
            # Penalize shield intervention so the actor learns to avoid requests that
            # would be overridden at deployment rather than relying on the shield.
            reward -= 0.06 * float(info.get("shield_override", 0.0) or 0.0)
            # Recovery reward is gated by all-arrival risk rather than delivered-packet delay.
            exposure = float(info.get("deadline_exposure_rate", 0.0) or 0.0)
            blocked = float(info.get("blocking_rate_so_far", 0.0) or 0.0)
            interrupted = float(info.get("interruption_rate_so_far", 0.0) or 0.0)
            safe = (exposure <= float(active.max_deadline_exposure_rate or 1.0)
                    and blocked <= float(active.block_target)
                    and interrupted <= float(active.intr_target))
            if safe:
                thr = _extract_step_throughput_mbps(env, info)
                floor = float(active.throughput_floor_mbps or 20.0)
                recovery = float(getattr(args, "intent_throughput_recovery_scale", 0.012)) * math.tanh(thr / max(floor, 1e-12))
                reward = float(reward) + recovery
        info["risk_memory_mean"] = recent
        info["risk_memory_cvar"] = tail
        info["compiled_step_violation"] = violation
        info["intent_outcome_pressure"] = outcome_pressure
        info["risk_signal"] = risk_signal
        return obs, float(np.clip(reward, -5.0, 5.0)), done, info

    env.step = wrapped_step
    env._agentic_reward_controller_installed = True
    env._agentic_reward_controller_mode = mode


def install_ppo_lagrangian_controller(env, intent: IntentProfile, args) -> None:
    """Projected primal-dual PPO baseline using the same measurable safety signals.

    It receives the public intent vector but has no risk memory, no action shield,
    and no compiler-specific guarded recovery.
    """
    if getattr(env, "_ppo_lagrangian_controller_installed", False):
        return
    base_step = env.step
    state = {
        "lambda_block": float(getattr(args, "lagrangian_lambda_init", 0.40)),
        "lambda_intr": float(getattr(args, "lagrangian_lambda_init", 0.40)),
        "lambda_deadline": float(getattr(args, "lagrangian_lambda_init", 0.40)),
    }
    dual_lr = float(getattr(args, "lagrangian_dual_lr", 0.012))
    lambda_max = float(getattr(args, "lagrangian_lambda_max", 25.0))

    def wrapped_step(action):
        obs, reward, done, info = base_step(action)
        info = dict(info) if isinstance(info, dict) else {}
        active = _active_intent(env, intent)
        block = max(0.0, _metric_or_trace(info, env, ("blocking_rate_so_far", "blocking_prob"), 0.0))
        intr = max(0.0, _metric_or_trace(info, env, ("interruption_rate_so_far", "interrupt_prob"), 0.0))
        deadline = max(0.0, _metric_or_trace(info, env, ("deadline_exposure_rate",), 0.0))
        deadline_target = float(active.max_deadline_exposure_rate if active.max_deadline_exposure_rate is not None else 0.20)
        for key, event, target in (("lambda_block", block, active.block_target), ("lambda_intr", intr, active.intr_target), ("lambda_deadline", deadline, deadline_target)):
            state[key] = float(np.clip(state[key] + dual_lr * (event - float(target)), 0.0, lambda_max))
        penalty = (
            state["lambda_block"] * max(0.0, block - active.block_target)
            + state["lambda_intr"] * max(0.0, intr - active.intr_target)
            + state["lambda_deadline"] * max(0.0, deadline - deadline_target)
        )
        info.update({
            "lagrangian_lambda_block": state["lambda_block"],
            "lagrangian_lambda_intr": state["lambda_intr"],
            "lagrangian_lambda_deadline": state["lambda_deadline"],
            "lagrangian_penalty": float(penalty),
        })
        return obs, float(np.clip(float(reward) - float(penalty), -5.0, 5.0)), done, info

    env.step = wrapped_step
    env._ppo_lagrangian_controller_installed = True
    env.ppo_lagrangian_state = state


def _enrich_agentic_episode_metrics(metrics: Dict[str, Any], env) -> None:
    rows = metrics.get("episode_metrics_std", []) if isinstance(metrics, dict) else []
    summaries = getattr(env, "agentic_episode_summaries", []) or []
    for row, summary in zip(rows, summaries):
        if isinstance(row, dict) and isinstance(summary, dict):
            row.update(summary)

# -----------------------------------------------------------------------------
# Learner dispatch
# -----------------------------------------------------------------------------

def _metric_or_trace(info: Dict[str, Any], env, aliases: Tuple[str, ...], default: float = 0.0) -> float:
    """Read a scalar event from info or the latest agentic trace."""
    for key in aliases:
        try:
            value = float(info.get(key, np.nan))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    trace = getattr(env, "agentic_trace", []) or []
    if trace:
        for key in aliases:
            try:
                value = float(trace[-1].get(key, np.nan))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return float(default)


FULL_POLICY_NO_SHIELD = "Intent-PPO-CMDP-FullPolicy-NoShield"

UNIVERSAL_PPO_SCHEMES = {
    "PPO", "PPO-Lagrangian", "PPO_CMDP",
    "Intent-PPO-CMDP-NoRisk", "Intent-PPO-CMDP-NoShield", "Intent-PPO-CMDP",
    FULL_POLICY_NO_SHIELD,
}

UNIVERSAL_VALUE_SCHEMES = {"SOTA-DQN", "SOTA-DuelingDDQN"}
UNIVERSAL_LEARNED_SCHEMES = UNIVERSAL_PPO_SCHEMES | UNIVERSAL_VALUE_SCHEMES

def _checkpoint_source_scheme(scheme: str) -> str:
    """Map causal deployment-only ablations to the checkpoint they must reuse."""
    return "Intent-PPO-CMDP" if str(scheme) == FULL_POLICY_NO_SHIELD else str(scheme)

def _training_steps(args) -> int:
    """Training horizon, decoupled from the held-out evaluation horizon."""
    value = getattr(args, "training_steps", None)
    return int(args.steps if value is None else value)


def _policy_fingerprint(policy) -> str:
    """Stable short hash of frozen network parameters for cross-context audits."""
    net = getattr(policy, "net", None)
    if net is None or not hasattr(net, "state_dict"):
        return ""
    h = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        h.update(str(name).encode("utf-8"))
        try:
            arr = tensor.detach().cpu().numpy()
            h.update(arr.tobytes(order="C"))
        except Exception:
            h.update(repr(tensor).encode("utf-8"))
    return h.hexdigest()[:20]


def configure_v44_training_curriculum(env, args, policy_seed: int) -> None:
    """Expose one policy to multiple intents, switch times, and stress profiles."""
    if not bool(getattr(args, "multi_intent_training", True)):
        return
    fn = getattr(env, "configure_multi_intent_training", None)
    if not callable(fn):
        return
    intent_names = split_csv(getattr(args, "training_intents", "balanced_agentic,reliability_first,latency_critical"))
    scenario_names = split_csv(getattr(args, "training_scenarios", "nominal,congestion_burst,emergency_surge,mixed_stress,admission_pressure"))
    intents = [INTENT_LIBRARY[x] for x in intent_names]
    scenarios = [SCENARIO_LIBRARY[x] for x in scenario_names]
    fracs = split_float_csv(getattr(args, "training_switch_fracs", "0.35,0.65"))
    fn(
        intents=intents,
        scenarios=scenarios,
        curriculum_seed=stable_int_seed("v44_curriculum", int(policy_seed)),
        switch_fractions=fracs,
        static_probability=float(getattr(args, "training_static_probability", 0.20)),
    )


def _universal_training_signature(args) -> str:
    payload = {
        "revision": "request_level_universal_v4_4",
        "episodes": int(args.episodes),
        "steps": int(_training_steps(args)),
        "n_devices": int(args.n_devices),
        "arrival_scale": float(args.arrival_scale),
        "bits_scale": float(args.bits_scale),
        "class_mix": list(args.class_mix),
        "device_skew": float(args.device_skew),
        "intent_reward_penalty_scale": float(args.intent_reward_penalty_scale),
        "intent_tail_window": int(args.intent_tail_window),
        "intent_tail_cvar_alpha": float(args.intent_tail_cvar_alpha),
        "intent_tail_penalty_scale": float(args.intent_tail_penalty_scale),
        "intent_throughput_recovery_scale": float(args.intent_throughput_recovery_scale),
        "shield_age_fraction": float(args.shield_age_fraction),
        "shield_queue_guard": float(args.shield_queue_guard),
        "training_intents": split_csv(args.training_intents),
        "training_scenarios": split_csv(args.training_scenarios),
        "training_switch_fracs": split_float_csv(args.training_switch_fracs),
        "training_static_probability": float(args.training_static_probability),
        "ia_lr": float(getattr(args, "ia_lr", 1e-4)),
        "baseline_lr": float(getattr(args, "baseline_lr", 2e-4)),
        "ia_entropy_start": float(args.ia_entropy_start),
        "ia_entropy_end": float(args.ia_entropy_end),
        "baseline_entropy_start": float(args.baseline_entropy_start),
        "baseline_entropy_end": float(args.baseline_entropy_end),
        "intent_profiles": {
            name: INTENT_LIBRARY[name].to_dict()
            for name in split_csv(args.training_intents)
        },
        "training_stress_profiles": {
            name: SCENARIO_LIBRARY[name].__dict__
            for name in split_csv(args.training_scenarios)
        },
        "lagrangian_dual_lr": float(args.lagrangian_dual_lr),
        "lagrangian_lambda_init": float(args.lagrangian_lambda_init),
        "lagrangian_lambda_max": float(args.lagrangian_lambda_max),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _universal_policy_checkpoint(args, seed: int, scheme: str) -> Path:
    root = Path(getattr(args, "policy_cache_dir", "policy_cache_v44"))
    signature = _universal_training_signature(args)
    source_scheme = _checkpoint_source_scheme(scheme)
    safe_scheme = str(source_scheme).replace("/", "_").replace(" ", "_")
    return root / signature / f"{safe_scheme}__seed{int(seed)}.pt"


def train_scheme(scheme: str, env, args, intent: IntentProfile, device, policy_seed: int):
    t0 = time.perf_counter()
    train_steps = _training_steps(args)
    cmdp_lr = args.cmdp_lr
    block_tgt = intent.block_target
    intr_tgt = intent.intr_target
    energy_tgt = args.energy_j_target

    # V9 avoids duplicate shaping inside the environment. All reward compilation
    # happens in an explicit wrapper below, preserving base simulator dynamics.
    setattr(env, "enable_intent_reward_shaping", False)
    setattr(env, "intent_reward_mode", "off")
    if hasattr(env, "configure_admission_shield"):
        env.configure_admission_shield(False)

    if scheme in UNIVERSAL_LEARNED_SCHEMES:
        configure_v44_training_curriculum(env, args, policy_seed)

    if scheme in BASELINE_POLICIES:
        # Heuristic schedulers have no training stage; they are evaluated on the
        # same held-out exogenous sequence as the learned policies below.
        metrics = {"rewards_per_episode": [], "q_value_max": [], "episode_metrics_std": [], "policy": None}
    elif scheme == "DoubleQ":
        metrics = hiot_doubleq_train(env, episodes=args.episodes, steps_per_ep=train_steps,
                                     block_target=block_tgt, intr_target=intr_tgt,
                                     energy_j_target=energy_tgt, lr=cmdp_lr, seed=policy_seed)
    elif scheme == "SOTA-DQN":
        metrics = train_value_dqn(
            env, episodes=args.episodes, steps_per_ep=train_steps, seed=policy_seed, device=device,
            architecture="standard", double_dqn=False, cfg=ValueDQNConfig(),
        )
    elif scheme == "SOTA-DuelingDDQN":
        metrics = train_value_dqn(
            env, episodes=args.episodes, steps_per_ep=train_steps, seed=policy_seed, device=device,
            architecture="dueling", double_dqn=True, cfg=ValueDQNConfig(),
        )
    elif scheme == "DQN":
        metrics = hiot_dqn_train(env, episodes=args.episodes, steps_per_ep=train_steps, cfg=DQNCfg(constr_lr=0.0), device=device)
    elif scheme == "Q-Learning":
        metrics = hiot_qlearning_train(env, episodes=args.episodes, steps_per_ep=train_steps, rng=np.random.default_rng(policy_seed))
    elif scheme in {"DQN_CMDP", "Intent-DQN-CMDP"}:
        metrics = hiot_dqn_train(env, episodes=args.episodes, steps_per_ep=train_steps,
                                 cfg=DQNCfg(constr_block_target=block_tgt, constr_intr_target=intr_tgt,
                                            constr_energy_j_target=energy_tgt, constr_lr=cmdp_lr), device=device)
    elif scheme == "DuelingDQN":
        metrics = hiot_dueling_dqn_train(env, episodes=args.episodes, steps_per_ep=train_steps,
                                         cfg=DuelingCfg(constr_lr=0.0), device=device)
    elif scheme in {"DuelingDQN_CMDP", "Intent-DuelingDQN-CMDP"}:
        metrics = hiot_dueling_dqn_train(env, episodes=args.episodes, steps_per_ep=train_steps,
                                         cfg=DuelingCfg(constr_block_target=block_tgt, constr_intr_target=intr_tgt,
                                                        constr_energy_j_target=energy_tgt, constr_lr=cmdp_lr), device=device)
    elif scheme == "ActorCritic":
        metrics = hiot_actor_critic_train(env, episodes=args.episodes, steps_per_ep=train_steps, seed=policy_seed, device=device)
    elif scheme == "PPO":
        metrics = hiot_ppo_train(env, episodes=args.episodes, steps_per_ep=train_steps, cfg=PPOCfg(lr=float(getattr(args, "baseline_lr", 2e-4)), ent_coef=float(getattr(args, "baseline_entropy_start", 0.030)), entropy_coef_end=float(getattr(args, "baseline_entropy_end", 0.008)), entropy_decay_episodes=max(1, int(args.episodes))), device=device)
    elif scheme == "PPO-Lagrangian":
        install_ppo_lagrangian_controller(env, intent, args)
        cfg = PPOCfg(lr=float(getattr(args, "baseline_lr", 2e-4)), ent_coef=float(getattr(args, "baseline_entropy_start", 0.030)), entropy_coef_end=float(getattr(args, "baseline_entropy_end", 0.008)), entropy_decay_episodes=max(1, int(args.episodes)))
        metrics = hiot_ppo_train(env, episodes=args.episodes, steps_per_ep=train_steps, cfg=cfg, device=device)
    elif scheme == "PPO_CMDP":
        install_agentic_reward_controller(env, intent, args, mode="reliability_only")
        cfg = PPOCfg(lr=float(getattr(args, "baseline_lr", 2e-4)), ent_coef=float(getattr(args, "baseline_entropy_start", 0.030)), entropy_coef_end=float(getattr(args, "baseline_entropy_end", 0.008)), entropy_decay_episodes=max(1, int(args.episodes)))
        metrics = hiot_ppo_train(env, episodes=args.episodes, steps_per_ep=train_steps, cfg=cfg, device=device)
    elif scheme in {"Intent-PPO-CMDP-NoRisk", "Intent-PPO-CMDP-NoShield", "Intent-PPO-CMDP", FULL_POLICY_NO_SHIELD}:
        if scheme == FULL_POLICY_NO_SHIELD:
            raise RuntimeError(
                "The same-checkpoint shield-off ablation is evaluation-only and must reuse "
                "the already trained Intent-PPO-CMDP checkpoint."
            )
        if scheme == "Intent-PPO-CMDP":
            env.configure_admission_shield(
                True,
                age_fraction=float(getattr(args, "shield_age_fraction", 0.70)),
                queue_guard=float(getattr(args, "shield_queue_guard", 0.35)),
            )
            mode = "full_risk"
        elif scheme == "Intent-PPO-CMDP-NoRisk":
            mode = "full_no_tail"
        else:
            mode = "full_risk"
        install_agentic_reward_controller(env, intent, args, mode=mode)
        cfg = PPOCfg(lr=float(getattr(args, "ia_lr", 1e-4)), ent_coef=float(getattr(args, "ia_entropy_start", 0.040)), entropy_coef_end=float(getattr(args, "ia_entropy_end", 0.010)), entropy_decay_episodes=max(1, int(args.episodes)))
        metrics = hiot_ppo_train(env, episodes=args.episodes, steps_per_ep=train_steps, cfg=cfg, device=device)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    _enrich_agentic_episode_metrics(metrics, env)
    train_time = time.perf_counter() - t0
    if isinstance(metrics, dict):
        metrics["train_time_s"] = float(train_time)
        metrics["policy_fingerprint"] = _policy_fingerprint(metrics.get("policy"))
    return metrics, train_time

def run_experiment(args, tag: str | None = None) -> Tuple[Path, List[Dict[str, Any]], List[str]]:
    outdir = resolve_output_dir(args, tag=tag)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = outdir.name

    device = get_device(args.device)
    seeds = [int(x) for x in split_csv(args.seeds)]
    intents = [INTENT_LIBRARY[x] for x in split_csv(args.intents)]
    scenarios = [SCENARIO_LIBRARY[x] for x in split_csv(args.scenarios)]
    schemes = split_csv(args.schemes)

    manifest = {
        "timestamp": ts,
        "profile": args.profile,
        "tag": tag,
        "resume_enabled": bool(getattr(args, "resume", False) or getattr(args, "resume_dir", None) or getattr(args, "run_id", None)),
        "resume_dir": str(getattr(args, "resume_dir", "") or ""),
        "run_id": str(getattr(args, "run_id", "") or ""),
        "episodes": args.episodes,
        "eval_episodes": int(getattr(args, "eval_episodes", 100)),
        "steps": args.steps,
        "seeds": seeds,
        "n_devices": args.n_devices,
        "arrival_scale": args.arrival_scale,
        "class_mix": tuple(args.class_mix),
        "burst_k": args.burst_k,
        "device_skew": args.device_skew,
        "bits_scale": args.bits_scale,
        "delay_penalty_w": args.delay_penalty_w,
        "energy_penalty_w": args.energy_penalty_w,
        "prio_w": tuple(args.prio_w),
        "cmdp_lr": args.cmdp_lr,
        "intent_reward_penalty_scale": args.intent_reward_penalty_scale,
        "intent_tail_window": getattr(args, "intent_tail_window", 45),
        "intent_tail_cvar_alpha": getattr(args, "intent_tail_cvar_alpha", 0.90),
        "intent_tail_penalty_scale": getattr(args, "intent_tail_penalty_scale", 0.14),
        "intent_throughput_recovery_scale": getattr(args, "intent_throughput_recovery_scale", 0.012),
        "ia_reliability_block_guard": getattr(args, "ia_reliability_block_guard", 0.020),
        "lagrangian_dual_lr": getattr(args, "lagrangian_dual_lr", 0.012),
        "lagrangian_lambda_init": getattr(args, "lagrangian_lambda_init", 0.40),
        "lagrangian_lambda_max": getattr(args, "lagrangian_lambda_max", 25.0),
        "shield_age_fraction": getattr(args, "shield_age_fraction", 0.70),
        "shield_queue_guard": getattr(args, "shield_queue_guard", 0.35),
        "intent_switch_to": getattr(args, "intent_switch_to", None),
        "intent_switch_frac": getattr(args, "intent_switch_frac", 0.50),
        "intent_switch_hold_steps": int(getattr(args, "intent_switch_hold_steps", 10)),
        "intent_switch_tolerance": float(getattr(args, "intent_switch_tolerance", 0.05)),
        "schemes": schemes,
        "intents": [i.to_dict() for i in intents],
        "scenarios": [s.__dict__ for s in scenarios],
        "device": str(device),
        "output_mode": args.output_mode,
        "save_episode_logs": bool(args.save_episode_logs),
        "save_trace_logs": bool(args.save_trace_logs),
        "save_seed_summary": bool(args.save_seed_summary),
        "save_full_aggregated": bool(args.save_full_aggregated),
        "save_live_summary": bool(args.save_live_summary),
        "make_figures": bool(args.make_figures),
        "runner_revision": "request_level_universal_v4_4EC300",
        "action_model": {
            "candidate_slots": 4,
            "modes": ["defer", "grant", "protect", "coexist", "reject"],
            "n_actions": 20,
            "handoff_modeled": False,
            "state_dependent_shield_mask": True,
            "reject_only_before_admission": True,
        },
        "request_level_accounting": True,
        "online_violation_estimator": "target-centred smoothed cumulative request rates",
        "switch_adaptation_metric": "10-step settling time to a post-switch steady-state band; post-switch violation is reported separately as the quality measure",
        "mask_protocol": "physical feasibility mask applied to all schemes; additional admission-safety filter applied only to Full IA; the FullPolicy-NoShield causal ablation uses the identical frozen Full IA checkpoint with only this deployment filter disabled",
        "shield_activity_metric": "intent-safety mask-active rate and mean filtered-action fraction beyond common feasibility",
        "sota_value_baseline_protocol": "SOTA-DQN uses standard target-network DQN; SOTA-DuelingDDQN uses a dueling network with Double-DQN targets; both use the common physical-feasibility mask, the unmodified base reward, universal multi-intent training, and greedy deployment",
        "multi_intent_training": bool(getattr(args, "multi_intent_training", True)),
        "training_intents": split_csv(getattr(args, "training_intents", "balanced_agentic,reliability_first,latency_critical")),
        "training_scenarios": split_csv(getattr(args, "training_scenarios", "nominal,congestion_burst,emergency_surge,mixed_stress,admission_pressure")),
        "training_switch_fracs": split_float_csv(getattr(args, "training_switch_fracs", "0.35,0.65")),
        "training_static_probability": float(getattr(args, "training_static_probability", 0.20)),
        "ia_entropy_schedule": [float(getattr(args, "ia_entropy_start", 0.040)), float(getattr(args, "ia_entropy_end", 0.010))],
        "baseline_entropy_schedule": [float(getattr(args, "baseline_entropy_start", 0.030)), float(getattr(args, "baseline_entropy_end", 0.008))],
        "universal_policy_cache_enabled": bool(getattr(args, "policy_cache", True)),
        "universal_policy_cache_dir": str(getattr(args, "policy_cache_dir", "policy_cache_v44")),
        "evaluation_action_mode": str(getattr(args, "eval_action_mode", "greedy")),
        "training_steps": int(_training_steps(args)),
        "evaluation_steps": int(args.steps),
        "terminal_drain_steps": int(getattr(args, "terminal_drain_steps", 0)),
        "terminal_drain_auto_extend": bool(getattr(args, "terminal_drain_auto_extend", False)),
        "terminal_drain_max_steps": int(_terminal_drain_limit(args)),
        "terminal_drain_rng_replay_preserved": bool(int(getattr(args, "terminal_drain_steps", 0)) > 0),
        "effective_environment_steps": int(args.steps) + int(_terminal_drain_limit(args)),
        "horizon_protocol": (
            f"{int(_training_steps(args))}-step V4.4 training horizon with {int(args.steps)}-step held-out arrival window"
            + (
                f" followed by at least {int(getattr(args, 'terminal_drain_steps', 0))} no-arrival terminal-drain steps"
                + (f" with adaptive extension up to {int(_terminal_drain_limit(args))} steps" if bool(getattr(args, "terminal_drain_auto_extend", False)) else "")
                if int(getattr(args, "terminal_drain_steps", 0)) > 0 else ""
            )
        ),
        "evaluation_sampling_protocol": "indexed common uniform shared across learned schemes within each intent/scenario/master-seed context",
        "require_cached_policy": bool(getattr(args, "require_cached_policy", False)),
        "ia_lr": float(getattr(args, "ia_lr", 1e-4)),
        "baseline_lr": float(getattr(args, "baseline_lr", 2e-4)),
        "universal_training_signature": _universal_training_signature(args),
        "seed_protocol": "topology/evaluation/action-sampling seeds are matched across schemes; V4.4 universal-policy initialization and curriculum seeds depend only on the master seed; the FullPolicy-NoShield causal ablation reuses the exact Full IA checkpoint",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
    }
    write_manifest(outdir, manifest)

    checkpoint_path = outdir / "_resume_checkpoint.csv"
    summary_rows: List[Dict[str, Any]] = read_rows_csv(checkpoint_path)
    # Keep only completed rows matching the current configured grid. This avoids
    # mixing old exploratory rows when a run is resumed with changed settings.
    expected_keys = {
        _task_key(intent.name, scenario.name, seed, scheme)
        for intent in intents for scenario in scenarios for seed in seeds for scheme in schemes
    }
    summary_rows = [r for r in summary_rows if _row_task_key(r) in expected_keys]
    completed = {_row_task_key(r) for r in summary_rows}
    if summary_rows:
        print(f"[resume] loaded {len(summary_rows)} completed task(s) from {checkpoint_path}")
        refresh_incremental_outputs(
            outdir, summary_rows,
            keep_seed_summary=bool(getattr(args, "save_seed_summary", False)),
            keep_full_aggregated=bool(getattr(args, "save_full_aggregated", False)),
        )
    failures: List[str] = []
    # V4.4 trains one PPO-family policy per master seed and scheme, then reuses
    # that frozen policy across evaluation intents and stress scenarios within
    # the run. If a run is resumed in a new process, deterministic retraining
    # reconstructs the same policy before the next incomplete context.
    universal_training_cache: Dict[Tuple[int, str], Tuple[Dict[str, Any], float]] = {}

    for intent in intents:
        for scenario in scenarios:
            for seed in seeds:
                for scheme in schemes:
                    key = _task_key(intent.name, scenario.name, seed, scheme)
                    if key in completed and not bool(getattr(args, "force_rerun_completed", False)):
                        print(f"[skip] already completed intent={intent.name} scenario={scenario.name} seed={seed} scheme={scheme}")
                        continue
                    print(f"[run] intent={intent.name} scenario={scenario.name} seed={seed} scheme={scheme}")
                    # Common-random-number evaluation protocol. V4.4 universal
                    # PPO-family policies are initialized and trained independently
                    # of the evaluation intent/scenario, so a policy fingerprint can
                    # be compared directly across static and switching contexts.
                    env_seed = stable_int_seed("environment_topology", seed)
                    universal = bool(getattr(args, "multi_intent_training", True)) and scheme in UNIVERSAL_LEARNED_SCHEMES
                    policy_seed = (
                        stable_int_seed("policy_universal_v44", seed)
                        if universal else
                        stable_int_seed("policy_context", seed, intent.name, scenario.name)
                    )
                    seed_everything(policy_seed)
                    try:
                        training_reused = 0
                        if universal:
                            cache_key = (int(seed), _checkpoint_source_scheme(str(scheme)))
                            if cache_key in universal_training_cache:
                                train_metrics, train_time = universal_training_cache[cache_key]
                                training_reused = 1
                            else:
                                checkpoint = _universal_policy_checkpoint(args, seed, scheme)
                                loaded = False
                                if bool(getattr(args, "policy_cache", True)) and checkpoint.exists():
                                    try:
                                        if scheme in UNIVERSAL_VALUE_SCHEMES:
                                            policy, meta = load_value_policy(checkpoint, device=device)
                                        else:
                                            policy, meta = load_ppo_policy(checkpoint, device=device)
                                        train_metrics = {
                                            "policy": policy,
                                            "policy_fingerprint": str(meta.get("policy_fingerprint", _policy_fingerprint(policy))),
                                            "train_time_s": float(meta.get("train_time_s", 0.0)),
                                            "rewards_per_episode": [],
                                            "q_value_max": [],
                                            "episode_metrics_std": [],
                                        }
                                        train_time = float(meta.get("train_time_s", 0.0))
                                        training_reused = 1
                                        loaded = True
                                        print(f"[policy-cache] loaded {checkpoint}")
                                    except Exception as cache_exc:
                                        if bool(getattr(args, "require_cached_policy", False)):
                                            raise RuntimeError(f"Required cached policy is unreadable: {checkpoint}: {cache_exc}") from cache_exc
                                        print(f"[policy-cache] ignored unreadable checkpoint {checkpoint}: {cache_exc}")
                                if not loaded and bool(getattr(args, "require_cached_policy", False)):
                                    raise FileNotFoundError(
                                        f"Frozen-policy evaluation requires checkpoint {checkpoint}, but it is missing. "
                                        "For SOTA-DQN baselines run the primary extension first; for the frozen PPO family retain policy_cache_v44."
                                    )
                                if not loaded and scheme == FULL_POLICY_NO_SHIELD:
                                    raise FileNotFoundError(
                                        f"Causal shield-off ablation requires the Full IA checkpoint {checkpoint}. "
                                        "Ensure Intent-PPO-CMDP is scheduled before the causal ablation for each final seed."
                                    )
                                if not loaded:
                                    train_env_seed = stable_int_seed("training_environment_v44", seed)
                                    train_env = build_env(
                                        args, INTENT_LIBRARY["balanced_agentic"],
                                        SCENARIO_LIBRARY["nominal"], train_env_seed,
                                        steps_override=_training_steps(args),
                                    )
                                    train_metrics, train_time = train_scheme(
                                        scheme, train_env, args, INTENT_LIBRARY["balanced_agentic"],
                                        device, policy_seed,
                                    )
                                    if bool(getattr(args, "policy_cache", True)) and isinstance(train_metrics, dict) and train_metrics.get("policy") is not None:
                                        cache_meta = {
                                            "policy_fingerprint": str(train_metrics.get("policy_fingerprint", "")),
                                            "train_time_s": float(train_time),
                                            "policy_seed": int(policy_seed),
                                            "training_signature": _universal_training_signature(args),
                                            "scheme": str(_checkpoint_source_scheme(scheme)),
                                            "seed": int(seed),
                                            "training_steps": int(_training_steps(args)),
                                        }
                                        if scheme in UNIVERSAL_VALUE_SCHEMES:
                                            cache_meta.update({
                                                "value_dqn_architecture": str(train_metrics.get("value_dqn_architecture", "")),
                                                "value_dqn_double_target": bool(train_metrics.get("value_dqn_double_target", False)),
                                                "value_dqn_config": dict(train_metrics.get("value_dqn_config", {})),
                                            })
                                            save_value_policy(train_metrics["policy"], checkpoint, metadata=cache_meta)
                                        else:
                                            save_ppo_policy(train_metrics["policy"], checkpoint, metadata=cache_meta)
                                        print(f"[policy-cache] saved {checkpoint}")
                                universal_training_cache[cache_key] = (train_metrics, train_time)
                        else:
                            train_env = build_env(args, intent, scenario, env_seed)
                            train_metrics, train_time = train_scheme(scheme, train_env, args, intent, device, policy_seed)

                        # Publication metrics are computed on a frozen-policy,
                        # held-out evaluation sequence shared by all schemes.
                        eval_env_seed = stable_int_seed("evaluation_sequence", seed)
                        terminal_drain_steps = int(max(0, getattr(args, "terminal_drain_steps", 0)))
                        terminal_drain_auto_extend = bool(getattr(args, "terminal_drain_auto_extend", False))
                        terminal_drain_max_steps = int(_terminal_drain_limit(args))
                        eval_env = build_env(
                            args, intent, scenario, eval_env_seed,
                            steps_override=int(args.steps) + terminal_drain_max_steps,
                        )
                        configure_evaluation_environment(scheme, eval_env, args)
                        if scheme in BASELINE_POLICIES:
                            if terminal_drain_steps > 0:
                                metrics = run_policy_baseline_terminal_drain(
                                    eval_env, BASELINE_POLICIES[scheme], intent,
                                    int(getattr(args, "eval_episodes", 100)), int(args.steps), terminal_drain_steps,
                                    auto_extend=terminal_drain_auto_extend, max_drain_steps=terminal_drain_max_steps,
                                )
                            else:
                                metrics = run_policy_baseline(
                                    eval_env, BASELINE_POLICIES[scheme], intent,
                                    int(getattr(args, "eval_episodes", 100)), args.steps,
                                )
                        else:
                            evaluation_action_seed = stable_int_seed(
                                "v44e_action_stream", seed, intent.name, scenario.name
                            )
                            if terminal_drain_steps > 0:
                                metrics = run_trained_policy_evaluation_terminal_drain(
                                    eval_env, train_metrics.get("policy") if isinstance(train_metrics, dict) else None,
                                    int(getattr(args, "eval_episodes", 100)), int(args.steps), terminal_drain_steps,
                                    action_mode=str(getattr(args, "eval_action_mode", "greedy")),
                                    sampling_seed=int(evaluation_action_seed),
                                    auto_extend=terminal_drain_auto_extend, max_drain_steps=terminal_drain_max_steps,
                                )
                            else:
                                metrics = run_trained_policy_evaluation(
                                    eval_env, train_metrics.get("policy") if isinstance(train_metrics, dict) else None,
                                    int(getattr(args, "eval_episodes", 100)), args.steps,
                                    action_mode=str(getattr(args, "eval_action_mode", "greedy")),
                                    sampling_seed=int(evaluation_action_seed),
                                )
                        # Preserve training-only convergence diagnostics without
                        # mixing training episodes into performance estimates.
                        if isinstance(train_metrics, dict):
                            for diag_key in (
                                "q_value_max", "lambda_block", "lambda_intr", "lambda_energy",
                                "ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_kl", "ppo_total_loss",
                            ):
                                if diag_key in train_metrics:
                                    metrics[diag_key] = train_metrics[diag_key]
                        _enrich_agentic_episode_metrics(metrics, eval_env)
                        env = eval_env
                    except Exception as e:
                        msg = f"{scheme} failed under {intent.name}/{scenario.name}/seed={seed}: {e.__class__.__name__}: {e}"
                        print(f"[warn] {msg}")
                        failures.append(msg)
                        append_failure(outdir, msg)
                        continue

                    eps = metrics.get("episode_metrics_std", []) if isinstance(metrics, dict) else []
                    episode_path = ""
                    if eps and bool(getattr(args, "save_episode_logs", False)):
                        ep_path = outdir / "episodes" / f"{intent.name}__{scenario.name}__{scheme}__seed{seed}.csv"
                        write_rows_csv(ep_path, eps)
                        episode_path = str(ep_path)
                    if getattr(env, "agentic_trace", None) and bool(getattr(args, "save_trace_logs", False)):
                        trace_path = outdir / "traces" / f"{intent.name}__{scenario.name}__{scheme}__seed{seed}.csv"
                        write_rows_csv(trace_path, env.agentic_trace)

                    env_summaries = getattr(env, "agentic_episode_summaries", []) or []
                    fine = summarize_agentic_metrics(metrics, intent, env_summaries)
                    row = {
                        "intent": intent.name,
                        "scenario": scenario.name,
                        "scheme": scheme,
                        "checkpoint_source_scheme": _checkpoint_source_scheme(scheme) if universal else str(scheme),
                        "seed": seed,
                        "env_seed": int(env_seed),
                        "evaluation_seed": int(eval_env_seed),
                        "policy_seed": int(policy_seed),
                        "policy_fingerprint": str(train_metrics.get("policy_fingerprint", "")) if isinstance(train_metrics, dict) else "",
                        "evaluation_action_mode": str(getattr(args, "eval_action_mode", "greedy")),
                        "training_steps": int(_training_steps(args)),
                        "evaluation_steps": int(args.steps),
                        "terminal_drain_steps": int(getattr(args, "terminal_drain_steps", 0)),
                        "terminal_drain_auto_extend": int(bool(getattr(args, "terminal_drain_auto_extend", False))),
                        "terminal_drain_max_steps": int(_terminal_drain_limit(args)),
                        "terminal_drain_rng_replay_preserved": int(int(getattr(args, "terminal_drain_steps", 0)) > 0),
                        "effective_environment_steps": int(args.steps) + int(_terminal_drain_limit(args)),
                        "terminal_drain_enabled": int(int(getattr(args, "terminal_drain_steps", 0)) > 0),
                        "horizon_protocol": (
                            f"{int(_training_steps(args))}-step training / {int(args.steps)}-step arrival window"
                            + (
                                f" + >= {int(getattr(args, 'terminal_drain_steps', 0))}-step no-arrival drain"
                                + (f" adaptively extended to cohort maturity (cap {int(_terminal_drain_limit(args))})" if bool(getattr(args, "terminal_drain_auto_extend", False)) else "")
                                if int(getattr(args, "terminal_drain_steps", 0)) > 0 else ""
                            )
                        ),
                        "evaluation_action_sampling_seed": int(stable_int_seed("v44e_action_stream", seed, intent.name, scenario.name)),
                        "universal_training_reused": int(training_reused),
                        "train_time_s": float(train_time),
                        **fine,
                    }
                    if episode_path:
                        row["episode_csv"] = episode_path
                    summary_rows.append(row)
                    completed.add(key)
                    refresh_incremental_outputs(
                        outdir, summary_rows,
                        keep_seed_summary=bool(getattr(args, "save_seed_summary", False)),
                        keep_full_aggregated=bool(getattr(args, "save_full_aggregated", False)),
                    )
                    if bool(getattr(args, "save_live_summary", False)):
                        write_rows_csv(outdir / "agentic_summary_live.csv", summary_rows)

    # Final refresh and optional figure generation. The checkpoint file is retained
    # intentionally; it is small and is the basis for future resume.
    refresh_incremental_outputs(
        outdir, summary_rows,
        keep_seed_summary=bool(getattr(args, "save_seed_summary", False)),
        keep_full_aggregated=True,
    )
    write_v9_validation_matrix(outdir, summary_rows)
    if str(getattr(args, "profile", "")) in {"calibration", "feasibility_audit"}:
        write_feasibility_calibration_report(outdir, summary_rows)
    if bool(getattr(args, "make_figures", True)):
        plot_summary(outdir / "agentic_summary_aggregated.csv", outdir / "figures", plot_style=args.plot_style)
    if not bool(getattr(args, "save_full_aggregated", False)):
        full_agg = outdir / "agentic_summary_aggregated.csv"
        if full_agg.exists():
            full_agg.unlink()
    if not bool(getattr(args, "save_seed_summary", False)):
        seed_summary_path = outdir / "agentic_seed_summary.csv"
        if seed_summary_path.exists():
            seed_summary_path.unlink()
    return outdir, summary_rows, failures


def write_v9_validation_matrix(outdir: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a compact seed-level validation matrix for V9 design gates."""
    if not rows:
        return
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        cols = [c for c in [
            "weighted_violation_mean", "soft_intent_compliance_score", "blocking_mean",
            "interruption_mean", "deadline_exposure_mean", "deadline_miss_proxy_mean",
            "service_success_rate_mean", "shield_override_rate_mean", "throughput_mean_mbps",
            "delay_mean_ms", "fairness_mean", "energy_per_mbit_j",
            "post_switch_violation_mean", "switch_adaptation_time_steps", "switch_adaptation_success",
        ] if c in df.columns]
        if not cols:
            return
        grouped = df.groupby(["intent", "scenario", "scheme"], dropna=False)[cols].agg(["mean", "std", "count"])
        grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
        grouped.reset_index().to_csv(outdir / "v9_validation_matrix.csv", index=False)
        if getattr(df, "empty", True):
            return
        if "profile" not in df.columns:
            pass
    except Exception as exc:
        append_failure(outdir, f"validation matrix write failed: {exc}")


def write_feasibility_calibration_report(outdir: Path, rows: List[Dict[str, Any]]) -> None:
    """Expose observed operating ranges; it does not silently rewrite targets."""
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if df.empty:
            return
        baseline = df[df["scheme"].isin(["MaxWeight", "PPO", "PPO-Lagrangian"])].copy()
        cols = [c for c in [
            "blocking_mean", "interruption_mean", "deadline_exposure_mean",
            "deadline_miss_proxy_mean", "service_success_rate_mean", "throughput_mean_mbps",
            "delay_mean_ms", "fairness_mean", "energy_per_mbit_j",
        ] if c in baseline.columns]
        if not cols:
            return
        report = baseline.groupby(["intent", "scenario"])[cols].agg(["min", "median", "max"])
        report.columns = [f"{a}_{b}" for a, b in report.columns]
        report.reset_index().to_csv(outdir / "feasibility_calibration_report.csv", index=False)
    except Exception as exc:
        append_failure(outdir, f"feasibility report write failed: {exc}")


def aggregate_summary(in_csv: Path, out_csv: Path) -> None:
    import pandas as pd
    if not in_csv.exists():
        return
    df = pd.read_csv(in_csv)
    if df.empty:
        return
    group_cols = ["intent", "scenario", "scheme"]
    ignore = set(group_cols + ["seed", "env_seed", "evaluation_seed", "policy_seed", "episode_csv"])
    numeric_cols = [c for c in df.columns if c not in ignore and np.issubdtype(df[c].dtype, np.number)]
    rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["num_seeds"] = int(g["seed"].nunique()) if "seed" in g.columns else int(len(g))
        for c in numeric_cols:
            vals = np.asarray(g[c].values, dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{c}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
            row[f"{c}_std"] = float(np.nanstd(vals)) if vals.size else np.nan
        rows.append(row)
    write_rows_csv(out_csv, rows)


def _scheme_category(name: str) -> str:
    name = str(name)
    if name in {"MaxWeight", "LyapunovDPP", "MPC-Lite", "PriorityAging", "ProportionalFair", "TDMA-Preempt"}:
        return "Published/legacy scheduler"
    if name == "SLA-Aware":
        return "Rule-based intent baseline"
    if name.startswith("Intent-"):
        return "Proposed agentic RL"
    if name == "PPO-Lagrangian":
        return "Constrained RL baseline"
    if name.endswith("_CMDP"):
        return "Internal CMDP ablation"
    return "Published RL family"


def _plot_grouped_metric_panel(df, metrics, titles, fig_path: Path, scheme_order: List[str], suptitle: str) -> None:
    n = len(metrics)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(max(12, 0.72 * len(scheme_order)), 4.6 * rows))
    axes = np.asarray(axes).reshape(-1)
    rank = {s: i for i, s in enumerate(scheme_order)}
    g = df.copy()
    g["_rank"] = g["scheme"].map(lambda s: rank.get(s, 10_000))
    g = g.sort_values(["_rank", "scheme"])
    schemes_for_plot = list(g["scheme"])
    labels = [scheme_label(s) for s in schemes_for_plot]
    x = np.arange(len(labels))
    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in g.columns:
            ax.axis("off")
            continue
        vals = g[metric].astype(float).values
        colors = [SCHEME_COLORS.get(s, None) for s in schemes_for_plot]
        ax.bar(x, vals, color=colors if not any(c is None for c in colors) else None)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            ymin, ymax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if ymin >= 0:
                ax.set_ylim(0.0, ymax * 1.14 if ymax > 0 else 1.0)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _plot_rl_vs_nonrl_overview(df, fig_path: Path, metric: str = "soft_intent_compliance_score_mean") -> None:
    if metric not in df.columns:
        return
    local = df.copy()
    local["category"] = local["scheme"].map(_scheme_category)
    rows = []
    for (intent, scenario, category), g in local.groupby(["intent", "scenario", "category"]):
        vals = g[metric].astype(float).values
        vals = vals[np.isfinite(vals)]
        if vals.size:
            rows.append({"intent": intent, "scenario": scenario, "category": category, "value": float(np.nanmean(vals))})
    if not rows:
        return
    import pandas as pd
    d = pd.DataFrame(rows)
    groups = list(d["intent"].astype(str) + "\n" + d["scenario"].astype(str))
    groups = list(dict.fromkeys(groups))
    cats = ["Published/legacy scheduler", "Rule-based intent baseline", "Published RL family", "Constrained RL baseline", "Internal CMDP ablation", "Proposed agentic RL"]
    x = np.arange(len(groups))
    width = 0.18
    plt.figure(figsize=(max(12, 1.2 * len(groups)), 6))
    for i, cat in enumerate(cats):
        vals = []
        for group in groups:
            intent, scenario = group.split("\n", 1)
            sub = d[(d["intent"] == intent) & (d["scenario"] == scenario) & (d["category"] == cat)]
            vals.append(float(sub["value"].iloc[0]) if len(sub) else np.nan)
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=cat)
    plt.xticks(x, groups, rotation=20, ha="right", fontsize=9)
    plt.ylabel("Soft Intent Compliance Score")
    plt.title("Paper-main comparison overview by method family")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_summary(agg_csv: Path, figdir: Path, plot_style: str = "composite") -> None:
    import pandas as pd
    if not agg_csv.exists():
        return
    df = pd.read_csv(agg_csv)
    if df.empty:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    scheme_order = split_csv(DEFAULT_PAPER_SCHEMES)

    # Composite figures: 4 figures per intent/scenario rather than dozens of single-metric plots.
    composite_groups = {
        "intent_compliance": (
            ["soft_intent_compliance_score_mean", "weighted_violation_mean_mean", "weighted_violation_cvar95_mean", "intent_regret_auc_mean"],
            ["Soft intent compliance", "Mean weighted violation", "Violation CVaR95", "Intent regret AUC"],
        ),
        "reliability_resilience": (
            ["interruption_mean_mean", "blocking_mean_mean", "reliability_safety_margin_mean_mean", "shock_preempt_rate_mean"],
            ["Interruption probability", "Blocking probability", "Reliability safety margin", "Shock preemption rate"],
        ),
        "service_tradeoff": (
            ["throughput_mean_mbps_mean", "delay_cvar95_ms_mean", "energy_per_mbit_j_mean", "fairness_mean_mean"],
            ["Throughput (Mbps)", "Delay CVaR95 (ms)", "Energy per Mbit (J)", "Jain fairness"],
        ),
        "learning_agentic_diagnostics": (
            ["action_entropy_norm_mean", "qmax_oscillation_std_mean", "lambda_pressure_mean_mean", "train_time_s_mean"],
            ["Action entropy", "Q/value oscillation", "Dual pressure", "Training time (s)"],
        ),
    }
    for (intent, scenario), g in df.groupby(["intent", "scenario"]):
        for group_name, (metrics, titles) in composite_groups.items():
            if not any(m in g.columns for m in metrics):
                continue
            safe_name = f"{group_name}__{intent}__{scenario}".replace("/", "_")
            _plot_grouped_metric_panel(
                g, metrics, titles, figdir / f"{safe_name}.png", scheme_order,
                f"{group_name.replace('_', ' ').title()}: {intent} / {scenario}"
            )

    _plot_rl_vs_nonrl_overview(df, figdir / "method_family_soft_compliance_overview.png")

    if plot_style != "single":
        return

    # Optional detailed single-metric mode, disabled by default to avoid 60+ plots.
    metrics = [
        ("soft_intent_compliance_score_mean", "Soft Intent Compliance Score"),
        ("weighted_violation_mean_mean", "Weighted Violation Score"),
        ("weighted_violation_cvar95_mean", "Violation CVaR95"),
        ("interruption_mean_mean", "Interruption Probability"),
        ("blocking_mean_mean", "Blocking Probability"),
        ("delay_cvar95_ms_mean", "Delay CVaR95 (ms)"),
        ("throughput_mean_mbps_mean", "Throughput (Mbps)"),
        ("fairness_mean_mean", "Jain Fairness"),
        ("energy_per_mbit_j_mean", "Energy per Mbit (J)"),
    ]
    rank = {s: i for i, s in enumerate(scheme_order)}
    for metric, ylabel in metrics:
        if metric not in df.columns:
            continue
        for (intent, scenario), g in df.groupby(["intent", "scenario"]):
            g = g.copy()
            g["_rank"] = g["scheme"].map(lambda s: rank.get(s, 10_000))
            g = g.sort_values(["_rank", "scheme"])
            schemes_for_plot = list(g["scheme"])
            labels = [scheme_label(s) for s in schemes_for_plot]
            vals = g[metric].astype(float).values
            if not np.any(np.isfinite(vals)):
                continue
            x = np.arange(len(labels))
            plt.figure(figsize=(max(8, 0.78 * len(labels)), 5.8))
            colors = [SCHEME_COLORS.get(s, None) for s in schemes_for_plot]
            plt.bar(x, vals, color=colors if not any(c is None for c in colors) else None)
            plt.xticks(x, labels, rotation=25, ha="right", fontsize=10)
            plt.ylabel(ylabel, fontsize=13)
            plt.xlabel("Scheme", fontsize=13)
            plt.title(f"{ylabel}: {intent} / {scenario}", fontsize=14)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            safe_name = f"single_{metric}__{intent}__{scenario}".replace("/", "_")
            plt.savefig(figdir / f"{safe_name}.png", dpi=300)
            plt.close()

def configure_profile(args):
    # Profiles define reproducible defaults; explicitly supplied CLI values
    # take precedence over those defaults.
    explicit_cli = set(getattr(args, "_explicit_cli", set()))
    restorable_fields = {
        "episodes", "eval_episodes", "steps", "training_steps", "terminal_drain_steps", "seeds", "n_devices",
        "intents", "scenarios", "schemes", "intent_switch_to",
        "intent_switch_frac", "intent_switch_hold_steps",
        "intent_switch_tolerance", "output_mode", "make_figures",
        "multi_intent_training", "training_intents", "training_scenarios",
        "training_switch_fracs", "training_static_probability",
        "ia_lr", "baseline_lr", "ia_entropy_start", "ia_entropy_end", "baseline_entropy_start", "baseline_entropy_end",
        "policy_cache", "policy_cache_dir",
    }
    explicit_values = {
        name: copy.deepcopy(getattr(args, name))
        for name in explicit_cli & restorable_fields
        if hasattr(args, name)
    }
    user_requested_custom_schemes = (getattr(args, "scheme_set", "paper_main") == "custom")
    if args.profile == "smoke":
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 6, 6, 40, "1", 12
        args.intents, args.scenarios, args.schemes = "balanced_agentic", "nominal", FAST_SMOKE_SCHEMES
        if args.output_mode == "auto": args.output_mode = "compact"
    elif args.profile == "metric_sanity":
        # Tiny monotonicity gate. All-deny must worsen blocking/deadline exposure
        # and reduce useful service relative to all-grant/MaxWeight.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 30, 30, 150, "1", 24
        args.intents = "latency_critical"
        args.scenarios = "nominal,mixed_stress"
        args.schemes = METRIC_SANITY_SCHEMES
        args.intent_switch_to = None
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "feasibility_audit":
        # No learning is needed. MaxWeight, CapacityFirst, and AllGrant estimate
        # the attainable service/deadline envelope under held-out traffic/radio
        # realizations before manuscript intent targets are frozen.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 1, 100, 150, "101,102,103", 24
        args.intents = "balanced_agentic,reliability_first,latency_critical"
        args.scenarios = "nominal,congestion_burst,emergency_surge,mixed_stress,admission_pressure"
        args.schemes = FEASIBILITY_AUDIT_SCHEMES
        args.intent_switch_to = None
        args.multi_intent_training = False
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "calibration":
        # Fast operating-range check. It establishes whether target scales and
        # newly instrumented metrics behave sensibly before policy tuning.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 100, 50, 150, "1,2,3", 24
        args.intents = "reliability_first,latency_critical"
        args.scenarios = "nominal,congestion_burst,mixed_stress"
        args.schemes = DIAGNOSTIC_CALIBRATION_SCHEMES
        args.intent_switch_to = None
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "development":
        # Mechanism test: retain MaxWeight/PPO as sanity anchors, then test
        # whether shield and risk memory add value beyond non-agentic constrained PPO.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 600, 100, 150, "101,102,103", 24
        args.intents = "balanced_agentic,reliability_first,latency_critical"
        args.scenarios = "nominal,congestion_burst,mixed_stress"
        args.schemes = FOCUSED_DEVELOPMENT_SCHEMES
        args.intent_switch_to = None
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "intent_switch":
        # V9.4 online adaptation test: policies are trained/evaluated with an
        # explicit context transition at 50% of each episode. Runtime is not the
        # limiting factor, so keep all mechanism ablations and use five seeds.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 300, 100, 150, "1,2,3,4,5", 24
        args.intents = "balanced_agentic"
        args.scenarios = "congestion_burst,emergency_surge,mixed_stress"
        args.schemes = INTENT_SWITCH_SCHEMES
        args.intent_switch_to = "reliability_first"
        args.intent_switch_frac = 0.50
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "intent_switch_latency":
        # Separate latency-focused switch: balanced operation changes to
        # latency-critical during congestion/radio stress. Run this only after
        # V9.3 verification passes.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 300, 100, 150, "1,2,3,4,5", 24
        args.intents = "balanced_agentic"
        args.scenarios = "congestion_burst,mixed_stress"
        args.schemes = INTENT_SWITCH_SCHEMES
        args.intent_switch_to = "latency_critical"
        args.intent_switch_frac = 0.50
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "final_verification":
        # Only use after calibration + development + intent-switch gates pass.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 600, 200, 150, "11,12,13,14,15", 24
        args.intents = "balanced_agentic,reliability_first,latency_critical"
        args.scenarios = "nominal,congestion_burst,mixed_stress"
        args.schemes = V9_ABLATION_SCHEMES
        args.intent_switch_to = None
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "standard":
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 180, 80, 120, "1,2,3", 24
        args.intents, args.scenarios, args.schemes = "reliability_first,balanced_agentic", "nominal,mixed_stress", PAPER_MAIN_SCHEMES
        if args.output_mode == "auto": args.output_mode = "compact"
    elif args.profile == "verification":
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = 300, 100, 150, "1,2,3,4,5", 24
        args.intents, args.scenarios, args.schemes = "reliability_first,balanced_agentic", "nominal,mixed_stress", PAPER_MAIN_SCHEMES
        if args.output_mode == "auto": args.output_mode = "analysis"
    elif args.profile == "manuscript":
        # Retained only for an explicitly frozen protocol. V9 should not enter
        # here until the validation README gates are satisfied.
        args.episodes, args.eval_episodes, args.steps, args.seeds, args.n_devices = max(args.episodes, 800), max(args.eval_episodes, 200), max(args.steps, 150), "1,2,3,4,5,6,7,8,9,10", 24
        args.intents = "reliability_first,latency_critical,balanced_agentic"
        args.scenarios = "nominal,congestion_burst,mixed_stress"
        args.schemes = PAPER_MAIN_SCHEMES
        if args.output_mode == "auto": args.output_mode = "analysis"

    if not user_requested_custom_schemes:
        scheme_set = getattr(args, "scheme_set", "paper_main")
        override_map = {
            "metric_sanity": METRIC_SANITY_SCHEMES,
            "feasibility_audit": FEASIBILITY_AUDIT_SCHEMES,
            "diagnostic_calibration": DIAGNOSTIC_CALIBRATION_SCHEMES,
            "focused_development": FOCUSED_DEVELOPMENT_SCHEMES,
            "v9_ablation": V9_ABLATION_SCHEMES,
            "intent_switch": INTENT_SWITCH_SCHEMES,
            "paper_main": PAPER_MAIN_SCHEMES,
            "ablation_full": ABLATION_FULL_SCHEMES,
            "rl_heavy": RL_HEAVY_PAPER_SCHEMES,
            "full_legacy": FULL_LEGACY_SCHEMES,
        }
        # Profiles own their scheme set; only custom explicitly overrides it.
        if args.profile in {"smoke", "metric_sanity", "feasibility_audit", "calibration", "development", "intent_switch", "intent_switch_latency", "final_verification"}:
            pass
        elif scheme_set in override_map:
            args.schemes = override_map[scheme_set]

    for name, value in explicit_values.items():
        setattr(args, name, value)

    if args.output_mode == "auto": args.output_mode = "compact"
    if args.output_mode == "compact":
        args.save_episode_logs = args.save_trace_logs = args.save_seed_summary = args.save_full_aggregated = args.save_live_summary = False
    elif args.output_mode == "analysis":
        args.save_episode_logs = args.save_trace_logs = False
        args.save_seed_summary = args.save_full_aggregated = True
        args.save_live_summary = False
    elif args.output_mode == "debug":
        args.save_episode_logs = args.save_trace_logs = args.save_seed_summary = args.save_full_aggregated = args.save_live_summary = True
    return args


def _running_without_cli_args() -> bool:
    return len(sys.argv) <= 1


def _apply_pycharm_green_run_defaults(args):
    """Apply top-of-file PyCharm defaults when the file is launched directly.

    This allows the PyCharm green Run button to behave like a fully specified
    command, including resume support and a stable output directory.
    """
    for key, value in PYCHARM_GREEN_RUN_DEFAULTS.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _explicit_cli_destinations(parser: argparse.ArgumentParser) -> set[str]:
    """Return argparse destination names explicitly supplied on the CLI."""
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit = set()
    for token in sys.argv[1:]:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default="auto", choices=["auto", "custom", "smoke", "metric_sanity", "feasibility_audit", "calibration", "development", "intent_switch", "intent_switch_latency", "final_verification", "debug", "standard", "verification", "manuscript"])
    p.add_argument("--auto_detail_profile", type=str, default="calibration", choices=["metric_sanity", "calibration", "development", "intent_switch", "intent_switch_latency", "final_verification", "standard", "verification"],
                   help="For profile='auto', selects the bounded detailed stage after the smoke test.")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--eval_episodes", type=int, default=100, help="Held-out frozen-policy evaluation episodes per seed.")
    p.add_argument("--eval_action_mode", type=str, default="greedy", choices=["greedy", "stochastic"],
                   help="Frozen-policy deployment rule. V4.4E uses stochastic categorical sampling; greedy is retained as sensitivity analysis.")
    p.add_argument("--require_cached_policy", action="store_true",
                   help="Evaluation-only safety gate: fail if a required V4.4 universal policy checkpoint is missing instead of retraining it.")
    p.add_argument("--steps", type=int, default=150, help="Held-out evaluation episode horizon in decision epochs.")
    p.add_argument("--training_steps", type=int, default=None,
                   help="Optional training horizon. V4.4E-Causal300 uses 150 training steps while evaluating frozen policies for 300 steps.")
    p.add_argument("--terminal_drain_steps", type=int, default=0,
                   help="Minimum evaluation-only no-arrival follow-up after --steps. This never changes the training signature.")
    p.add_argument("--terminal_drain_auto_extend", action="store_true",
                   help="After the minimum drain, continue with arrivals disabled only while pending cohort requests remain right-censored.")
    p.add_argument("--terminal_drain_max_steps", type=int, default=500,
                   help="Safety cap for adaptive terminal drain. V2 uses 500 steps to cover realized deadline jitter without changing training.")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--n_devices", type=int, default=24)
    p.add_argument("--arrival_scale", type=float, default=1.0, help="Dimensionless multiplier on the explicit per-device request rates.")
    p.add_argument("--bits_scale", type=float, default=2.0)
    p.add_argument("--delay_penalty_w", type=float, default=0.0025)
    p.add_argument("--energy_penalty_w", type=float, default=0.0002)
    p.add_argument("--prio_w", type=float, nargs=3, default=(1.4, 1.0, 0.85))
    p.add_argument("--class_mix", type=float, nargs=3, default=(3.0, 1.0, 1.0))
    p.add_argument("--burst_k", type=float, default=None)
    p.add_argument("--device_skew", type=float, default=0.15)
    p.add_argument("--cmdp_lr", type=float, default=5e-3)
    p.add_argument("--intent_reward_penalty_scale", type=float, default=0.30)
    p.add_argument("--intent_tail_window", type=int, default=45)
    p.add_argument("--intent_tail_cvar_alpha", type=float, default=0.90)
    p.add_argument("--intent_tail_penalty_scale", type=float, default=0.18)
    p.add_argument("--intent_throughput_recovery_scale", type=float, default=0.012)
    p.add_argument("--ia_reliability_block_guard", type=float, default=0.020,
                   help="Legacy compatibility value; V9 uses the admission shield and calibrated intent targets.")
    p.add_argument("--shield_age_fraction", type=float, default=0.70,
                   help="Shield activates preemption for deny-on-backlog when queue age reaches this fraction of its deadline.")
    p.add_argument("--shield_queue_guard", type=float, default=0.35,
                   help="Shield activates service when queued-device fraction exceeds this value.")
    p.add_argument("--intent_switch_to", type=str, default=None, choices=list(INTENT_LIBRARY.keys()),
                   help="Optional online intent target applied partway through every episode.")
    p.add_argument("--intent_switch_frac", type=float, default=0.50,
                   help="Fraction of the episode at which the optional intent transition occurs.")
    p.add_argument("--intent_switch_hold_steps", type=int, default=10,
                   help="Consecutive-step averaging window used for post-switch settling time.")
    p.add_argument("--intent_switch_tolerance", type=float, default=0.05,
                   help="Absolute minimum half-width of the steady-state violation band used for settling time.")
    p.add_argument("--multi_intent_training", action=argparse.BooleanOptionalAction, default=True,
                   help="Train PPO-family policies once on a multi-intent/multi-stress curriculum before frozen evaluation.")
    p.add_argument("--training_intents", type=str, default="balanced_agentic,reliability_first,latency_critical",
                   help="Intent pool sampled during V4.4 universal-policy training.")
    p.add_argument("--training_scenarios", type=str, default="nominal,congestion_burst,emergency_surge,mixed_stress,admission_pressure",
                   help="Stress-scenario pool sampled during V4.4 universal-policy training.")
    p.add_argument("--training_switch_fracs", type=str, default="0.35,0.65",
                   help="Training switch fractions. Final evaluation at 0.50 is intentionally held out by default.")
    p.add_argument("--training_static_probability", type=float, default=0.20,
                   help="Probability that a curriculum episode keeps one sampled intent for the full episode.")
    p.add_argument("--ia_lr", type=float, default=1e-4, help="Learning rate for IA-PPO-CMDP and its mechanism ablations.")
    p.add_argument("--baseline_lr", type=float, default=2e-4, help="Learning rate for PPO-family baseline policies.")
    p.add_argument("--ia_entropy_start", type=float, default=0.040)
    p.add_argument("--ia_entropy_end", type=float, default=0.010)
    p.add_argument("--baseline_entropy_start", type=float, default=0.030)
    p.add_argument("--baseline_entropy_end", type=float, default=0.008)
    p.add_argument("--policy_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Persist V4.4 universal PPO policies so static/reliability/latency evaluations reuse exactly the same trained network.")
    p.add_argument("--policy_cache_dir", type=str, default="policy_cache_v44")
    p.add_argument("--lagrangian_dual_lr", type=float, default=0.012,
                   help="Projected dual step size for PPO-Lagrangian.")
    p.add_argument("--lagrangian_lambda_init", type=float, default=0.40,
                   help="Initial projected dual multiplier for PPO-Lagrangian.")
    p.add_argument("--lagrangian_lambda_max", type=float, default=25.0,
                   help="Maximum projected dual multiplier for PPO-Lagrangian.")
    p.add_argument("--energy_j_target", type=float, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--intents", type=str, default="reliability_first,latency_critical,balanced_agentic")
    p.add_argument("--scenarios", type=str, default="nominal,congestion_burst,sinr_drop,emergency_surge,mixed_stress")
    p.add_argument("--schemes", type=str, default=DEFAULT_PAPER_SCHEMES)
    p.add_argument("--scheme_set", type=str, default="paper_main", choices=["metric_sanity", "feasibility_audit", "diagnostic_calibration", "focused_development", "v9_ablation", "intent_switch", "paper_main", "ablation_full", "rl_heavy", "full_legacy", "custom"],
                   help="paper_main uses the manuscript comparison set; ablation_full compares same-family RL/CMDP/agentic variants; rl_heavy keeps additional diagnostics; full_legacy also includes older heuristic baselines.")
    p.add_argument("--plot_style", type=str, default="composite", choices=["composite", "single"],
                   help="composite produces few multi-panel paper figures; single additionally generates detailed single-metric plots.")
    p.add_argument("--output_mode", type=str, default="auto", choices=["auto", "compact", "analysis", "debug"],
                   help="compact saves one main analysis CSV; analysis also keeps seed/full aggregate CSVs; debug also stores episode/trace logs.")
    p.add_argument("--save_episode_logs", action="store_true", help="Store one per-episode CSV for every scheme/seed. Disabled by default to avoid crowded outputs.")
    p.add_argument("--save_trace_logs", action="store_true", help="Store step-level trace CSVs. Disabled by default because these create many files.")
    p.add_argument("--save_seed_summary", action="store_true", help="Keep seed-level summary CSV in addition to the compact main CSV.")
    p.add_argument("--save_full_aggregated", action="store_true", help="Keep the full aggregated CSV with all computed diagnostics.")
    p.add_argument("--save_live_summary", action="store_true", help="Continuously update a live seed-level summary during execution.")
    p.add_argument("--make_figures", action=argparse.BooleanOptionalAction, default=True, help="Generate composite figures. Use --no-make_figures for CSV-only runs.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run by skipping completed intent/scenario/seed/scheme tasks.")
    p.add_argument("--resume_dir", type=str, default=None, help="Path to an existing outputs_agentic run directory to resume.")
    p.add_argument("--run_id", type=str, default=None, help="Stable output folder name under outputs_agentic. Use with --resume for long manuscript runs.")
    p.add_argument("--force_rerun_completed", action="store_true", help="Ignore the resume checkpoint and recompute completed tasks.")
    p.add_argument("--skip_smoke", action="store_true", help="For --profile auto, skip the pre-run smoke test.")
    explicit_cli = _explicit_cli_destinations(p)
    args = p.parse_args()
    args._explicit_cli = explicit_cli
    if _running_without_cli_args():
        args = _apply_pycharm_green_run_defaults(args)
        print("[pycharm] No command-line arguments detected; using PYCHARM_GREEN_RUN_DEFAULTS from viot_agentic_runner.py")
        print(f"[pycharm] profile={args.profile} run_id={args.run_id} resume={args.resume} output_mode={args.output_mode} make_figures={args.make_figures}")
    return args


def main():
    args = parse_args()
    if args.profile == "auto":
        if not args.skip_smoke:
            smoke_args = copy.deepcopy(args)
            smoke_args.profile = "smoke"
            if getattr(args, "run_id", None):
                smoke_args.run_id = f"{args.run_id}_smoke"
            smoke_args = configure_profile(smoke_args)
            print("[auto] Stage 1/2: running smoke test before the detailed run.")
            smoke_out, smoke_rows, smoke_failures = run_experiment(smoke_args, tag="smoke")
            print(f"[auto] Smoke output: {smoke_out}")
            if smoke_failures or not smoke_rows:
                print("[auto] Smoke test did not complete cleanly. Detailed run is not started.")
                print(f"[auto] failures={len(smoke_failures)} rows={len(smoke_rows)}")
                return
        detail_args = copy.deepcopy(args)
        detail_args.profile = args.auto_detail_profile
        detail_args = configure_profile(detail_args)
        print(f"[auto] Stage 2/2: smoke test passed; starting {detail_args.profile} detailed run.")
        outdir, rows, failures = run_experiment(detail_args, tag=detail_args.profile)
    else:
        args = configure_profile(args)
        outdir, rows, failures = run_experiment(args, tag=args.profile)

    print(f"\n[done] output directory: {outdir}")
    print(f"[done] main summary: {outdir / 'agentic_main_summary.csv'}")
    print(f"[done] seed-level runs completed: {len(rows)}")
    print(f"[done] failures: {len(failures)}")


if __name__ == "__main__":
    main()
