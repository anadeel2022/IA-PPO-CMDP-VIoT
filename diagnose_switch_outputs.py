#!/usr/bin/env python3
"""Check whether held-out switch metrics distinguish schemes within matched contexts."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

METRICS = [
    "weighted_violation_mean",
    "weighted_violation_cvar95",
    "post_switch_violation_mean",
    "switch_adaptation_time_steps",
    "switch_adaptation_success",
    "blocking_mean",
    "service_success_rate_mean",
    "throughput_mean_mbps",
    "feasibility_filter_active_rate",
    "feasibility_filtered_action_fraction",
    "shield_filter_active_rate",
    "shield_filtered_action_fraction",
    "shield_override_rate_mean",
    "intent_action_shift_l1",
    "intent_mode_shift_l1",
    "counterfactual_intent_policy_l1",
    "counterfactual_mode_policy_l1",
    "counterfactual_argmax_change_rate",
    "frozen_policy_entropy_norm",
    "pre_mask_reject_probability",
    "feasible_reject_probability",
    "pre_mask_unsafe_probability_mass",
    "shield_filtered_probability_mass",
    "actual_shield_prevention_rate",
    "stochastic_action_entropy",
    "policy_mode_prob_defer",
    "policy_mode_prob_grant",
    "policy_mode_prob_protect",
    "policy_mode_prob_coexist",
    "policy_mode_prob_reject",
    "deny_action_rate",
    "action_entropy_norm",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    path = args.run_dir / "agentic_seed_summary.csv"
    if not path.exists():
        path = args.run_dir / "_resume_checkpoint.csv"
    df = pd.read_csv(path)
    context = ["intent", "scenario", "seed"]
    print(f"Rows: {len(df)}")
    print(f"Matched contexts: {df.groupby(context).ngroups}")
    for metric in METRICS:
        if metric not in df.columns:
            continue
        nunique = df.groupby(context)[metric].nunique(dropna=False)
        identical = int((nunique == 1).sum())
        total = int(len(nunique))
        print(f"{metric}: identical across all schemes in {identical}/{total} contexts")
    if "switch_adaptation_success" in df.columns:
        x = pd.to_numeric(df["switch_adaptation_success"], errors="coerce")
        print(f"adaptation success-rate range: {x.min():.3f} to {x.max():.3f}")
        if x.notna().any() and float(x.max()) <= 0.0:
            print("WARNING: no episode reached the declared settling band; inspect adaptation references/bands before a final run.")
    if "feasibility_filter_active_rate" in df.columns:
        x = pd.to_numeric(df["feasibility_filter_active_rate"], errors="coerce")
        print(f"common feasibility-mask active range: {x.min():.3f} to {x.max():.3f}")
    if "shield_filter_active_rate" in df.columns:
        x = pd.to_numeric(df["shield_filter_active_rate"], errors="coerce")
        print(f"shield mask-active range: {x.min():.3f} to {x.max():.3f}")
    if "policy_fingerprint" in df.columns:
        learned = df[df["policy_fingerprint"].fillna("").astype(str).str.len() > 0]
        if not learned.empty:
            bad = learned.groupby(["seed", "scheme"])["policy_fingerprint"].nunique()
            print(f"universal policy fingerprint mismatches: {int((bad != 1).sum())}/{len(bad)} seed/scheme groups")
    proposed = df[df["scheme"] == "Intent-PPO-CMDP"]
    if not proposed.empty and "intent_action_shift_l1" in proposed.columns:
        x = pd.to_numeric(proposed["intent_action_shift_l1"], errors="coerce")
        print(f"Full IA observed intent-action shift L1: mean={x.mean():.3f}, range={x.min():.3f} to {x.max():.3f}")
    for metric in (
        "counterfactual_intent_policy_l1", "counterfactual_mode_policy_l1",
        "counterfactual_argmax_change_rate", "frozen_policy_entropy_norm",
        "pre_mask_reject_probability", "feasible_reject_probability",
        "pre_mask_unsafe_probability_mass", "shield_filtered_probability_mass",
        "actual_shield_prevention_rate", "stochastic_action_entropy",
    ):
        if not proposed.empty and metric in proposed.columns:
            x = pd.to_numeric(proposed[metric], errors="coerce")
            if x.notna().any():
                print(f"Full IA {metric}: mean={x.mean():.3f}, range={x.min():.3f} to {x.max():.3f}")
    if not proposed.empty and all(f"policy_mode_prob_{m}" in proposed.columns for m in ("defer", "grant", "protect", "coexist", "reject")):
        vals = {m: pd.to_numeric(proposed[f"policy_mode_prob_{m}"], errors="coerce").mean() for m in ("defer", "grant", "protect", "coexist", "reject")}
        print("Full IA stochastic deployment mode probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in vals.items()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
