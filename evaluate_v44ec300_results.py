#!/usr/bin/env python3
"""Evaluate the V4.4E-Causal300 confirmation/final result set.

This evaluator intentionally does not tune the learning algorithm.  It checks
structural validity, exact same-checkpoint shield causality, persistence of the
V4.4E tail-risk and intent-conditioning mechanisms at a 300-step held-out
horizon, and reports mean-risk and unfinished-request behavior descriptively.

Development confirmation is a go/no-go gate only for implementation/mechanism
integrity.  Final-seed results are confirmatory and must not trigger parameter
retuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validate_submission_results import validate
from evaluate_v44e_results import (
    _load,
    _gate_pair,
    _mean_numeric,
    _conditioned_mode_shift,
    _fingerprint_mismatches,
)

FULL = "Intent-PPO-CMDP"
NO_TAIL = "Intent-PPO-CMDP-NoRisk"
CMDP = "PPO_CMDP"
CAUSAL_OFF = "Intent-PPO-CMDP-FullPolicy-NoShield"


def _paired(df: pd.DataFrame, comparator: str, metric: str) -> pd.DataFrame:
    key = ["intent", "scenario", "seed"]
    a = df[df.scheme == FULL][key + [metric]].rename(columns={metric: "full"})
    b = df[df.scheme == comparator][key + [metric]].rename(columns={metric: "comp"})
    return a.merge(b, on=key, how="inner").dropna()


def _ratio(df: pd.DataFrame, comparator: str, metric: str) -> float:
    m = _paired(df, comparator, metric)
    if m.empty:
        return float("nan")
    denom = max(float(m.comp.mean()), 1e-12)
    return float(m.full.mean() / denom)


def _seed_ratios(df: pd.DataFrame, comparator: str, metric: str) -> dict[str, float]:
    m = _paired(df, comparator, metric)
    out: dict[str, float] = {}
    for seed, g in m.groupby("seed"):
        denom = max(float(g.comp.mean()), 1e-12)
        out[str(int(seed))] = float(g.full.mean() / denom)
    return out


def _same_checkpoint_identity(frames: Iterable[pd.DataFrame]) -> tuple[bool, int, int]:
    rows = []
    for df in frames:
        if "policy_fingerprint" not in df.columns:
            continue
        key = ["intent", "scenario", "seed"]
        a = df[df.scheme == FULL][key + ["policy_fingerprint"]].rename(columns={"policy_fingerprint": "full_fp"})
        b = df[df.scheme == CAUSAL_OFF][key + ["policy_fingerprint"]].rename(columns={"policy_fingerprint": "off_fp"})
        if not a.empty and not b.empty:
            rows.append(a.merge(b, on=key, how="inner"))
    if not rows:
        return False, 0, 0
    m = pd.concat(rows, ignore_index=True)
    mismatches = int((m.full_fp.astype(str) != m.off_fp.astype(str)).sum())
    return mismatches == 0 and len(m) > 0, mismatches, int(len(m))


def _manifest_protocol_ok(run_dirs: list[Path], *, expected_eval_steps: int = 300, expected_train_steps: int = 150) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for d in run_dirs:
        p = d / "manifest.json"
        if not p.exists():
            errors.append(f"{d}: missing manifest.json")
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        eval_steps = int(m.get("evaluation_steps", m.get("steps", -1)))
        train_steps = int(m.get("training_steps", m.get("steps", -1)))
        mode = str(m.get("evaluation_action_mode", "")).lower()
        if eval_steps != expected_eval_steps:
            errors.append(f"{d}: evaluation_steps={eval_steps}, expected {expected_eval_steps}")
        if train_steps != expected_train_steps:
            errors.append(f"{d}: training_steps={train_steps}, expected {expected_train_steps}")
        if mode != "stochastic":
            errors.append(f"{d}: evaluation_action_mode={mode}, expected stochastic")
    return len(errors) == 0, errors


def _nominal_feasibility(feasibility_dir: Path) -> tuple[bool, list[str], dict[str, float]]:
    """Return the finite-anchor attainability diagnostic for nominal operation.

    V4.4 targets and the cached development policies were fixed before the
    300-step confirmation.  The diagnostic therefore must not be converted into
    a post-hoc target-tuning gate.  When available, an expected-metric convex
    mixture certificate is accepted in addition to a single deterministic
    anchor.
    """
    audit_path = feasibility_dir / "v44ec300_feasibility_audit.csv"
    if not audit_path.exists():
        candidates = list(feasibility_dir.glob("*feasibility_audit.csv"))
        if candidates:
            audit_path = candidates[0]
    if not audit_path.exists():
        return False, [], {}
    audit = pd.read_csv(audit_path)
    required = {"balanced_agentic", "reliability_first", "latency_critical"}
    nominal = audit[(audit.scenario == "nominal") & audit.intent.isin(required)].copy()
    cert_col = (
        "target_feasible_by_anchor_or_mixture"
        if "target_feasible_by_anchor_or_mixture" in nominal.columns
        else "simultaneous_target_feasible_by_anchor"
    )
    feasible = set(nominal.loc[nominal[cert_col] == 1, "intent"].astype(str))
    gaps: dict[str, float] = {}
    if "closest_max_normalized_target_gap" in nominal.columns:
        for _, r in nominal.iterrows():
            try:
                gaps[str(r["intent"])] = float(r["closest_max_normalized_target_gap"])
            except Exception:
                pass
    return required.issubset(feasible), sorted(feasible), gaps


def _feasibility_horizon_shift(current_dir: Path, previous_dir: Path | None) -> dict:
    """Compare nominal anchor means between the original 150-step and 300-step audits.

    This is descriptive evidence of horizon sensitivity.  It never changes
    targets, policies, or the go/no-go mechanism gate.
    """
    if previous_dir is None or not previous_dir.exists():
        return {}
    cur_path = current_dir / "agentic_seed_summary.csv"
    old_path = previous_dir / "agentic_seed_summary.csv"
    if not cur_path.exists() or not old_path.exists():
        return {}
    cur = pd.read_csv(cur_path)
    old = pd.read_csv(old_path)
    wanted = {
        "balanced_agentic": "MaxWeight",
        "reliability_first": "MaxWeight",
        "latency_critical": "CapacityFirst",
    }
    metrics = [
        "interruption_mean", "deadline_exposure_mean", "service_success_rate_mean",
        "fairness_mean", "throughput_mean_mbps", "unfinished_rate_mean",
    ]
    out: dict[str, dict] = {}
    for intent, scheme in wanted.items():
        a = old[(old.intent == intent) & (old.scenario == "nominal") & (old.scheme == scheme)]
        b = cur[(cur.intent == intent) & (cur.scenario == "nominal") & (cur.scheme == scheme)]
        if a.empty or b.empty:
            continue
        rec = {"anchor": scheme}
        for metric in metrics:
            if metric not in a.columns or metric not in b.columns:
                continue
            v150 = float(pd.to_numeric(a[metric], errors="coerce").mean())
            v300 = float(pd.to_numeric(b[metric], errors="coerce").mean())
            rec[metric] = {
                "steps150": v150,
                "steps300": v300,
                "relative_change": float((v300 - v150) / max(abs(v150), 1e-12)),
            }
        out[intent] = rec
    return out


def _causal_shield(admission: pd.DataFrame) -> dict[str, float | bool]:
    full = admission[admission.scheme == FULL]
    off = admission[admission.scheme == CAUSAL_OFF]
    m_reject = _paired(admission, CAUSAL_OFF, "deny_action_rate")
    m_block = _paired(admission, CAUSAL_OFF, "blocking_mean")
    m_viol = _paired(admission, CAUSAL_OFF, "weighted_violation_mean")
    m_cvar = _paired(admission, CAUSAL_OFF, "weighted_violation_cvar95")

    def diff(m: pd.DataFrame) -> float:
        return float((m.comp - m.full).mean()) if not m.empty else float("nan")

    filtered = _mean_numeric(full, "shield_filtered_probability_mass")
    prevention = _mean_numeric(full, "actual_shield_prevention_rate")
    reject_reduction = diff(m_reject)
    blocking_reduction = diff(m_block)
    violation_change = diff(m_viol)  # positive => shield lowers violation
    cvar_change = diff(m_cvar)
    # Mechanism gate: shield must actually remove policy mass and, on the same
    # checkpoint, must not increase admission blocking/rejection on average.
    ok = bool(
        np.isfinite(filtered) and filtered > 1e-4
        and np.isfinite(prevention) and prevention > 1e-4
        and np.isfinite(reject_reduction) and reject_reduction >= -1e-6
        and np.isfinite(blocking_reduction) and blocking_reduction >= -1e-6
    )
    return {
        "ok": ok,
        "filtered_probability_mass": filtered,
        "sampled_prevention_rate": prevention,
        "reject_reduction_off_minus_on": reject_reduction,
        "blocking_reduction_off_minus_on": blocking_reduction,
        "violation_reduction_off_minus_on": violation_change,
        "cvar_reduction_off_minus_on": cvar_change,
        "full_blocking": _mean_numeric(full, "blocking_mean"),
        "off_blocking": _mean_numeric(off, "blocking_mean"),
        "full_reject": _mean_numeric(full, "deny_action_rate"),
        "off_reject": _mean_numeric(off, "deny_action_rate"),
    }


def _unfinished_summary(main_df: pd.DataFrame, admission: pd.DataFrame, previous150: list[Path]) -> dict[str, float | None]:
    full = pd.concat([main_df, admission], ignore_index=True)
    full = full[full.scheme == FULL]
    current = _mean_numeric(full, "unfinished_rate_mean")
    out: dict[str, float | None] = {"full_mean_300": None if not np.isfinite(current) else float(current)}
    if previous150:
        parts = []
        for p in previous150:
            if p and p.exists():
                parts.append(_load(p))
        if parts:
            old = pd.concat(parts, ignore_index=True)
            old = old[old.scheme == FULL]
            old_mean = _mean_numeric(old, "unfinished_rate_mean")
            out["full_mean_150"] = None if not np.isfinite(old_mean) else float(old_mean)
            if np.isfinite(current) and np.isfinite(old_mean):
                out["relative_change_300_vs_150"] = float((current - old_mean) / max(abs(old_mean), 1e-12))
    return out


def _write_md(path: Path, result: dict) -> None:
    lines = [
        "# V4.4E-Causal300 evaluation report",
        "",
        f"Stage: **{result.get('stage')}**",
        f"Decision: **{result.get('decision')}**",
        "",
        "## Core checks",
    ]
    for key in [
        "structural_validation", "protocol_150_train_300_eval", "nominal_anchor_feasibility_diagnostic",
        "universal_policy_fingerprint", "causal_same_checkpoint_identity", "tail_risk_gate",
        "counterfactual_intent_use", "stochastic_entropy_ok", "causal_shield_gate",
    ]:
        lines.append(f"- {key}: {result.get(key)}")
    lines += [
        "",
        "Finite-anchor target attainability is a diagnostic only. Non-demonstration by the limited anchor set is not a proof of infeasibility and does not block the implementation/mechanism confirmation. Targets are frozen to preserve the semantics of the cached V4.4 policies.",
        "",
        "## Descriptive performance",
    ]
    for key in [
        "mean_violation_ratio_full_vs_ppo_cmdp", "tail_cvar_mean_ratio",
        "tail_post_switch_mean_ratio", "counterfactual_intent_policy_l1",
        "counterfactual_mode_policy_l1", "conditioned_mode_shift_reliability_l1",
        "conditioned_mode_shift_latency_l1", "stochastic_action_entropy",
    ]:
        lines.append(f"- {key}: {result.get(key)}")
    if "mean_violation_seed_ratios" in result:
        lines.append(f"- mean_violation_seed_ratios: {result['mean_violation_seed_ratios']}")
    lines += ["", "## Nominal anchor attainability diagnostic"]
    lines.append(f"- fully demonstrated by finite anchors/mixtures: {result.get('nominal_anchor_feasibility_diagnostic')}")
    lines.append(f"- demonstrated intents: {result.get('nominal_feasible_intents')}")
    lines.append(f"- closest normalized gaps: {result.get('nominal_feasibility_closest_normalized_gaps')}")
    lines.append("- This diagnostic is not a hard gate and does not trigger target retuning.")
    if result.get("feasibility_horizon_shift_150_to_300"):
        lines.append(f"- 150-to-300-step anchor shift: {result.get('feasibility_horizon_shift_150_to_300')}")
    if "causal_shield" in result:
        lines += ["", "## Same-checkpoint shield causality"]
        for k, v in result["causal_shield"].items():
            lines.append(f"- {k}: {v}")
    if "unfinished" in result:
        lines += ["", "## Horizon accounting"]
        for k, v in result["unfinished"].items():
            lines.append(f"- {k}: {v}")
        lines.append("- Pending requests remain reported as unfinished and remain in the all-arrival service-success denominator; the 300-step horizon reduces, but does not mathematically eliminate, end-of-horizon pending requests.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["development", "final"], default="development")
    ap.add_argument("--feasibility", type=Path, required=True)
    ap.add_argument("--static", type=Path, required=True)
    ap.add_argument("--reliability", type=Path, required=True)
    ap.add_argument("--latency", type=Path, required=True)
    ap.add_argument("--admission", type=Path, required=True)
    ap.add_argument("--previous150-static", type=Path)
    ap.add_argument("--previous150-reliability", type=Path)
    ap.add_argument("--previous150-latency", type=Path)
    ap.add_argument("--previous150-admission", type=Path)
    ap.add_argument("--previous150-feasibility", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    run_dirs = [args.static, args.reliability, args.latency, args.admission]
    reports = {str(d): validate(d) for d in run_dirs}
    structural_ok = all(r["status"] == "PASS" for r in reports.values())
    protocol_ok, protocol_errors = _manifest_protocol_ok(run_dirs)

    static = _load(args.static)
    reliability = _load(args.reliability)
    latency = _load(args.latency)
    admission = _load(args.admission)
    main_df = pd.concat([static, reliability, latency], ignore_index=True)
    switch_df = pd.concat([reliability, latency], ignore_index=True)

    feasibility_ok, feasible_intents, feasibility_gaps = _nominal_feasibility(args.feasibility)
    feasibility_shift = _feasibility_horizon_shift(args.feasibility, args.previous150_feasibility)
    fp_mismatch = _fingerprint_mismatches(run_dirs and [static, reliability, latency, admission])
    fp_ok = fp_mismatch == 0
    causal_fp_ok, causal_fp_mismatch, causal_pairs = _same_checkpoint_identity([static, reliability, latency, admission])

    # Mechanism persistence. Thresholds are confirmation tolerances, not tuning targets.
    tail_cvar = _gate_pair(main_df, NO_TAIL, "weighted_violation_cvar95", min_win_frac=0.45, max_mean_ratio=1.02)
    tail_post = _gate_pair(switch_df, NO_TAIL, "post_switch_violation_mean", min_win_frac=0.45, max_mean_ratio=1.02)
    tail_ok = bool(tail_cvar[0] and tail_post[0])

    full = main_df[main_df.scheme == FULL]
    cf_l1 = _mean_numeric(full, "counterfactual_intent_policy_l1")
    cf_mode = _mean_numeric(full, "counterfactual_mode_policy_l1")
    entropy = _mean_numeric(full, "stochastic_action_entropy")
    rel_mode_shift = _conditioned_mode_shift(reliability, "reliability_first")
    lat_mode_shift = _conditioned_mode_shift(latency, "latency_critical")
    intent_ok = bool(np.isfinite(cf_l1) and cf_l1 >= 0.04 and np.isfinite(cf_mode) and cf_mode >= 0.02)
    entropy_ok = bool(np.isfinite(entropy) and entropy >= 0.05)

    mean_ratio = _ratio(main_df, CMDP, "weighted_violation_mean")
    seed_ratios = _seed_ratios(main_df, CMDP, "weighted_violation_mean")
    causal = _causal_shield(admission)

    previous = [p for p in [args.previous150_static, args.previous150_reliability, args.previous150_latency, args.previous150_admission] if p is not None]
    unfinished = _unfinished_summary(main_df, admission, previous)

    # Development gate deliberately excludes an arbitrary mean-risk superiority
    # threshold.  Mean-risk is reported and will frame the manuscript claim.
    # The finite-anchor feasibility audit is intentionally diagnostic rather
    # than a hard gate. A failure only says that the limited anchor set did not
    # demonstrate the complete target vector at this horizon; it is not an
    # implementation/mechanism failure and is not a proof of infeasibility.
    # Targets are frozen because changing them now would invalidate the semantics
    # of the already-trained V4.4 checkpoints.
    mechanism_pass = bool(
        structural_ok and protocol_ok and fp_ok and causal_fp_ok
        and tail_ok and intent_ok and entropy_ok and bool(causal["ok"])
    )
    if args.stage == "final":
        # Final results are never used for tuning. Only data-integrity failures
        # invalidate the run; scientific effect sizes are reported as observed.
        overall_pass = bool(structural_ok and protocol_ok and fp_ok and causal_fp_ok)
        decision = "FINAL DATA INTEGRITY PASS - FREEZE RESULTS" if overall_pass else "FINAL DATA INTEGRITY FAILURE"
    else:
        overall_pass = mechanism_pass
        decision = "PROCEED TO UNTOUCHED FINAL SEEDS" if overall_pass else "REVIEW IMPLEMENTATION/MECHANISM BEFORE FINAL SEEDS"

    result = {
        "stage": args.stage,
        "overall_pass": overall_pass,
        "decision": decision,
        "structural_validation": structural_ok,
        "protocol_150_train_300_eval": protocol_ok,
        "protocol_errors": protocol_errors,
        "nominal_target_feasibility": feasibility_ok,
        "nominal_anchor_feasibility_diagnostic": feasibility_ok,
        "nominal_feasible_intents": feasible_intents,
        "nominal_feasibility_closest_normalized_gaps": feasibility_gaps,
        "nominal_feasibility_is_hard_gate": False,
        "nominal_feasibility_note": "Finite-anchor non-demonstration is diagnostic only and is not a proof of physical infeasibility. Targets and cached V4.4 policy semantics remain frozen; no post-hoc target retuning is performed after changing the held-out horizon.",
        "feasibility_horizon_shift_150_to_300": feasibility_shift,
        "universal_policy_fingerprint": fp_ok,
        "policy_fingerprint_mismatches": int(fp_mismatch),
        "causal_same_checkpoint_identity": causal_fp_ok,
        "causal_checkpoint_mismatches": int(causal_fp_mismatch),
        "causal_checkpoint_pairs": int(causal_pairs),
        "tail_risk_gate": tail_ok,
        "tail_cvar_wins": int(tail_cvar[1]),
        "tail_cvar_pairs": int(tail_cvar[2]),
        "tail_cvar_mean_ratio": None if not np.isfinite(tail_cvar[3]) else float(tail_cvar[3]),
        "tail_post_switch_wins": int(tail_post[1]),
        "tail_post_switch_pairs": int(tail_post[2]),
        "tail_post_switch_mean_ratio": None if not np.isfinite(tail_post[3]) else float(tail_post[3]),
        "mean_violation_ratio_full_vs_ppo_cmdp": None if not np.isfinite(mean_ratio) else float(mean_ratio),
        "mean_violation_seed_ratios": seed_ratios,
        "counterfactual_intent_use": intent_ok,
        "counterfactual_intent_policy_l1": None if not np.isfinite(cf_l1) else float(cf_l1),
        "counterfactual_mode_policy_l1": None if not np.isfinite(cf_mode) else float(cf_mode),
        "conditioned_mode_shift_reliability_l1": None if not np.isfinite(rel_mode_shift) else float(rel_mode_shift),
        "conditioned_mode_shift_latency_l1": None if not np.isfinite(lat_mode_shift) else float(lat_mode_shift),
        "stochastic_action_entropy": None if not np.isfinite(entropy) else float(entropy),
        "stochastic_entropy_ok": entropy_ok,
        "causal_shield_gate": bool(causal["ok"]),
        "causal_shield": causal,
        "unfinished": unfinished,
        "note": "The 300-step evaluation reduces horizon sensitivity but pending requests can still exist for late arrivals; they are reported as unfinished and remain in the all-arrival service-success denominator.",
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(args.report.with_suffix(".md"), result)

    print("\nV4.4E-Causal300 evaluation")
    print(f"1. Structural validation: {'PASS' if structural_ok else 'FAIL'}")
    print(f"2. Protocol 150-step training / 300-step stochastic evaluation: {'PASS' if protocol_ok else 'FAIL'}")
    print(f"3. Nominal finite-anchor attainability at 300 steps: {'DEMONSTRATED' if feasibility_ok else 'NOT FULLY DEMONSTRATED'} ({len(feasible_intents)}/3 intents); diagnostic only, not a hard gate")
    if feasibility_gaps:
        print(f"   Closest normalized target gaps: {feasibility_gaps}")
    print(f"4. Frozen-policy consistency: {'PASS' if fp_ok else 'FAIL'}; Full==CausalOff fingerprints: {'PASS' if causal_fp_ok else 'FAIL'} ({causal_pairs} matched rows)")
    print(f"5. Tail-risk persistence: CVaR wins={tail_cvar[1]}/{tail_cvar[2]}, ratio={tail_cvar[3]:.3f}; post-switch wins={tail_post[1]}/{tail_post[2]}, ratio={tail_post[3]:.3f} -> {'PASS' if tail_ok else 'REVIEW'}")
    print(f"6. Mean violation Full/PPO-CMDP={mean_ratio:.3f}; seed ratios={seed_ratios} (descriptive, not a tuning gate)")
    print(f"7. Intent use: action-L1={cf_l1:.3f}, mode-L1={cf_mode:.3f}, reliability mode shift={rel_mode_shift:.3f}, latency mode shift={lat_mode_shift:.3f}, entropy={entropy:.3f} -> {'PASS' if intent_ok and entropy_ok else 'REVIEW'}")
    print(f"8. Same-checkpoint shield: filtered mass={causal['filtered_probability_mass']:.4f}, prevention={causal['sampled_prevention_rate']:.4f}, reject reduction={causal['reject_reduction_off_minus_on']:.5f}, blocking reduction={causal['blocking_reduction_off_minus_on']:.5f}, violation reduction={causal['violation_reduction_off_minus_on']:.5f} -> {'PASS' if causal['ok'] else 'REVIEW'}")
    print(f"9. Full IA unfinished-rate mean at 300 steps={unfinished.get('full_mean_300')}")
    if unfinished.get("full_mean_150") is not None:
        print(f"   Previous 150-step mean={unfinished.get('full_mean_150')}; relative change={unfinished.get('relative_change_300_vs_150'):.3f}")
    print(f"\nDecision: {decision}")

    if overall_pass:
        return 0
    return 3 if args.stage == "development" else 4


if __name__ == "__main__":
    raise SystemExit(main())
