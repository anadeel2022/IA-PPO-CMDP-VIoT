#!/usr/bin/env python3
"""Evaluate V4.4E stochastic frozen-policy development results.

V4.4E is an evaluation-layer revision. It does not change the V4.4 simulator,
intent targets, reward compiler, action model, or learned checkpoints. The gate
therefore focuses on whether stochastic deployment exposes the mechanisms that
were hidden by greedy argmax evaluation:

* tail-risk benefit relative to the no-tail ablation;
* non-inferior mean violation relative to PPO-CMDP;
* direct counterfactual intent sensitivity of the frozen actor;
* probability mass removed by the admission-safety shield and realized
  rejection/blocking separation under admission pressure;
* exact checkpoint reuse relative to the V4.4 greedy development run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validate_submission_results import validate

FULL = "Intent-PPO-CMDP"
NO_TAIL = "Intent-PPO-CMDP-NoRisk"
NO_SHIELD = "Intent-PPO-CMDP-NoShield"
CMDP = "PPO_CMDP"
MODE_NAMES = ("defer", "grant", "protect", "coexist", "reject")


def _load(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "agentic_seed_summary.csv"
    if not path.exists():
        path = run_dir / "_resume_checkpoint.csv"
    if not path.exists():
        raise FileNotFoundError(f"No seed summary found in {run_dir}")
    df = pd.read_csv(path)
    df["run_dir"] = str(run_dir)
    return df


def _paired(df: pd.DataFrame, comparator: str, metric: str) -> pd.DataFrame:
    key = ["run_dir", "intent", "scenario", "seed"]
    a = df[df.scheme == FULL][key + [metric]].rename(columns={metric: "full"})
    b = df[df.scheme == comparator][key + [metric]].rename(columns={metric: "comp"})
    out = a.merge(b, on=key, how="inner")
    out["full"] = pd.to_numeric(out["full"], errors="coerce")
    out["comp"] = pd.to_numeric(out["comp"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _gate_pair(
    df: pd.DataFrame,
    comparator: str,
    metric: str,
    *,
    min_win_frac: float = 0.50,
    max_mean_ratio: float = 0.98,
):
    m = _paired(df, comparator, metric)
    if m.empty:
        return False, 0, 0, np.nan
    wins = int((m.full < m.comp).sum())
    frac = wins / len(m)
    ratio = float(np.mean(m.full) / max(float(np.mean(m.comp)), 1e-12))
    return bool(frac >= min_win_frac and ratio <= max_mean_ratio), wins, int(len(m)), ratio


def _mean_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def _paired_noninferiority(df: pd.DataFrame, margin: float = 0.05, bootstrap_reps: int = 10000):
    """Paired 5% non-inferiority diagnostic with seed-cluster bootstrap.

    Development gating uses the predeclared point-estimate margin. The clustered
    bootstrap upper bound is reported as an uncertainty diagnostic; with only
    three development seeds it is intentionally not used as a hard gate.
    """
    m = _paired(df, CMDP, "weighted_violation_mean").copy()
    if m.empty:
        return {
            "ok": False, "mean_relative_difference": np.nan,
            "mean_ratio": np.nan, "median_relative_difference": np.nan,
            "seed_cluster_bootstrap_upper95": np.nan, "n": 0,
        }
    denom = np.maximum(np.abs(m["comp"].to_numpy(dtype=float)), 1e-12)
    m["relative_difference"] = (m["full"].to_numpy(dtype=float) - m["comp"].to_numpy(dtype=float)) / denom
    mean_rel = float(m["relative_difference"].mean())
    med_rel = float(m["relative_difference"].median())
    ratio = float(m["full"].mean() / max(float(m["comp"].mean()), 1e-12))

    seed_means = m.groupby("seed", dropna=False)["relative_difference"].mean().to_numpy(dtype=float)
    upper = np.nan
    if seed_means.size:
        rng = np.random.default_rng(440044)
        boot = np.empty(int(bootstrap_reps), dtype=float)
        for i in range(int(bootstrap_reps)):
            boot[i] = float(np.mean(rng.choice(seed_means, size=seed_means.size, replace=True)))
        upper = float(np.quantile(boot, 0.95))
    return {
        "ok": bool(np.isfinite(mean_rel) and mean_rel <= float(margin)),
        "mean_relative_difference": mean_rel,
        "mean_ratio": ratio,
        "median_relative_difference": med_rel,
        "seed_cluster_bootstrap_upper95": upper,
        "n": int(len(m)),
    }


def _conditioned_mode_shift(df: pd.DataFrame, target_intent: str) -> float:
    full = df[df.scheme == FULL].copy()
    rows = []
    for _, r in full.iterrows():
        a = []
        b = []
        for mode in MODE_NAMES:
            x = pd.to_numeric(pd.Series([r.get(f"intent_mode_prob__balanced_agentic__{mode}", np.nan)]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([r.get(f"intent_mode_prob__{target_intent}__{mode}", np.nan)]), errors="coerce").iloc[0]
            a.append(float(x) if np.isfinite(x) else np.nan)
            b.append(float(y) if np.isfinite(y) else np.nan)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if np.isfinite(a).all() and np.isfinite(b).all():
            rows.append(float(np.sum(np.abs(a - b))))
    return float(np.mean(rows)) if rows else np.nan


def _fingerprint_mismatches(frames: Iterable[pd.DataFrame]) -> int:
    df = pd.concat(list(frames), ignore_index=True)
    if "policy_fingerprint" not in df.columns:
        return 999
    learned = df[df["policy_fingerprint"].fillna("").astype(str).str.len() > 0]
    if learned.empty:
        return 999
    nuniq = learned.groupby(["seed", "scheme"], dropna=False)["policy_fingerprint"].nunique(dropna=False)
    return int((nuniq != 1).sum())


def _source_checkpoint_mismatches(stochastic: pd.DataFrame, greedy: pd.DataFrame) -> int:
    if "policy_fingerprint" not in stochastic.columns or "policy_fingerprint" not in greedy.columns:
        return 999
    def compact(df: pd.DataFrame, label: str) -> pd.DataFrame:
        x = df[df["policy_fingerprint"].fillna("").astype(str).str.len() > 0].copy()
        if x.empty:
            return x
        g = x.groupby(["seed", "scheme"], dropna=False)["policy_fingerprint"].agg(lambda z: sorted(set(map(str, z))))
        rows = []
        for (seed, scheme), vals in g.items():
            rows.append({"seed": seed, "scheme": scheme, label: vals[0] if len(vals) == 1 else "MULTIPLE:" + "|".join(vals)})
        return pd.DataFrame(rows)
    a = compact(stochastic, "fp_stochastic")
    b = compact(greedy, "fp_greedy")
    if a.empty or b.empty:
        return 999
    m = a.merge(b, on=["seed", "scheme"], how="inner")
    if m.empty:
        return 999
    return int((m.fp_stochastic != m.fp_greedy).sum())


def _greedy_sensitivity(stochastic: pd.DataFrame, greedy: pd.DataFrame) -> dict:
    out = {}
    if greedy.empty:
        return out
    key = ["intent", "scenario", "seed", "scheme"]
    metrics = ["weighted_violation_mean", "weighted_violation_cvar95", "blocking_mean", "throughput_mean_mbps"]
    for metric in metrics:
        if metric not in stochastic.columns or metric not in greedy.columns:
            continue
        a = stochastic[key + [metric]].rename(columns={metric: "stoch"})
        b = greedy[key + [metric]].rename(columns={metric: "greedy"})
        m = a.merge(b, on=key, how="inner").dropna()
        if m.empty:
            continue
        denom = max(float(np.mean(np.abs(m.greedy))), 1e-12)
        out[f"greedy_sensitivity::{metric}::mean_abs_difference"] = float(np.mean(np.abs(m.stoch - m.greedy)))
        out[f"greedy_sensitivity::{metric}::relative_mean_abs_difference"] = float(np.mean(np.abs(m.stoch - m.greedy)) / denom)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feasibility", type=Path, required=True, help="Existing V4.4 feasibility development directory.")
    ap.add_argument("--static", type=Path, required=True)
    ap.add_argument("--reliability", type=Path, required=True)
    ap.add_argument("--latency", type=Path, required=True)
    ap.add_argument("--admission", type=Path, required=True)
    ap.add_argument("--greedy-static", type=Path)
    ap.add_argument("--greedy-reliability", type=Path)
    ap.add_argument("--greedy-latency", type=Path)
    ap.add_argument("--greedy-admission", type=Path)
    ap.add_argument("--report", type=Path, default=Path("outputs_agentic/v44e_development_gate.json"))
    args = ap.parse_args()

    stochastic_dirs = [args.static, args.reliability, args.latency, args.admission]
    reports = {str(d): validate(d) for d in stochastic_dirs}
    structural_ok = all(r["status"] == "PASS" for r in reports.values())
    for d, r in reports.items():
        print(f"{d}: validator={r['status']}")

    static = _load(args.static)
    reliability = _load(args.reliability)
    latency = _load(args.latency)
    admission = _load(args.admission)
    main_df = pd.concat([static, reliability, latency], ignore_index=True)
    switch_df = pd.concat([reliability, latency], ignore_index=True)
    all_df = pd.concat([main_df, admission], ignore_index=True)

    result: dict[str, object] = {}
    print("\nV4.4E stochastic frozen-policy development gates")
    print(f"1. Structural validation: {'PASS' if structural_ok else 'FAIL'}")
    result["structural_validation"] = bool(structural_ok)

    # Reuse the frozen V4.4 target-feasibility decision. No target is changed in V4.4E.
    audit_path = args.feasibility / "v44_feasibility_audit.csv"
    feasibility_ok = False
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        required = {"balanced_agentic", "reliability_first", "latency_critical"}
        nominal = audit[(audit.scenario == "nominal") & audit.intent.isin(required)]
        feasible = set(nominal.loc[nominal.simultaneous_target_feasible_by_anchor == 1, "intent"].astype(str))
        feasibility_ok = required.issubset(feasible)
    print(f"2. Frozen V4.4 nominal target feasibility: {'PASS' if feasibility_ok else 'FAIL'}")
    result["nominal_target_feasibility"] = bool(feasibility_ok)

    mismatch = _fingerprint_mismatches([static, reliability, latency, admission])
    fp_ok = mismatch == 0
    print(f"3. One frozen checkpoint across V4.4E contexts: {'PASS' if fp_ok else 'FAIL'} ({mismatch} mismatched seed/scheme groups)")
    result["universal_policy_fingerprint"] = bool(fp_ok)
    result["policy_fingerprint_mismatches"] = int(mismatch)

    # If old greedy outputs are supplied, require exact fingerprint identity.
    greedy_parts = []
    for p in [args.greedy_static, args.greedy_reliability, args.greedy_latency, args.greedy_admission]:
        if p is not None and p.exists():
            greedy_parts.append(_load(p))
    source_fp_ok = True
    source_mismatch = 0
    greedy_df = pd.concat(greedy_parts, ignore_index=True) if greedy_parts else pd.DataFrame()
    if not greedy_df.empty:
        source_mismatch = _source_checkpoint_mismatches(all_df, greedy_df)
        source_fp_ok = source_mismatch == 0
        print(f"4. V4.4 checkpoint identity vs greedy source runs: {'PASS' if source_fp_ok else 'FAIL'} ({source_mismatch} mismatched rows)")
    else:
        print("4. V4.4 checkpoint identity vs greedy source runs: NOT CHECKED (greedy directories not supplied)")
    result["source_checkpoint_identity"] = bool(source_fp_ok)
    result["source_checkpoint_mismatches"] = int(source_mismatch)

    # Tail-risk mechanism remains the primary ablation claim.
    tail_cvar = _gate_pair(main_df, NO_TAIL, "weighted_violation_cvar95", min_win_frac=0.50, max_mean_ratio=0.98)
    tail_post = _gate_pair(switch_df, NO_TAIL, "post_switch_violation_mean", min_win_frac=0.50, max_mean_ratio=0.98)
    tail_ok = bool(tail_cvar[0] and tail_post[0])
    print(
        f"5. Tail-risk ablation: CVaR wins={tail_cvar[1]}/{tail_cvar[2]}, ratio={tail_cvar[3]:.3f}; "
        f"post-switch wins={tail_post[1]}/{tail_post[2]}, ratio={tail_post[3]:.3f} -> {'PASS' if tail_ok else 'REVIEW'}"
    )
    result["tail_risk_gate"] = tail_ok
    result["tail_cvar_mean_ratio"] = float(tail_cvar[3]) if np.isfinite(tail_cvar[3]) else None
    result["tail_post_switch_mean_ratio"] = float(tail_post[3]) if np.isfinite(tail_post[3]) else None

    # Mean-risk comparison is treated as paired non-inferiority, not as an
    # optimization target. V4.4E must not tune to an arbitrary mean-ratio cutoff.
    ni = _paired_noninferiority(main_df, margin=0.05)
    print(
        "6. Paired mean-violation non-inferiority vs PPO-CMDP: "
        f"mean relative difference={ni['mean_relative_difference']:.3f}, ratio={ni['mean_ratio']:.3f}, "
        f"seed-cluster bootstrap upper95={ni['seed_cluster_bootstrap_upper95']:.3f} "
        f"-> {'PASS' if ni['ok'] else 'REVIEW'} (5% development margin)"
    )
    result["mean_violation_noninferiority"] = bool(ni["ok"])
    for k, v in ni.items():
        if k != "ok":
            result[f"mean_violation_noninferiority::{k}"] = None if isinstance(v, float) and not np.isfinite(v) else v

    # In stochastic deployment argmax-change is no longer the operative policy
    # response. Use total variation implied by the same-state probability L1.
    full = main_df[main_df.scheme == FULL]
    cf_l1 = _mean_numeric(full, "counterfactual_intent_policy_l1")
    cf_mode = _mean_numeric(full, "counterfactual_mode_policy_l1")
    cf_argmax = _mean_numeric(full, "counterfactual_argmax_change_rate")
    entropy = _mean_numeric(full, "stochastic_action_entropy")
    rel_mode_shift = _conditioned_mode_shift(reliability, "reliability_first")
    lat_mode_shift = _conditioned_mode_shift(latency, "latency_critical")
    # L1=0.04 corresponds to a 2 percentage-point total-variation change in the
    # complete action distribution; mode L1=0.02 corresponds to 1 percentage point.
    intent_ok = bool(np.isfinite(cf_l1) and cf_l1 >= 0.04 and np.isfinite(cf_mode) and cf_mode >= 0.02)
    entropy_ok = bool(np.isfinite(entropy) and entropy >= 0.05)
    print(
        "7. Same-state intent-conditioned stochastic policy use: "
        f"action-L1={cf_l1:.3f} (TV={0.5*cf_l1:.3f}), mode-L1={cf_mode:.3f}, "
        f"argmax-change={cf_argmax:.3f} -> {'PASS' if intent_ok else 'REVIEW'}"
    )
    print(
        f"   Observed conditioned mode shift L1 reliability={rel_mode_shift:.3f}, latency={lat_mode_shift:.3f}; "
        f"stochastic entropy={entropy:.3f} -> {'PASS' if entropy_ok else 'REVIEW'}"
    )
    result["counterfactual_intent_use"] = intent_ok
    result["counterfactual_intent_policy_l1"] = None if not np.isfinite(cf_l1) else float(cf_l1)
    result["counterfactual_mode_policy_l1"] = None if not np.isfinite(cf_mode) else float(cf_mode)
    result["counterfactual_argmax_change_rate"] = None if not np.isfinite(cf_argmax) else float(cf_argmax)
    result["conditioned_mode_shift_reliability_l1"] = None if not np.isfinite(rel_mode_shift) else float(rel_mode_shift)
    result["conditioned_mode_shift_latency_l1"] = None if not np.isfinite(lat_mode_shift) else float(lat_mode_shift)
    result["stochastic_action_entropy"] = None if not np.isfinite(entropy) else float(entropy)
    result["stochastic_entropy_ok"] = entropy_ok

    # Admission shield: probability-mass evidence plus realized stochastic action
    # behavior. The latter must create a separation from the separately trained
    # no-shield ablation without causing >5% mean violation degradation.
    adm_full = admission[admission.scheme == FULL]
    adm_ns = admission[admission.scheme == NO_SHIELD]
    filtered_mass = _mean_numeric(adm_full, "shield_filtered_probability_mass")
    prevention = _mean_numeric(adm_full, "actual_shield_prevention_rate")
    unsafe_mass = _mean_numeric(adm_full, "pre_mask_unsafe_probability_mass")
    reject_prob = _mean_numeric(adm_full, "feasible_reject_probability")
    reject_full = _mean_numeric(adm_full, "deny_action_rate")
    reject_ns = _mean_numeric(adm_ns, "deny_action_rate")
    block_full = _mean_numeric(adm_full, "blocking_mean")
    block_ns = _mean_numeric(adm_ns, "blocking_mean")
    viol_full = _mean_numeric(adm_full, "weighted_violation_mean")
    viol_ns = _mean_numeric(adm_ns, "weighted_violation_mean")
    reject_gap = reject_ns - reject_full if np.isfinite(reject_ns) and np.isfinite(reject_full) else np.nan
    block_gap = block_ns - block_full if np.isfinite(block_ns) and np.isfinite(block_full) else np.nan
    safety_not_worse = bool(np.isfinite(viol_full) and np.isfinite(viol_ns) and viol_full <= 1.05 * max(viol_ns, 1e-12))
    realized_separation = bool(
        (np.isfinite(reject_gap) and reject_gap >= 0.001)
        or (np.isfinite(block_gap) and block_gap >= 0.001)
    )
    shield_ok = bool(
        np.isfinite(filtered_mass) and filtered_mass >= 0.005
        and np.isfinite(prevention) and prevention >= 0.005
        and realized_separation
        and safety_not_worse
    )
    print(
        "8. Admission-safety shield under stochastic deployment: "
        f"filtered probability mass={filtered_mass:.4f}, sampled prevention={prevention:.4f}, "
        f"feasible reject probability={reject_prob:.4f}, unsafe mass={unsafe_mass:.4f}; "
        f"reject Full/NoShield={reject_full:.4f}/{reject_ns:.4f}, "
        f"blocking Full/NoShield={block_full:.4f}/{block_ns:.4f}, "
        f"violation Full/NoShield={viol_full:.3f}/{viol_ns:.3f} -> {'PASS' if shield_ok else 'REVIEW'}"
    )
    result["admission_shield_gate"] = shield_ok
    result["shield_filtered_probability_mass"] = None if not np.isfinite(filtered_mass) else float(filtered_mass)
    result["actual_shield_prevention_rate"] = None if not np.isfinite(prevention) else float(prevention)
    result["admission_reject_gap"] = None if not np.isfinite(reject_gap) else float(reject_gap)
    result["admission_blocking_gap"] = None if not np.isfinite(block_gap) else float(block_gap)
    result["shield_safety_not_worse"] = safety_not_worse

    # NoShield performance is a secondary mechanism diagnostic. Require that the
    # proposed shield is not materially worse on tail/post-switch risk; the
    # primary shield gate above comes from probability and realized blocking.
    ns_cvar = _gate_pair(main_df, NO_SHIELD, "weighted_violation_cvar95", min_win_frac=0.40, max_mean_ratio=1.05)
    ns_post = _gate_pair(switch_df, NO_SHIELD, "post_switch_violation_mean", min_win_frac=0.40, max_mean_ratio=1.05)
    noshield_perf_ok = bool(ns_cvar[0] and ns_post[0])
    print(
        f"9. Full IA vs NoShield risk guard: CVaR wins={ns_cvar[1]}/{ns_cvar[2]}, ratio={ns_cvar[3]:.3f}; "
        f"post-switch wins={ns_post[1]}/{ns_post[2]}, ratio={ns_post[3]:.3f} -> {'PASS' if noshield_perf_ok else 'REVIEW'}"
    )
    result["noshield_risk_guard"] = noshield_perf_ok

    if not greedy_df.empty:
        sens = _greedy_sensitivity(all_df, greedy_df)
        result.update(sens)
        print("10. Greedy sensitivity: recorded for secondary analysis (not a development gate).")
    else:
        print("10. Greedy sensitivity: not available; stochastic gate remains valid.")

    overall = bool(
        structural_ok and feasibility_ok and fp_ok and source_fp_ok
        and tail_ok and ni["ok"] and intent_ok and entropy_ok
        and shield_ok and noshield_perf_ok
    )
    decision = "PROCEED TO UNTOUCHED FINAL SEEDS" if overall else "REVIEW V4.4E BEFORE FINAL SEEDS"
    result["overall_pass"] = overall
    result["decision"] = decision
    print(f"\nOverall V4.4E development decision: {decision}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = ["# V4.4E stochastic evaluation gate", "", f"Decision: **{decision}**", ""]
    md += [f"- `{k}`: `{v}`" for k, v in result.items() if k != "decision"]
    args.report.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0 if overall else (2 if not structural_ok else 3)


if __name__ == "__main__":
    raise SystemExit(main())
