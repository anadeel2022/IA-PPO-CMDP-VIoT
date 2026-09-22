from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

RUNS = {
    "static": "td4_final_static",
    "reliability": "td4_final_reliability",
    "latency": "td4_final_latency",
    "admission": "td4_final_admission",
}

ORIGINAL_RUNS = {
    "static": "v44ec300_final_static",
    "reliability": "v44ec300_final_reliability",
    "latency": "v44ec300_final_latency",
    "admission": "v44ec300_final_admission",
}

MEASUREMENT_IDENTITY_METRICS = {
    "measurement_blocking_mean": "blocking_mean",
    "measurement_throughput_mean_mbps": "throughput_mean_mbps",
    "measurement_service_success_rate_mean": "service_success_rate_mean",
    "measurement_deadline_exposure_mean": "deadline_exposure_mean",
    "measurement_unfinished_rate_mean": "unfinished_rate_mean",
}

FULL = "Intent-PPO-CMDP"
CAUSAL_OFF = "Intent-PPO-CMDP-FullPolicy-NoShield"
PPO_CMDP = "PPO_CMDP"
NO_TAIL = "Intent-PPO-CMDP-NoRisk"


def _read_seed(root: Path, run_id: str) -> pd.DataFrame:
    path = root / run_id / "agentic_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required sensitivity output: {path}")
    df = pd.read_csv(path)
    df["sensitivity_run"] = run_id
    return df


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return float("nan")
    x = pd.to_numeric(df[col], errors="coerce")
    return float(x.mean())


def _max_abs(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return float("nan")
    x = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmax(np.abs(x.to_numpy(dtype=float)))) if x.notna().any() else float("nan")


def _paired_delta(df: pd.DataFrame, a: str, b: str, metric: str) -> Dict[str, float]:
    keys = [c for c in ["intent", "scenario", "seed"] if c in df.columns]
    sub = df[df["scheme"].isin([a, b])]
    if sub.empty or metric not in sub.columns:
        return {"n": 0, "a_mean": np.nan, "b_mean": np.nan, "a_minus_b": np.nan, "b_minus_a": np.nan}
    p = sub.pivot_table(index=keys, columns="scheme", values=metric, aggfunc="first").dropna()
    if a not in p.columns or b not in p.columns:
        return {"n": 0, "a_mean": np.nan, "b_mean": np.nan, "a_minus_b": np.nan, "b_minus_a": np.nan}
    return {
        "n": int(len(p)),
        "a_mean": float(p[a].mean()),
        "b_mean": float(p[b].mean()),
        "a_minus_b": float((p[a] - p[b]).mean()),
        "b_minus_a": float((p[b] - p[a]).mean()),
    }


def _measurement_window_identity(output_root: Path, td_dfs: Dict[str, pd.DataFrame]) -> Dict:
    """Verify that the first 300 steps are identical to the frozen primary run.

    The terminal drain begins only after the measurement window. Therefore these
    raw request-level metrics must match the already-frozen 300-step outputs.
    This is a stronger integrity check than re-validating policy-distribution
    diagnostics inside the drain run.
    """
    per_run = {}
    all_pass = True
    total_pairs = 0
    max_abs = 0.0
    for name, original_run in ORIGINAL_RUNS.items():
        td = td_dfs[name].copy()
        orig_path = output_root / original_run / "agentic_seed_summary.csv"
        if not orig_path.exists():
            per_run[name] = {"status": "NOT_CHECKED", "reason": f"missing {orig_path}"}
            all_pass = False
            continue
        orig = pd.read_csv(orig_path)
        keys = [c for c in ["intent", "scenario", "seed", "scheme"] if c in td.columns and c in orig.columns]
        if len(keys) != 4:
            per_run[name] = {"status": "FAIL", "reason": "missing identity key columns"}
            all_pass = False
            continue
        merged = td.merge(orig, on=keys, how="inner", suffixes=("_td", "_orig"))
        expected_pairs = len(td)
        metric_report = {}
        run_pass = len(merged) == expected_pairs
        for td_col, orig_col in MEASUREMENT_IDENTITY_METRICS.items():
            td_name = td_col if td_col in merged.columns else f"{td_col}_td"
            orig_name = orig_col if orig_col in merged.columns else f"{orig_col}_orig"
            if td_name not in merged.columns or orig_name not in merged.columns:
                metric_report[td_col] = {"status": "MISSING"}
                run_pass = False
                continue
            a = pd.to_numeric(merged[td_name], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(merged[orig_name], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            if not finite.all():
                metric_report[td_col] = {"status": "NONFINITE", "finite_pairs": int(finite.sum()), "pairs": int(len(a))}
                run_pass = False
                continue
            diff = np.abs(a - b)
            md = float(diff.max()) if diff.size else 0.0
            metric_report[td_col] = {"status": "PASS" if md <= 1e-10 else "FAIL", "max_abs_diff": md}
            max_abs = max(max_abs, md)
            if md > 1e-10:
                run_pass = False
        per_run[name] = {
            "status": "PASS" if run_pass else "FAIL",
            "matched_rows": int(len(merged)),
            "expected_rows": int(expected_pairs),
            "metrics": metric_report,
        }
        total_pairs += int(len(merged))
        all_pass = all_pass and run_pass
    return {
        "overall_pass": bool(all_pass),
        "matched_rows": int(total_pairs),
        "max_abs_difference": float(max_abs),
        "per_run": per_run,
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_audit(output_root: Path, share_dir: Path) -> Dict:
    dfs = {name: _read_seed(output_root, run_id) for name, run_id in RUNS.items()}
    all_eval = pd.concat([dfs[k] for k in ["static", "reliability", "latency", "admission"]], ignore_index=True)

    measurement_identity = _measurement_window_identity(output_root, dfs)

    required_cols = [
        "right_censored_rate_mean", "measurement_right_censored_rate_mean",
        "unfinished_rate_mean", "measurement_unfinished_rate_mean",
        "service_success_rate_mean", "measurement_service_success_rate_mean",
        "deadline_exposure_mean", "measurement_deadline_exposure_mean",
        "request_conservation_error_max_abs",
    ]
    missing = sorted({c for c in required_cols if c not in all_eval.columns})

    rng_preserved = bool(
        "measurement_rng_stream_preserved" in all_eval.columns
        and pd.to_numeric(all_eval["measurement_rng_stream_preserved"], errors="coerce").fillna(0).eq(1).all()
    )

    overall = {
        "rows": int(len(all_eval)),
        "measurement_rng_stream_preserved": rng_preserved,
        "missing_required_columns": missing,
        "right_censored_rate_mean": _mean(all_eval, "right_censored_rate_mean"),
        "right_censored_rate_max": float(pd.to_numeric(all_eval.get("right_censored_rate_mean"), errors="coerce").max()) if "right_censored_rate_mean" in all_eval else np.nan,
        "measurement_right_censored_rate_mean": _mean(all_eval, "measurement_right_censored_rate_mean"),
        "measurement_unfinished_rate_mean": _mean(all_eval, "measurement_unfinished_rate_mean"),
        "terminal_unfinished_rate_mean": _mean(all_eval, "unfinished_rate_mean"),
        "unfinished_rate_change": _mean(all_eval, "unfinished_rate_mean") - _mean(all_eval, "measurement_unfinished_rate_mean"),
        "measurement_service_success_rate_mean": _mean(all_eval, "measurement_service_success_rate_mean"),
        "terminal_service_success_rate_mean": _mean(all_eval, "service_success_rate_mean"),
        "service_success_change": _mean(all_eval, "service_success_rate_mean") - _mean(all_eval, "measurement_service_success_rate_mean"),
        "measurement_deadline_exposure_mean": _mean(all_eval, "measurement_deadline_exposure_mean"),
        "terminal_deadline_exposure_mean": _mean(all_eval, "deadline_exposure_mean"),
        "deadline_exposure_change": _mean(all_eval, "deadline_exposure_mean") - _mean(all_eval, "measurement_deadline_exposure_mean"),
        "request_conservation_error_max_abs": _max_abs(all_eval, "request_conservation_error_max_abs"),
        "terminal_drain_actual_steps_mean": _mean(all_eval, "terminal_drain_actual_steps_mean"),
        "terminal_drain_actual_steps_max": float(pd.to_numeric(all_eval.get("terminal_drain_actual_steps_max"), errors="coerce").max()) if "terminal_drain_actual_steps_max" in all_eval else np.nan,
        "terminal_drain_extension_steps_mean": _mean(all_eval, "terminal_drain_extension_steps_mean"),
        "terminal_drain_extension_steps_max": float(pd.to_numeric(all_eval.get("terminal_drain_extension_steps_max"), errors="coerce").max()) if "terminal_drain_extension_steps_max" in all_eval else np.nan,
        "max_realized_device_deadline_steps_max": float(pd.to_numeric(all_eval.get("max_realized_device_deadline_steps_max"), errors="coerce").max()) if "max_realized_device_deadline_steps_max" in all_eval else np.nan,
    }

    full = all_eval[all_eval["scheme"] == FULL]
    overall["full_ia"] = {
        "measurement_unfinished_rate_mean": _mean(full, "measurement_unfinished_rate_mean"),
        "terminal_unfinished_rate_mean": _mean(full, "unfinished_rate_mean"),
        "right_censored_rate_mean": _mean(full, "right_censored_rate_mean"),
        "right_censored_rate_max": float(pd.to_numeric(full.get("right_censored_rate_mean"), errors="coerce").max()) if "right_censored_rate_mean" in full else np.nan,
        "measurement_service_success_rate_mean": _mean(full, "measurement_service_success_rate_mean"),
        "terminal_service_success_rate_mean": _mean(full, "service_success_rate_mean"),
        "measurement_deadline_exposure_mean": _mean(full, "measurement_deadline_exposure_mean"),
        "terminal_deadline_exposure_mean": _mean(full, "deadline_exposure_mean"),
    }

    admission = dfs["admission"]
    shield = {
        "blocking": _paired_delta(admission, FULL, CAUSAL_OFF, "blocking_mean"),
        "service_success": _paired_delta(admission, FULL, CAUSAL_OFF, "service_success_rate_mean"),
        "deadline_exposure": _paired_delta(admission, FULL, CAUSAL_OFF, "deadline_exposure_mean"),
        "unfinished": _paired_delta(admission, FULL, CAUSAL_OFF, "unfinished_rate_mean"),
        "weighted_violation": _paired_delta(admission, FULL, CAUSAL_OFF, "weighted_violation_mean"),
        "cvar95": _paired_delta(admission, FULL, CAUSAL_OFF, "weighted_violation_cvar95"),
    }

    static = dfs["static"]
    comparisons = {
        "full_vs_ppo_cmdp_weighted_violation": _paired_delta(static, FULL, PPO_CMDP, "weighted_violation_mean"),
        "full_vs_ppo_cmdp_cvar95": _paired_delta(static, FULL, PPO_CMDP, "weighted_violation_cvar95"),
        "full_vs_no_tail_weighted_violation": _paired_delta(static, FULL, NO_TAIL, "weighted_violation_mean"),
        "full_vs_no_tail_cvar95": _paired_delta(static, FULL, NO_TAIL, "weighted_violation_cvar95"),
    }

    per_run = {}
    for name, df in dfs.items():
        per_run[name] = {
            "rows": int(len(df)),
            "right_censored_rate_mean": _mean(df, "right_censored_rate_mean"),
            "measurement_unfinished_rate_mean": _mean(df, "measurement_unfinished_rate_mean"),
            "terminal_unfinished_rate_mean": _mean(df, "unfinished_rate_mean"),
            "service_success_change": _mean(df, "service_success_rate_mean") - _mean(df, "measurement_service_success_rate_mean"),
            "deadline_exposure_change": _mean(df, "deadline_exposure_mean") - _mean(df, "measurement_deadline_exposure_mean"),
            "terminal_drain_actual_steps_mean": _mean(df, "terminal_drain_actual_steps_mean"),
            "terminal_drain_actual_steps_max": float(pd.to_numeric(df.get("terminal_drain_actual_steps_max"), errors="coerce").max()) if "terminal_drain_actual_steps_max" in df else np.nan,
            "terminal_drain_extension_steps_mean": _mean(df, "terminal_drain_extension_steps_mean"),
            "terminal_drain_extension_steps_max": float(pd.to_numeric(df.get("terminal_drain_extension_steps_max"), errors="coerce").max()) if "terminal_drain_extension_steps_max" in df else np.nan,
        }

    pass_censoring = (not missing) and np.isfinite(overall["right_censored_rate_max"]) and overall["right_censored_rate_max"] <= 1e-12
    pass_conservation = np.isfinite(overall["request_conservation_error_max_abs"]) and overall["request_conservation_error_max_abs"] <= 1e-12
    audit = {
        "protocol": {
            "training_steps": 150,
            "arrival_window_steps": 300,
            "terminal_drain_minimum_steps": 250,
            "terminal_drain_auto_extend": True,
            "terminal_drain_safety_cap_steps": 500,
            "terminal_total_steps": "300 + realized adaptive drain",
            "training_changed": False,
            "policy_parameters_changed": False,
            "arrival_during_drain": False,
            "purpose": "right-censoring sensitivity for the frozen final learned/heuristic evaluation policies; feasibility-anchor calibration is not rerun because it is diagnostic only",
        },
        "integrity": {
            "right_censoring_eliminated": bool(pass_censoring),
            "request_conservation_pass": bool(pass_conservation),
            "measurement_window_identity_pass": bool(measurement_identity["overall_pass"]),
            "measurement_rng_stream_preserved": bool(rng_preserved),
            "overall_pass": bool(pass_censoring and pass_conservation and measurement_identity["overall_pass"] and rng_preserved),
        },
        "measurement_window_identity": measurement_identity,
        "overall": overall,
        "per_run": per_run,
        "same_checkpoint_shield_admission": shield,
        "static_comparisons": comparisons,
    }

    share_dir.mkdir(parents=True, exist_ok=True)
    (share_dir / "terminal_drain_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=True), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Terminal-drain sensitivity audit")
    lines.append("")
    lines.append("Frozen protocol: 150-step training, 300-step arrival/measurement window, then at least 250 no-arrival follow-up steps, adaptively extended only until all pending cohort requests have received their complete realized deadline opportunity (500-step safety cap). No policy is retrained or updated.")
    lines.append("")
    lines.append(f"- Right-censoring eliminated: {'PASS' if pass_censoring else 'REVIEW'} (max terminal right-censored rate = {overall['right_censored_rate_max']:.6g})")
    lines.append(f"- Request conservation: {'PASS' if pass_conservation else 'REVIEW'} (max absolute error = {overall['request_conservation_error_max_abs']:.6g})")
    lines.append(f"- Frozen 300-step measurement-window identity: {'PASS' if measurement_identity['overall_pass'] else 'REVIEW'} (max absolute difference = {measurement_identity['max_abs_difference']:.3g})")
    lines.append(f"- Measurement RNG replay preservation: {'PASS' if rng_preserved else 'FAIL'}")
    lines.append(f"- Mean unfinished rate: {overall['measurement_unfinished_rate_mean']:.4f} at 300 steps -> {overall['terminal_unfinished_rate_mean']:.4f} after drain")
    lines.append(f"- Adaptive drain actually used: mean {overall['terminal_drain_actual_steps_mean']:.2f} steps; maximum {overall['terminal_drain_actual_steps_max']:.0f}; maximum extension beyond 250 = {overall['terminal_drain_extension_steps_max']:.0f} steps")
    lines.append(f"- Maximum realized per-device deadline observed in evaluated policies: {overall['max_realized_device_deadline_steps_max']:.0f} steps")
    lines.append(f"- Mean service success: {overall['measurement_service_success_rate_mean']:.4f} -> {overall['terminal_service_success_rate_mean']:.4f}")
    lines.append(f"- Mean deadline exposure: {overall['measurement_deadline_exposure_mean']:.4f} -> {overall['terminal_deadline_exposure_mean']:.4f}")
    f = overall["full_ia"]
    lines.append(f"- Full IA unfinished: {f['measurement_unfinished_rate_mean']:.4f} -> {f['terminal_unfinished_rate_mean']:.4f}; terminal right-censored rate = {f['right_censored_rate_mean']:.6g}")
    lines.append("")
    lines.append("## Same-checkpoint shield sensitivity under admission pressure")
    for metric, vals in shield.items():
        lines.append(f"- {metric}: Full={vals['a_mean']:.4f}, ShieldOff={vals['b_mean']:.4f}, ShieldOff-Full={vals['b_minus_a']:.4f}, n={vals['n']}")
    lines.append("")
    lines.append("These are sensitivity results. The original frozen 300-step results remain the primary manuscript results unless the terminal accounting check materially changes the interpretation.")
    (share_dir / "terminal_drain_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # One combined seed-level file is convenient for independent verification.
    combined = pd.concat(dfs.values(), ignore_index=True)
    combined.to_csv(share_dir / "terminal_drain_all_seed_summaries.csv", index=False)

    # Copy the compact and seed-level files needed to inspect every sensitivity run.
    for name, run_id in RUNS.items():
        src = output_root / run_id
        dst = share_dir / name
        for filename in ["agentic_seed_summary.csv", "agentic_summary_aggregated.csv", "agentic_main_summary.csv", "manifest.json", "submission_validation_report.json", "_resume_checkpoint.csv", "v44_feasibility_audit.csv"]:
            _copy_if_exists(src / filename, dst / filename)

    # Include the already-frozen primary final summaries/audit when they are available,
    # so the user only needs to share this single ZIP back for comparison.
    original_dir = share_dir / "original_frozen_final_reference"
    for run_id in ["v44ec300_final_static", "v44ec300_final_reliability", "v44ec300_final_latency", "v44ec300_final_admission", "v44ec300_feasibility_final"]:
        src = output_root / run_id
        for filename in ["agentic_seed_summary.csv", "agentic_summary_aggregated.csv", "agentic_main_summary.csv", "manifest.json"]:
            _copy_if_exists(src / filename, original_dir / run_id / filename)
    for filename in ["v44ec300_final_audit.json", "v44ec300_final_audit.md"]:
        _copy_if_exists(output_root / filename, original_dir / filename)

    return audit


def package_share(share_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(share_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("terminal_drain_sensitivity_share_v4") / path.relative_to(share_dir)))
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs_agentic")
    ap.add_argument("--share-dir", default="terminal_drain_sensitivity_share_v4")
    ap.add_argument("--zip", default="terminal_drain_sensitivity_share_v4.zip")
    args = ap.parse_args()
    root = Path(args.output_root)
    share = Path(args.share_dir)
    if share.exists():
        shutil.rmtree(share)
    audit = build_audit(root, share)
    zip_path = package_share(share, Path(args.zip))
    print(json.dumps(audit["integrity"], indent=2))
    print(f"SHARE THIS FILE: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
