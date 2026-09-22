#!/usr/bin/env python3
"""Audit intent-target attainability using deterministic service anchors.

The audit is diagnostic.  It asks whether the complete target vector is
*demonstrated* by the finite set of deterministic anchors used in the
experiment.  Failure is not a proof of physical infeasibility.

For the 300-step confirmation, the script reports two certificates:
  1. a single-anchor certificate; and
  2. an expected-metric convex-mixture certificate over the anchor policies.

The mixture certificate is useful because a randomized/time-sharing scheduler
can operate between deterministic anchor points.  It is deliberately reported
as an expected-metric diagnostic and is not used as an implementation gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANCHORS = {"MaxWeight", "CapacityFirst", "AllGrant"}
EPS = 1e-12
MIXTURE_GRID_STEP = 0.005
METRICS = [
    "blocking_mean",
    "interruption_mean",
    "deadline_exposure_mean",
    "service_success_rate_mean",
    "fairness_mean",
    "throughput_mean_mbps",
]


def _intent_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(x.get("name")): x for x in manifest.get("intents", []) if x.get("name")}


def _meets(row: pd.Series | dict[str, float], target: dict[str, Any]) -> bool:
    checks = [
        float(row.get("blocking_mean", np.inf)) <= float(target.get("block_target", np.inf)) + EPS,
        float(row.get("interruption_mean", np.inf)) <= float(target.get("intr_target", np.inf)) + EPS,
        float(row.get("deadline_exposure_mean", np.inf)) <= float(target.get("max_deadline_exposure_rate", np.inf)) + EPS,
        float(row.get("service_success_rate_mean", -np.inf)) >= float(target.get("min_service_success_rate", -np.inf)) - EPS,
        float(row.get("fairness_mean", -np.inf)) >= float(target.get("min_fairness", -np.inf)) - EPS,
    ]
    floor = target.get("throughput_floor_mbps")
    if floor is not None:
        checks.append(float(row.get("throughput_mean_mbps", -np.inf)) >= float(floor) - EPS)
    return bool(all(checks))


def _normalized_gaps(values: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    """Positive normalized target misses; zero means the target is met."""
    block_t = float(target.get("block_target", np.inf))
    intr_t = float(target.get("intr_target", np.inf))
    dead_t = float(target.get("max_deadline_exposure_rate", np.inf))
    succ_t = float(target.get("min_service_success_rate", -np.inf))
    fair_t = float(target.get("min_fairness", -np.inf))
    thr_t = target.get("throughput_floor_mbps")
    thr_t = float(thr_t) if thr_t is not None else -np.inf

    scales = [max(abs(block_t), 0.01), max(abs(intr_t), 0.01), max(abs(dead_t), 0.01),
              max(abs(succ_t), 0.01), max(abs(fair_t), 0.01), max(abs(thr_t), 0.10)]
    raw = [
        max(0.0, float(values[0]) - block_t),
        max(0.0, float(values[1]) - intr_t),
        max(0.0, float(values[2]) - dead_t),
        max(0.0, succ_t - float(values[3])),
        max(0.0, fair_t - float(values[4])),
        max(0.0, thr_t - float(values[5])) if np.isfinite(thr_t) else 0.0,
    ]
    return np.asarray([x / s for x, s in zip(raw, scales)], dtype=float)


def _weight_vectors(n: int, step: float = MIXTURE_GRID_STEP):
    if n <= 0:
        return
    if n == 1:
        yield np.asarray([1.0], dtype=float)
        return
    units = int(round(1.0 / step))
    if n == 2:
        for i in range(units + 1):
            yield np.asarray([i / units, 1.0 - i / units], dtype=float)
        return
    if n == 3:
        for i in range(units + 1):
            for j in range(units - i + 1):
                k = units - i - j
                yield np.asarray([i, j, k], dtype=float) / units
        return
    # Generic recursive fallback is unnecessary for the current three anchors.
    raise ValueError(f"Convex audit supports up to three anchors, got {n}")


def _convex_audit(scheme_means: pd.DataFrame, target: dict[str, Any]) -> dict[str, Any]:
    if scheme_means.empty or not target:
        return {
            "feasible": False,
            "feasible_weights": {},
            "closest_weights": {},
            "closest_max_normalized_gap": np.nan,
            "closest_values": np.full(len(METRICS), np.nan),
        }
    sm = scheme_means.sort_values("scheme").reset_index(drop=True)
    schemes = sm["scheme"].astype(str).tolist()
    A = sm[METRICS].to_numpy(dtype=float)
    best = None
    feasible = None
    for w in _weight_vectors(len(schemes)):
        values = w @ A
        gaps = _normalized_gaps(values, target)
        score = float(np.max(gaps))
        if best is None or score < best[0] - 1e-15:
            best = (score, w.copy(), values.copy(), gaps.copy())
        if score <= 1e-12:
            feasible = (w.copy(), values.copy())
            break
    assert best is not None
    fweights = {}
    if feasible is not None:
        fweights = {s: float(x) for s, x in zip(schemes, feasible[0]) if x > 1e-12}
    cweights = {s: float(x) for s, x in zip(schemes, best[1]) if x > 1e-12}
    return {
        "feasible": feasible is not None,
        "feasible_weights": fweights,
        "closest_weights": cweights,
        "closest_max_normalized_gap": float(best[0]),
        "closest_values": best[2],
        "closest_gaps": best[3],
    }


def _missed_metric_names(values: np.ndarray, target: dict[str, Any]) -> str:
    gaps = _normalized_gaps(values, target)
    return ";".join(m for m, g in zip(METRICS, gaps) if g > 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    path = args.run_dir / "agentic_seed_summary.csv"
    if not path.exists():
        path = args.run_dir / "_resume_checkpoint.csv"
    df = pd.read_csv(path)
    df = df[df["scheme"].isin(ANCHORS)].copy()
    targets = _intent_map(manifest)

    rows = []
    for (intent, scenario), g in df.groupby(["intent", "scenario"], dropna=False):
        target = targets.get(str(intent), {})
        scheme_means = g.groupby("scheme", as_index=False).mean(numeric_only=True)
        simultaneous = []
        for _, r in scheme_means.iterrows():
            if target and _meets(r, target):
                simultaneous.append(str(r["scheme"]))

        convex = _convex_audit(scheme_means, target)
        best_violation_row = (
            scheme_means.loc[scheme_means["weighted_violation_mean"].idxmin()]
            if "weighted_violation_mean" in scheme_means else None
        )
        closest_values = np.asarray(convex["closest_values"], dtype=float)
        row = {
            "intent": intent,
            "scenario": scenario,
            "n_seeds": int(g["seed"].nunique()),
            "simultaneous_target_feasible_by_anchor": int(bool(simultaneous)),
            "feasible_anchor_schemes": ";".join(simultaneous),
            "target_feasible_by_anchor_mixture": int(bool(convex["feasible"])),
            "feasible_anchor_mixture": json.dumps(convex["feasible_weights"], sort_keys=True),
            "target_feasible_by_anchor_or_mixture": int(bool(simultaneous) or bool(convex["feasible"])),
            "closest_anchor_mixture": json.dumps(convex["closest_weights"], sort_keys=True),
            "closest_max_normalized_target_gap": float(convex["closest_max_normalized_gap"]),
            "closest_missed_metrics": _missed_metric_names(closest_values, target),
            "best_weighted_violation": float(best_violation_row["weighted_violation_mean"]) if best_violation_row is not None else np.nan,
            "best_violation_scheme": str(best_violation_row["scheme"]) if best_violation_row is not None else "",
            "best_blocking": float(scheme_means["blocking_mean"].min()),
            "best_interruption": float(scheme_means["interruption_mean"].min()),
            "best_deadline_exposure": float(scheme_means["deadline_exposure_mean"].min()),
            "best_service_success": float(scheme_means["service_success_rate_mean"].max()),
            "best_fairness": float(scheme_means["fairness_mean"].max()),
            "best_throughput_mbps": float(scheme_means["throughput_mean_mbps"].max()),
            "closest_mix_blocking": float(closest_values[0]),
            "closest_mix_interruption": float(closest_values[1]),
            "closest_mix_deadline_exposure": float(closest_values[2]),
            "closest_mix_service_success": float(closest_values[3]),
            "closest_mix_fairness": float(closest_values[4]),
            "closest_mix_throughput_mbps": float(closest_values[5]),
            "target_blocking": target.get("block_target", np.nan),
            "target_interruption": target.get("intr_target", np.nan),
            "target_deadline_exposure": target.get("max_deadline_exposure_rate", np.nan),
            "target_service_success": target.get("min_service_success_rate", np.nan),
            "target_fairness": target.get("min_fairness", np.nan),
            "target_throughput_mbps": target.get("throughput_floor_mbps", np.nan),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = args.run_dir / "v44_feasibility_audit.csv"
    out.to_csv(out_csv, index=False)

    single = int(out["simultaneous_target_feasible_by_anchor"].sum()) if not out.empty else 0
    mix = int(out["target_feasible_by_anchor_or_mixture"].sum()) if not out.empty else 0
    total = int(len(out))
    lines = [
        "# V4.4E-Causal300 intent-target attainability audit", "",
        f"Complete target vector demonstrated by a single deterministic anchor: **{single}/{total}** contexts.",
        f"Complete target vector demonstrated by a single anchor or expected-metric convex anchor mixture: **{mix}/{total}** contexts.", "",
        "This is a diagnostic over a finite anchor set. A failed context is not a proof of physical infeasibility. "
        "The convex certificate represents expected metrics obtainable by randomized/time-sharing use of the deterministic anchor policies; it does not claim per-episode simultaneous satisfaction.", "",
    ]
    for _, r in out.iterrows():
        if int(r["simultaneous_target_feasible_by_anchor"]):
            status = f"SINGLE-ANCHOR DEMONSTRATED ({r['feasible_anchor_schemes']})"
        elif int(r["target_feasible_by_anchor_mixture"]):
            status = f"MIXTURE DEMONSTRATED {r['feasible_anchor_mixture']}"
        else:
            status = (
                f"NOT DEMONSTRATED; closest normalized gap={r['closest_max_normalized_target_gap']:.4f}; "
                f"missed={r['closest_missed_metrics']}; closest mix={r['closest_anchor_mixture']}"
            )
        lines.append(
            f"- {r['intent']} / {r['scenario']}: {status}; "
            f"best weighted violation={r['best_weighted_violation']:.4f} ({r['best_violation_scheme']})."
        )
    (args.run_dir / "v44_feasibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nWrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
