#!/usr/bin/env python3
"""Final manuscript plot generator for the frozen V-IoT V4.4E-Causal300 study.

This script supersedes the older make_viot_manuscript_figures.py for the final paper.
It merges the frozen V4.4E-Causal300 outputs with the matched SOTA-DQN extension,
uses the five master seeds as the independent aggregation unit, and generates the
result figures currently intended for the manuscript.

Figures generated
-----------------
Fig02_Final_Static_Benchmark    : 4-panel matched static benchmark
Fig03_Final_Context_Heatmap     : 7 x 9 weighted-violation context heatmap
Fig04_Final_Intent_Switching    : post-switch violation, mode shift, adaptation time
Fig05_Final_Causal_Shield       : same-checkpoint shield ON/OFF under terminal-complete admission pressure
Fig06_Final_Terminal_Drain      : primary 300-step vs terminal-complete IA sensitivity

Usage
-----
python make_viot_final_manuscript_plots.py
python make_viot_final_manuscript_plots.py --root outputs_agentic --out final_manuscript_plots --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER_SEEDS = [11, 12, 13, 14, 15]

BASE_SCHEMES = ["MaxWeight", "PPO", "PPO-Lagrangian", "PPO_CMDP", "Intent-PPO-CMDP"]
DQN_SCHEMES = ["SOTA-DQN", "SOTA-DuelingDDQN"]
FINAL_SCHEMES = ["MaxWeight", "SOTA-DQN", "SOTA-DuelingDDQN", "PPO", "PPO-Lagrangian", "PPO_CMDP", "Intent-PPO-CMDP"]

LABELS = {
    "MaxWeight": "MaxWeight",
    "SOTA-DQN": "DQN",
    "SOTA-DuelingDDQN": "Dueling DDQN",
    "PPO": "PPO",
    "PPO-Lagrangian": "PPO-Lagrangian",
    "PPO_CMDP": "PPO-CMDP",
    "Intent-PPO-CMDP": "IA-PPO-CMDP",
    "Intent-PPO-CMDP-FullPolicy-NoShield": "Shield OFF",
}

COLORS = {
    "MaxWeight": "#595959",
    "SOTA-DQN": "#4472C4",
    "SOTA-DuelingDDQN": "#5B9BD5",
    "PPO": "#70AD47",
    "PPO-Lagrangian": "#ED7D31",
    "PPO_CMDP": "#A5A5A5",
    "Intent-PPO-CMDP": "#7030A0",
    "Intent-PPO-CMDP-FullPolicy-NoShield": "#C00000",
}

INTENT_LABEL = {
    "balanced_agentic": "Balanced",
    "reliability_first": "Reliability-first",
    "latency_critical": "Latency-critical",
}
SCENARIO_LABEL = {
    "nominal": "Nominal",
    "congestion_burst": "Congestion",
    "mixed_stress": "Mixed stress",
    "emergency_surge": "Emergency surge",
    "admission_pressure": "Admission pressure",
}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs_agentic"))
    p.add_argument("--out", type=Path, default=Path("final_manuscript_plots"))
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--bootstrap", type=int, default=20000)
    return p.parse_args()


def read_seed(root: Path, run: str) -> pd.DataFrame:
    path = root / run / "agentic_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = set(MASTER_SEEDS) - set(df["seed"].unique())
    if missing:
        raise ValueError(f"{run}: missing master seeds {sorted(missing)}")
    return df[df["seed"].isin(MASTER_SEEDS)].copy()


def merge_matched(root: Path, base_run: str, dqn_run: str) -> pd.DataFrame:
    base = read_seed(root, base_run)
    base = base[base["scheme"].isin(BASE_SCHEMES)].copy()
    dqn = read_seed(root, dqn_run)
    dqn = dqn[dqn["scheme"].isin(DQN_SCHEMES)].copy()
    common = sorted(set(base.columns) & set(dqn.columns))
    out = pd.concat([base[common], dqn[common]], ignore_index=True)
    return out[out["scheme"].isin(FINAL_SCHEMES)].copy()


def seed_aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Average all requested contexts within each scheme x independent master seed."""
    return df.groupby(["scheme", "seed"], as_index=False)[metrics].mean()


def bootstrap_mean_ci(x, n_boot=20000, seed=260910):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_by_scheme(seed_df: pd.DataFrame, metric: str, n_boot: int):
    rows = []
    for i, scheme in enumerate(FINAL_SCHEMES):
        vals = seed_df.loc[seed_df.scheme == scheme, metric].to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=260910 + i)
        rows.append({"scheme": scheme, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return pd.DataFrame(rows)


def save(fig, outdir: Path, stem: str, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def errorbar_panel(ax, summary: pd.DataFrame, ylabel: str, title: str, percent=False):
    x = np.arange(len(summary))
    y = summary["mean"].to_numpy(float)
    lo = summary["ci_low"].to_numpy(float)
    hi = summary["ci_high"].to_numpy(float)
    if percent:
        y, lo, hi = 100*y, 100*lo, 100*hi
    yerr = np.vstack([y-lo, hi-y])
    colors = [COLORS[s] for s in summary.scheme]
    ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.55, width=0.72)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in summary.scheme], rotation=36, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)


def fig02_static(df: pd.DataFrame, outdir: Path, tabdir: Path, dpi: int, n_boot: int):
    metrics = ["weighted_violation_mean", "weighted_violation_cvar95", "service_success_rate_mean", "throughput_mean_mbps"]
    sd = seed_aggregate(df, metrics)
    summaries = {m: summarize_by_scheme(sd, m, n_boot) for m in metrics}
    for m, s in summaries.items():
        s.to_csv(tabdir / f"Fig02_{m}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    errorbar_panel(axes[0,0], summaries[metrics[0]], "Weighted violation", "(a) Mean weighted violation")
    errorbar_panel(axes[0,1], summaries[metrics[1]], r"CVaR$_{95}$", r"(b) Episode-level CVaR$_{95}$")
    errorbar_panel(axes[1,0], summaries[metrics[2]], "Service success (%)", "(c) Service-success rate", percent=True)
    errorbar_panel(axes[1,1], summaries[metrics[3]], "Throughput (Mbit/s)", "(d) Mean throughput")
    fig.tight_layout(w_pad=1.1, h_pad=1.4)
    save(fig, outdir, "Fig02_Final_Static_Benchmark", dpi)


def fig03_heatmap(df: pd.DataFrame, outdir: Path, tabdir: Path, dpi: int):
    # context mean is first averaged across the five independent seeds
    g = df.groupby(["scheme", "intent", "scenario"], as_index=False)["weighted_violation_mean"].mean()
    intent_order = ["balanced_agentic", "reliability_first", "latency_critical"]
    scenario_order = ["nominal", "congestion_burst", "mixed_stress"]
    contexts = [(i,s) for i in intent_order for s in scenario_order]
    arr = np.full((len(FINAL_SCHEMES), len(contexts)), np.nan)
    for r, scheme in enumerate(FINAL_SCHEMES):
        for c, (intent, scenario) in enumerate(contexts):
            q = g[(g.scheme==scheme)&(g.intent==intent)&(g.scenario==scenario)]
            if len(q): arr[r,c] = float(q.weighted_violation_mean.iloc[0])

    export = pd.DataFrame(arr, index=[LABELS[s] for s in FINAL_SCHEMES],
                          columns=[f"{INTENT_LABEL[i]} | {SCENARIO_LABEL[s]}" for i,s in contexts])
    export.to_csv(tabdir / "Fig03_Context_WeightedViolation.csv")

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    im = ax.imshow(arr, cmap="YlGnBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(FINAL_SCHEMES)))
    ax.set_yticklabels([LABELS[s] for s in FINAL_SCHEMES])
    ax.set_xticks(np.arange(len(contexts)))
    ax.set_xticklabels([f"{INTENT_LABEL[i]}\n{SCENARIO_LABEL[s]}" for i,s in contexts], rotation=35, ha="right")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            val = arr[r,c]
            if np.isfinite(val):
                # contrast text according to normalized cell intensity
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("Weighted violation (lower is better)")
    ax.set_title("Context-wise weighted violation across matched static evaluation")
    fig.tight_layout()
    save(fig, outdir, "Fig03_Final_Context_Heatmap", dpi)


def switch_summary(root: Path, base_run: str, dqn_run: str, n_boot: int):
    df = merge_matched(root, base_run, dqn_run)
    metrics = ["post_switch_violation_mean", "intent_mode_shift_l1", "switch_adaptation_time_steps"]
    sd = seed_aggregate(df, metrics)
    return sd, {m: summarize_by_scheme(sd, m, n_boot) for m in metrics}


def fig04_switch(root: Path, outdir: Path, tabdir: Path, dpi: int, n_boot: int):
    rel_sd, rel = switch_summary(root, "v44ec300_final_reliability", "sota_dqn_final_reliability", n_boot)
    lat_sd, lat = switch_summary(root, "v44ec300_final_latency", "sota_dqn_final_latency", n_boot)
    metrics = ["post_switch_violation_mean", "intent_mode_shift_l1", "switch_adaptation_time_steps"]
    for m in metrics:
        rr = rel[m].copy(); rr["switch"] = "Balanced to reliability-first"
        ll = lat[m].copy(); ll["switch"] = "Balanced to latency-critical"
        pd.concat([rr,ll], ignore_index=True).to_csv(tabdir / f"Fig04_{m}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.35))
    x = np.arange(len(FINAL_SCHEMES)); w = 0.36
    panel_defs = [
        ("post_switch_violation_mean", "Post-switch weighted violation", "(a) Post-switch service risk"),
        ("intent_mode_shift_l1", r"Mode-shift $L_1$", "(b) Scheduling response magnitude"),
        ("switch_adaptation_time_steps", "Adaptation time (steps)", "(c) Adaptation time"),
    ]
    for ax, (m, ylabel, title) in zip(axes, panel_defs):
        r = rel[m].set_index("scheme").reindex(FINAL_SCHEMES)
        l = lat[m].set_index("scheme").reindex(FINAL_SCHEMES)
        ry, ly = r["mean"].to_numpy(), l["mean"].to_numpy()
        re = np.vstack([ry-r.ci_low.to_numpy(), r.ci_high.to_numpy()-ry])
        le = np.vstack([ly-l.ci_low.to_numpy(), l.ci_high.to_numpy()-ly])
        axes_colors = [COLORS[s] for s in FINAL_SCHEMES]
        # Same scheme colors; switch type distinguished by hatch/alpha.
        b1 = ax.bar(x-w/2, ry, w, color=axes_colors, edgecolor="black", linewidth=.45, alpha=.72, label="Reliability-first")
        b2 = ax.bar(x+w/2, ly, w, color=axes_colors, edgecolor="black", linewidth=.45, alpha=1.0, hatch="//", label="Latency-critical")
        ax.errorbar(x-w/2, ry, yerr=re, fmt="none", ecolor="black", elinewidth=.7, capsize=2)
        ax.errorbar(x+w/2, ly, yerr=le, fmt="none", ecolor="black", elinewidth=.7, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in FINAL_SCHEMES], rotation=39, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.1)
    save(fig, outdir, "Fig04_Final_Intent_Switching", dpi)


def paired_metric_summary(df: pd.DataFrame, scheme: str, metrics: list[str], n_boot: int):
    sdf = df[df.scheme==scheme].groupby("seed", as_index=False)[metrics].mean()
    out = []
    for j,m in enumerate(metrics):
        mean, lo, hi = bootstrap_mean_ci(sdf[m], n_boot=n_boot, seed=260930+j)
        out.append({"metric":m, "mean":mean, "ci_low":lo, "ci_high":hi})
    return sdf, pd.DataFrame(out)


def fig05_shield(root: Path, outdir: Path, tabdir: Path, dpi: int, n_boot: int):
    df = read_seed(root, "td4_final_admission")
    on = "Intent-PPO-CMDP"; off = "Intent-PPO-CMDP-FullPolicy-NoShield"
    metrics = ["blocking_mean", "service_success_rate_mean", "weighted_violation_mean", "weighted_violation_cvar95"]
    on_seed, on_sum = paired_metric_summary(df, on, metrics, n_boot)
    off_seed, off_sum = paired_metric_summary(df, off, metrics, n_boot)
    on_sum["condition"]="Shield ON"; off_sum["condition"]="Shield OFF"
    pd.concat([on_sum,off_sum], ignore_index=True).to_csv(tabdir / "Fig05_Causal_Shield_Summary.csv", index=False)
    paired = on_seed[["seed"]].copy()
    for m in metrics:
        paired[f"{m}_on"] = on_seed[m].to_numpy()
        paired[f"{m}_off"] = off_seed[m].to_numpy()
        paired[f"{m}_on_minus_off"] = paired[f"{m}_on"] - paired[f"{m}_off"]
    paired.to_csv(tabdir / "Fig05_Causal_Shield_PairedSeeds.csv", index=False)

    fig, axes = plt.subplots(2,2,figsize=(9.2,6.6))
    titles = ["(a) Blocking rate", "(b) Service-success rate", "(c) Weighted violation", r"(d) CVaR$_{95}$"]
    ylabels = ["Blocking (%)","Service success (%)","Weighted violation",r"CVaR$_{95}$"]
    for ax,m,title,ylabel in zip(axes.ravel(),metrics,titles,ylabels):
        a = on_sum[on_sum.metric==m].iloc[0]; b = off_sum[off_sum.metric==m].iloc[0]
        y = np.array([a["mean"],b["mean"]],float); lo=np.array([a["ci_low"],b["ci_low"]],float); hi=np.array([a["ci_high"],b["ci_high"]],float)
        if m in ["blocking_mean","service_success_rate_mean"]:
            y*=100; lo*=100; hi*=100
        err=np.vstack([y-lo,hi-y])
        ax.bar([0,1],y,color=[COLORS[on],COLORS[off]],edgecolor="black",linewidth=.6,width=.62)
        ax.errorbar([0,1],y,yerr=err,fmt="none",ecolor="black",capsize=3,elinewidth=.8)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Shield ON","Shield OFF"])
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y",alpha=.22); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2,h_pad=1.3)
    save(fig,outdir,"Fig05_Final_Causal_Shield",dpi)


def fig06_drain(root: Path, outdir: Path, tabdir: Path, dpi: int):
    primary = read_seed(root, "v44ec300_final_static")
    drain = read_seed(root, "td4_final_static")
    metrics = ["unfinished_rate_mean", "service_success_rate_mean", "deadline_exposure_mean"]
    primary = primary[primary.scheme=="Intent-PPO-CMDP"].groupby("seed",as_index=False)[metrics].mean().sort_values("seed")
    drain = drain[drain.scheme=="Intent-PPO-CMDP"].groupby("seed",as_index=False)[metrics].mean().sort_values("seed")
    if primary.seed.tolist()!=drain.seed.tolist(): raise ValueError("Terminal-drain seed pairing mismatch")
    export = pd.DataFrame({"seed":primary.seed})
    for m in metrics:
        export[f"{m}_primary300"] = primary[m].to_numpy()
        export[f"{m}_terminal_complete"] = drain[m].to_numpy()
    export.to_csv(tabdir / "Fig06_Terminal_Drain_PairedSeeds.csv",index=False)

    fig, axes = plt.subplots(1,3,figsize=(10.8,3.8))
    titles = ["(a) Unfinished requests", "(b) Service-success rate", "(c) Deadline exposure"]
    for ax,m,title in zip(axes,metrics,titles):
        a=100*primary[m].to_numpy(); b=100*drain[m].to_numpy()
        for i,seed in enumerate(primary.seed):
            ax.plot([0,1],[a[i],b[i]],marker="o",linewidth=1.1,alpha=.65)
        ax.plot([0,1],[a.mean(),b.mean()],marker="D",markersize=7,linewidth=2.5,color="black",label="Mean")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Primary\n300-step","Terminal\ncomplete"])
        ax.set_ylabel("Rate (%)"); ax.set_title(title); ax.grid(axis="y",alpha=.22); ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.25)
    save(fig,outdir,"Fig06_Final_Terminal_Drain",dpi)


def main():
    args=parse_args(); set_style()
    root=args.root; out=args.out; figdir=out/"figures"; tabdir=out/"plot_data"
    figdir.mkdir(parents=True,exist_ok=True); tabdir.mkdir(parents=True,exist_ok=True)

    static=merge_matched(root,"v44ec300_final_static","sota_dqn_final_static")
    fig02_static(static,figdir,tabdir,args.dpi,args.bootstrap)
    fig03_heatmap(static,figdir,tabdir,args.dpi)
    fig04_switch(root,figdir,tabdir,args.dpi,args.bootstrap)
    fig05_shield(root,figdir,tabdir,args.dpi,args.bootstrap)
    fig06_drain(root,figdir,tabdir,args.dpi)

    manifest={
        "independent_unit":"master seed",
        "master_seeds":MASTER_SEEDS,
        "static_runs":["v44ec300_final_static","sota_dqn_final_static"],
        "reliability_switch_runs":["v44ec300_final_reliability","sota_dqn_final_reliability"],
        "latency_switch_runs":["v44ec300_final_latency","sota_dqn_final_latency"],
        "causal_shield_run":"td4_final_admission",
        "terminal_drain_pair":["v44ec300_final_static","td4_final_static"],
        "bootstrap_resamples":args.bootstrap,
        "note":"Static/switch figures merge the frozen V4.4E-Causal300 controllers with the matched SOTA DQN extension. CI bars are seed-cluster percentile bootstrap intervals after context averaging within seed."
    }
    (out/"PLOT_GENERATION_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Generated final manuscript plots in {out.resolve()}")

if __name__=="__main__": main()
