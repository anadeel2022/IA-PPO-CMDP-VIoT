# hiot_utils.py — unified metrics + helpers + plotting utilities (ROBUST STANDARDIZATION)
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
import matplotlib.ticker as mticker

_DEF_FIGSIZE = (7, 6)
_FONTS = dict(title=20, label=16, tick=12, legend=12)

# If _FONTS doesn’t exist yet, provide safe defaults
try:
    _FONTS
except NameError:
    _FONTS = dict(title=20, label=16, tick=12, legend=12)

def _prep_ax(ax, title, xlabel, ylabel):
    """Shared axes styling used by legacy plots."""
    ax.set_title(title, fontsize=_FONTS["title"])
    ax.set_xlabel(xlabel, fontsize=_FONTS["label"])
    ax.set_ylabel(ylabel, fontsize=_FONTS["label"])
    ax.tick_params(axis="both", labelsize=_FONTS["tick"])
    ax.grid(alpha=0.3)

EPS = 1e-12

# ---- Metrics hygiene toggles ----
MIN_SUPPORT_FOR_RATE = 5
RATE_SMOOTH_ALPHA = 0.5
DROP_ZERO_THROUGHPUT_WITH_LOW_SUPPORT = False

# ---------- unified metric keys ----------
@dataclass(frozen=True)
class MetricKeys:
    BITS_TX = "bits_tx_ep"          # bits (per episode)
    THR_Mbps = "throughput_mbps"    # Mbps (per episode)
    DELAY_ms = "avg_delay_ms"       # ms (per episode)
    BLOCK = "blocking_prob"         # [0,1] (NaN if arrivals==0)
    INTR = "interrupt_prob"         # [0,1] (NaN if scheduled==0)
    ENERGY_J = "energy_j"           # Joule (per episode)
    EE_BPJ = "energy_eff_bpj"       # bits/J (NaN if energy==0)
    FAIR = "jain_fairness"          # [0,1] across classes (NaN if no served bits)
    UTIL = "utilization"            # [0,1] (NaN if capacity undefined)
    QMAX = "q_value_max"            # learner-specific (optional)

def safe_div(n, d, eps=EPS):
    d = d if abs(d) > eps else np.nan
    return n / d

def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    if s2 <= EPS:
        return np.nan
    return (s1 * s1) / (len(x) * s2)

# ---------- episode accumulator ----------
def new_episode_acc(n_classes: int = 3):
    return {
        "bits": 0.0,
        "delays_ms": [],                 # list of floats
        "arrivals": 0,
        "blocked": 0,
        "preempted": 0,
        "interrupted": 0,
        "scheduled": 0,
        "slots_used": 0,
        "energy_j": 0.0,
        "served_bits_per_class": np.zeros(n_classes, dtype=float),
        "completed_requests": 0,
        "deadline_exposed_new": 0,
    }

def accumulate_from_info(acc: dict, info: dict, n_classes: int = 3):
    if not info:
        return

    def _to_float(x, scale=1.0):
        try:
            if x is None:
                return None
            v = float(x) * scale
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            return None

    def _to_float_list(x, scale=1.0):
        if x is None:
            return []
        if isinstance(x, (list, tuple, np.ndarray)):
            out = []
            for xi in x:
                v = _to_float(xi, scale)
                if v is not None:
                    out.append(v)
            return out
        v = _to_float(x, scale)
        return [v] if v is not None else []

    # Bits delivered (common aliases)
    for k in ("delivered_bits", "bits_delivered", "bits_tx"):
        if k in info:
            v = _to_float(info.get(k, 0.0), 1.0)
            if v is not None:
                acc["bits"] += v
            break

    # Delays (normalize to ms; accept many aliases)
    delays_ms = []
    if isinstance(info.get("delays_ms"), (list, tuple, np.ndarray)):
        delays_ms.extend(_to_float_list(info.get("delays_ms"), 1.0))
    for k in ("delay_ms", "latency_ms", "pkt_delay_ms"):
        if k in info:
            delays_ms.extend(_to_float_list(info.get(k), 1.0))
    for k in ("delay_s", "latency_s"):
        if k in info:
            delays_ms.extend(_to_float_list(info.get(k), 1e3))
    for k in ("delay", "latency"):        # unit inference
        if k in info:
            vals = _to_float_list(info.get(k), 1.0)
            if vals:
                m = float(np.nanmean(vals))
                if m <= 1.0:
                    vals = [v * 1e3 for v in vals]
            delays_ms.extend(vals)
    if delays_ms:
        acc["delays_ms"].extend(delays_ms)

    # Arrivals / blocking / preemptions
    acc["arrivals"] += int(info.get("arrivals", 0) or 0)
    acc["blocked"]  += int(info.get("blocked", info.get("denies", info.get("deny", 0))) or 0)
    acc["preempted"] += int(info.get("preempted", info.get("preempt", 0)) or 0)
    acc["interrupted"] += int(info.get("interrupted", info.get("interrupt_event", 0)) or 0)
    acc["completed_requests"] += int(info.get("completed_requests", 0) or 0)
    acc["deadline_exposed_new"] += int(info.get("deadline_exposed_new", 0) or 0)

    # Scheduling / slots / energy
    acc["scheduled"]  += int(info.get("scheduled", 0) or 0)
    acc["slots_used"] += int(info.get("slots_used", 0) or 0)

    if "energy_j" in info:
        v = _to_float(info.get("energy_j"), 1.0)
        if v is not None:
            acc["energy_j"] += v
    elif "energy_mj" in info:
        v = _to_float(info.get("energy_mj"), 1e-3)  # mJ -> J
        if v is not None:
            acc["energy_j"] += v

    # Per-class served bits; keep vector length and zero-fill NaNs
    cls_bits = info.get("served_bits_per_class", None)
    if cls_bits is not None:
        v = np.array(cls_bits, dtype=float).ravel()
        v[~np.isfinite(v)] = 0.0
        if v.size != acc["served_bits_per_class"].size:
            w = np.zeros_like(acc["served_bits_per_class"])
            m = min(v.size, w.size); w[:m] = v[:m]; v = w
        acc["served_bits_per_class"] += v

# ---------- finalize episode metrics ----------
def finalize_episode_metrics(env,
                             steps_per_ep: int,
                             num_channels: int = None,
                             slot_s: float = None,
                             ep_acc: dict | None = None):
    """Compute per-episode metrics with robust handling of undefined cases."""
    try:
        from phy_layer import SLOT_DURATION_S as SLOT_DEFAULT
    except Exception:
        SLOT_DEFAULT = 0.001

    slot_s = float(slot_s if slot_s is not None else getattr(env, "slot_s", SLOT_DEFAULT))
    num_channels = int(num_channels if num_channels is not None
                       else getattr(env, "n_channels", getattr(env, "num_channels", 1)))
    rate_bps = float(getattr(env, "channel_rate_bps",
                     getattr(env, "rate_bps",
                     getattr(env, "CHANNEL_RATE_BPS", 0.0))))

    def _get(name, default):
        if hasattr(env, name):
            return getattr(env, name)
        if ep_acc is not None:
            m = {
                "bits_tx_ep": "bits",
                "energy_j_ep": "energy_j",
                "delay_samples_ms_ep": "delays_ms",
                "total_arrivals_ep": "arrivals",
                "blocked_arrivals_ep": "blocked",
                "preemptions_ep": "preempted",
                "interrupt_events": "interrupted",
                "scheduled_tx_ep": "scheduled",
                "slots_used_ep": "slots_used",
                "served_bits_per_class_ep": "served_bits_per_class",
                "completed_requests_ep": "completed_requests",
                "deadline_exposed_requests_ep": "deadline_exposed_new",
            }
            key = m.get(name, None)
            if key is not None and key in ep_acc:
                return ep_acc[key]
        return default

    bits = float(_get("bits_tx_ep", 0.0))
    energy = float(_get("energy_j_ep", 0.0))
    delays = np.asarray(_get("delay_samples_ms_ep", []), dtype=float)
    if delays.size:
        delays = delays[np.isfinite(delays) & (delays >= 0)]

    total_arr = int(_get("total_arrivals_ep", 0))
    blocked = int(_get("blocked_arrivals_ep", 0))
    preempts = int(_get("preemptions_ep", 0))
    # Actual interruption is a scheduled backlog transmission with no service.
    # Successful preemption is a scheduling action, not a reliability failure.
    interrupted_events = int(_get("interrupt_events", 0))
    scheduled = int(_get("scheduled_tx_ep", 0))
    slots_used = int(_get("slots_used_ep", 0))
    served_bits_per_class = np.asarray(_get("served_bits_per_class_ep", []), dtype=float).ravel()
    completed_requests = int(_get("completed_requests_ep", 0))
    deadline_exposed_requests = int(_get("deadline_exposed_requests_ep", 0))
    try:
        queues = getattr(env, "request_queues", [])
        pending_requests = int(sum(len(q) for q in queues))
        now_step = int(getattr(env, "t", steps_per_ep))
        right_censored_requests = int(sum(
            1 for q in queues for req in q
            if bool(getattr(req, "active", True))
            and (int(getattr(req, "arrival_step", 0)) + int(getattr(req, "deadline_steps", 0)) > now_step)
        ))
        pending_after_deadline_requests = int(max(pending_requests - right_censored_requests, 0))
    except Exception:
        pending_requests = max(0, total_arr - blocked - completed_requests)
        right_censored_requests = pending_requests
        pending_after_deadline_requests = 0

    # Throughput (Mbps)
    thr_mbps = safe_div(bits, steps_per_ep * slot_s) / 1e6
    if DROP_ZERO_THROUGHPUT_WITH_LOW_SUPPORT and (scheduled < MIN_SUPPORT_FOR_RATE or bits <= EPS):
        thr_mbps = float('nan')

    # Delay: prefer per-packet delays; if none, fall back to step-averaged queue age (steps → ms)
    had_traffic = (bits > 0.0) or (scheduled > 0) or (total_arr > 0)
    if delays.size > 0:
        avg_delay_ms = float(np.nanmean(delays))
    else:
        try:
            qs = getattr(env, "delay_steps_ep", [])
            if qs and np.any(np.isfinite(qs)):
                avg_age_steps = float(np.nanmean([q for q in qs if np.isfinite(q)]))
                avg_delay_ms = float(avg_age_steps * slot_s * 1e3)
            else:
                # conservative: 0 for traffic/no-samples so plots stay populated
                avg_delay_ms = 0.0 if had_traffic else 0.0
        except Exception:
            avg_delay_ms = 0.0 if had_traffic else 0.0

    # Exact request-level blocking and scheduled-attempt interruption rates.
    blocking = float(blocked / total_arr) if total_arr > 0 else np.nan
    intr = float(interrupted_events / scheduled) if scheduled > 0 else 0.0
    service_success_rate = float(completed_requests / total_arr) if total_arr > 0 else np.nan
    deadline_exposure_rate = float(deadline_exposed_requests / total_arr) if total_arr > 0 else np.nan
    unfinished_rate = float(pending_requests / total_arr) if total_arr > 0 else np.nan
    right_censored_rate = float(right_censored_requests / total_arr) if total_arr > 0 else np.nan
    pending_after_deadline_rate = float(pending_after_deadline_requests / total_arr) if total_arr > 0 else np.nan
    request_conservation_error = int(total_arr - completed_requests - blocked - pending_requests)

    # Energy efficiency: undefined when energy is zero or missing
    ee_bpj = (bits / energy) if energy > EPS else np.nan

    # Fairness: prefer the per-device delivered-service vector used in the paper.
    served_bits_per_device = np.asarray(getattr(env, "served_bits_per_device_ep", []), dtype=float).ravel()
    if served_bits_per_device.size and np.sum(served_bits_per_device) > EPS:
        fairness = jain_index(served_bits_per_device)
    else:
        fairness = jain_index(served_bits_per_class) if served_bits_per_class.size and np.sum(served_bits_per_class) > EPS else 0.0

    # Capacity-based utilization with safe fallback
    if rate_bps > 0.0:
        capacity_bits = num_channels * steps_per_ep * slot_s * rate_bps
        util = float(np.clip(safe_div(bits, capacity_bits), 0.0, 1.0)) if capacity_bits > 0 else np.nan
    else:
        denom = num_channels * steps_per_ep
        util = float(np.clip(safe_div(slots_used, denom), 0.0, 1.0)) if denom > 0 else np.nan
    # Helper counters for runner diagnostics
    attempts_on_backlog_ep = int(blocked + scheduled)  # transmit attempts + explicit denies
    denies_on_backlog_ep = int(blocked)

    return {
        MetricKeys.BITS_TX: bits,
        # helper counters for runner diagnostics (attempts/denies on backlog)
        # We approximate: attempts_on_backlog = blocked + scheduled
        # (i.e., every scheduled tx or denial happens only when backlog exists)
        'attempts_on_backlog_ep': attempts_on_backlog_ep,
        'denies_on_backlog_ep':   denies_on_backlog_ep,

        
        MetricKeys.THR_Mbps: thr_mbps,
        MetricKeys.DELAY_ms: avg_delay_ms,
        MetricKeys.BLOCK: blocking,
        MetricKeys.INTR: intr,
        MetricKeys.ENERGY_J: energy,
        MetricKeys.EE_BPJ: ee_bpj,
        MetricKeys.FAIR: fairness,
        MetricKeys.UTIL: util,
        "service_success_rate": service_success_rate,
        "deadline_exposure_rate": deadline_exposure_rate,
        "unfinished_rate": unfinished_rate,
        "right_censored_rate": right_censored_rate,
        "pending_after_deadline_rate": pending_after_deadline_rate,
        "right_censored_requests_ep": right_censored_requests,
        "pending_after_deadline_requests_ep": pending_after_deadline_requests,
        "request_conservation_error": request_conservation_error,
        "completed_requests_ep": completed_requests,
        "deadline_exposed_requests_ep": deadline_exposed_requests,
        "pending_requests_ep": pending_requests,

        # add raw counters so plots can fall back to totals
        "total_arrivals_ep": total_arr,
        "blocked_arrivals_ep": blocked,
        "preemptions_ep": preempts,
        "interrupt_events_ep": interrupted_events,
        "scheduled_tx_ep": scheduled,
        "slots_used_ep": slots_used,
    }


# ---------- CSV saver ----------
def save_metrics_csv(path: str, metrics: list[dict]):
    """
    Robust CSV writer:
      • Uses the union of keys across all episode dicts for the header.
      • Fills missing keys per row with 0.0 (or "" for non-numeric if needed).
      • Converts NaNs to empty strings to avoid "nan" text in plots.
    """
    if not metrics:
        return

    # 1) union of keys across all episodes
    all_keys = set()
    for row in metrics:
        if isinstance(row, dict):
            all_keys |= set(row.keys())
    cols = list(all_keys)

    # (Optional) stable ordering – put the common standardized keys first
    preferred = [
        "throughput_mbps", "avg_delay_ms", "energy_eff_bpj", "jain_fairness",
        "blocking_prob", "interrupt_prob", "utilization",
        "bits_tx_ep", "energy_j",
        "total_arrivals_ep", "blocked_arrivals_ep",
        "scheduled_tx_ep", "preemptions_ep", "slots_used_ep",
        "attempts_on_backlog_ep", "denies_on_backlog_ep",
        "q_value_max"
    ]
    ordered = [k for k in preferred if k in all_keys] + [k for k in sorted(all_keys) if k not in preferred]

    def _clean_row(row: dict) -> dict:
        out = {}
        for k in ordered:
            v = row.get(k, 0.0)
            # turn NaN into "" for CSV friendliness, keep numerics otherwise
            try:
                if isinstance(v, float) and not np.isfinite(v):
                    out[k] = ""
                else:
                    out[k] = v
            except Exception:
                out[k] = v if v is not None else ""
        return out

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in metrics:
            if not isinstance(row, dict):
                continue
            w.writerow(_clean_row(row))


# ============================ plotting utilities =============================



SCHEME_COLORS = {
    "Q-Learning":      "#1f77b4",
    "DoubleQ":         "#ff7f0e",
    "DoubleQ_CMDP":    "#ffbb78",  # lighter orange for CMDP variant
    "DQN":             "#2ca02c",
    "DQN_CMDP":        "#98df8a",  # lighter green
    "DuelingDQN":      "#d62728",
    "DuelingDQN_CMDP": "#ff9896",  # lighter red
    "ActorCritic":     "#9467bd",
    "PPO":             "#8c564b",
}


def _ts(): return time.strftime("%Y%m%d_%H%M%S")
def _ensure_outdir(outdir) -> Path:
    p = Path(outdir) if outdir else Path("./outputs")
    p.mkdir(parents=True, exist_ok=True); return p

# Map legacy plot series -> standardized episode metric keys
_STD_SYNONYMS = {
    "delay_per_episode":      MetricKeys.DELAY_ms,
    "energy_efficiency_per_episode": MetricKeys.EE_BPJ,
    "fairness_per_episode":   MetricKeys.FAIR,
    "throughput_mbps":        MetricKeys.THR_Mbps,
}

def _from_std(m: dict, std_key: str) -> np.ndarray:
    eps = m.get("episode_metrics_std", []) or []
    if not eps:
        return np.array([], dtype=float)
    arr = np.array([row.get(std_key, np.nan) for row in eps], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr

# -------- CDF / mean-bar (prefer standardized metrics) --------
def plot_cdf(results, series_key, xlabel, outdir=None, figsize=_DEF_FIGSIZE, lw=2.0):
    """
    Plot ECDFs. If data are highly quantized (few unique values), fall back to a
    quantile-interpolated CDF to avoid big vertical/horizontal “steps”.
    """
    out = _ensure_outdir(outdir);
    ts = _ts()

    # 1) Load all series first so we can compute pooled percentiles
    series = []
    names = []
    for name, m in results:
        s = np.asarray(m.get(series_key, []), dtype=float)
        if s.size == 0 or not np.any(np.isfinite(s)):
            std_key = _STD_SYNONYMS.get(series_key, None)
            if std_key is not None:
                s = _from_std(m, std_key)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        series.append(s);
        names.append(name)

    if not series:
        return

    pooled = np.concatenate(series)
    fig, ax = plt.subplots(figsize=figsize)

    # 2) Plot each CDF (quantile-smoothed if heavy ties)
    for name, s in zip(names, series):
        uniq = np.unique(s)
        tie_ratio = (uniq.size / s.size) if s.size > 0 else 1.0
        color = SCHEME_COLORS.get(name)

        if tie_ratio < 0.25:
            q = np.linspace(0.0, 1.0, 201)
            xq = np.quantile(s, q)
            ax.plot(xq, q, label=name, linewidth=lw, color=color)
        else:
            x = np.sort(s)
            y = np.linspace(0.0, 1.0, x.size, endpoint=True)
            ax.step(x, y, where="post", label=name, linewidth=lw, color=color)

    # # 3) Auto-focus on the middle 90% to reveal differences
    # if pooled.size >= 40:  # need enough samples
    #     p5, p95 = np.nanpercentile(pooled, [5, 95])
    #     if np.isfinite(p5) and np.isfinite(p95) and p95 > p5:
    #         ax.set_xlim(p5, p95)

    ax.set_title(f"CDF of {xlabel}", fontsize=_FONTS["title"])
    ax.set_xlabel(xlabel, fontsize=_FONTS["label"])
    ax.set_ylabel("CDF", fontsize=_FONTS["label"])
    ax.tick_params(axis="both", labelsize=_FONTS["tick"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=_FONTS["legend"])
    fig.tight_layout()
    fig.savefig(out / f"CDF_{series_key.replace('_', '-')}_{ts}.png", dpi=300)
    plt.close(fig)


def plot_mean_bar(results, series_key, ylabel, outdir=None, figsize=_DEF_FIGSIZE):
    out = _ensure_outdir(outdir); ts = _ts()
    labels, means, colors = [], [], []
    for name, m in results:
        std_key = _STD_SYNONYMS.get(series_key, None)
        s = _from_std(m, std_key) if std_key is not None else np.array([])
        if s.size == 0 or not np.any(np.isfinite(s)):
            s = np.asarray(m.get(series_key, []), dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        labels.append(name); means.append(float(np.nanmean(s))); colors.append(SCHEME_COLORS.get(name, None))
    if not labels:
        return
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=figsize)
    if all(c is not None for c in colors): ax.bar(x, means, color=colors)
    else: ax.bar(x, means)
    # --- Auto-zoom when bars are close (no numeric labels) ---
    ymax = float(max(means)) if means else 0.0
    ymin = float(min(means)) if means else 0.0
    span = ymax - ymin
    if span <= 0.15 * (abs(ymax) + 1e-9):  # bars within ~15% → zoom and pad
        pad = (0.20 * span) if span > 0 else max(0.02 * (abs(ymax) + 1e-9), 0.1)
        ax.set_ylim(ymin - 0.10 * pad, ymax + 0.90 * pad)

    ax.set_title(f"Mean {ylabel}", fontsize=_FONTS["title"])
    ax.set_xlabel("Scheme", fontsize=_FONTS["label"])
    ax.set_ylabel(ylabel, fontsize=_FONTS["label"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=_FONTS["tick"])
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"mean_bar_{series_key.replace('_','-')}_{ts}.png", dpi=300); plt.close(fig)

# -------- Throughput in Mbps (axes formatted as integers) --------
def plot_throughput_mbps_cdf(results, outdir, figsize=_DEF_FIGSIZE):
    out = _ensure_outdir(outdir); ts = _ts()
    fig, ax = plt.subplots(figsize=figsize)

    xmax_seen = 0.0
    for name, m in results:
        s = _from_std(m, MetricKeys.THR_Mbps)
        if s.size == 0:
            s = np.asarray(m.get("throughput_per_episode", []), float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        x = np.sort(s); y = np.linspace(0, 1, x.size, endpoint=True)
        ax.step(x, y, where="post", label=name, linewidth=2.0, color=SCHEME_COLORS.get(name))
        if x.size:
            xmax_seen = max(xmax_seen, float(x[-1]))

    ax.set_title("CDF of Throughput (Mbps)", fontsize=_FONTS["title"])
    ax.set_xlabel("Throughput (Mbps)", fontsize=_FONTS["label"])
    ax.set_ylabel("CDF", fontsize=_FONTS["label"])
    ax.grid(alpha=0.3); ax.legend(fontsize=_FONTS["legend"])

    # >>> smart tick formatting (no more all-zero labels)
    import matplotlib.ticker as mticker
    if xmax_seen < 1:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    elif xmax_seen < 10:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.tight_layout(); fig.savefig(out / f"CDF_throughput_mbps_{ts}.png", dpi=300); plt.close(fig)

def plot_throughput_mbps_bar(results, outdir, figsize=_DEF_FIGSIZE):
    out = _ensure_outdir(outdir); ts = _ts()
    labels, means, colors = [], [], []

    for name, m in results:
        s = _from_std(m, MetricKeys.THR_Mbps)
        if s.size == 0:
            s = np.asarray(m.get("throughput_per_episode", []), float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        labels.append(name)
        means.append(float(np.nanmean(s)))
        # fall back to default color if scheme color missing
        colors.append(SCHEME_COLORS.get(name, None))

    if not labels:
        return

    ymax_seen = max(means) if means else 0.0
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)

    # if any None in colors, let matplotlib cycle colors
    if any(c is None for c in colors):
        ax.bar(x, means)  # default cycle
    else:
        ax.bar(x, means, color=colors)

    # Auto-zoom like other bars
    ymax = float(max(means)) if means else 0.0
    ymin = float(min(means)) if means else 0.0
    span = ymax - ymin
    if span <= 0.15 * (abs(ymax) + 1e-9):
        pad = (0.20 * span) if span > 0 else max(0.02 * (abs(ymax)+1e-9), 0.1)
        ax.set_ylim(ymin - 0.10*pad, ymax + 0.90*pad)
    else:
        # modest headroom if not zooming
        ax.set_ylim(0.0, 1.10 * ymax_seen if ymax_seen > 0 else 1.0)

    ax.set_title("Mean Throughput (Mbps)", fontsize=_FONTS["title"])
    ax.set_xlabel("Scheme", fontsize=_FONTS["label"])
    ax.set_ylabel("Throughput (Mbps)", fontsize=_FONTS["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=_FONTS["tick"])
    ax.grid(axis="y", alpha=0.3)

    # Smart tick formatting WITHOUT resetting limits
    import matplotlib.ticker as mticker
    if ymax_seen < 1:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    elif ymax_seen < 10:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(out / f"mean_bar_throughput_mbps_{ts}.png", dpi=300)
    plt.close(fig)

# --- replace your plot_blocking_mean_std in hiot_utils.py with this ---

def plot_blocking_mean_std(results, outdir=None, figsize=_DEF_FIGSIZE):
    out = _ensure_outdir(outdir); ts = _ts()

    labels, means, ylo, yhi, colors = [], [], [], [], []
    for name, m in results:
        # 1) Prefer standardized per-episode metrics
        eps = m.get("episode_metrics_std", []) or []
        vals = np.array(
            [float(row.get("blocking_prob", np.nan)) for row in eps],
            dtype=float
        )
        vals = vals[np.isfinite(vals)]

        # 2) Fallback to legacy schema (denies/arrivals) if needed
        if vals.size == 0:
            den = np.asarray(m.get("denies_per_episode", []), float)
            arr = np.asarray(m.get("arrivals_per_episode", []), float)
            n = min(den.size, arr.size)
            if n > 0:
                vals = np.clip(den[:n] / np.maximum(arr[:n], 1.0), 0.0, 1.0)

        if vals.size == 0:
            continue  # nothing to show for this scheme

        vals = np.clip(vals, 0.0, 1.0)  # probabilities
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))

        # Bound mean±std to [0, 1], then convert to non-negative two-sided yerr
        low  = max(0.0, mu - sd)
        high = min(1.0, mu + sd)
        labels.append(name)
        means.append(mu)
        ylo.append(mu - low)   # non-negative
        yhi.append(high - mu)  # non-negative
        colors.append(SCHEME_COLORS.get(name))

    if not labels:
        print("plot_blocking_mean_std: no finite blocking data found")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    yerr = np.vstack([ylo, yhi])  # shape (2, N) and non-negative
    ax.bar(x, means, yerr=yerr, capsize=4, color=colors)

    _prep_ax(ax, "Mean Blocking Probability ± Std", "Scheme", "Blocking Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=_FONTS["tick"])
    fig.tight_layout()
    fig.savefig(out / f"blocking_prob_mean_std_{ts}.png", dpi=300)
    plt.close(fig)


# -------- Channel utilization --------
def plot_channel_utilization(results, n_channels=None, steps_per_ep=None, outdir=None, figsize=_DEF_FIGSIZE):
    """
    Channel utilization boxplot with sensible axes:
      • Prefer standardized utilization in [0,1].
      • Fall back to legacy throughput/(n_channels*steps) only if it looks like a fraction.
      • Set y-limits from the data with padding so the top whiskers are never clipped.
    """
    out = _ensure_outdir(outdir); ts = _ts()
    labels, data, all_vals = [], [], []

    for name, m in results:
        u = _from_std(m, MetricKeys.UTIL)

        if u.size == 0:
            thr = np.asarray(m.get("throughput_per_episode", []), float)
            if thr.size:
                denom = float(max(1, (n_channels or 1) * (steps_per_ep or 1)))
                u_try = thr / denom
                # Accept the legacy fallback only if it behaves like occupancy
                if np.nanpercentile(u_try, 95) <= 1.5:
                    u = u_try

        u = u[np.isfinite(u)]
        if u.size == 0:
            continue

        labels.append(name)
        data.append(u)
        all_vals.append(u)

    if not data:
        return

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        whis=(0, 100),
        boxprops=dict(edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, name in zip(bp["boxes"], labels):
        patch.set_facecolor(SCHEME_COLORS.get(name))
        patch.set_alpha(0.85)

    # Dynamic y-limits with headroom so upper caps never sit on the frame
    all_concat = np.concatenate(all_vals)
    y_lo = float(np.nanmin(all_concat))
    y_hi = float(np.nanmax(all_concat))
    span = max(y_hi - y_lo, 1e-6)
    pad = 0.10 * span
    ax.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad))

    _prep_ax(ax, "Distribution of Channel Utilization", "Scheme", "Channel Utilization (normalized)")
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=_FONTS["tick"])

    fig.tight_layout()
    fig.savefig(out / f"channel_utilization_{ts}.png", dpi=300)
    plt.close(fig)

def moving_average(x, w: int):
    """Simple trailing moving average with padding so output length == input length."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or w is None or w <= 1:
        return x
    # ignore NaNs in convolution by pre-filling with the first finite value
    if not np.any(np.isfinite(x)):
        return x
    first = x[np.isfinite(x)][0]
    x_fill = np.where(np.isfinite(x), x, first)
    kern = np.ones(int(w), dtype=float) / float(w)
    m = np.convolve(x_fill, kern, mode="valid")
    # left-pad so length matches input
    pad = np.full(w - 1, m[0], dtype=float)
    return np.concatenate([pad, m], axis=0)

def episodes_to_convergence(series,
                            frac: float = 0.95,
                            window: int = 20,
                            band: float = 0.05,
                            streak: int = 5):
    """
    Return the first episode index where the moving-average stays within
    a (1±band) window around 'frac' of the best observed moving-average
    for 'streak' consecutive episodes.

    More forgiving than before:
      • If crossing happens near the very end (fewer than 'streak' points left),
        we treat the LAST episode as converged instead of returning None.
      • Ignores NaNs. If series is empty or entirely NaN, returns None.
    """
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return None
    ma = moving_average(s, window)
    if not np.any(np.isfinite(ma)):
        return None

    # Target is a fraction of the best smoothed value observed so far
    target = frac * np.nanmax(ma)
    lower = target * (1.0 - band)

    run = 0
    last_idx = None
    for i, v in enumerate(ma):
        if np.isfinite(v) and v >= lower:
            run += 1
            last_idx = i
            if run >= streak:
                return i
        else:
            run = 0

    # Fallback: crossed but too close to the end to accumulate 'streak'
    return last_idx  # may be None if we truly never got close

# ============================ new plots used by runner =============================

def _collect_series_by_scheme_across_seeds(all_runs, series_key: str):
    """
    all_runs: list over seeds -> [(name, metrics_dict), ...]
    Returns dict: name -> list[np.ndarray] (one per seed)
    """
    by_name = {}
    for run in all_runs:
        for name, m in run:
            s = np.asarray(m.get(series_key, []), dtype=float)
            if s.size == 0:
                # try standardized mapping if applicable
                std_key = _STD_SYNONYMS.get(series_key, None)
                if std_key is not None:
                    s = _from_std(m, std_key)
            by_name.setdefault(name, []).append(s)
    return by_name

def _stack_with_nan_padding(arr_list):
    """
    Right-pad arrays in arr_list with NaNs to the same length and stack -> [K, T]
    """
    if not arr_list:
        return np.zeros((0,0), dtype=float)
    L = max(a.size for a in arr_list)
    out = np.full((len(arr_list), L), np.nan, dtype=float)
    for i, a in enumerate(arr_list):
        if a.size:
            out[i, :a.size] = a
    return out

def _plot_mavg_lines(ax, lines_dict, window: int, rebase_to_min: bool, ylabel: str):
    xmax = 0
    for name, series_list in lines_dict.items():
        # smooth each seed then aggregate
        smoothed = []
        for s in series_list:
            if s.size == 0:
                smoothed.append(s); continue
            ma = moving_average(s, max(1, int(window)))
            if rebase_to_min and np.any(np.isfinite(ma)):
                first = ma[np.isfinite(ma)][0]
                ma = ma - first

            smoothed.append(ma)
        M = _stack_with_nan_padding(smoothed)  # [seeds, T]
        if M.size == 0:
            continue
        mu = np.nanmean(M, axis=0)
        lo = np.nanpercentile(M, 25, axis=0)
        hi = np.nanpercentile(M, 75, axis=0)
        xs = np.arange(M.shape[1])
        ax.plot(xs, mu, label=name, linewidth=2.0, color=SCHEME_COLORS.get(name))
        ax.fill_between(xs, lo, hi, alpha=0.15, color=SCHEME_COLORS.get(name))
        xmax = max(xmax, xs[-1] if xs.size else 0)
    ax.set_xlabel("Episode", fontsize=_FONTS["label"])
    ax.set_ylabel(ylabel, fontsize=_FONTS["label"])
    ax.grid(alpha=0.3)
    ax.margins(y=0.05)
    ax.tick_params(axis="both", labelsize=_FONTS["tick"])
    ax.legend(fontsize=_FONTS["legend"])
    ax.set_xlim(0, xmax)



def plot_mavg_across_seeds(all_runs, key, ylabel, outdir=None, window=50, ema_alpha=0.20):
    """
    Plot smoothed trends across seeds.

    Input shape expected by the runner:
        all_runs = [  # list over seeds
            [ (scheme_name, metrics_dict), ... ],   # seed 1
            [ (scheme_name, metrics_dict), ... ],   # seed 2
            ...
        ]

    For rewards (key contains 'reward'), we show a *running mean across episodes*
    to produce the desired "rise -> settle" shape. For other series we apply a
    trailing moving average followed by a light EMA. Finally, we subtract the
    minimum of the first 10% of points (per seed) so curves start near a common
    baseline without distorting early dynamics.
    """
    out = _ensure_outdir(outdir); ts = _ts()

    def _ma(x, k):
        x = np.asarray(x, dtype=float)
        if k <= 1 or x.size < k: return x
        w = np.ones(int(k), float) / float(k)
        m = np.convolve(x, w, mode="valid")
        # left-pad to keep length comparable across keys
        pad = np.full(k-1, m[0], dtype=float) if m.size else np.array([], float)
        return np.concatenate([pad, m]) if pad.size else m

    def _ema(x, a):
        x = np.asarray(x, dtype=float)
        if x.size == 0: return x
        y = np.empty_like(x); y[0] = x[0]
        for i in range(1, x.size): y[i] = a*x[i] + (1-a)*y[i-1]
        return y

    use_running = ("reward" in key)

    # Gather series per scheme across seeds
    by_name = {}
    for seed_runs in all_runs:            # seed_runs is list[(name, metrics)]
        for name, m in seed_runs:
            s = np.asarray(m.get(key, []), dtype=float)
            if s.size == 0:
                # allow standardized synonym (used by some drivers)
                std_key = _STD_SYNONYMS.get(key, None)
                if std_key is not None:
                    s = _from_std(m, std_key)
            by_name.setdefault(name, []).append(s)

    if not by_name:
        return

    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE)

    for name, series_list in by_name.items():
        smoothed = []
        for s in series_list:
            if s.size == 0:
                continue

            if use_running:
                c = np.cumsum(s)
                denom = np.arange(1, s.size+1, dtype=float)
                y = c / denom
            else:
                y = _ma(s, int(max(1, window)))

            y = _ema(y, float(ema_alpha))

            # rebase to the minimum of first 10% to emphasize the rise
            if y.size:
                k = max(1, int(0.1 * y.size))
                base = np.nanmin(y[:k])
                y = y - base
            smoothed.append(y)

        if not smoothed:
            continue

        # Align lengths and average across seeds
        L = min((arr.size for arr in smoothed), default=0)
        if L <= 0:
            continue
        stack = np.stack([arr[:L] for arr in smoothed], axis=0)
        mu = np.nanmean(stack, axis=0)

        ax.plot(mu, label=name, linewidth=2.0, color=SCHEME_COLORS.get(name))

    ax.set_title(f"{ylabel} Moving Average", fontsize=_FONTS["title"])
    ax.set_xlabel("Episode", fontsize=_FONTS["label"])
    ax.set_ylabel(ylabel, fontsize=_FONTS["label"])
    ax.grid(alpha=0.3); ax.legend(fontsize=_FONTS["legend"])
    ax.margins(y=0.05)
    fig.tight_layout(); fig.savefig(out / f"{key.replace('_','-')}_ma_{ts}.png", dpi=300); plt.close(fig)


def plot_q_value_convergence_across_seeds(all_runs, window: int = 20, outdir=None):
    """
    Convenience wrapper specialized for 'q_value_max'.
    """
    out = _ensure_outdir(outdir); ts = _ts()
    lines = _collect_series_by_scheme_across_seeds(all_runs, "q_value_max")
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE)
    _plot_mavg_lines(ax, lines, window, rebase_to_min=False, ylabel="Max Q / Value")
    ax.set_title("Q/Value Convergence", fontsize=_FONTS["title"])
    fig.tight_layout(); fig.savefig(out / f"q_value_convergence_{ts}.png", dpi=300); plt.close(fig)

def plot_ip_cdf(results, outdir=None, figsize=_DEF_FIGSIZE, lw=2.0):
    """
    CDF of interruption probability. Prefer standardized 'interrupt_prob' from episode_metrics_std;
    fall back to preempt_counts_per_episode / scheduled if available, otherwise arrivals.
    """
    out = _ensure_outdir(outdir); ts = _ts()
    fig, ax = plt.subplots(figsize=figsize)
    for name, m in results:
        s = _from_std(m, MetricKeys.INTR)
        if s.size == 0:
            # fallback manual compute
            pre = np.asarray(m.get("preempt_counts_per_episode", []), float)
            arr = np.asarray(m.get("arrivals_per_episode", []), float)
            # scheduled is not tracked outside std metrics; arrivals is a loose proxy
            den = np.where(arr > 0, arr, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.clip(pre / den, 0.0, 1.0)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        x = np.sort(s); y = np.linspace(0, 1, x.size, endpoint=True)
        ax.step(x, y, where="post", label=name, linewidth=lw, color=SCHEME_COLORS.get(name))
    ax.set_title("CDF of Interruption Probability", fontsize=_FONTS["title"])
    ax.set_xlabel("Interrupt Probability", fontsize=_FONTS["label"])
    ax.set_ylabel("CDF", fontsize=_FONTS["label"])
    ax.grid(alpha=0.3); ax.legend(fontsize=_FONTS["legend"])
    fig.tight_layout(); fig.savefig(out / f"CDF_interrupt_prob_{ts}.png", dpi=300); plt.close(fig)

def plot_convergence_and_time(conv_mean: dict, time_mean: dict, outdir=None, figsize=_DEF_FIGSIZE):
    """
    Single combined figure:
      • Bars (left y-axis): Episodes to Convergence
      • Dashed line (right y-axis): Training Time (seconds)
    """
    out = _ensure_outdir(outdir); ts = _ts()

    # Order by our canonical palette, keeping only keys we actually have.
    labels = [k for k in SCHEME_COLORS.keys() if (k in conv_mean) or (k in time_mean)]
    if not labels:
        return

    conv_vals = [int(conv_mean.get(k, 0)) for k in labels]
    time_vals = [float(time_mean.get(k, 0.0)) for k in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    # Bars: episodes to convergence
    colors = [SCHEME_COLORS.get(k) for k in labels]
    bars = ax.bar(x, conv_vals, color=colors, zorder=2)

    # Line: training time
    line, = ax2.plot(x, time_vals, linestyle="--", marker="o", linewidth=2.0,
                     color="black", label="Training Time (s)", zorder=3)

    # Axes styling
    ax.set_title("Convergence Speed and Training Time", fontsize=_FONTS["title"])
    ax.set_xlabel("Scheme", fontsize=_FONTS["label"])
    ax.set_ylabel("Episodes to Convergence", fontsize=_FONTS["label"])
    ax2.set_ylabel("Training Time (s)", fontsize=_FONTS["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=_FONTS["tick"])
    ax.tick_params(axis="y", labelsize=_FONTS["tick"])
    ax2.tick_params(axis="y", labelsize=_FONTS["tick"])

    # Nice ranges
    if any(v > 0 for v in conv_vals):
        ax.set_ylim(0, max(conv_vals) * 1.15)
    if any(v > 0 for v in time_vals):
        ax2.set_ylim(0, max(time_vals) * 1.20)

    # Grid under bars
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend for the line (bars are explained by x-tick labels)
    ax2.legend(loc="upper left", fontsize=_FONTS["legend"])

    fig.tight_layout()
    fig.savefig(out / f"convergence_time_{ts}.png", dpi=300)
    plt.close(fig)


def plot_ppo_diagnostics(diag, outdir=None, figsize=(12, 9)):
    out = _ensure_outdir(outdir); ts = _ts()
    color = SCHEME_COLORS.get("PPO", "#8c564b")

    pl = np.asarray(diag.get("ppo_policy_loss", diag.get("policy_loss", [])), float)
    vl = np.asarray(diag.get("ppo_value_loss",  diag.get("value_loss",  [])), float)
    en = np.asarray(diag.get("ppo_entropy",     diag.get("entropy",     [])), float)
    kl = np.asarray(diag.get("ppo_kl",          diag.get("kl",          [])), float)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    (ax1, ax2), (ax3, ax4) = axs

    ax1.plot(pl, linewidth=1.8, color=color); _prep_ax(ax1, "Policy Loss", "Episode", "Loss")
    ax2.plot(vl, linewidth=1.8, color=color); _prep_ax(ax2, "Value Loss",  "Episode", "Loss")
    ax3.plot(en, linewidth=1.8, color=color); _prep_ax(ax3, "Entropy",     "Episode", "Entropy")
    ax4.plot(kl, linewidth=1.8, color=color); _prep_ax(ax4, "KL Divergence","Episode","KL")

    fig.suptitle("PPO Diagnostics", fontsize=_FONTS["title"] + 6)
    fig.savefig(out / f"ppo_diagnostics_{ts}.png", dpi=300); plt.close(fig)

def plot_ppo_diagnostics_multi(diags, outdir=None, figsize=(12, 9)):
    """
    diags: list of dicts, one per seed. Keys: ppo_policy_loss, ppo_value_loss, ppo_entropy, ppo_kl
    Overlays all seeds (light alpha) and draws a thick mean line to avoid the 'repeated' look.
    """
    out = _ensure_outdir(outdir); ts = _ts()
    color = SCHEME_COLORS.get("PPO", "#8c564b")

    keys = ["ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_kl"]
    series = {k: [] for k in keys}
    for d in diags:
        for k in keys:
            arr = np.asarray(d.get(k, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                series[k].append(arr)

    def _overlay(ax, arrs, title, ylabel):
        if not arrs:
            return
        max_len = max(a.size for a in arrs)
        # plot each seed (thin, transparent)
        for a in arrs:
            ax.plot(a, linewidth=1.0, alpha=0.35, color=color)
        # pad with NaN and take nanmean to get a mean curve
        stacked = np.full((len(arrs), max_len), np.nan, dtype=float)
        for i, a in enumerate(arrs):
            stacked[i, :a.size] = a
        mean_curve = np.nanmean(stacked, axis=0)
        ax.plot(mean_curve, linewidth=2.4, color=color)
        _prep_ax(ax, title, "Episode", ylabel)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    (ax1, ax2), (ax3, ax4) = axs

    _overlay(ax1, series["ppo_policy_loss"], "Policy Loss", "Loss")
    _overlay(ax2, series["ppo_value_loss"],  "Value Loss",  "Loss")
    _overlay(ax3, series["ppo_entropy"],     "Entropy",     "Entropy")
    _overlay(ax4, series["ppo_kl"],          "KL Divergence","KL")

    fig.suptitle("PPO Diagnostics", fontsize=_FONTS["title"] + 6)
    fig.savefig(out / f"ppo_diagnostics_{ts}.png", dpi=300); plt.close(fig)
