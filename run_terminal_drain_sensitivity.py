from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
FINAL_SCHEMES = "MaxWeight,PPO,PPO-Lagrangian,PPO_CMDP,Intent-PPO-CMDP-NoRisk,Intent-PPO-CMDP,Intent-PPO-CMDP-FullPolicy-NoShield"


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def cache_check() -> None:
    root = Path("policy_cache_v44")
    if not root.exists():
        raise SystemExit("policy_cache_v44 is missing. Run this patch inside the completed final Causal300 folder.")
    required_source_schemes = ["PPO", "PPO-Lagrangian", "PPO_CMDP", "Intent-PPO-CMDP-NoRisk", "Intent-PPO-CMDP"]
    seeds = [11, 12, 13, 14, 15]
    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        ok = all((d / f"{scheme}__seed{seed}.pt").exists() for scheme in required_source_schemes for seed in seeds)
        if ok:
            candidates.append(d)
    if not candidates:
        raise SystemExit(
            "No V4.4 cache signature contains all final seed 11-15 PPO checkpoints. "
            "Do not rerun training; restore policy_cache_v44 from the completed final folder."
        )
    print("CHECK PASS: final frozen checkpoints found in:")
    for d in candidates:
        print(f"  {d}")


def common(device: str):
    return [
        "--episodes", "600", "--eval_episodes", "200", "--steps", "300", "--training_steps", "150",
        "--terminal_drain_steps", "250", "--terminal_drain_auto_extend", "--terminal_drain_max_steps", "500", "--seeds", "11,12,13,14,15",
        "--eval_action_mode", "stochastic", "--policy_cache_dir", "policy_cache_v44",
        "--require_cached_policy", "--resume", "--output_mode", "analysis", "--no-make_figures", "--device", device,
    ]


def validate_run(run_id: str):
    run([PYTHON, "validate_submission_results.py", str(Path("outputs_agentic") / run_id)])


def main():
    ap = argparse.ArgumentParser(description="Terminal-drain accounting sensitivity V4 with exact 300-step RNG replay for frozen V4.4E-Causal300 final policies.")
    ap.add_argument("--mode", choices=["check", "run", "package"], default="run")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    if args.mode in {"check", "run"}:
        run([PYTHON, "-m", "pytest", "-q"])
        run([PYTHON, "validate_action_semantics.py"])
        cache_check()
        if args.mode == "check":
            print("CHECK PASS: ready for terminal-drain accounting sensitivity V4. No training will be performed.")
            return

    if args.mode == "run":
        # V4 reruns only the four frozen final evaluation families. The feasibility-anchor
        # audit is diagnostic and does not need another expensive terminal replay.

        # Static final contexts.
        run([PYTHON, "viot_agentic_runner.py", "--profile", "final_verification", "--run_id", "td4_final_static",
             *common(args.device), "--intents", "balanced_agentic,reliability_first,latency_critical",
             "--scenarios", "nominal,congestion_burst,mixed_stress", "--schemes", FINAL_SCHEMES, "--scheme_set", "custom"])
        validate_run("td4_final_static")

        # Held-out balanced -> reliability switch.
        run([PYTHON, "viot_agentic_runner.py", "--profile", "intent_switch", "--run_id", "td4_final_reliability",
             *common(args.device), "--intents", "balanced_agentic",
             "--scenarios", "congestion_burst,emergency_surge,mixed_stress", "--schemes", FINAL_SCHEMES, "--scheme_set", "custom",
             "--intent_switch_to", "reliability_first", "--intent_switch_frac", "0.50"])
        validate_run("td4_final_reliability")

        # Held-out balanced -> latency switch.
        run([PYTHON, "viot_agentic_runner.py", "--profile", "intent_switch_latency", "--run_id", "td4_final_latency",
             *common(args.device), "--intents", "balanced_agentic",
             "--scenarios", "congestion_burst,mixed_stress", "--schemes", FINAL_SCHEMES, "--scheme_set", "custom",
             "--intent_switch_to", "latency_critical", "--intent_switch_frac", "0.50"])
        validate_run("td4_final_latency")

        # Same-checkpoint safety-shield stress.
        run([PYTHON, "viot_agentic_runner.py", "--profile", "final_verification", "--run_id", "td4_final_admission",
             *common(args.device), "--intents", "balanced_agentic,reliability_first,latency_critical",
             "--scenarios", "admission_pressure", "--schemes", FINAL_SCHEMES, "--scheme_set", "custom"])
        validate_run("td4_final_admission")

    # package mode can be used after a restart once the four V4 output folders exist.
    run([PYTHON, "terminal_drain_audit.py", "--output-root", "outputs_agentic",
         "--share-dir", "terminal_drain_sensitivity_share_v4", "--zip", "terminal_drain_sensitivity_share_v4.zip"])


if __name__ == "__main__":
    main()
