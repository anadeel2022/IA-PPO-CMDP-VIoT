#!/usr/bin/env python3
"""Publication wrapper for the frozen IA-PPO-CMDP V-IoT manuscript protocol.

This wrapper changes orchestration only. It does not modify the frozen simulator,
policy, intent, reward/compiler, shield, PHY, action-space, or seed definitions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
PPO_SCHEMES = (
    "MaxWeight,PPO,PPO-Lagrangian,PPO_CMDP,"
    "Intent-PPO-CMDP-NoRisk,Intent-PPO-CMDP,"
    "Intent-PPO-CMDP-FullPolicy-NoShield"
)
DQN_SCHEMES = "SOTA-DQN,SOTA-DuelingDDQN"
SEEDS = "11,12,13,14,15"


def call(args: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def validate_primary(run_id: str) -> None:
    call([PYTHON, "validate_submission_results.py", str(ROOT / "outputs_agentic" / run_id)])


def validate_dqn(run_id: str) -> None:
    call([PYTHON, "validate_sota_baseline_results.py", str(ROOT / "outputs_agentic" / run_id)])


def common_ppo(device: str, require_cache: bool = False) -> list[str]:
    out = [
        "--episodes", "600", "--eval_episodes", "200",
        "--steps", "300", "--training_steps", "150",
        "--seeds", SEEDS, "--schemes", PPO_SCHEMES, "--scheme_set", "custom",
        "--eval_action_mode", "stochastic",
        "--policy_cache_dir", "policy_cache_v44",
        "--resume", "--output_mode", "analysis", "--no-make_figures",
        "--device", device,
    ]
    if require_cache:
        out.append("--require_cached_policy")
    return out


def run_ppo(device: str) -> None:
    jobs = [
        ("final_verification", "v44ec300_final_static",
         ["--intents", "balanced_agentic,reliability_first,latency_critical",
          "--scenarios", "nominal,congestion_burst,mixed_stress"], False),
        ("intent_switch", "v44ec300_final_reliability",
         ["--intents", "balanced_agentic",
          "--scenarios", "congestion_burst,emergency_surge,mixed_stress",
          "--intent_switch_to", "reliability_first", "--intent_switch_frac", "0.50"], True),
        ("intent_switch_latency", "v44ec300_final_latency",
         ["--intents", "balanced_agentic",
          "--scenarios", "congestion_burst,mixed_stress",
          "--intent_switch_to", "latency_critical", "--intent_switch_frac", "0.50"], True),
        ("final_verification", "v44ec300_final_admission",
         ["--intents", "balanced_agentic,reliability_first,latency_critical",
          "--scenarios", "admission_pressure"], True),
    ]
    for profile, run_id, extra, require_cache in jobs:
        call([PYTHON, "viot_agentic_runner.py", "--profile", profile, "--run_id", run_id,
              *common_ppo(device, require_cache=require_cache), *extra])
        validate_primary(run_id)
        if "switch" in profile:
            call([PYTHON, "diagnose_switch_outputs.py", str(ROOT / "outputs_agentic" / run_id)])


def common_dqn(device: str, require_cache: bool = False) -> list[str]:
    out = [
        "--episodes", "600", "--eval_episodes", "200",
        "--steps", "300", "--training_steps", "150",
        "--seeds", SEEDS, "--schemes", DQN_SCHEMES, "--scheme_set", "custom",
        "--eval_action_mode", "greedy",
        "--policy_cache_dir", "policy_cache_sota_dqn",
        "--resume", "--output_mode", "analysis", "--no-make_figures",
        "--device", device,
    ]
    if require_cache:
        out.append("--require_cached_policy")
    return out


def run_dqn(device: str) -> None:
    jobs = [
        ("final_verification", "sota_dqn_final_static",
         ["--intents", "balanced_agentic,reliability_first,latency_critical",
          "--scenarios", "nominal,congestion_burst,mixed_stress"], False),
        ("intent_switch", "sota_dqn_final_reliability",
         ["--intents", "balanced_agentic",
          "--scenarios", "congestion_burst,emergency_surge,mixed_stress",
          "--intent_switch_to", "reliability_first", "--intent_switch_frac", "0.50"], True),
        ("intent_switch_latency", "sota_dqn_final_latency",
         ["--intents", "balanced_agentic",
          "--scenarios", "congestion_burst,mixed_stress",
          "--intent_switch_to", "latency_critical", "--intent_switch_frac", "0.50"], True),
        ("final_verification", "sota_dqn_final_admission",
         ["--intents", "balanced_agentic,reliability_first,latency_critical",
          "--scenarios", "admission_pressure"], True),
    ]
    for profile, run_id, extra, require_cache in jobs:
        call([PYTHON, "viot_agentic_runner.py", "--profile", profile, "--run_id", run_id,
              *common_dqn(device, require_cache=require_cache), *extra])
        validate_dqn(run_id)


def run_drain(device: str) -> None:
    call([PYTHON, "run_terminal_drain_sensitivity.py", "--mode", "run", "--device", device])


def run_figures() -> None:
    call([PYTHON, "make_viot_final_manuscript_plots.py",
          "--root", "outputs_agentic", "--out", "reproduced_manuscript_plots",
          "--dpi", "600", "--bootstrap", "20000"])


def check() -> None:
    call([PYTHON, "-m", "pytest", "-q"])
    call([PYTHON, "validate_action_semantics.py"])
    call([PYTHON, "make_viot_final_manuscript_plots.py",
          "--root", "results/processed", "--out", "_quick_plot_check",
          "--dpi", "120", "--bootstrap", "1000"])
    print("\nCHECK PASS: tests, action semantics, and committed-result plotting succeeded.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["check", "ppo", "dqn", "drain", "figures", "all"], default="check")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    if args.mode == "check":
        check()
    elif args.mode == "ppo":
        run_ppo(args.device)
    elif args.mode == "dqn":
        run_dqn(args.device)
    elif args.mode == "drain":
        run_drain(args.device)
    elif args.mode == "figures":
        run_figures()
    else:
        check()
        run_ppo(args.device)
        run_dqn(args.device)
        run_drain(args.device)
        run_figures()


if __name__ == "__main__":
    main()
