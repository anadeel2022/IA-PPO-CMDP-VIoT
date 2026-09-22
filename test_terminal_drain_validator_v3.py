import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_submission_results import validate


def test_terminal_drain_validator_does_not_require_primary_policy_diagnostics(tmp_path: Path):
    run = tmp_path / "td2_final_static"
    run.mkdir()
    manifest = {
        "runner_revision": "request_level_universal_v4_4EC300",
        "request_level_accounting": True,
        "action_model": {"n_actions": 20},
        "seeds": [11],
        "schemes": ["Intent-PPO-CMDP", "Intent-PPO-CMDP-FullPolicy-NoShield"],
        "intents": ["balanced_agentic"],
        "scenarios": ["nominal"],
        "eval_episodes": 200,
        "training_steps": 150,
        "evaluation_steps": 300,
        "steps": 300,
        "terminal_drain_steps": 250,
        "terminal_drain_auto_extend": True,
        "terminal_drain_max_steps": 500,
        "effective_environment_steps": 800,
        "evaluation_action_mode": "stochastic",
        "multi_intent_training": True,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = []
    for scheme in manifest["schemes"]:
        row = {
            "intent": "balanced_agentic",
            "scenario": "nominal",
            "seed": 11,
            "scheme": scheme,
            "env_seed": 100,
            "evaluation_seed": 200,
            "policy_seed": 300,
            "episodes": 200,
            "training_steps": 150,
            "evaluation_steps": 300,
            "policy_fingerprint": "same-frozen-checkpoint",
            "unfinished_rate_mean": 0.0,
            "right_censored_rate_mean": 0.0,
            "measurement_right_censored_rate_mean": 0.3,
            "measurement_unfinished_rate_mean": 0.3,
            "measurement_service_success_rate_mean": 0.7,
            "measurement_deadline_exposure_mean": 0.4,
            "terminal_drain_actual_steps_mean": 275.0,
            "terminal_drain_actual_steps_max": 300.0,
            "terminal_drain_extension_steps_mean": 25.0,
            "terminal_drain_extension_steps_max": 50.0,
            "max_realized_device_deadline_steps_max": 300.0,
            "weighted_violation_mean": 0.1,
            "weighted_violation_cvar95": 0.2,
            "blocking_mean": 0.01,
            "interruption_mean": 0.02,
            "delay_mean_ms": 10.0,
            "throughput_mean_mbps": 2.5,
            "fairness_mean": 0.8,
            "service_success_rate_mean": 0.95,
            "request_conservation_error_max_abs": 0.0,
            "intent_switch_active": 0,
            "evaluation_action_sampling_seed": 123,
            # Intentionally unavailable in the accounting-only sensitivity summary.
            "counterfactual_intent_policy_l1": np.nan,
            "counterfactual_mode_policy_l1": np.nan,
            "counterfactual_argmax_change_rate": np.nan,
            "frozen_policy_entropy_norm": np.nan,
            "pre_mask_reject_probability": np.nan,
            "feasible_reject_probability": np.nan,
            "pre_mask_unsafe_probability_mass": np.nan,
            "shield_filtered_probability_mass": np.nan,
            "actual_shield_prevention_rate": np.nan,
            "stochastic_action_entropy": np.nan,
            "policy_mode_prob_defer": np.nan,
            "policy_mode_prob_grant": np.nan,
            "policy_mode_prob_protect": np.nan,
            "policy_mode_prob_coexist": np.nan,
            "policy_mode_prob_reject": np.nan,
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(run / "agentic_seed_summary.csv", index=False)

    report = validate(run)
    assert report["status"] == "PASS", report["errors"]
    assert report["checks"]["terminal_drain_sensitivity_mode"] is True
    assert report["checks"]["policy_distribution_diagnostics_revalidated_in_terminal_drain"] is False
    assert report["checks"]["causal_same_checkpoint_mismatches"] == 0
