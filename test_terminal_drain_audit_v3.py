from pathlib import Path

import pandas as pd

from terminal_drain_audit import _measurement_window_identity, ORIGINAL_RUNS


def test_measurement_window_identity_matches_frozen_primary(tmp_path: Path):
    keys = {"intent": "balanced_agentic", "scenario": "nominal", "seed": 11, "scheme": "PPO_CMDP"}
    td_dfs = {}
    for name, run_id in ORIGINAL_RUNS.items():
        original_dir = tmp_path / run_id
        original_dir.mkdir(parents=True)
        original = pd.DataFrame([{**keys,
            "blocking_mean": 0.01,
            "throughput_mean_mbps": 2.5,
            "service_success_rate_mean": 0.7,
            "deadline_exposure_mean": 0.4,
            "unfinished_rate_mean": 0.3,
        }])
        original.to_csv(original_dir / "agentic_seed_summary.csv", index=False)
        td_dfs[name] = pd.DataFrame([{**keys,
            "blocking_mean": 0.02,
            "throughput_mean_mbps": 2.6,
            "service_success_rate_mean": 0.9,
            "deadline_exposure_mean": 0.5,
            "unfinished_rate_mean": 0.0,
            "measurement_blocking_mean": 0.01,
            "measurement_throughput_mean_mbps": 2.5,
            "measurement_service_success_rate_mean": 0.7,
            "measurement_deadline_exposure_mean": 0.4,
            "measurement_unfinished_rate_mean": 0.3,
        }])
    report = _measurement_window_identity(tmp_path, td_dfs)
    assert report["overall_pass"] is True
    assert report["max_abs_difference"] == 0.0
