import numpy as np

from viot_agentic_intent import INTENT_LIBRARY, SCENARIO_LIBRARY, make_agentic_env
from viot_agentic_runner import (
    policy_max_weight,
    run_policy_baseline,
    run_policy_baseline_terminal_drain,
)


def _env(steps):
    return make_agentic_env(
        intent=INTENT_LIBRARY["balanced_agentic"],
        scenario=SCENARIO_LIBRARY["nominal"],
        steps_per_ep=steps,
        n_devices=24,
        seed=123456,
        arrival_scale=1.0,
        class_mix=(3.0, 1.0, 1.0),
        burst_k=None,
        device_skew=0.15,
        bits_scale=2.0,
        delay_penalty_w=0.0025,
        energy_penalty_w=0.0002,
        prio_w=(1.4, 1.0, 0.85),
    )


def test_terminal_drain_preserves_next_episode_measurement_rng_stream():
    intent = INTENT_LIBRARY["balanced_agentic"]
    primary = run_policy_baseline(_env(300), policy_max_weight, intent, episodes=5, steps_per_ep=300)
    drained = run_policy_baseline_terminal_drain(
        _env(800), policy_max_weight, intent, episodes=5,
        arrival_window_steps=300, drain_steps=20,
        auto_extend=False, max_drain_steps=20,
    )

    mapping = {
        "blocking_prob": "measurement_blocking_prob",
        "throughput_mbps": "measurement_throughput_mbps",
        "service_success_rate": "measurement_service_success_rate",
        "deadline_exposure_rate": "measurement_deadline_exposure_rate",
        "unfinished_rate": "measurement_unfinished_rate",
    }
    for p, d in zip(primary["episode_metrics_std"], drained["episode_metrics_std"]):
        assert d["measurement_rng_stream_preserved"] == 1
        for pkey, dkey in mapping.items():
            assert np.isclose(float(p[pkey]), float(d[dkey]), rtol=0.0, atol=1e-12), (pkey, p[pkey], d[dkey])

class _UniformPolicy:
    def action_probabilities(self, obs, action_mask=None):
        p = np.ones(20, dtype=float)
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=bool).ravel()
            p[~m] = 0.0
        if p.sum() <= 0:
            p[:] = 1.0
        return p / p.sum()

    def act(self, obs, action_mask=None):
        if action_mask is None:
            return 0
        idx = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        return int(idx[0]) if idx.size else 0


def test_stochastic_policy_terminal_branch_preserves_measurement_stream():
    from viot_agentic_runner import run_trained_policy_evaluation, run_trained_policy_evaluation_terminal_drain

    policy = _UniformPolicy()
    primary = run_trained_policy_evaluation(
        _env(300), policy, episodes=4, steps_per_ep=300,
        action_mode="stochastic", sampling_seed=987654,
    )
    drained = run_trained_policy_evaluation_terminal_drain(
        _env(800), policy, episodes=4, arrival_window_steps=300, drain_steps=15,
        action_mode="stochastic", sampling_seed=987654,
        auto_extend=False, max_drain_steps=15,
    )
    mapping = {
        "blocking_prob": "measurement_blocking_prob",
        "throughput_mbps": "measurement_throughput_mbps",
        "service_success_rate": "measurement_service_success_rate",
        "deadline_exposure_rate": "measurement_deadline_exposure_rate",
        "unfinished_rate": "measurement_unfinished_rate",
    }
    for p, d in zip(primary["episode_metrics_std"], drained["episode_metrics_std"]):
        for pkey, dkey in mapping.items():
            assert np.isclose(float(p[pkey]), float(d[dkey]), rtol=0.0, atol=1e-12), (pkey, p[pkey], d[dkey])
