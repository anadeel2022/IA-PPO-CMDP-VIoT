from viot_agentic_intent import INTENT_LIBRARY, SCENARIO_LIBRARY, make_agentic_env
from hiot_utils import finalize_episode_metrics
from viot_agentic_runner import _pending_right_censored_count


def _env(steps=20):
    return make_agentic_env(
        intent=INTENT_LIBRARY["balanced_agentic"],
        scenario=SCENARIO_LIBRARY["nominal"],
        steps_per_ep=steps,
        n_devices=12,
        seed=7,
        arrival_scale=1.0,
        class_mix=(3.0, 1.0, 1.0),
        burst_k=None,
        device_skew=0.15,
        bits_scale=2.0,
        delay_penalty_w=0.0025,
        energy_penalty_w=0.0002,
        prio_w=(1.4, 1.0, 0.85),
        intent_switch_to=None,
        intent_switch_frac=0.5,
        intent_switch_hold_steps=10,
        intent_switch_tolerance=0.05,
    )


def test_arrivals_can_be_disabled_without_dropping_existing_cohort():
    env = _env(steps=20)
    env.reset(seed=7)
    before = env.total_arrivals_ep
    env.arrivals_enabled = False
    for _ in range(5):
        env.step(0)
    assert env.total_arrivals_ep == before


def test_fixed_250_can_leave_jittered_deadline_right_censored():
    """A realized deadline can exceed the nominal 250-step BE value."""
    env = _env(steps=400)
    env.reset(seed=8)
    env.request_queues = [type(q)() for q in env.request_queues]
    env._deadline_heap = []
    env.queue_bits[:] = 0.0
    env.total_arrivals_ep = 0
    env.completed_requests_ep = 0
    env.blocked_arrivals_ep = 0
    env.deadline_steps[0] = 275
    env._enqueue_request(0, arrival_step=0, count_arrival=True)
    env.arrivals_enabled = False

    for _ in range(250):
        env.step(0)
    assert _pending_right_censored_count(env) == 1

    for _ in range(25):
        env.step(0)
    assert _pending_right_censored_count(env) == 0
    m = finalize_episode_metrics(env, steps_per_ep=275, num_channels=env.n_channels, slot_s=env.slot_s)
    assert m["right_censored_requests_ep"] == 0
    assert m["right_censored_rate"] == 0.0


def test_realized_device_deadlines_can_be_checked_directly():
    env = _env(steps=300)
    env.reset(seed=9)
    assert int(env.deadline_steps.max()) >= 1
