"""Deterministic gate for the corrected request-level action model.

Run before any expensive training:
    python validate_action_semantics.py

The script exits non-zero if defer, grant, protect, coexist, and reject do not
produce operationally distinct outcomes from a common copied state.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from hiot_env import (
    HIoTEnv,
    MODE_COEXIST,
    MODE_DEFER,
    MODE_GRANT,
    MODE_NAMES,
    MODE_PROTECT,
    MODE_REJECT,
)


def build_state() -> HIoTEnv:
    env = HIoTEnv(
        steps_per_ep=40,
        n_devices=8,
        n_channels=3,
        seed=17,
        arrival_scale=0.0,
        class_mix=(3, 1, 1),
        warm_start_backlog=False,
        device_jitter_cv=0.0,
    )
    env.reset(seed=17)
    for dev in range(4):
        env._enqueue_request(dev, arrival_step=-3, count_arrival=True)
    env.current_sinr_db[:] = 40.0
    env._sync_aggregate_state()
    return env


def main() -> int:
    base = build_state()
    rows = []
    signatures = set()
    for mode in (MODE_DEFER, MODE_GRANT, MODE_PROTECT, MODE_COEXIST, MODE_REJECT):
        env = copy.deepcopy(base)
        _, reward, _, info = env.step(env.encode_action(0, mode))
        row = {
            "mode": MODE_NAMES[mode],
            "delivered_bits": round(float(info.get("delivered_bits", 0.0)), 6),
            "channels_used": int(info.get("channels_used", 0)),
            "completed_requests": int(info.get("completed_requests", 0)),
            "blocked": int(info.get("blocked", 0)),
            "invalid_reject": int(info.get("invalid_reject", 0)),
            "preempted": int(info.get("preempted", 0)),
            "secondary_users": len(info.get("secondary_dev_indices", [])),
            "reward": round(float(reward), 6),
            "request_conservation_error": env.request_accounting()["arrivals"]
                - env.request_accounting()["completed"]
                - env.request_accounting()["rejected"]
                - env.request_accounting()["pending"],
        }
        rows.append(row)
        signatures.add((
            row["delivered_bits"], row["channels_used"], row["blocked"],
            row["preempted"], row["secondary_users"],
        ))

    errors = []
    by_mode = {row["mode"]: row for row in rows}
    if by_mode["defer"]["delivered_bits"] != 0.0:
        errors.append("defer transmitted data")
    if by_mode["reject"]["blocked"] != 1:
        errors.append("reject did not block a not-yet-admitted request")
    if by_mode["grant"]["delivered_bits"] <= 0.0:
        errors.append("grant did not provide ordinary service")
    if by_mode["protect"]["channels_used"] < 2:
        errors.append("protect did not reserve repetition resources")
    if by_mode["coexist"]["channels_used"] < 2:
        errors.append("coexist did not serve multiple users/channels")
    if len(signatures) < 4:
        errors.append("fewer than four operationally distinct action signatures")
    if any(row["request_conservation_error"] != 0 for row in rows):
        errors.append("request accounting is not conservative")

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "actions": rows}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
