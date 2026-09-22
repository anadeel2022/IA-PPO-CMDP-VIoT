#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "checkpoint_manifest.csv"
EXPECTED_SIGNATURE = "89b3003a441eeb26"
EXPECTED_SEEDS = {11, 12, 13, 14, 15}
EXPECTED_SCHEMES = {
    "PPO",
    "PPO-Lagrangian",
    "PPO_CMDP",
    "Intent-PPO-CMDP-NoRisk",
    "Intent-PPO-CMDP",
    "SOTA-DQN",
    "SOTA-DuelingDDQN",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()

    for row in rows:
        path = ROOT / row["path"]
        scheme = row["scheme"]
        seed = int(row["seed"])
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        if sha256(path) != row["sha256"]:
            errors.append(f"sha256 mismatch: {path}")
            continue

        obj = torch.load(path, map_location="cpu", weights_only=True)
        md = obj.get("metadata", {})
        checks = {
            "scheme": str(md.get("scheme")) == scheme,
            "seed": int(md.get("seed", -1)) == seed,
            "signature": str(md.get("training_signature")) == EXPECTED_SIGNATURE,
            "fingerprint": str(md.get("policy_fingerprint")) == row["policy_fingerprint"],
            "n_in": int(obj.get("n_in", -1)) == 60,
            "n_actions": int(obj.get("n_actions", -1)) == 20,
        }
        for key, ok in checks.items():
            if not ok:
                errors.append(f"{key} mismatch: {path}")
        seen.add((scheme, seed))

    expected = {(scheme, seed) for scheme in EXPECTED_SCHEMES for seed in EXPECTED_SEEDS}
    if seen != expected:
        errors.append(
            f"checkpoint grid mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}"
        )

    if errors:
        print("CHECKPOINT VALIDATION: FAIL")
        for error in errors:
            print(" -", error)
        raise SystemExit(2)

    print(
        f"CHECKPOINT VALIDATION: PASS ({len(rows)} files, signature={EXPECTED_SIGNATURE})"
    )
    print(
        "Same-checkpoint shield ablation correctly reuses Intent-PPO-CMDP checkpoints; "
        "no separate FullPolicy-NoShield file is expected."
    )


if __name__ == "__main__":
    main()
