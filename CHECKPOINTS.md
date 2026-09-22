# Frozen policy checkpoints

This repository includes the complete **paper-seed checkpoint set** needed to evaluate the learning methods without retraining.

All committed checkpoints use training signature `89b3003a441eeb26` and manuscript master seeds `11, 12, 13, 14, 15`.

## PPO-family cache

Directory: `policy_cache_v44/89b3003a441eeb26/`

Included schemes:

- `PPO`
- `PPO-Lagrangian`
- `PPO_CMDP`
- `Intent-PPO-CMDP-NoRisk`
- `Intent-PPO-CMDP`

Total: 25 checkpoints.

## Value-based cache

Directory: `policy_cache_sota_dqn/89b3003a441eeb26/`

Included schemes:

- `SOTA-DQN`
- `SOTA-DuelingDDQN`

Total: 10 checkpoints.

## Same-checkpoint shield ablation

There is intentionally **no separate checkpoint** for `Intent-PPO-CMDP-FullPolicy-NoShield`. The code maps that deployment-only ablation to the corresponding `Intent-PPO-CMDP__seedXX.pt` checkpoint. Thus, the trained network is identical and only deployment admission shielding is disabled.

## Excluded checkpoint files

The development cache also contained development seeds `101--103` and a superseded training-signature directory `f763c5ad138c5b5b`. These are not part of the manuscript evaluation and are deliberately excluded.

## Integrity

`checkpoint_manifest.csv` records the scheme, seed, training signature, policy fingerprint, input/action dimensions, SHA-256 digest, and byte size for every committed checkpoint.

Run:

```bash
python verify_checkpoints.py
```

to validate all 35 checkpoint files.
