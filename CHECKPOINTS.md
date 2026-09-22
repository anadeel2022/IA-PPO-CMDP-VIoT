# Frozen policy checkpoints

This repository includes the complete **paper-seed checkpoint set** needed to evaluate the learning methods without retraining.

## Included checkpoint families

All committed checkpoints use training signature `89b3003a441eeb26` and manuscript master seeds `11, 12, 13, 14, 15`.

### PPO-family cache: `policy_cache_v44/89b3003a441eeb26/`

- `PPO`
- `PPO-Lagrangian`
- `PPO_CMDP`
- `Intent-PPO-CMDP-NoRisk`
- `Intent-PPO-CMDP`

There are 25 PPO-family checkpoints: 5 schemes × 5 paper seeds.

### Value-based cache: `policy_cache_sota_dqn/89b3003a441eeb26/`

- `SOTA-DQN`
- `SOTA-DuelingDDQN`

There are 10 value-based checkpoints: 2 schemes × 5 paper seeds.

## Same-checkpoint shield ablation

There is intentionally **no separate checkpoint** for `Intent-PPO-CMDP-FullPolicy-NoShield`. The code maps that deployment-only ablation to the corresponding `Intent-PPO-CMDP__seedXX.pt` file. This is required for the causal same-checkpoint experiment: the learned network is identical and only deployment admission shielding is disabled.

## Excluded checkpoint files

The uploaded development cache also contained:

- development seeds `101--103`; and
- a superseded training-signature directory `f763c5ad138c5b5b`.

Those files are not part of the manuscript evaluation and are deliberately excluded from the publication package.

## Integrity manifest

`checkpoint_manifest.csv` records for every committed checkpoint:

- scheme;
- seed;
- training signature;
- stored policy fingerprint;
- input/action dimensions;
- SHA-256 file digest; and
- byte size.

Run:

```bash
python verify_checkpoints.py
```

to verify the committed caches before evaluation.
