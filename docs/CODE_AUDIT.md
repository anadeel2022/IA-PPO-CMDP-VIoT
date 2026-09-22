# Publication code audit

Audit date: 2026-09-22

The uploaded development package was reviewed before staging this publication repository.

## Checks passed

- Python test suite in the supplied package: **10 passed**.
- `validate_action_semantics.py`: **PASS** for defer, grant, protect, coexist, and reject actions.
- Primary frozen result families: submission validation **PASS**.
- DQN/Dueling Double DQN extension families: baseline validation **PASS**.
- Terminal-drain audit: **PASS** for elimination of residual right censoring, exact request conservation, 300-step measurement-window identity, and RNG-stream preservation.
- Final manuscript plot data were regenerated from the curated result folders and matched the supplied manuscript `plot_data` CSVs exactly.
- No API keys, passwords, access tokens, or author e-mail addresses were detected in the staged source files.

## Items intentionally excluded

- Python virtual environments and caches.
- Full development `outputs_agentic/` trees.
- Development-only checkpoints (seeds 101--103 and superseded training signatures).
- Obsolete intermediate scripts, historical changelogs, and superseded plotting scripts.
- Validation-report JSON files containing absolute local Windows paths.
- Development documentation containing machine-specific local paths.
- Complexity/scalability utilities not used in the final manuscript.

## Frozen checkpoints included

The publication package now includes the complete checkpoint grid required for the manuscript seeds under training signature `89b3003a441eeb26`:

- 25 PPO-family checkpoints: `PPO`, `PPO-Lagrangian`, `PPO_CMDP`, `Intent-PPO-CMDP-NoRisk`, and `Intent-PPO-CMDP`, each for seeds 11--15;
- 10 value-based checkpoints: `SOTA-DQN` and `SOTA-DuelingDDQN`, each for seeds 11--15.

The `Intent-PPO-CMDP-FullPolicy-NoShield` causal ablation intentionally has no separate checkpoint. It reuses the corresponding `Intent-PPO-CMDP` checkpoint, as required by the same-checkpoint design. Checkpoint metadata, policy fingerprints, and SHA-256 digests are recorded in `checkpoint_manifest.csv` and validated by `verify_checkpoints.py`.

## Important implementation note

`viot_agentic_runner.py` is retained from the frozen development code and contains historical no-argument/PyCharm defaults and legacy scheme definitions. For paper reproduction, do **not** rely on those defaults. Use `run_paper_reproduction.py`, which supplies the exact frozen manuscript command-line configuration explicitly.

The publication wrapper changes orchestration only; it does not alter simulator, policy, reward, intent, shield, PHY, or action semantics.
