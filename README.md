# IA-PPO-CMDP-VIoT

Code and reproducibility package for the manuscript:

**Intent-Aware Risk-Constrained Reinforcement Learning for Request-Level Service Scheduling in RSU-Assisted Vehicular IoT**

Target journal: *Internet of Things* (Elsevier).

## What is included

This repository contains the frozen simulation and learning code used for the paper, publication orchestration scripts, validation tests, curated seed-level result summaries, and the scripts/data used to generate the manuscript result figures.

IA-PPO-CMDP combines a universal intent-conditioned PPO scheduler with service-risk compilation, rolling risk memory, an upper-tail training term, and a deployment admission shield. The simulator operates at the request level and enforces exact request conservation.

The repository intentionally does **not** claim that IA-PPO-CMDP is uniformly superior across all static metrics. The manuscript reports the observed trade-offs against MaxWeight, DQN, Dueling Double DQN, PPO, PPO-Lagrangian, and PPO-CMDP.

## Frozen paper configuration

- 24 vehicles
- 3 orthogonal 1-MHz channels
- 1-ms scheduling slots
- 4 dynamically constructed request candidates
- 5 modes: defer, grant, protect, coexist, reject
- 20 candidate-mode actions
- 45-dimensional network/request state
- 15-dimensional service-intent context
- 60-dimensional policy input
- 5 independent master seeds: 11--15
- training: 600 episodes × 150 steps
- held-out evaluation: 200 episodes × 300 steps

Evaluated intents: **Balanced**, **Reliability-First**, and **Latency-Critical**.

## Quick validation

Create a clean Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m pytest -q
python validate_action_semantics.py
python run_paper_reproduction.py --mode check
```

The staged package was audited with **10 passing pytest tests** and a passing action-semantics validation.

## Reproduce manuscript figures without retraining

Curated seed-level summaries are committed under `results/processed/`. Generate the result figures with:

```bash
python make_viot_final_manuscript_plots.py \
  --root results/processed \
  --out reproduced_manuscript_plots \
  --dpi 600 \
  --bootstrap 20000
```

The committed `results/plot_data/` files are the numerical data used for the manuscript result plots.

## Full experiment reproduction

Use the publication wrapper rather than running `viot_agentic_runner.py` without arguments:

```bash
python run_paper_reproduction.py --mode ppo --device cpu
python run_paper_reproduction.py --mode dqn --device cpu
python run_paper_reproduction.py --mode drain --device cpu
python run_paper_reproduction.py --mode figures --device cpu
```

Or run all stages:

```bash
python run_paper_reproduction.py --mode all --device cpu
```

The full workflow is computationally expensive because it retrains five independent seeds for the learning methods. All experiment definitions remain frozen; do not retune parameters on seeds 11--15.

## Same-checkpoint deployment-shield experiment

The causal shield ablation reuses the **identical trained IA-PPO-CMDP checkpoint** and disables only the deployment admission shield. It must not be replaced by independently trained shield-on/shield-off policies.

## Terminal-drain sensitivity

The primary comparison remains the 300-step held-out deployment experiment. Terminal-drain runs disable new arrivals after that window and allow the existing request cohort to terminate while preserving the original measurement trajectory and random-number stream.

## Repository map

- Core simulator/learning code: top-level `*.py` modules.
- `run_paper_reproduction.py`: publication orchestration.
- `make_viot_final_manuscript_plots.py`: manuscript result plots.
- `results/processed/`: curated seed-level final outputs.
- `results/plot_data/`: exact plotting data committed with the paper package.
- `figures/`: final result figures in PDF/PNG/SVG.
- `docs/ENVIRONMENT.md`: recorded final execution environment.
- `docs/RESULTS_PROVENANCE.md`: result-folder mapping.
- `docs/CODE_AUDIT.md`: publication-safety audit.
- `REPRODUCIBILITY.md`: detailed frozen protocol.

## Checkpoints

Model checkpoint caches are intentionally not committed to Git because the complete paper package should not mix only a subset of trained policies. If checkpoint archival is required, the full frozen checkpoint set should be archived together with a versioned release/DOI rather than partially uploaded.

## Citation and license

`CITATION.cff` and the software license will be finalized after the paper author list and repository-release policy are confirmed. Until a license is added, this private staging repository should not be interpreted as granting reuse rights.
