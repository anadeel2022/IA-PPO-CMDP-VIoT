# IA-PPO-CMDP-VIoT

Code and reproducibility package for the manuscript:

**Intent-Aware Risk-Constrained Reinforcement Learning for Request-Level Service Scheduling in RSU-Assisted Vehicular IoT**

## Repository status

This repository is currently a **private staging repository** for the paper code. The scientific configuration and manuscript results are frozen. Source code, processed result files, validation scripts, and figure-generation scripts will be added only after a publication-safety audit.

The repository will be made public once the reproducibility package has been checked against the final manuscript.

## Method overview

IA-PPO-CMDP is a request-level scheduler for centralized RSU-assisted Vehicular Internet of Things (V-IoT). The frozen implementation uses:

- 24 vehicular devices and 3 orthogonal 1-MHz channels;
- 1-ms scheduling slots;
- 4 dynamically constructed request candidates;
- 5 scheduling modes: defer, grant, protect, coexist, and reject;
- a 20-action candidate-mode space;
- a 45-dimensional network/request state;
- a 15-dimensional service-intent context;
- a universal PPO actor-critic;
- intent-conditioned service-risk compilation;
- a 45-step rolling risk memory;
- a 90th-percentile upper-tail training term;
- a deployment admission shield;
- exact request-level conservation;
- terminal-drain sensitivity for finite-horizon right censoring.

The evaluated intents are **Balanced**, **Reliability-First**, and **Latency-Critical**.

## Frozen evaluation protocol

The final paper evaluation uses:

- master seeds: 11, 12, 13, 14, 15;
- training: 600 episodes × 150 steps;
- held-out evaluation: 200 episodes × 300 steps;
- matched traffic, PHY conditions, candidate construction, and physical feasibility masks;
- PPO-family stochastic categorical deployment;
- DQN-family greedy masked argmax deployment;
- terminal-drain sensitivity with arrivals disabled after the primary 300-step window.

External comparators in the manuscript are:

- MaxWeight
- DQN
- Dueling Double DQN
- PPO
- PPO-Lagrangian
- PPO-CMDP

The repository will preserve the manuscript claim discipline: IA-PPO-CMDP is **not** presented as uniformly superior across all static metrics.

## Planned repository structure

```text
IA-PPO-CMDP-VIoT/
├── README.md
├── REPRODUCIBILITY.md
├── .gitignore
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── src/
├── experiments/
├── validation/
├── analysis/
├── results/
│   ├── processed/
│   └── plot_data/
├── figures/
├── tests/
└── docs/
```

The exact structure may be adjusted during the code audit so that the public package remains minimal and reproducible.

## Reproducibility policy

The public repository will include the scripts and processed numerical data required to reproduce the manuscript tables and figures. Large development outputs, virtual environments, temporary logs, obsolete runs, and local machine paths will not be published.

The same-checkpoint deployment-shield ablation will be preserved as a separate reproducibility target because it reuses the identical trained IA-PPO-CMDP checkpoint while changing only the deployment shield state.

## Citation

A `CITATION.cff` file and archival DOI will be added after the final author list and publication metadata are frozen.

## License

A software license will be selected before the repository is made public. Until then, this private staging repository should not be treated as granting reuse rights.

## Paper status

Target journal: *Internet of Things* (Elsevier).

The manuscript, author metadata, and submission declarations are maintained separately from this code repository.
