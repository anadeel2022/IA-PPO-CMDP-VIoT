# Reproducibility Guide

This document records the frozen experimental protocol for the manuscript **Intent-Aware Risk-Constrained Reinforcement Learning for Request-Level Service Scheduling in RSU-Assisted Vehicular IoT**.

The code and commands below will be finalized after the source audit. The numerical protocol itself is frozen and must not be retuned to improve the final seeds.

## 1. Core system

The simulator is request-level and uses a centralized RSU scheduler.

- Devices: 24
- Channels: 3
- Channel bandwidth: 1 MHz each
- Slot duration: 1 ms
- Candidate requests per decision: 4
- Modes: defer, grant, protect, coexist, reject
- Discrete actions: 4 × 5 = 20
- Base state: 45 dimensions
- Intent context: 15 dimensions
- Policy input: 60 dimensions

Exact request conservation is enforced:

```text
arrivals = completed + rejected + pending
```

## 2. Final service intents

### Balanced

- Blocking target: 0.060
- Interruption target: 0.200
- Effective deadline: 80 ms
- Deadline-exposure target: 0.70
- Minimum service success: 0.60
- Minimum fairness: 0.50
- Throughput floor: 2.50 Mbps

### Reliability-First

- Blocking target: 0.030
- Interruption target: 0.160
- Effective deadline: 80 ms
- Deadline-exposure target: 0.72
- Minimum service success: 0.60
- Minimum fairness: 0.50
- Throughput floor: 2.40 Mbps

### Latency-Critical

- Blocking target: 0.100
- Interruption target: 0.350
- Effective deadline: 70 ms
- Deadline-exposure target: 0.60
- Minimum service success: 0.75
- Minimum fairness: 0.50
- Throughput floor: 3.00 Mbps

Energy efficiency is inactive for the three manuscript intents. The minimum-utilization target is 0.10.

## 3. Training and evaluation

Final independent master seeds:

```text
11, 12, 13, 14, 15
```

Training:

```text
600 episodes
150 steps per episode
```

Held-out evaluation:

```text
200 episodes
300 steps per episode
```

The training curriculum includes nominal, congestion-burst, emergency-surge, mixed-stress, and admission-pressure conditions. Twenty percent of training episodes retain a static intent; other episodes switch at 35% or 65% of the training horizon.

The held-out switching evaluation uses a 50% switch point.

## 4. Proposed method

The manuscript method is **IA-PPO-CMDP**. Internal code identifiers may use `Intent-PPO-CMDP`.

The frozen implementation combines:

- universal intent-conditioned PPO;
- target-centered online service-risk estimates;
- outcome-pressure compilation;
- a 45-step rolling risk memory;
- a 90th-percentile upper-tail training statistic;
- a deployment admission shield.

The training upper-tail statistic is distinct from the final episode-level CVaR95 evaluation metric.

## 5. Same-checkpoint shield experiment

The primary causal shield experiment reuses the exact trained IA-PPO-CMDP policy checkpoint and changes only deployment-time shield activation.

This experiment must never be replaced by a comparison of independently trained shield-on and shield-off networks.

## 6. Terminal-drain sensitivity

The primary deployment measurement window remains 300 steps.

Afterward:

- new arrivals are disabled;
- a minimum 250-step drain is applied;
- the drain may auto-extend up to 500 steps;
- the RNG state is saved before the drain branch and restored afterward.

The terminal drain is an accounting sensitivity and does not replace the primary 300-step benchmark.

## 7. Statistical unit

The five master seeds are the independent experimental units.

Repeated episodes, intents, scenarios, and operating contexts are not treated as independent replicates.

Primary uncertainty and paired comparisons use seed-level aggregation.

## 8. Scripts to be staged

The publication package is expected to retain cleaned versions of the following functional roles:

```text
viot_agentic_runner.py
viot_agentic_intent.py
run_q1_value_baselines.py
run_terminal_drain_sensitivity.py
evaluate_v44ec300_results.py
validate_action_semantics.py
validate_submission_results.py
make_viot_manuscript_figures.py
```

Names may be reorganized into `src/`, `experiments/`, `validation/`, and `analysis/` during the audit.

## 9. Files that will not be published directly

The repository should not contain:

- Python virtual environments;
- `__pycache__`;
- the full `outputs_agentic/` development tree;
- obsolete development runs;
- temporary logs;
- machine-specific absolute Windows paths;
- private credentials or tokens;
- raw checkpoint caches unless explicitly selected for archival release.

Curated processed results and plot-data files required to reproduce the manuscript figures and tables will be included.

## 10. Final archival plan

Before manuscript submission:

1. audit and stage the final source;
2. reproduce manuscript figures from the staged package;
3. run validation/tests from a clean environment;
4. freeze a paper release;
5. make the repository public;
6. archive the release with a DOI where possible;
7. cite the archived version in the manuscript Code Availability statement.
