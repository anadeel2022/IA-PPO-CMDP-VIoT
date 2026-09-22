# Reproducibility Guide

This document records the frozen protocol for **Intent-Aware Risk-Constrained Reinforcement Learning for Request-Level Service Scheduling in RSU-Assisted Vehicular IoT**.

## Core system

The simulator is request-level and uses a centralized RSU scheduler.

- Devices: 24
- Channels: 3 × 1 MHz
- Slot: 1 ms
- Candidate requests: 4
- Modes: defer, grant, protect, coexist, reject
- Actions: 20
- Base state: 45 dimensions
- Intent context: 15 dimensions
- Policy input: 60 dimensions

Request accounting satisfies exactly:

```text
arrivals = completed + rejected + pending
```

## Service classes

| Class | Rate (req/s) | Base payload (bits) | Deadline (ms) | Tx power (dBm) | Battery (J) | Priority |
|---|---:|---:|---:|---:|---:|---:|
| Safety/Emergency | 20 | 4000 | 20 | 18 | 50 | 1.40 |
| Telemetry | 10 | 2500 | 100 | 12 | 30 | 1.00 |
| Best Effort | 4 | 1800 | 250 | 8 | 20 | 0.85 |

The paper uses `bits_scale=2.0`, giving reference payloads of 8000, 5000, and 3600 bits before device-level jitter.

## Final intents

### Balanced

- blocking ≤ 0.060
- interruption ≤ 0.200
- effective deadline = 80 ms
- deadline exposure ≤ 0.70
- service success ≥ 0.60
- fairness ≥ 0.50
- throughput ≥ 2.50 Mbps

Weights: block 1.00, interruption 1.15, deadline exposure 1.20, success 1.05, fairness 0.65, energy 0.20, utilization 0.20, throughput 0.95.

### Reliability-First

- blocking ≤ 0.030
- interruption ≤ 0.160
- effective deadline = 80 ms
- deadline exposure ≤ 0.72
- service success ≥ 0.60
- fairness ≥ 0.50
- throughput ≥ 2.40 Mbps

Weights: block 1.30, interruption 2.20, deadline exposure 1.10, success 1.00, fairness 0.35, energy 0.10, utilization 0.20, throughput 0.65.

### Latency-Critical

- blocking ≤ 0.100
- interruption ≤ 0.350
- effective deadline = 70 ms
- deadline exposure ≤ 0.60
- service success ≥ 0.75
- fairness ≥ 0.50
- throughput ≥ 3.00 Mbps

Weights: block 0.55, interruption 0.55, deadline exposure 2.40, success 1.65, fairness 0.30, energy 0.10, utilization 0.20, throughput 1.35.

The energy-efficiency objective is inactive for these three manuscript intents. Minimum utilization is 0.10.

## IA-PPO-CMDP

The manuscript method is `IA-PPO-CMDP`; the internal code identifier is `Intent-PPO-CMDP`.

The frozen implementation combines:

- universal intent-conditioned PPO;
- target-centered cumulative service-risk estimates with κ=10;
- outcome-pressure compilation;
- risk signal `rho = 0.55 * nu + 0.45 * omega`;
- 45-step rolling risk memory;
- 90th-percentile upper-tail training statistic;
- deployment admission shield;
- PPO reward clipping to [-5, 5].

The training upper-tail mechanism is distinct from the episode-level CVaR95 evaluation metric.

## PPO training protocol

- hidden layers: 128, 128
- gamma: 0.99
- GAE lambda: 0.95
- PPO clip: 0.15
- value coefficient: 0.5
- epochs per rollout: 4
- minibatch fraction: 0.5
- value clip: 0.25
- gradient norm: 0.5
- IA learning rate: 1e-4
- baseline PPO learning rate: 2e-4
- IA entropy: 0.040 → 0.010
- baseline PPO entropy: 0.030 → 0.008
- PPO-Lagrangian dual LR: 0.012
- initial multiplier: 0.40
- maximum multiplier: 25

## DQN-family protocol

- hidden layers: 128, 128
- gamma: 0.99
- Adam LR: 5e-4
- batch size: 128
- replay capacity: 50,000
- warm-up: 1,000 steps
- train every 4 environment steps
- target sync every 500 steps
- epsilon: 1.00 → 0.05 over 70% of planned training
- gradient norm limit: 10
- deployment: greedy masked argmax

## Seeds and horizons

Independent master seeds:

```text
11, 12, 13, 14, 15
```

Training: 600 episodes × 150 steps.

Held-out evaluation: 200 episodes × 300 steps.

PPO-family deployment is stochastic categorical using matched indexed uniforms within context. DQN-family deployment is greedy masked argmax. MaxWeight is deterministic.

## Training curriculum

Training scenarios:

- nominal
- congestion burst
- emergency surge
- mixed stress
- admission pressure

Twenty percent of episodes retain a static intent; otherwise the intent switches at 35% or 65% of the training horizon. Held-out switching uses a 50% switch point.

## Same-checkpoint shield experiment

The primary shield experiment reuses the exact trained `Intent-PPO-CMDP` checkpoint and changes only deployment shield activation through `Intent-PPO-CMDP-FullPolicy-NoShield`.

## Frozen checkpoint package

The repository includes the complete manuscript-seed checkpoint grid under training signature `89b3003a441eeb26`. PPO-family checkpoints are stored in `policy_cache_v44/89b3003a441eeb26/` and DQN-family checkpoints in `policy_cache_sota_dqn/89b3003a441eeb26/`. Development seeds and superseded training signatures are excluded. Run `python verify_checkpoints.py` before cached-policy evaluation.

The deployment-only `Intent-PPO-CMDP-FullPolicy-NoShield` condition reuses the corresponding full IA checkpoint and therefore has no independent model file.

## Terminal drain

After the primary 300-step trajectory:

- arrivals are disabled;
- minimum drain = 250 steps;
- automatic extension is enabled;
- maximum drain = 500 steps;
- RNG state is saved before the drain and restored afterward.

The terminal drain is a sensitivity analysis, not a replacement for the primary benchmark.

## Statistics

The five master seeds are the independent experimental units. Contexts and episodes are not treated as independent replicates. Aggregate static metrics are first averaged within each seed across the nine intent-scenario conditions. The manuscript uses seed-cluster percentile bootstrap intervals, paired seed differences, exact two-sided sign-flip tests, and Holm correction over the external comparisons for the primary risk endpoints.
