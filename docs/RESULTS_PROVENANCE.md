# Results provenance

This repository contains curated summary outputs from the frozen paper experiments so that manuscript plots can be reproduced without rerunning training.

## Primary 300-step PPO-family / MaxWeight runs

- `results/processed/v44ec300_final_static`
- `results/processed/v44ec300_final_reliability`
- `results/processed/v44ec300_final_latency`
- `results/processed/v44ec300_final_admission`

## Matched value-based extension

- `results/processed/sota_dqn_final_static`
- `results/processed/sota_dqn_final_reliability`
- `results/processed/sota_dqn_final_latency`
- `results/processed/sota_dqn_final_admission`

These contain DQN and Dueling Double DQN under the same final seeds and held-out contexts.

## Terminal-drain sensitivity

- `results/processed/td4_final_static`
- `results/processed/td4_final_reliability`
- `results/processed/td4_final_latency`
- `results/processed/td4_final_admission`

The terminal-drain runs reuse frozen PPO-family policies and preserve the original 300-step measurement trajectory before disabling new arrivals and allowing the existing cohort to terminate.

## Figure reproduction

Run:

```bash
python make_viot_final_manuscript_plots.py \
  --root results/processed \
  --out reproduced_manuscript_plots \
  --dpi 600 \
  --bootstrap 20000
```

The generated `plot_data/*.csv` files should numerically match the committed files under `results/plot_data/`.
