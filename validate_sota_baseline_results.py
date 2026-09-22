"""Strict validator for the DQN / Dueling Double-DQN benchmark extension."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_SCHEMES = {"SOTA-DQN", "SOTA-DuelingDDQN"}
CORE_METRICS = [
    "weighted_violation_mean", "weighted_violation_cvar95", "blocking_mean",
    "interruption_mean", "deadline_exposure_mean", "service_success_rate_mean",
    "unfinished_rate_mean", "throughput_mean_mbps", "fairness_mean",
]


def validate(run_dir: Path) -> dict:
    errors=[]; warnings=[]; checks={}
    manifest_path=run_dir/'manifest.json'
    seed_path=run_dir/'agentic_seed_summary.csv'
    fail_path=run_dir/'failures.log'
    if not manifest_path.exists(): errors.append('manifest.json missing')
    if not seed_path.exists(): errors.append('agentic_seed_summary.csv missing')
    if errors: return {'status':'FAIL','errors':errors,'warnings':warnings,'checks':checks}
    m=json.load(open(manifest_path,encoding='utf-8'))
    df=pd.read_csv(seed_path)

    schemes=set(df['scheme'].astype(str)) if 'scheme' in df else set()
    checks['schemes']=sorted(schemes)
    if schemes != EXPECTED_SCHEMES:
        errors.append(f'Expected schemes {sorted(EXPECTED_SCHEMES)}, observed {sorted(schemes)}')

    for key, expected in [('training_steps',150),('evaluation_steps',300),('eval_episodes',200)]:
        val=m.get(key, m.get('steps') if key=='evaluation_steps' else None)
        checks[key]=val
        if int(val) != expected:
            errors.append(f'{key}={val}, expected {expected}')
    checks['evaluation_action_mode']=m.get('evaluation_action_mode')
    if str(m.get('evaluation_action_mode','')).lower() != 'greedy':
        errors.append('SOTA DQN baselines must use greedy deployment.')
    if not bool(m.get('multi_intent_training',False)):
        errors.append('Universal multi-intent training must be enabled.')

    # Task completeness from manifest dimensions.
    n_int=len(m.get('intents',[])); n_scn=len(m.get('scenarios',[])); n_seed=len(m.get('seeds',[])); n_sch=len(m.get('schemes',[]))
    expected_rows=n_int*n_scn*n_seed*n_sch
    checks['expected_rows']=expected_rows; checks['observed_rows']=len(df)
    if len(df) != expected_rows:
        errors.append(f'Observed {len(df)} rows, expected {expected_rows}.')
    keycols=['intent','scenario','seed','scheme']
    if all(c in df for c in keycols):
        dup=int(df.duplicated(keycols).sum()); checks['duplicate_rows']=dup
        if dup: errors.append(f'{dup} duplicate task rows.')

    # Frozen universal checkpoint identity across contexts.
    if 'policy_fingerprint' not in df:
        errors.append('policy_fingerprint missing')
    else:
        bad=0; empty=0
        for (seed,scheme),g in df.groupby(['seed','scheme']):
            fps=[str(x) for x in g['policy_fingerprint'].fillna('') if str(x)]
            if not fps: empty += 1
            elif len(set(fps)) != 1: bad += 1
        checks['seed_scheme_groups_with_missing_fingerprint']=empty
        checks['seed_scheme_groups_with_multiple_fingerprints']=bad
        if empty or bad: errors.append('Universal frozen-policy fingerprint consistency failed.')

    # Exact request conservation and finite metrics.
    if 'request_conservation_error_max_abs' not in df:
        errors.append('request_conservation_error_max_abs missing')
    else:
        rc=pd.to_numeric(df['request_conservation_error_max_abs'],errors='coerce').to_numpy(float)
        mx=float(np.nanmax(np.abs(rc))) if np.isfinite(rc).any() else np.inf
        checks['request_conservation_error_max_abs']=mx
        if not np.isfinite(mx) or mx > 1e-12: errors.append(f'Request conservation error {mx}.')
    for c in CORE_METRICS:
        if c not in df:
            errors.append(f'{c} missing'); continue
        x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
        frac=float(np.isfinite(x).mean()) if x.size else 0.0
        checks[f'finite_fraction::{c}']=frac
        if frac < 1.0: errors.append(f'{c} contains non-finite values.')

    # Baselines must never activate the proposed admission shield.
    for c in ['shield_override_rate_mean','shield_filter_active_rate','shield_filtered_action_fraction']:
        if c in df:
            x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
            finite=x[np.isfinite(x)]
            mx=float(np.max(np.abs(finite))) if finite.size else 0.0
            checks[f'max_abs::{c}']=mx
            if mx > 1e-12: errors.append(f'{c} is nonzero for SOTA value baseline ({mx}).')

    # Sanity: learned policies trained only once per seed/scheme and reused elsewhere.
    if 'universal_training_reused' in df:
        reuse=pd.to_numeric(df['universal_training_reused'],errors='coerce').fillna(0)
        checks['rows_marked_training_reused']=int((reuse>0).sum())

    if fail_path.exists() and fail_path.read_text(encoding='utf-8').strip():
        errors.append('failures.log is non-empty.')

    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'warnings':warnings,'checks':checks}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_dir')
    args=ap.parse_args(); run_dir=Path(args.run_dir)
    report=validate(run_dir)
    out=run_dir/'sota_baseline_validation.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 2)

if __name__=='__main__': main()
