import pandas as pd
import numpy as np
from pathlib import Path


def main() -> None:
    base = Path('outputs/dumps/task1_release_pd2_log.csv')
    out = Path('outputs/reports/gpt/snippet_release_guard_fail.csv')
    if not base.exists():
        print('[ERROR] release log missing:', base)
        return
    df = pd.read_csv(base)
    # Compute arrivals_bps from A_bits and slot_sec
    if 'A_bits' in df.columns and 'slot_sec' in df.columns:
        slot = df['slot_sec'].replace(0, np.nan)
        df['arrivals_bps'] = df['A_bits'] / slot
    else:
        df['arrivals_bps'] = np.nan
    # Queue enforcement fail: feasible_queue_guard==1 and S_eff_bps < arrivals_bps
    feasible_queue = df.get('feasible_queue_guard', pd.Series([0]*len(df))).astype(float)
    S_eff = df.get('S_eff_bps', pd.Series([np.nan]*len(df))).astype(float)
    mask_q = (feasible_queue == 1.0) & (S_eff < df['arrivals_bps'].astype(float))
    # Energy enforcement fail: feasible_energy_guard==1 and E_slot_J > energy_budget_per_slot_j*(1-epsilon_energy)
    eps = df.get('epsilon_energy', pd.Series([0.0]*len(df))).astype(float)
    budget = df.get('energy_budget_per_slot_j', pd.Series([np.nan]*len(df))).astype(float)
    tight = budget * (1.0 - eps.clip(0.0, 1.0))
    feasible_energy = df.get('feasible_energy_guard', pd.Series([0]*len(df))).astype(float)
    E_vals = df.get('E_slot_J', pd.Series([np.nan]*len(df))).astype(float)
    mask_e = (feasible_energy == 1.0) & (E_vals > tight)
    mask = (mask_q | mask_e)
    cols = [c for c in ['t','feasible_sem_guard','feasible_queue_guard','feasible_energy_guard',
                        'P','B','E_slot_J','S_eff_bps','arrivals_bps','Q_bits','Q_next_bits','A_bits','slot_sec','epsilon_energy'] if c in df.columns]
    snip = df.loc[mask, cols].head(100)
    out.parent.mkdir(parents=True, exist_ok=True)
    snip.to_csv(out, index=False)
    print('[SAVE]', out)


if __name__ == '__main__':
    main()