"""Task-1 Compliance Proof Script

Extracts quantitative evidence for Task-1 compliance requirements:
- Preflight feasibility intersection rate
- Unit/identity checks (B_eff_Hz coverage)
- V-scan consistency
- Guard enforcement (fallback_used and action_in_guard_set)
- QoE target metrics
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_preflight_results(preflight_path: Path) -> dict:
    """Parse preflight resolved YAML file."""
    with open(preflight_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract guard feasibility rates
    guard_energy = data.get('guard', {}).get('energy', False)
    guard_queue = data.get('guard', {}).get('queue', False)
    guard_semantic = data.get('guard', {}).get('semantic', False)
    
    # Compute intersection rate (AND of all guards)
    if all([guard_energy, guard_queue, guard_semantic]):
        intersection_rate = 1.0
    else:
        # Use the provided feasible_intersection_rate if available
        intersection_rate = data.get('feasible_intersection_rate', 0.0)
    
    return {
        'pass': data.get('pass', False),
        'guard_energy': guard_energy,
        'guard_queue': guard_queue,
        'guard_semantic': guard_semantic,
        'feasible_intersection_rate': intersection_rate,
        'knobs': {
            'V': data.get('V', 0.0),
            'slot_sec': data.get('slot_sec', 0.0),
            'arrivals_bps': data.get('arrivals_bps', 0.0),
            'energy_budget_per_slot_j': data.get('energy_budget_per_slot_j', 0.0),
            'latency_budget_q_bps': data.get('latency_budget_q_bps', 0.0),
            'semantic_budget': data.get('semantic_budget', 0.0)
        }
    }


def check_unit_coverage(log_path: Path) -> dict:
    """Check B_eff_Hz >= B_min_kHz*1000 coverage."""
    df = pd.read_csv(log_path)
    
    if not all(col in df.columns for col in ['B_eff_Hz', 'B_min_kHz']):
        return {'coverage': 0.0, 'violation_sample': None, 'status': 'MISSING_COLUMNS'}
    
    # Calculate coverage
    lhs = df['B_eff_Hz'].to_numpy(dtype=float)
    rhs = df['B_min_kHz'].to_numpy(dtype=float) * 1000.0
    coverage = (lhs >= rhs).mean()
    
    # Find first violation if any
    violation_sample = None
    if coverage < 0.99:
        violations = np.where(lhs < rhs)[0]
        if len(violations) > 0:
            idx = violations[0]
            violation_sample = {
                'slot': int(df.get('t', pd.Series([idx])).iloc[idx]) if 't' in df.columns else idx,
                'lhs': lhs[idx],
                'rhs': rhs[idx] / 1000.0  # Convert back to kHz for display
            }
    
    return {
        'coverage': coverage,
        'violation_sample': violation_sample,
        'status': 'PASS' if coverage >= 0.99 else 'FAIL'
    }


def check_vscan_consistency(vscan_path: Path) -> dict:
    """Check V-scan summary.csv consistency."""
    df = pd.read_csv(vscan_path)
    
    required_columns = ['agg_version', 'warmup_skip']
    has_required_cols = all(col in df.columns for col in required_columns)
    
    return {
        'row_count': len(df),
        'has_required_columns': has_required_cols,
        'sample_rows': df.head(2).to_dict('records') if len(df) > 0 else [],
        'status': 'PASS' if len(df) >= 3 and has_required_cols else 'FAIL'
    }


def check_guard_enforcement(log_path: Path) -> dict:
    """Check guard enforcement (fallback_used and action_in_guard_set)."""
    df = pd.read_csv(log_path)
    
    fallback_sum = 0
    action_not_in_guard_sum = 0
    leakage_sample = None
    
    if 'fallback_used' in df.columns:
        fallback_sum = int(df['fallback_used'].sum())
    
    if 'action_in_guard_set' in df.columns:
        # Convert to boolean properly
        action_in_guard_bool = df['action_in_guard_set'].astype(bool)
        action_not_in_guard_sum = int((~action_in_guard_bool).sum())
        
        # Find first leakage sample if any
        if action_not_in_guard_sum > 0:
            violations = np.where(~action_in_guard_bool)[0]
            if len(violations) > 0:
                idx = violations[0]
                leakage_sample = {
                    'slot': int(df.get('t', pd.Series([idx])).iloc[idx]) if 't' in df.columns else idx,
                    'fallback_used': int(df['fallback_used'].iloc[idx]) if 'fallback_used' in df.columns else 0,
                    'action_in_guard_set': bool(df['action_in_guard_set'].iloc[idx])
                }
    
    return {
        'fallback_used_sum': fallback_sum,
        'action_not_in_guard_sum': action_not_in_guard_sum,
        'leakage_sample': leakage_sample,
        'status': 'PASS' if fallback_sum == 0 and action_not_in_guard_sum == 0 else 'FAIL'
    }


def extract_qoe_metrics(log_path: Path, vscan_path: Path) -> dict:
    """Extract QoE metrics from log and vscan data."""
    df_log = pd.read_csv(log_path)
    df_vscan = pd.read_csv(vscan_path) if vscan_path.exists() else pd.DataFrame()
    
    # Extract MOS and energy metrics
    mos_mean = df_log['mos'].mean() if 'mos' in df_log.columns else None
    e_slot_j_mean = df_log['E_slot_J'].mean() if 'E_slot_J' in df_log.columns else None
    
    # Extract from vscan if available
    vscan_metrics = {}
    if len(df_vscan) > 0:
        vscan_metrics = {
            'MOS_mean': df_vscan['MOS_mean'].iloc[0] if 'MOS_mean' in df_vscan.columns else None,
            'MOS_ci_lo': df_vscan['MOS_ci_lo'].iloc[0] if 'MOS_ci_lo' in df_vscan.columns else None,
            'MOS_ci_hi': df_vscan['MOS_ci_hi'].iloc[0] if 'MOS_ci_hi' in df_vscan.columns else None,
            'E_slot_J_mean': df_vscan['E_slot_J_mean'].iloc[0] if 'E_slot_J_mean' in df_vscan.columns else None,
            'E_slot_J_ci_lo': df_vscan['E_slot_J_ci_lo'].iloc[0] if 'E_slot_J_ci_lo' in df_vscan.columns else None,
            'E_slot_J_ci_hi': df_vscan['E_slot_J_ci_hi'].iloc[0] if 'E_slot_J_ci_hi' in df_vscan.columns else None
        }
    
    return {
        'log_metrics': {
            'MOS_mean': mos_mean,
            'E_slot_J_mean': e_slot_j_mean
        },
        'vscan_metrics': vscan_metrics
    }


def generate_compliance_report() -> dict:
    """Generate comprehensive compliance report."""
    base_path = Path(__file__).parent.parent.parent.parent
    
    # Define file paths
    preflight_path = base_path / 'outputs' / 'dumps' / 'task1_release_pd2_preflight_resolved.yaml'
    log_path = base_path / 'outputs' / 'dumps' / 'task1_release_pd2_log.csv'
    vscan_path = base_path / 'outputs' / 'vscan' / 'summary.csv'
    
    # Run all checks
    preflight_results = parse_preflight_results(preflight_path)
    unit_coverage = check_unit_coverage(log_path)
    vscan_consistency = check_vscan_consistency(vscan_path)
    guard_enforcement = check_guard_enforcement(log_path)
    qoe_metrics = extract_qoe_metrics(log_path, vscan_path)
    
    # Determine overall status
    all_checks = [
        preflight_results['feasible_intersection_rate'] >= 0.30,
        unit_coverage['status'] == 'PASS',
        vscan_consistency['status'] == 'PASS',
        guard_enforcement['status'] == 'PASS'
    ]
    
    overall_status = 'PASS' if all(all_checks) else 'FAIL'
    
    return {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'preflight': preflight_results,
        'unit_coverage': unit_coverage,
        'vscan_consistency': vscan_consistency,
        'guard_enforcement': guard_enforcement,
        'qoe_metrics': qoe_metrics,
        'file_paths': {
            'preflight': str(preflight_path),
            'log': str(log_path),
            'vscan': str(vscan_path)
        }
    }


def main():
    """Main function to generate compliance proof."""
    report = generate_compliance_report()
    
    # Print concise one-liners for validation log
    print(f"[PREFLIGHT] pass={report['preflight']['pass']}, intersection_rate={report['preflight']['feasible_intersection_rate']:.3f}")
    print(f"[UNIT_COVERAGE] {report['unit_coverage']['status']} (coverage={report['unit_coverage']['coverage']:.3f})")
    print(f"[VSCAN] {report['vscan_consistency']['status']} (rows={report['vscan_consistency']['row_count']})")
    print(f"[GUARD_ENFORCEMENT] {report['guard_enforcement']['status']} (fallback_sum={report['guard_enforcement']['fallback_used_sum']}, action_not_in_guard_sum={report['guard_enforcement']['action_not_in_guard_sum']})")
    
    # Print violation samples if any
    if report['unit_coverage']['violation_sample']:
        sample = report['unit_coverage']['violation_sample']
        print(f"[UNIT_VIOLATION] slot={sample['slot']}, B_eff_Hz={sample['lhs']:.3f}, B_min_kHz={sample['rhs']:.3f}")
    
    if report['guard_enforcement']['leakage_sample']:
        sample = report['guard_enforcement']['leakage_sample']
        print(f"[GUARD_LEAKAGE] slot={sample['slot']}, fallback_used={sample['fallback_used']}, action_in_guard_set={sample['action_in_guard_set']}")
    
    # Print vscan sample rows
    if report['vscan_consistency']['sample_rows']:
        print("[VSCAN_SAMPLES]")
        for i, row in enumerate(report['vscan_consistency']['sample_rows']):
            print(f"  Row {i+1}: {row}")
    
    # Print QoE metrics
    if report['qoe_metrics']['log_metrics']['MOS_mean']:
        print(f"[QOE] MOS_mean={report['qoe_metrics']['log_metrics']['MOS_mean']:.3f}, E_slot_J_mean={report['qoe_metrics']['log_metrics']['E_slot_J_mean']:.3f}")
    
    if report['qoe_metrics']['vscan_metrics']:
        vscan = report['qoe_metrics']['vscan_metrics']
        if vscan.get('MOS_mean'):
            print(f"[VSCAN_QOE] MOS_mean={vscan['MOS_mean']:.3f} [{vscan['MOS_ci_lo']:.3f}, {vscan['MOS_ci_hi']:.3f}], E_slot_J_mean={vscan['E_slot_J_mean']:.3f}")
    
    return report


if __name__ == '__main__':
    main()