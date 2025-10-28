#!/usr/bin/env python3
"""Task-1 Compliance Proof Checker

Validates the four PD9 gates:
1. Preflight feasibility: rate ≥ 0.30
2. Unit/identity: coverage ≥ 99% (B_eff_Hz ≥ B_min_kHz*1000)
3. V-scan consistency: columns contain agg_version,warmup_skip and rows ≥ 3
4. Guard enforcement: fallback_used.sum==0 and (~action_in_guard_set).sum==0
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_preflight_feasibility():
    """Check preflight feasibility rate ≥ 0.30"""
    preflight_file = "outputs/dumps/task1_release_pd2_preflight_resolved.yaml"
    
    if not os.path.exists(preflight_file):
        print("FAIL: Preflight file not found")
        return False, "Preflight file missing"
    
    with open(preflight_file, 'r') as f:
        data = yaml.safe_load(f)
    
    pass_status = data.get('pass', False)
    feasible_rate = data.get('feasible_intersection_rate', 0.0)
    P_max_dBm = data.get('link', {}).get('P_max_dBm', 0.0)
    B_min_kHz = data.get('search', {}).get('B_min_kHz', 0.0)
    B_grid_k = data.get('search', {}).get('B_grid_k', 0)
    
    print(f"Preflight: pass={pass_status}, P_max_dBm={P_max_dBm}, B_min_kHz={B_min_kHz}, B_grid_k={B_grid_k}, feasible_intersection_rate={feasible_rate}")
    
    verdict = feasible_rate >= 0.30
    if not verdict:
        return False, f"Preflight feasibility rate {feasible_rate} < 0.30"
    
    return True, "PASS"

def check_unit_identity():
    """Check unit/identity coverage ≥ 99% (B_eff_Hz ≥ B_min_kHz*1000)"""
    log_file = "outputs/dumps/task1_release_pd2_log.csv"
    
    if not os.path.exists(log_file):
        print("FAIL: Action log file not found")
        return False, "Action log file missing"
    
    df = pd.read_csv(log_file)
    
    # Check if required columns exist
    if 'B' not in df.columns or 'B_min_kHz' not in df.columns:
        print("FAIL: Missing required columns (B, B_min_kHz)")
        return False, "Missing required columns"
    
    # Convert B from MHz to Hz (B is in MHz, so B_eff_Hz = B * 1e6)
    # B_min_kHz is in kHz, so B_min_Hz = B_min_kHz * 1000
    df['B_eff_Hz'] = df['B'] * 1e6
    df['B_min_Hz'] = df['B_min_kHz'] * 1000
    
    # Check identity condition
    identity_ok = df['B_eff_Hz'] >= df['B_min_Hz']
    coverage = identity_ok.mean()
    
    print(f"Unit/identity: coverage={coverage:.4f} ({identity_ok.sum()}/{len(df)})")
    
    if coverage < 0.99:
        # Find first violation
        violations = df[~identity_ok]
        if len(violations) > 0:
            first_violation = violations.iloc[0]
            t = first_violation.get('t', 0) if 't' in df.columns else first_violation.name
            lhs = first_violation['B_eff_Hz']
            rhs = first_violation['B_min_Hz']
            print(f"FIRST violating slot: t={t}, lhs={lhs:.2f}, rhs={rhs:.2f}")
        return False, f"Unit/identity coverage {coverage:.4f} < 0.99"
    
    return True, "PASS"

def check_vscan_consistency():
    """Check V-scan consistency: columns contain agg_version,warmup_skip and rows ≥ 3"""
    vscan_file = "outputs/vscan/summary.csv"
    
    if not os.path.exists(vscan_file):
        print("FAIL: V-scan summary file not found")
        return False, "V-scan summary file missing"
    
    df = pd.read_csv(vscan_file)
    
    # Check required columns
    required_cols = ['agg_version', 'warmup_skip']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"FAIL: Missing columns: {missing_cols}")
        return False, f"Missing columns: {missing_cols}"
    
    # Check row count
    row_count = len(df)
    print(f"V-scan: rows={row_count}, columns={list(df.columns)}")
    
    if row_count < 3:
        print(f"FAIL: Only {row_count} rows, need at least 3")
        return False, f"Only {row_count} rows, need at least 3"
    
    # Print first 2 data rows
    print("First 2 data rows:")
    for i in range(min(2, len(df))):
        row = df.iloc[i]
        print(f"  Row {i}: V={row.get('V', 'N/A')}, agg_version={row.get('agg_version', 'N/A')}, warmup_skip={row.get('warmup_skip', 'N/A')}")
    
    return True, "PASS"

def check_guard_enforcement():
    """Check guard enforcement: fallback_used.sum==0 and (~action_in_guard_set).sum==0"""
    log_file = "outputs/dumps/task1_release_pd2_log.csv"
    
    if not os.path.exists(log_file):
        print("FAIL: Action log file not found")
        return False, "Action log file missing"
    
    df = pd.read_csv(log_file)
    
    # Check if required columns exist
    if 'fallback_used' not in df.columns or 'action_in_guard_set' not in df.columns:
        print("FAIL: Missing required columns (fallback_used, action_in_guard_set)")
        return False, "Missing required columns"
    
    fallback_sum = df['fallback_used'].sum()
    guard_violations = (~df['action_in_guard_set'].astype(bool)).sum()
    
    print(f"Guard enforcement: fallback_used.sum={fallback_sum}, (~action_in_guard_set).sum={guard_violations}")
    
    if fallback_sum > 0 or guard_violations > 0:
        # Find first leakage sample
        leakage_mask = (df['fallback_used'] > 0) | (~df['action_in_guard_set'].astype(bool))
        if leakage_mask.any():
            first_leakage = df[leakage_mask].iloc[0]
            t = first_leakage.get('t', 0) if 't' in df.columns else first_leakage.name
            action_in_guard_set = first_leakage['action_in_guard_set']
            fallback_used = first_leakage['fallback_used']
            bad_energy = first_leakage.get('bad_energy', 0)
            bad_queue = first_leakage.get('bad_queue', 0)
            bad_sem = first_leakage.get('bad_sem', 0)
            P_star = first_leakage.get('P', 0)
            B_star = first_leakage.get('B', 0)
            
            print(f"FIRST leakage sample: slot={t}, action_in_guard_set={action_in_guard_set}, fallback_used={fallback_used}")
            print(f"  bad_energy={bad_energy}, bad_queue={bad_queue}, bad_sem={bad_sem}, P*={P_star}, B*={B_star}")
        
        return False, f"Guard enforcement failed: fallback_used.sum={fallback_sum}, guard_violations={guard_violations}"
    
    return True, "PASS"

def main():
    """Main validation function"""
    print("=== Task-1 Compliance Proof Checker ===\n")
    
    results = []
    
    # Check preflight feasibility
    print("1. Preflight feasibility:")
    ok1, msg1 = check_preflight_feasibility()
    results.append(("Preflight feasibility", ok1, msg1))
    print()
    
    # Check unit/identity
    print("2. Unit/identity:")
    ok2, msg2 = check_unit_identity()
    results.append(("Unit/identity", ok2, msg2))
    print()
    
    # Check V-scan consistency
    print("3. V-scan consistency:")
    ok3, msg3 = check_vscan_consistency()
    results.append(("V-scan consistency", ok3, msg3))
    print()
    
    # Check guard enforcement
    print("4. Guard enforcement:")
    ok4, msg4 = check_guard_enforcement()
    results.append(("Guard enforcement", ok4, msg4))
    print()
    
    # Summary
    print("=== Summary ===")
    all_pass = all(ok for _, ok, _ in results)
    
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"{name}: {status} - {msg}")
    
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    
    # Return exit code
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())