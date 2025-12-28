# TWC-6G-Networking

**Lyapunov-driven cross-layer optimization for low-bit-rate satellite / NTN semantic voice communications.**

## 1. Project Overview

This repository implements a **Lyapunov-driven online learning framework** for semantic-aware resource allocation in Non-Terrestrial Networks (NTN). The core problem addresses **per-slot continuous resource decision-making** (Transmission Power $P_t$, Bandwidth $B_t$) under **long-term constraints** (Queue Stability, Energy Budget, Semantic Distortion) within an environment characterized by **incomplete information** and **non-stationary channel dynamics**.

**Key Features:**
- **Online Learning, Not RL**: Solves the constrained optimization problem via a safety-aware bandit approach rather than deep reinforcement learning.
- **Strict Constraints**: Enforces hard safety constraints on semantic queue backlogs and energy consumption.
- **Reproducibility**: All results are generated via deterministic scripts and traceable CSV logs.

## 2. Algorithmic Framework

The solution architecture integrates **Lyapunov Optimization** with **Multi-Armed Bandits (MAB)** to decouple long-term coupled constraints into per-slot instantaneous sub-problems.

### 2.1 Lyapunov-based Decomposition
- **Outer Loop**: Drift-Plus-Penalty framework.
- **Virtual Queues**: Tracks violations for semantic data backlog ($Q^{sem}$) and energy usage ($Q^{eng}$).
- **Objective**: At each time slot $t$, minimize the upper bound of the drift-plus-penalty term:
  $$ \min_{P_t, B_t} V \cdot \text{Cost}(P_t, B_t) + Q^{sem}_t \cdot \Delta Q^{sem}_t + Q^{eng}_t \cdot \Delta Q^{eng}_t $$

### 2.2 Core Algorithm: RA-UCB++
**Resource-Aware UCB with Zooming & Safety** (`task3/algorithms/rucb_plus_impl/`)

RA-UCB++ is designed to handle the non-stationary bandit problem derived from the Lyapunov decomposition:
1.  **Sliding Window UCB (SW-UCB)**: Adapts to channel non-stationarity by discarding stale history.
2.  **Feasible Region Constraints**: Dynamically masks arms that violate immediate safety thresholds based on current queue states.
3.  **Arm Elimination & Zooming**: Progressively refines the discretization of the continuous $(P, B)$ space, eliminating suboptimal regions to accelerate convergence.
4.  **Change Detection**: Integrates **Page–Hinkley (PH)** test to reset bandit priors upon detecting abrupt environmental shifts.

### 2.3 Critical Implementation Details
- **Safety Shield**: A post-decision filter that overrides unsafe agent actions.
  - *Note*: The mapping between the *agent-selected arm* and the *actually executed action* is strictly tracked in logs to ensure valid off-policy updates.
- **Regret Definition**: Calculated against a **Best Fixed Feasible Arm Oracle**, which has full knowledge of channel statistics but respects the same safety constraints.
- **Non-stationarity**: In rapidly varying channels, strict $O(\log T)$ regret is not guaranteed; performance is evaluated against the dynamic oracle gap.

## 3. Experiments & Baselines

All experiments are conducted under **Task 3** specifications.

### 3.1 Baselines
The framework is benchmarked against the following algorithms:
*   **SafeOpt-GP**: Gaussian Process-based Bayesian Optimization with safety constraints.
*   **Safe-LinUCB**: Contextual linear bandit with safety barriers.
*   **SW-UCB**: Standard sliding-window UCB (baseline without explicit safety awareness).
*   **Lagrangian-PPO**: Proximal Policy Optimization with Lagrangian relaxation (RL baseline).
*   **Lyapunov-Greedy-Oracle**: Offline "teacher" model with perfect 1-step lookahead.

### 3.2 Experimental Setup
*   **Action Space**:
    *   Power $P_t \in [20, 33]$ dBm (Approximated via discrete arms + zooming).
    *   Bandwidth $B_t \in [5, 20]$ kHz.
*   **Lyapunov Parameter ($V$)**: Scanned to analyze the Trade-off between Cost (QoE) and Queue Stability.
*   **Environment**:
    *   Non-stationary fading channels.
    *   Stochastic packet arrivals and energy harvesting.

### 3.3 Evaluation Metrics
*   **Semantic Performance**: QoE (MOS), sWER (Semantic Word Error Rate).
*   **System Stability**: Queue Backlog, Energy Consumption, Constraint Violation Rate.
*   **Learning Efficiency**: Cumulative Regret, Decision Latency.

## 4. Repository Structure

This repository contains only implemented and verified modules.

```text
TWC-6G-Networking/
├── task3/
│   ├── algorithms/              # Core implementations
│   │   ├── run_experiments.py   # Main entry point for Task 3
│   │   ├── rucb_plus_impl/      # RA-UCB++ specific logic
│   │   ├── safeopt_gp.py        # SafeOpt baseline wrapper
│   │   ├── lagrangian_ppo.py    # PPO baseline wrapper
│   │   └── ...
│   ├── fig01_regret_vs_time/    # Figure 1 generation (Regret/Convergence)
│   │   ├── fig01_regret_vs_time.py
│   │   └── inputs/              # Local CSV dumps for plotting
│   └── ...
├── outputs/
│   ├── dumps/                   # Raw experiment logs (CSV)
│   └── figs/                    # Generated publication-ready figures
└── ...
```

## 5. Figures & Deliverables

Task 3 generates **18** distinct figures to validate the framework.

*   **Fig. 01: Regret vs Time**: Demonstrates convergence speed and cumulative regret against baselines.
*   **Steady-state Performance**: Metrics (QoE, Queue, Energy) vs Lyapunov $V$.
*   **Robustness**: Performance under different non-stationarity levels (Total Variation).
*   **Policy Visualization**: Heatmaps of Power-Bandwidth allocation policies.

**Note**: All figures are generated programmatically via the scripts in `task3/`. No manual plotting is involved.

## 6. Usage

### 6.1 Run Full Experiment (Task 3)
Execute the main experiment script to generate traces for all algorithms:

```bash
python task3/algorithms/run_experiments.py \
  --horizon 10000 \
  --V 800 --w 512 --g 10 \
  --seeds 0 1 2 \
  --algorithms raucb_plus sw_ucb safe_linucb safeopt_gp lagrangian_ppo best_fixed_arm_oracle lyapunov_greedy_oracle
```

### 6.2 Generate Figure 1 (Regret)
Process the generated logs to produce the Regret vs Time plot:

```bash
python task3/fig01_regret_vs_time/fig01_regret_vs_time.py --T 10000
```
*Output: `fig01_regret_vs_time.png`, `fig01_regret_vs_time_context_gap.png`*
