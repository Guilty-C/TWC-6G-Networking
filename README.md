
# Lyapunov + UCB for Semantic-Aware NTN Voice Communication

This repository contains the codebase for our ongoing research project targeting **IEEE TWC submission**.  
The project focuses on **cross-layer optimization of voice communication over 6G-NTN (direct GEO satellite links)**, with strong coupling between **application-layer voice coding (q_t)** and **physical-layer resources** (transmit power p_t, bandwidth allocation b_t, MCS m_t).

## Research Scope
- **Scenario**: Voice-over-NTN with semantic-aware QoE (streaming ASR + lightweight semantic weighting).
- **Objective**: Long-term minimization of **Semantic-Weighted WER (SWWER)** under constraints (energy, delay, distortion).
- **Methodology**:  
  - Lyapunov drift-plus-penalty for handling long-term constraints.  
  - **AI-assisted UCB (Sem-UCB++)** for inner-slot decisions (q, p, b, m).  
  - Lightweight LLM module (LLM-Lite) for semantic importance extraction.  

## Structure
```

6G/
├── semntn/              # Main project code
│   ├── configs/         # YAML configs for experiments
│   ├── src/             # Core source code
│   ├── data/            # Shared channel traces
│   ├── outputs/         # (ignored) experiment outputs
│   └── requirements.txt # Python dependencies
└── .gitignore

````

## Quick Start
To launch the end-to-end semantic evaluation workflow, run the following command from the repository root:

```bash
python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml
```

This command produces bucketed WER/SWWER CDF plots, stability curves, and raw metric dumps under `semntn/outputs/.../{dumps,figs}` as specified in the configuration file.

## Key Experiments
1. **V-scan**: SWWER vs 1/V, Q/J vs V.
2. **Semantic weighting**: WER-CDF by semantic bucket.
3. **UCB regret curves** (extension, to be run jointly).

## Current Deliverables
- **Student A (Zhu Yizhen)**: Semantic weighting module (LLM-Lite) + Lyapunov outer loop + mock inner API.
- **Student B (Guo Fangyu)**: UCB-based inner API (inner_api_ucb.py), integrated with Lyapunov loop.
- **Physics-Informed PESQ Surrogate**: End-to-end tooling for training, evaluating, and reporting constrained PESQ surrogate models with single-variable verification curves.

## Progress Log
- **2025-10-08**: Completed physics-informed PESQ surrogate modeling workflow, including model family comparison reporting, configuration-driven CLI orchestration, and automatic exclusion of generated binary artefacts from version control.
- **2025-10-09**: Documented the quick-start command and expected outputs for the semantic evaluation pipeline.

## Next Steps
- Refine PER/SINR mapping with more realistic channel models.
- Integrate true speech codec traces for stronger evaluation.
- Prepare reproducibility package (configs + scripts + logs).
