
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
- **2025-10-11**: Improved the ucb algorithm. Fixed the graphs.
  <img width="1046" height="615" alt="a4fd636d762a38540091dc5326ea364f" src="https://github.com/user-attachments/assets/6da54835-0d65-4305-828d-b06f950a9248" />
<img width="1041" height="634" alt="429b3e692f335e65c14916a7944b0513" src="https://github.com/user-attachments/assets/c22f2f5b-9d64-47c3-9b6f-c2d8d7c9b3a9" />

## Next Steps
- Refine PER/SINR mapping with more realistic channel models.
- Integrate true speech codec traces for stronger evaluation.
- Prepare reproducibility package (configs + scripts + logs).

### Regret 验证（与验收口径对齐）

运行一个最小的 K 臂 bandit 验证，生成累计遗憾曲线，并自动判断与理论阶数 \(O(\log T)\) 的匹配程度：

```bash
python semntn/src/run_regret.py --config configs/regret.yaml
```

输出目录：`semntn/outputs/regret/exp001/`

- `regret_curve.csv`
- `Regret_vs_T.png`
- `Regret_vs_logT.png`

命令行会打印形如：
```json
{"slope": 1.97, "C*": 8.42, "pass": true}
```

- `slope` 基于 \(R(T)\) 对 \(\log T\) 的线性回归斜率（对应对数级增长）。
- `C*` 为 \(\max_{t\le T} \frac{R(t)}{\sum_k (\ln t)/\Delta_k}\) 的估计值，用作“边界常数”指标。  
当 `pass=true` 时，表示**累计遗憾增长阶数与理论吻合**，满足“边界跟紧”的验收要求。
