
# Lyapunov + UCB for Semantic-Aware NTN Voice Communication

This repository contains the codebase for our ongoing research project targeting **IEEE TWC submission**.  
The project focuses on **cross-layer optimization of voice communication over 6G-NTN (direct GEO satellite links)**, with strong coupling between **application-layer voice coding (q_t)** and **physical-layer resources** (transmit power p_t, bandwidth allocation b_t, MCS m_t).

<!-- DOCS:START TOC -->
**Contents**
- [Getting Started](#getting-started)
- [V-scan Demo (UCB inner)](#v-scan-demo-ucb-inner)
- [Semantic Evaluation Pipeline](#semantic-evaluation-pipeline)
- [Regret 验证（与验收口径对齐）](#regret-验证与验收口径对齐)
- [配置文件位置说明](#配置文件位置说明)
<!-- DOCS:END TOC -->

<!-- DOCS:START GettingStarted -->
## Getting Started

### 1) Environment
Python 3.10 recommended.

```bash
# create & activate venv
python -m venv .venv
# Windows
.\\.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

# install deps
pip install -r semntn/requirements.txt
```
<!-- DOCS:END GettingStarted -->

<!-- DOCS:START VScan -->
## V-scan Demo (UCB inner)

运行依赖轻的内层 UCB 模式，复现 SWWER vs 1/V 以及 Q/J vs V：

```bash
python run_vscan.py --inner_mode ucb --config configs/vscan.yaml
```

**Outputs**

* `outputs/figs/Fig1_SWWER_vs_invV.png`
* `outputs/figs/Fig2_QJ_vs_V.png`
* `outputs/dumps/vscan_stats.csv`

> 调参入口：`configs/vscan.yaml` 中的 `V`, `Q_scale`, `J_scale`。
<!-- DOCS:END VScan -->

<!-- DOCS:START SemEval -->
## Semantic Evaluation Pipeline

端到端语义评估（WER/SWWER CDF、稳定性曲线与数据导出）：

```bash
python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml
```

输出位于 `semntn/outputs/{figs,dumps}/...`。
<!-- DOCS:END SemEval -->

<!-- DOCS:START Regret -->
## Regret 验证（与验收口径对齐）

最小 K 臂 bandit 验证，生成**累计遗憾曲线**并自动判断是否与理论 \(O(\log T)\) 阶数匹配：

```bash
python semntn/src/run_regret.py --config configs/regret.yaml
```

**Outputs**（默认示例：`semntn/outputs/regret/exp001/`）

* `regret_curve.csv`
* `Regret_vs_T.png`
* `Regret_vs_logT.png`
* `Regret_over_logT.png`

**CLI 报告**（示例）：

```json
{"slope": 1.97, "C*": 8.42, "pass": true}
```

* `slope`：对 ((\log T, R(T))) 的线性回归斜率（反映对数级增长）。
* `C*`：(\max_{t\le T}\frac{R(t)}{\sum_k (\ln t)/\Delta_k}) 的估计，用作“边界常数”指标。
  当 `pass=true` 时，表示**累计遗憾增长阶数与理论吻合**，满足“边界跟紧”的验收要求。

> 相关参数（算法族/阈值/作图）可在 `configs/regret.yaml` 中调节。
<!-- DOCS:END Regret -->

<!-- DOCS:START ConfigMap -->
## 配置文件位置说明

| 使用场景 | 配置路径 | 入口命令 | 主要输出 |
|---|---|---|---|
| V-scan（UCB inner） | `configs/vscan.yaml` | `python run_vscan.py --inner_mode ucb --config configs/vscan.yaml` | `outputs/figs/*`, `outputs/dumps/vscan_stats.csv` |
| 语义评估 | `semntn/configs/sem_eval.yaml` | `python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml` | `semntn/outputs/{figs,dumps}/...` |
| Regret 验证 | `configs/regret.yaml` | `python semntn/src/run_regret.py --config configs/regret.yaml` | `semntn/outputs/regret/...` |

> 约定：快速实验优先放在仓库根的 `configs/`；端到端语义评估的配置放在 `semntn/configs/`。
<!-- DOCS:END ConfigMap -->

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
