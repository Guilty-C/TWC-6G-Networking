
# Lyapunov + UCB for Semantic-Aware NTN Voice Communication

This repository contains the codebase for our ongoing research project targeting **IEEE TWC submission**.  
The project focuses on **cross-layer optimization of voice communication over 6G-NTN (direct GEO satellite links)**, with strong coupling between **application-layer voice coding (q_t)** and **physical-layer resources** (transmit power p_t, bandwidth allocation b_t, MCS m_t).

<!-- DOCS:START TOC -->
**Contents**
- [Getting Started](#getting-started)
- [V-scan Demo (UCB inner)](#v-scan-demo-ucb-inner)
- [Semantic Evaluation Pipeline](#semantic-evaluation-pipeline)
- [Regret 楠岃瘉锛堜笌楠屾敹鍙ｅ緞瀵归綈锛塢(#regret-楠岃瘉涓庨獙鏀跺彛寰勫榻?
- [閰嶇疆鏂囦欢浣嶇疆璇存槑](#閰嶇疆鏂囦欢浣嶇疆璇存槑)
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

杩愯渚濊禆杞荤殑鍐呭眰 UCB 妯″紡锛屽鐜?SWWER vs 1/V 浠ュ強 Q/J vs V锛?

```bash
python run_vscan.py --inner_mode ucb --config configs/vscan.yaml
```

**Outputs**

* `outputs/figs/Fig1_SWWER_vs_invV.png`
* `outputs/figs/Fig2_QJ_vs_V.png`
* `outputs/dumps/vscan_stats.csv`

> 璋冨弬鍏ュ彛锛歚configs/vscan.yaml` 涓殑 `V`, `Q_scale`, `J_scale`銆?
<!-- DOCS:END VScan -->

<!-- DOCS:START SemEval -->
## Semantic Evaluation Pipeline

绔埌绔涔夎瘎浼帮紙WER/SWWER CDF銆佺ǔ瀹氭€ф洸绾夸笌鏁版嵁瀵煎嚭锛夛細

```bash
python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml
```

杈撳嚭浣嶄簬 `semntn/outputs/{figs,dumps}/...`銆?
<!-- DOCS:END SemEval -->

<!-- DOCS:START Regret -->
## Regret 楠岃瘉锛堜笌楠屾敹鍙ｅ緞瀵归綈锛?

鏈€灏?K 鑷?bandit 楠岃瘉锛岀敓鎴?*绱閬楁喚鏇茬嚎**骞惰嚜鍔ㄥ垽鏂槸鍚︿笌鐞嗚 \(O(\log T)\) 闃舵暟鍖归厤锛?

```bash
python semntn/src/run_regret.py --config configs/regret.yaml
```

**Outputs**锛堥粯璁ょず渚嬶細`semntn/outputs/regret/exp001/`锛?

* `regret_curve.csv`
* `Regret_vs_T.png`
* `Regret_vs_logT.png`
* `Regret_over_logT.png`

**CLI 鎶ュ憡**锛堢ず渚嬶級锛?

```json
{"slope": 1.97, "C*": 8.42, "pass": true}
```

* `slope`锛氬 ((\log T, R(T))) 鐨勭嚎鎬у洖褰掓枩鐜囷紙鍙嶆槧瀵规暟绾у闀匡級銆?
* `C*`锛?\max_{t\le T}\frac{R(t)}{\sum_k (\ln t)/\Delta_k}) 鐨勪及璁★紝鐢ㄤ綔鈥滆竟鐣屽父鏁扳€濇寚鏍囥€?
  褰?`pass=true` 鏃讹紝琛ㄧず**绱閬楁喚澧為暱闃舵暟涓庣悊璁哄惢鍚?*锛屾弧瓒斥€滆竟鐣岃窡绱р€濈殑楠屾敹瑕佹眰銆?

> 鐩稿叧鍙傛暟锛堢畻娉曟棌/闃堝€?浣滃浘锛夊彲鍦?`configs/regret.yaml` 涓皟鑺傘€?
<!-- DOCS:END Regret -->

<!-- DOCS:START ConfigMap -->
## 閰嶇疆鏂囦欢浣嶇疆璇存槑

| 浣跨敤鍦烘櫙 | 閰嶇疆璺緞 | 鍏ュ彛鍛戒护 | 涓昏杈撳嚭 |
|---|---|---|---|
| V-scan锛圲CB inner锛?| `configs/vscan.yaml` | `python run_vscan.py --inner_mode ucb --config configs/vscan.yaml` | `outputs/figs/*`, `outputs/dumps/vscan_stats.csv` |
| 璇箟璇勪及 | `semntn/configs/sem_eval.yaml` | `python semntn/src/run_sem_eval.py --config semntn/configs/sem_eval.yaml` | `semntn/outputs/{figs,dumps}/...` |
| Regret 楠岃瘉 | `configs/regret.yaml` | `python semntn/src/run_regret.py --config configs/regret.yaml` | `semntn/outputs/regret/...` |

> 绾﹀畾锛氬揩閫熷疄楠屼紭鍏堟斁鍦ㄤ粨搴撴牴鐨?`configs/`锛涚鍒扮璇箟璇勪及鐨勯厤缃斁鍦?`semntn/configs/`銆?
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
鈹溾攢鈹€ semntn/              # Main project code
鈹?  鈹溾攢鈹€ configs/         # YAML configs for experiments
鈹?  鈹溾攢鈹€ src/             # Core source code
鈹?  鈹溾攢鈹€ data/            # Shared channel traces
鈹?  鈹溾攢鈹€ outputs/         # (ignored) experiment outputs
鈹?  鈹斺攢鈹€ requirements.txt # Python dependencies
鈹斺攢鈹€ .gitignore

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

<!-- TWC6G: TASK1_GUIDE START -->
## Task-1 (PD9) 鈥?How We Achieved and Verified PASS

### 1) Problem & Acceptance (PD9 gates)
- **Preflight**: feasible_intersection_rate 鈮?0.30.
- **Unit/Identity**: coverage of (B_eff_Hz 鈮?B_min_kHz脳1000) 鈮?0.99; print first violating sample on failure.
- **V-scan Consistency**: summary.csv has [agg_version, warmup_skip] and rows 鈮?3.
- **Guard Enforcement**: fallback_used.sum == 0 AND (~action_in_guard_set).sum == 0; print first leakage sample on failure.

### 2) Final PASS Baseline (from configs & preflight)
- arrivals_bps: **400.0**
- semantic_budget: **0.70**
- pd.delta_queue: **0.00**
- P_max_dBm: **47.0**
- B_min_kHz: **6.25**
- B_grid_k: **16**

### 3) Implementation Highlights
- **Guard projection**: escalate B along B_grid when window empty or queue guard infeasible, then project P; recompute guards using (P*,B*).
- **Unit/Identity**: compute `B_eff_Hz` strictly from emitted (P*,B*); enforce numeric types; NaN is immediate violation with first-sample print.
- **V-scan**: write `agg_version` and `warmup_skip`; keep row gate (鈮?3) consistent with analyzer.

### 4) Reproduce (one-shot RUNBOOK)
```bash
rm -rf outputs/vscan || true
python -m semntn.src.pd9_preflight --config configs/task1_release_pd2.yaml
python -m semntn.src.run_task1 --config configs/task1_wirecheck.yaml --mode wirecheck
python -m semntn.src.run_task1 --config configs/task1_release_pd2.yaml --mode release
python -m semntn.src.run_vscan --base-config configs/task1_release_pd2.yaml --vscan-config configs/vscan.yaml
python -m semntn.src.analyze_task1 --outdir outputs
```
<!-- TWC6G: TASK1_GUIDE END -->
