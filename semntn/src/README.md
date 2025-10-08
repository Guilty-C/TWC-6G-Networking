# SEMNTN 项目源代码目录

## PESQ 代理建模工具

- `pesq_surrogate.py`：包含具有物理约束和单调形状约束的PESQ代理模型，实现训练、评估、单变量验证曲线生成等功能。
- `model_family_report.py`：负责将不同模型族的量化指标和验证图表汇总成Markdown报告。
- `run_pesq_surrogate.py`：命令行入口，读取YAML配置，训练多个模型族并生成报告和验证曲线。

运行示例：

```bash
python semntn/src/run_pesq_surrogate.py --config semntn/configs/pesq_surrogate.yaml
```

运行脚本后会在 `semntn/reports/pesq_surrogate/` 目录下生成最优模型参数、单变量验证曲线以及模型族比较报告。该目录已在仓库 `.gitignore` 中忽略，避免将自动产出的曲线（PNG）等二进制文件提交至代码库。
