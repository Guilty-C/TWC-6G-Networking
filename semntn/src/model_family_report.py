"""Utilities to compile PESQ surrogate model comparison reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd


@dataclass
class ModelMetrics:
    """Summary metrics for one model family."""

    family: str
    type: str
    num_basis: int
    iterations: int
    train_rmse: float
    train_mae: float
    train_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float

    def as_row(self) -> List[object]:
        return [
            self.family,
            self.type,
            self.num_basis,
            self.iterations,
            self.train_rmse,
            self.train_mae,
            self.train_r2,
            self.test_rmse,
            self.test_mae,
            self.test_r2,
        ]


def _metrics_to_dataframe(metrics: Sequence[ModelMetrics]) -> pd.DataFrame:
    columns = [
        "family",
        "type",
        "num_basis",
        "iterations",
        "train_rmse",
        "train_mae",
        "train_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]
    data = [metric.as_row() for metric in metrics]
    df = pd.DataFrame(data, columns=columns)
    return df.sort_values("test_rmse").reset_index(drop=True)


def render_markdown_report(
    metrics: Sequence[ModelMetrics],
    best_family: str,
    curves: Sequence[Path],
    output_dir: Path,
    config_path: Path | None = None,
) -> Path:
    """Render a Markdown report comparing model families."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = _metrics_to_dataframe(metrics)

    def _format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _dataframe_to_markdown(frame: pd.DataFrame) -> List[str]:
        headers = list(frame.columns)
        header_line = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for _, row in frame.iterrows():
            formatted = [_format_value(row[col]) for col in headers]
            rows.append("| " + " | ".join(formatted) + " |")
        return [header_line, separator, *rows]

    md_lines: List[str] = ["# PESQ Surrogate Model Comparison", ""]
    if config_path is not None:
        md_lines.append(f"- Configuration: `{config_path}`")
    md_lines.append(f"- Best model family: **{best_family}**")
    md_lines.append("")

    md_lines.append("## Quantitative summary")
    md_lines.append("")
    md_lines.extend(_dataframe_to_markdown(df))
    md_lines.append("")

    if curves:
        md_lines.append("## Verification curves")
        md_lines.append("")
        for path in curves:
            md_lines.append(f"![{path.stem}]({path.name})")
        md_lines.append("")

    report_path = output_dir / "model_family_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_path
