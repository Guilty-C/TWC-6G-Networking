"""CLI to train physics-informed PESQ surrogate models and produce reports."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

try:  # pragma: no cover - import hook for script execution
    from .model_family_report import ModelMetrics, render_markdown_report
    from .pesq_surrogate import FeatureSpec, ModelFamily, PhysicsInformedSurrogate, train_test_split
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from model_family_report import ModelMetrics, render_markdown_report
    from pesq_surrogate import FeatureSpec, ModelFamily, PhysicsInformedSurrogate, train_test_split


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_feature_specs(config: Dict[str, object]) -> List[FeatureSpec]:
    features_cfg = config["data"]["features"]
    specs: List[FeatureSpec] = []
    for item in features_cfg:
        specs.append(
            FeatureSpec(
                name=item["name"],
                monotonic=item.get("monotonic", "none"),
                lower=item.get("lower"),
                upper=item.get("upper"),
            )
        )
    return specs


def _parse_families(config: Dict[str, object]) -> List[ModelFamily]:
    families_cfg = config["model"]["families"]
    families: List[ModelFamily] = []
    for item in families_cfg:
        params = {k: v for k, v in item.items() if k not in {"name", "type"}}
        families.append(ModelFamily(name=item["name"], type=item["type"], parameters=params))
    return families


def run(config_path: Path) -> None:
    config = _load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    ver_cfg = config["verification"]

    csv_path_cfg = Path(data_cfg["csv_path"])
    if not csv_path_cfg.is_absolute():
        candidate = (config_path.parent / csv_path_cfg).resolve()
        if candidate.exists():
            csv_path = candidate
        else:
            csv_path = (config_path.parent.parent / csv_path_cfg).resolve()
    else:
        csv_path = csv_path_cfg

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate dataset at {csv_path}")

    df = pd.read_csv(csv_path)

    target = data_cfg["target"]
    feature_specs = _parse_feature_specs(config)
    families = _parse_families(config)

    train_df, test_df = train_test_split(
        df,
        test_size=float(ver_cfg["test_size"]),
        seed=int(ver_cfg.get("random_seed", 0)),
    )

    regularization = float(model_cfg.get("regularization", 0.0))
    learning_rate = float(model_cfg.get("learning_rate", 0.01))
    max_iter = int(model_cfg.get("max_iter", 2000))
    tol = float(model_cfg.get("tol", 1e-6))

    metrics: List[ModelMetrics] = []
    models: Dict[str, PhysicsInformedSurrogate] = {}
    best_family_name = None
    best_rmse = float("inf")

    for family in families:
        model = PhysicsInformedSurrogate(
            feature_specs=feature_specs,
            family=family,
            regularization=regularization,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
        )
        model.fit(train_df, target)
        train_metrics = model.evaluate(train_df, target)
        test_metrics = model.evaluate(test_df, target)

        exported = model.export_parameters()
        metrics.append(
            ModelMetrics(
                family=family.name,
                type=family.type,
                num_basis=len(exported["basis_names"]),
                iterations=int(exported["iterations"]),
                train_rmse=train_metrics["rmse"],
                train_mae=train_metrics["mae"],
                train_r2=train_metrics["r2"],
                test_rmse=test_metrics["rmse"],
                test_mae=test_metrics["mae"],
                test_r2=test_metrics["r2"],
            )
        )
        models[family.name] = model

        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]
            best_family_name = family.name

    if best_family_name is None:
        raise RuntimeError("No model families were evaluated")

    output_dir_cfg = Path(ver_cfg["output_dir"])
    if not output_dir_cfg.is_absolute():
        candidate_out = (config_path.parent / output_dir_cfg).resolve()
        if candidate_out.exists() or candidate_out.parent.exists():
            output_dir = candidate_out
        else:
            output_dir = (config_path.parent.parent / output_dir_cfg).resolve()
    else:
        output_dir = output_dir_cfg
    curves_dir = output_dir / "curves"
    best_model = models[best_family_name]
    curves = best_model.generate_single_variable_curves(
        df=df,
        target=target,
        output_dir=curves_dir,
        curve_points=int(ver_cfg.get("curve_points", 60)),
        bins=int(ver_cfg.get("bins", 12)),
    )

    params_path = output_dir / f"{best_family_name}_parameters.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        yaml.safe_dump(best_model.export_parameters(), sort_keys=False),
        encoding="utf-8",
    )

    report_path = render_markdown_report(
        metrics=metrics,
        best_family=best_family_name,
        curves=[path.relative_to(output_dir) for path in curves],
        output_dir=output_dir,
        config_path=config_path,
    )

    print(f"Saved model comparison report to {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
