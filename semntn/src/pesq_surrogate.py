"""Physics-informed, shape-constrained PESQ surrogate modelling tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

FeatureMonotonicity = Literal["increasing", "decreasing", "none"]


@dataclass
class FeatureSpec:
    """Describe one input feature and its physical constraints."""

    name: str
    monotonic: FeatureMonotonicity = "none"
    lower: float | None = None
    upper: float | None = None

    def as_dict(self) -> Dict[str, float | str | None]:
        """Return a serialisable representation of the feature specification."""

        return {
            "name": self.name,
            "monotonic": self.monotonic,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass
class ModelFamily:
    """Configuration for a model family to be evaluated."""

    name: str
    type: str
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class FitHistory:
    """Keep track of optimisation progress for debugging/reporting."""

    losses: List[float]


@dataclass
class FitResult:
    """Structured result of fitting the surrogate model."""

    intercept: float
    weights: np.ndarray
    basis_names: List[str]
    history: FitHistory
    iterations: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "intercept": float(self.intercept),
            "weights": [float(w) for w in self.weights],
            "basis_names": self.basis_names,
            "history": {"losses": [float(v) for v in self.history.losses]},
            "iterations": int(self.iterations),
        }


class PhysicsInformedSurrogate:
    """PESQ surrogate model with monotonic shape constraints."""

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        family: ModelFamily,
        regularization: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 2000,
        tol: float = 1e-6,
        target_bounds: Tuple[float, float] = (1.0, 4.5),
    ) -> None:
        self.feature_specs = list(feature_specs)
        self.family = family
        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.target_bounds = target_bounds

        self._intercept: float | None = None
        self._weights: np.ndarray | None = None
        self._basis_names: List[str] = []
        self._weight_signs: List[Literal["positive", "negative", "free"]] = []
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._history: FitHistory | None = None
        self._iterations: int = 0

    # ------------------------------------------------------------------
    # Normalisation utilities
    # ------------------------------------------------------------------
    def _record_feature_stats(self, df: pd.DataFrame) -> None:
        for spec in self.feature_specs:
            series = df[spec.name]
            self._feature_stats[spec.name] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0) or 1.0),
            }

    def _normalise(self, values: np.ndarray, spec: FeatureSpec) -> np.ndarray:
        stats = self._feature_stats[spec.name]
        lower = spec.lower if spec.lower is not None else stats["min"]
        upper = spec.upper if spec.upper is not None else stats["max"]
        scale = upper - lower
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            scale = stats["std"] or 1.0
        centred = values - lower
        normalised = centred / scale
        return np.clip(normalised, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------
    def _sign_for_feature(self, spec: FeatureSpec) -> Literal["positive", "negative", "free"]:
        if spec.monotonic == "increasing":
            return "positive"
        if spec.monotonic == "decreasing":
            return "negative"
        return "free"

    def _linear_basis(self, values: np.ndarray) -> List[np.ndarray]:
        return [values, values ** 2, np.sqrt(values + 1e-12)]

    def _logistic_basis(self, values: np.ndarray, scale: float) -> List[np.ndarray]:
        shifted = values - 0.5
        logistic = 1.0 / (1.0 + np.exp(-scale * shifted))
        return [values, logistic, logistic ** 2]

    def _rational_basis(self, values: np.ndarray) -> List[np.ndarray]:
        eps = 1e-6
        return [values, values / (values + 1.0 + eps), values / (values + 0.5 + eps)]

    def _build_design_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        basis_columns: List[np.ndarray] = []
        basis_names: List[str] = []
        weight_signs: List[Literal["positive", "negative", "free"]] = []

        family_type = self.family.type.lower()
        scale = float(self.family.parameters.get("logistic_scale", 5.0))

        for spec in self.feature_specs:
            values = self._normalise(df[spec.name].to_numpy(dtype=float), spec)
            sign = self._sign_for_feature(spec)
            if family_type == "linear":
                bases = self._linear_basis(values)
                names = [f"{spec.name}_linear", f"{spec.name}_quadratic", f"{spec.name}_sqrt"]
            elif family_type == "logistic":
                bases = self._logistic_basis(values, scale)
                names = [f"{spec.name}_lin", f"{spec.name}_logistic", f"{spec.name}_logistic_sq"]
            elif family_type == "rational":
                bases = self._rational_basis(values)
                names = [f"{spec.name}_lin", f"{spec.name}_r1", f"{spec.name}_r2"]
            else:
                raise ValueError(f"Unsupported model family type: {self.family.type}")

            basis_columns.extend(bases)
            basis_names.extend(names)
            weight_signs.extend([sign] * len(bases))

        design_matrix = np.column_stack(basis_columns) if basis_columns else np.empty((len(df), 0))
        self._weight_signs = weight_signs
        return design_matrix, basis_names

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def _project_weights(self, weights: np.ndarray) -> None:
        for idx, sign in enumerate(self._weight_signs):
            if sign == "positive":
                weights[idx] = max(weights[idx], 0.0)
            elif sign == "negative":
                weights[idx] = min(weights[idx], 0.0)

    def fit(self, df: pd.DataFrame, target: str) -> FitResult:
        """Fit the surrogate model to the provided data."""

        if target not in df:
            raise KeyError(f"Target column '{target}' not present in data frame")

        self._record_feature_stats(df)
        X, basis_names = self._build_design_matrix(df)
        y = df[target].to_numpy(dtype=float)

        n_samples = len(df)
        n_features = X.shape[1]
        intercept = np.array(0.5 * (self.target_bounds[0] + self.target_bounds[1]), dtype=float)
        weights = np.zeros(n_features, dtype=float)
        history: List[float] = []

        for iteration in range(1, self.max_iter + 1):
            preds = intercept + X @ weights
            preds = np.clip(preds, self.target_bounds[0], self.target_bounds[1])
            residual = preds - y
            loss = 0.5 * float(np.mean(residual ** 2)) + 0.5 * self.regularization * float(np.dot(weights, weights))
            history.append(loss)

            grad_intercept = float(np.mean(residual))
            grad_weights = (X.T @ residual) / n_samples + self.regularization * weights

            intercept -= self.learning_rate * grad_intercept
            weights -= self.learning_rate * grad_weights
            self._project_weights(weights)

            grad_norm = float(np.linalg.norm(np.concatenate(([grad_intercept], grad_weights))))
            if grad_norm * self.learning_rate < self.tol:
                self._iterations = iteration
                break
        else:
            self._iterations = self.max_iter

        self._intercept = float(intercept)
        self._weights = weights
        self._basis_names = basis_names
        self._history = FitHistory(history)

        return FitResult(
            intercept=self._intercept,
            weights=self._weights.copy(),
            basis_names=basis_names,
            history=self._history,
            iterations=self._iterations,
        )

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------
    def _ensure_fitted(self) -> None:
        if self._intercept is None or self._weights is None:
            raise RuntimeError("Model must be fitted before prediction")

    def _design_matrix_from_values(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_design_matrix(df)
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict PESQ scores for a new dataframe."""

        self._ensure_fitted()
        X = self._design_matrix_from_values(df)
        preds = self._intercept + X @ self._weights
        return np.clip(preds, self.target_bounds[0], self.target_bounds[1])

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Compute standard regression metrics on a dataset."""

        preds = self.predict(df)
        y = df[target].to_numpy(dtype=float)
        residual = preds - y
        mse = float(np.mean(residual ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residual)))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(residual ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ------------------------------------------------------------------
    # Visual verification
    # ------------------------------------------------------------------
    def generate_single_variable_curves(
        self,
        df: pd.DataFrame,
        target: str,
        output_dir: Path,
        curve_points: int = 50,
        bins: int = 10,
    ) -> List[Path]:
        """Generate verification curves for each feature and save plots."""

        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        centres_records: Dict[str, pd.DataFrame] = {}
        saved_paths: List[Path] = []

        base_row = {spec.name: float(df[spec.name].median()) for spec in self.feature_specs}

        for spec in self.feature_specs:
            xs = np.linspace(float(df[spec.name].min()), float(df[spec.name].max()), curve_points)
            inputs = []
            for val in xs:
                row = base_row.copy()
                row[spec.name] = float(val)
                inputs.append(row)
            curve_df = pd.DataFrame(inputs)
            preds = self.predict(curve_df)

            binned = df[[spec.name, target]].copy()
            binned["bin"] = pd.cut(binned[spec.name], bins=bins, duplicates="drop")
            grouped = binned.groupby("bin", observed=True)[target].agg(["mean", "count"])
            grouped["centre"] = grouped.index.map(lambda interval: interval.mid)
            centres_records[spec.name] = grouped

            fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
            ax.plot(xs, preds, label="Surrogate", color="#0072B2", linewidth=2.0)
            ax.scatter(df[spec.name], df[target], s=15, alpha=0.3, label="Samples", color="#999999")
            ax.scatter(grouped["centre"], grouped["mean"], s=40, color="#D55E00", label="Binned mean")
            ax.set_xlabel(spec.name.replace("_", " "))
            ax.set_ylabel("PESQ")
            ax.set_title(f"{self.family.name}: {spec.name}")
            ax.set_ylim(self.target_bounds[0] - 0.1, self.target_bounds[1] + 0.1)
            ax.grid(True, alpha=0.2)
            ax.legend()

            file_path = output_dir / f"{self.family.name}_{spec.name}.png"
            fig.tight_layout()
            fig.savefig(file_path)
            plt.close(fig)
            saved_paths.append(file_path)

            grouped.to_csv(output_dir / f"{self.family.name}_{spec.name}_bins.csv")

        return saved_paths

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def export_parameters(self) -> Dict[str, object]:
        """Return a serialisable snapshot of the fitted model."""

        self._ensure_fitted()
        assert self._intercept is not None
        assert self._weights is not None
        return {
            "family": self.family.name,
            "type": self.family.type,
            "intercept": float(self._intercept),
            "weights": [float(x) for x in self._weights],
            "basis_names": list(self._basis_names),
            "weight_signs": list(self._weight_signs),
            "feature_stats": self._feature_stats,
            "history": self._history.losses if self._history else [],
            "iterations": self._iterations,
            "target_bounds": list(self.target_bounds),
        }


def train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple train-test split implementation without external dependencies."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(len(df) * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
