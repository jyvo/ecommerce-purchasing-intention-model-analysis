from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config, features, persistence, data as data_io
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def parse_curve(value) -> np.ndarray:
    """one stored ROC array, as floats"""
    if not isinstance(value, str):
        return np.asarray(value, dtype=float)
    if "..." in value:
        raise ValueError("stored ROC array is elided ('...') and cannot be reconstructed; re-run `uv run pi-pipeline`")
    return np.fromstring(value.strip().lstrip("[").rstrip("]"), sep=" ")


def attach_confusion_counts(results: pd.DataFrame, *, n_total: int, n_positive: int, tolerance: float = 1e-6) -> pd.DataFrame:
    """Add TN | FP | FN | TP columns to a grid of metrics"""
    n_negative = n_total - n_positive
    frame = results.copy()

    tp = (frame["Recall"] * n_positive).round()
    tn = (frame["Accuracy"] * n_total).round() - tp
    fp = n_negative - tn
    fn = n_positive - tp

    flagged = (tp + fp).replace(0, np.nan)
    precision = (tp / flagged).fillna(0.0)
    denominator = (precision + frame["Recall"]).replace(0, np.nan)
    f1 = (2 * precision * frame["Recall"] / denominator).fillna(0.0)

    for name, recomputed, stored in (
        ("precision", precision, frame["Precision"]),
        ("F1", f1, frame["F1-Score"]),
    ):
        worst = float((recomputed - stored).abs().max())
        if worst > tolerance:
            raise AssertionError(
                f"confusion-matrix reconstruction disagrees with the stored "
                f"{name} by {worst:.2e} (> {tolerance:.0e}). The partition "
                f"totals used ({n_positive} of {n_total}) may not be the ones "
                "these metrics were computed on."
            )

    frame["TP"] = tp.astype(int)
    frame["TN"] = tn.astype(int)
    frame["FP"] = fp.astype(int)
    frame["FN"] = fn.astype(int)
    return frame


@dataclass(frozen=True)
class GridSpec:
    """one arm x one feature set (its results, its sweep and its bundle)"""

    key: str
    arm: str
    tier: str
    feature_set: str
    label: str
    results_csv: Path
    sweep_csv: Path
    bundle_path: Path


GRID_SPECS: tuple[GridSpec, ...] = (
    GridSpec(
        key="production_full",
        arm=config.ARM_PRODUCTION,
        tier=config.TIER_PRODUCTION,
        feature_set="full",
        label="Production (telemetry only, all 64 features)",
        results_csv=config.RESULTS_TELEMETRY_CSV,
        sweep_csv=config.THRESHOLD_SWEEP_TELEMETRY_CSV,
        bundle_path=config.PRODUCTION_FULL_BUNDLE,
    ),
    GridSpec(
        key="production_top10",
        arm=config.ARM_PRODUCTION,
        tier=config.TIER_PRODUCTION,
        feature_set="top10",
        label="Production (telemetry only, top-10 selected)",
        results_csv=config.RESULTS_TELEMETRY_TOP10_CSV,
        sweep_csv=config.THRESHOLD_SWEEP_TELEMETRY_TOP10_CSV,
        bundle_path=config.PRODUCTION_TOP10_BUNDLE,
    ),
    GridSpec(
        key="ceiling_full",
        arm=config.ARM_ANALYSIS,
        tier=config.TIER_ANALYSIS,
        feature_set="full",
        label="Analysis ceiling (with PageValues, all 65 features)",
        results_csv=config.RESULTS_FULL_CSV,
        sweep_csv=config.THRESHOLD_SWEEP_CSV,
        bundle_path=config.CEILING_FULL_BUNDLE,
    ),
    GridSpec(
        key="ceiling_top10",
        arm=config.ARM_ANALYSIS,
        tier=config.TIER_ANALYSIS,
        feature_set="top10",
        label="Analysis ceiling (with PageValues, top-10 selected)",
        results_csv=config.RESULTS_TOP10_CSV,
        sweep_csv=config.THRESHOLD_SWEEP_TOP10_CSV,
        bundle_path=config.CEILING_TOP10_BUNDLE,
    ),
)


def grid_key(tier: str, feature_set: str) -> str:
    """GRID_SPECS key for a tier and feature set"""
    prefix = "production" if tier == config.TIER_PRODUCTION else "ceiling"
    return f"{prefix}_{'full' if feature_set == 'full' else 'top10'}"



@dataclass
class Grid:
    """loaded grid: 19 configurations (threshold sweep and its bundle)"""

    spec: GridSpec
    results: pd.DataFrame
    sweep: pd.DataFrame
    bundle: dict

    @property
    def winner(self) -> pd.Series:
        """the configuration this grid's bundle was fitted from"""
        meta = self.bundle["metadata"]
        entry = self.results[
            (self.results["Model"] == meta["model"])
            & (self.results["SamplingTechnique"] == meta["sampling_regime"])
        ]
        return entry.iloc[0]

    @property
    def winner_label(self) -> str:
        meta = self.bundle["metadata"]
        return f"{meta['model']} / {meta['sampling_regime']}"



@dataclass
class DatasetStages:
    """the three correlation matrices and the partition the grids were scored on"""

    corr_raw: pd.DataFrame
    corr_engineered: pd.DataFrame
    corr_pruned: pd.DataFrame
    train_medians: pd.Series
    n_total: int
    n_positive: int

    @property
    def base_rate(self) -> float:
        return self.n_positive / self.n_total


def dataset_stages() -> DatasetStages:
    """replay the deterministic half of the pipeline: fetch, engineer, encode, split"""
    frame, _ = data_io.fetch_dataset()
    frame = frame.copy()

    corr_raw = features.correlation_matrix(frame)
    frame = features.engineer_features(frame)
    corr_engineered = features.correlation_matrix(frame)
    frame = features.eliminate_correlated(frame)
    corr_pruned = features.correlation_matrix(frame)

    frame = features.encode(frame)
    X = frame.drop(config.TARGET, axis=1)
    y = frame[config.TARGET]
    X_train, _, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return DatasetStages(
        corr_raw=corr_raw,
        corr_engineered=corr_engineered,
        corr_pruned=corr_pruned,
        train_medians=X_train.median(),
        n_total=int(len(y_test)),
        n_positive=int(y_test.sum()),
    )


def importance_frame(bundle: dict) -> tuple[pd.DataFrame | None, str]:
    """(frame, kind) for a bundle's fitted classifier, or (None, reason)
        - nothing is refitted
        - only the four persisted bundles carry a fitted estimator
    """
    clf = bundle["pipeline"].named_steps["clf"]
    names = bundle["selected_features"] or bundle["feature_order"]

    values = getattr(clf, "feature_importances_", None)
    if values is not None:
        kind = config.IMPORTANCE_KIND_TREE
    else:
        coef = getattr(clf, "coef_", None)
        if coef is None:
            return None, (
                f"{type(clf).__name__} exposes neither feature_importances_ nor "
                "coef_; a distance-based or kernel model has no per-feature "
                "attribution to show, and refitting a surrogate in the app would "
                "publish a number no artifact contains."
            )
        values = np.asarray(coef).ravel()
        kind = config.IMPORTANCE_KIND_LINEAR

    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(names):
        return None, (
            f"estimator reports {len(values)} attributions against "
            f"{len(names)} contracted feature names; refusing to guess the mapping"
        )
    return pd.DataFrame({"Feature": list(names), "Value": values}), kind



@dataclass
class AppData:
    """everything the render targets need (loaded once)"""

    grids: dict[str, Grid]
    comparison: pd.DataFrame
    smote: pd.DataFrame
    stages: DatasetStages
    warnings: list[str] = field(default_factory=list)

    @property
    def production(self) -> Grid:
        """the bundle the prediction form scores against"""
        return self.grids["production_full"]

    @property
    def ceiling(self) -> Grid:
        return self.grids["ceiling_full"]

    def grid(self, tier: str, feature_set: str) -> Grid:
        return self.grids[grid_key(tier, feature_set)]

    @property
    def headline(self) -> dict:
        """the paired headline AUCs + deployability reason (if any)"""
        return {
            "production_auc": float(self.production.bundle["metadata"]["metrics"]["auc"]),
            "ceiling_auc": float(self.ceiling.bundle["metadata"]["metrics"]["auc"]),
            "production_label": self.production.winner_label,
            "ceiling_label": self.ceiling.winner_label,
            "not_deployable_reason": self.ceiling.bundle["not_deployable_reason"],
            "ablation": self.ceiling.bundle["metadata"]["pagevalues_ablation"],
        }


def load_app_data() -> AppData:
    """load every published artifact once at startup"""
    stages = dataset_stages()
    warnings: list[str] = []
    grids: dict[str, Grid] = {}

    for spec in GRID_SPECS:
        results = pd.read_csv(spec.results_csv)
        results["FPR"] = results["FPR"].map(parse_curve)
        results["TPR"] = results["TPR"].map(parse_curve)
        results = attach_confusion_counts(
            results, n_total=stages.n_total, n_positive=stages.n_positive
        )
        bundle = persistence.load_bundle(spec.bundle_path, arm=spec.arm)
        if bundle["tier"] != spec.tier:
            raise ValueError(f"{spec.bundle_path} declares tier {bundle['tier']!r}, expected {spec.tier!r}; "
                             "the artifact and the layout disagree about what is deployable")
        grids[spec.key] = Grid(
            spec=spec,
            results=results,
            sweep=pd.read_csv(spec.sweep_csv),
            bundle=bundle,
        )
        for label, point in bundle["operating_points"].items():
            if point is None:
                warnings.append(f"{spec.key}: no threshold satisfies the {label} constraint; the point is absent, not zero")

    return AppData(
        grids=grids,
        comparison=pd.read_csv(config.REGIME_THRESHOLD_COMPARISON_CSV),
        smote=pd.read_csv(config.SMOTE_CHECK_CSV),
        stages=stages,
        warnings=warnings,
    )
