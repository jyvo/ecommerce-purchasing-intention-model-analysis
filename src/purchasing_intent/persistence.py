from __future__ import annotations

import importlib.metadata
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from . import config


def library_versions() -> dict[str, str]:
    versions = {}
    for name in config.BUNDLE_TRACKED_LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _operating_point(row, threshold_source: str, constraint: str) -> dict | None:
    """a single sweep row as a plain dict of floats"""
    if row is None:
        return None
    return {
        "threshold": float(row["Threshold"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "f1": float(row["F1-Score"]),
        "lift": float(row["Lift"]),
        "predicted_positive": int(row["PredictedPositive"]),
        "threshold_source": threshold_source,
        "constraint": constraint,
        "selection_caveat": ("in-sample: threshold selected on the test set it is scored on. Measured selection "
            f"optimism ±0.0004; the constraint carries a {config.OPERATING_MARGIN_SE:g}-SE stability margin."),
    }


def _verification_sample(pipe, X_reference: pd.DataFrame):
    """rows at the 0/25/50/75/100th percentiles of the model's own probabilities"""
    proba = pipe.predict_proba(X_reference)[:, 1]
    order = proba.argsort()
    positions = sorted({int(round(q * (len(order) - 1))) for q in (0, 0.25, 0.5, 0.75, 1)})
    picked = order[positions]

    rows = X_reference.iloc[picked]
    return rows, [float(v) for v in proba[picked]]


def build_bundle(pipe, *, model_name: str, regime: str, arm: str, feature_set: str, X_reference: pd.DataFrame, 
                 metrics: dict, operating_points: dict, constraints: dict, threshold_source: str, 
                 base_rate: float, selection: dict | None = None, selected_features: list[str] | None = None, 
                 ablation: dict | None = None, notes: dict | None = None) -> dict:
    """assemble the serializable bundle for one fitted pipeline"""

    sample, sample_proba = _verification_sample(pipe, X_reference)
    tier = config.ARM_TIER[arm]

    return {
        "bundle_format_version": config.BUNDLE_FORMAT_VERSION,
        "pipeline": pipe,
        "feature_order": list(X_reference.columns),
        "feature_dtypes": {c: str(t) for c, t in X_reference.dtypes.items()},
        "selected_features": selected_features or [],
        "tier": tier,
        "arm": arm,
        "not_deployable_reason": (None if tier == config.TIER_PRODUCTION else config.NOT_DEPLOYABLE_REASON),
        "operating_points": {
            label: _operating_point(row, threshold_source, constraints.get(label, ""))
            for label, row in operating_points.items()
        },
        "default_threshold": config.DEFAULT_THRESHOLD,
        "base_rate": float(base_rate),
        "metadata": {
            "model": model_name,
            "sampling_regime": regime,
            "arm": arm,
            "tier": tier,
            "feature_set": feature_set,
            "n_input_features": X_reference.shape[1],
            "trained_on": date.today().isoformat(),
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "dataset": f"UCI ID {config.UCI_DATASET_ID} (Sakar & Kastro, 2018)",
            "library_versions": library_versions(),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "selection": selection or {},
            "pagevalues_ablation": ablation,
            "notes": notes or {},
        },
        "verification": {
            "sample_rows": sample,
            "expected_proba": sample_proba,
            "tolerance": config.BUNDLE_PROBA_TOLERANCE,
        },
    }


def save_bundle(bundle: dict, path: Path) -> Path:
    """write bundle to path (models/) via joblib"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=config.BUNDLE_COMPRESS_LEVEL)
    return path


def assert_arm(bundle: dict, arm: str | None, path=None) -> str:
    """check a bundle's declared arm against the arm the caller expects"""
    declared = bundle.get("arm")
    where = f"{path}: " if path else ""
    if declared not in config.ARM_FEATURE_COUNT:
        raise ValueError(f"{where}bundle declares arm={declared!r}, which is not a known arm")

    expected = config.ARM_FEATURE_COUNT[declared]
    found = len(bundle["feature_order"])
    if found != expected:
        raise ValueError(f"{where}bundle declares arm={declared!r} ({expected} features) but its feature "
            f"contract has {found}. The artifact disagrees with itself; re-run `uv run pi-pipeline`.")
    if arm is not None and arm != declared:
        raise ValueError(f"{where}caller asked for arm={arm!r} but this bundle is {declared!r} ({found} features, "
            f"tier {bundle.get('tier')!r}). Scoring a session against the wrong arm returns a probability that is silently wrong.")
    return declared


def load_bundle(path: Path, *, arm: str | None = None) -> dict:
    """load a compatible bundle"""
    bundle = joblib.load(path)
    found = bundle.get("bundle_format_version")
    if found != config.BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"{path} has bundle_format_version={found!r}, this build expects "
            f"{config.BUNDLE_FORMAT_VERSION!r}; re-run `uv run pi-pipeline`."
        )
    assert_arm(bundle, arm, path)
    return bundle


def frame_from_mapping(bundle: dict, values: dict, *, arm: str) -> pd.DataFrame:
    """build a one row frame that satisfies bundle's feature contract"""
    assert_arm(bundle, arm)

    unknown = set(values) - set(bundle["feature_order"])
    if unknown:
        raise KeyError(f"not features of this bundle ({arm}): {sorted(unknown)}")

    row = {name: values.get(name, 0) for name in bundle["feature_order"]}
    frame = pd.DataFrame([row], columns=bundle["feature_order"])
    return frame.astype(bundle["feature_dtypes"])


def predict_proba(bundle: dict, X, *, arm: str) -> float | list[float]:
    """positive-class probability for rows already matching the contract"""
    assert_arm(bundle, arm)
    expected = list(bundle["feature_order"])
    if list(X.columns) != expected:
        raise ValueError(f"frame does not match this bundle's contract ({arm}, {len(expected)} "
            f"features): got {len(X.columns)} columns. Build it with frame_from_mapping.")
    return bundle["pipeline"].predict_proba(X)[:, 1]


def verify_bundle(path: Path, *, arm: str | None = None) -> dict:
    """load path and re-score its fixture (raise if it does not round-trip)"""
    bundle = load_bundle(path, arm=arm)
    fixture = bundle["verification"]

    actual = [float(v) for v in bundle["pipeline"].predict_proba(fixture["sample_rows"])[:, 1]]
    expected = [float(v) for v in fixture["expected_proba"]]
    deltas = [abs(a - e) for a, e in zip(actual, expected)]
    worst = max(deltas)

    if worst > fixture["tolerance"]:
        raise AssertionError(
            f"{path.name} does not round-trip: expected {expected!r}, "
            f"got {actual!r} (max |delta|={worst:.3e} > {fixture['tolerance']:.0e})"
        )
    return {
        "path": str(path),
        "expected": expected,
        "actual": actual,
        "max_delta": worst,
        "rows_checked": len(expected),
        "feature_count": len(bundle["feature_order"]),
        "model": bundle["metadata"]["model"],
        "regime": bundle["metadata"]["sampling_regime"],
        "arm": bundle["arm"],
        "tier": bundle["tier"],
    }
