import pandas as pd

from . import config, thresholds

COMPARISON_COLUMNS = [
    "Arm",
    "Tier",
    "FeatureSet",
    "Model",
    "SamplingTechnique",
    "AUC",
    "F1@0.5",
    "BestF1",
    "BestThreshold",
    "F1Gain",
]

SMOTE_COLUMNS = ["Arm", "Model", "SamplingTechnique", "AUC", "F1@0.5", "BestF1", "BestThreshold"]


def _row(arm_label, tier, feature_set, model, regime, auc, y_test, y_proba) -> list:
    at_default = thresholds.sweep(y_test, y_proba, [config.DEFAULT_THRESHOLD])
    f1_default = float(at_default["F1-Score"].iloc[0])
    best, best_t = thresholds.best_f1(y_test, y_proba)
    return [
        arm_label,
        tier,
        feature_set,
        model,
        regime,
        float(auc),
        f1_default,
        best,
        best_t,
        best - f1_default,
    ]


def comparison_rows(arm_label, tier, feature_set, results, probas, y_test) -> list[list]:
    """one row per configuration of one grid"""
    auc_of = {
        (row["Model"], row["SamplingTechnique"]): row["AUC"] for _, row in results.iterrows()
    }
    rows = []
    for (model, regime), y_proba in probas.items():
        rows.append(
            _row(
                arm_label,
                tier,
                feature_set,
                model,
                regime,
                auc_of[(model, regime)],
                y_test,
                y_proba,
            )
        )
    return rows


def write_comparison(rows: list[list], csv_path=config.REGIME_THRESHOLD_COMPARISON_CSV):
    """persist the comparison and print the per-grid spread that motivates"""
    frame = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    print(
        f"\nRegime × threshold comparison -> {config.display_path(csv_path)} "
        f"({len(frame)} rows)"
    )

    print("\nSpread across the four regimes, per model x arm x feature set:")
    spread = (
        frame.groupby(["Arm", "FeatureSet", "Model"])
        .agg(
            regimes=("SamplingTechnique", "count"),
            spread_at_default=("F1@0.5", lambda s: s.max() - s.min()),
            spread_at_best=("BestF1", lambda s: s.max() - s.min()),
            auc_range=("AUC", lambda s: s.max() - s.min()),
        )
        .round(4)
    )
    print(spread.to_string())
    return frame
