from pathlib import Path
from typing import NamedTuple

from . import comparison, config, features, persistence, plots_mpl, thresholds, data as data_io
from .evaluate import best_configuration, build_registries, fit_and_evaluate, legacy_best_configuration, run_ablation, run_grid
from .models import sampler_registry
from .pipeline import build_pipeline, selected_features
from .tuning import tune_random_forest

from sklearn.model_selection import train_test_split
import numpy as np


def _refit_winner(winner, X_train, X_test, y_train, y_test, select_k=None):
    """rebuild and refit the winning configuration, returning its probabilities"""
    model_name = winner["Model"]
    regime = winner["SamplingTechnique"]
    clf = build_registries(y_train)[regime][model_name]
    pipe = build_pipeline(clf, sampler=sampler_registry()[regime], select_k=select_k)
    _, _, y_proba = fit_and_evaluate(
        pipe, model_name, X_train, X_test, y_train, y_test, regime
    )
    return pipe, y_proba


def _metrics_of(row) -> dict:
    """the five metrics of a results row, without the ROC arrays"""
    return {
        "accuracy": row["Accuracy"],
        "precision": row["Precision"],
        "recall": row["Recall"],
        "f1": row["F1-Score"],
        "auc": row["AUC"],
    }


def run_pipeline() -> None:
    """fetch -> engineer -> eliminate -> encode -> split -> grid -> tune -> sweep -> ablate"""
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)

    data, metadata = data_io.fetch_dataset()
    print(metadata)
    data.info()

    corr = features.correlation_matrix(data)
    plots_mpl.correlation_heatmap(corr, config.IMG_DIR / "heatmap1.png")

    data = features.engineer_features(data)
    print(data.isnull().sum())

    corr = features.correlation_matrix(data)
    plots_mpl.correlation_heatmap(corr, config.IMG_DIR / "heatmap2.png")

    # identify highly correlated features (threshold in config)
    high_corr = features.detect_correlated(corr)
    print(high_corr)
    if sorted(high_corr) != sorted(config.CORRELATED_DROP):
        print(
            "NOTE: correlation scan disagrees with config.CORRELATED_DROP — "
            f"scan={sorted(high_corr)} dropped={sorted(config.CORRELATED_DROP)}"
        )

    data = features.eliminate_correlated(data)

    corr = features.correlation_matrix(data)
    plots_mpl.correlation_heatmap(
        corr, config.IMG_DIR / "heatmap3.png", figsize=config.HEATMAP_FIGSIZE_REDUCED
    )

    data_io.write_raw(data)

    # preprocessing
    # encoding is a deterministic per row, transform and stays outside the pipeline
    data = features.encode(data)
    data.info()

    # check for data imbalance
    print(data[config.TARGET].value_counts().to_string())

    neg, pos = np.bincount(data[config.TARGET])
    total = neg + pos
    print(f"Minority class at {100 * pos / total:.2f}% of total")

    X = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, 
                                                        random_state=config.RANDOM_STATE, stratify=y)
    print("Train set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    comparison_rows = []
    smote_rows = []

    # note: arm 1 is the analysis tier, arm 2 is the prod tier (not secondary ablation)
    for arm in (WITH_PAGEVALUES, TELEMETRY_ONLY):
        rows, smote_row = _run_arm(X_train, X_test, y_train, y_test, arm)
        comparison_rows.extend(rows)
        smote_rows.append(smote_row)

    comparison.write_comparison(comparison_rows)
    comparison.write_smote_check(smote_rows)


class Arm(NamedTuple):
    """an E2E pass of the grid over a chosen feature space"""

    label: str
    tier: str
    drop: tuple[str, ...]
    full_tag: str
    top10_tag: str
    roc_prefix: str
    roc_top10_prefix: str
    results_csv: Path
    results_top10_csv: Path
    sweep_csv: Path
    sweep_top10_csv: Path
    primary_bundle: Path
    secondary_bundle: Path
    run_tuning: bool
    run_ablation: bool


WITH_PAGEVALUES = Arm(
    label=config.ARM_ANALYSIS,
    tier=config.TIER_ANALYSIS,
    drop=(),
    full_tag=config.FULL_TAG,
    top10_tag=config.TOP10_TAG,
    roc_prefix="",
    roc_top10_prefix="top10_",
    results_csv=config.RESULTS_FULL_CSV,
    results_top10_csv=config.RESULTS_TOP10_CSV,
    sweep_csv=config.THRESHOLD_SWEEP_CSV,
    sweep_top10_csv=config.THRESHOLD_SWEEP_TOP10_CSV,
    primary_bundle=config.CEILING_FULL_BUNDLE,
    secondary_bundle=config.CEILING_TOP10_BUNDLE,
    run_tuning=True,
    run_ablation=True,
)

# no pagevalues feat
TELEMETRY_ONLY = Arm(
    label=config.ARM_PRODUCTION,
    tier=config.TIER_PRODUCTION,
    drop=tuple(config.TELEMETRY_DROP),
    full_tag=config.TELEMETRY_FULL_TAG,
    top10_tag=config.TELEMETRY_TOP10_TAG,
    roc_prefix="nopv_",
    roc_top10_prefix="nopv_top10_",
    results_csv=config.RESULTS_TELEMETRY_CSV,
    results_top10_csv=config.RESULTS_TELEMETRY_TOP10_CSV,
    sweep_csv=config.THRESHOLD_SWEEP_TELEMETRY_CSV,
    sweep_top10_csv=config.THRESHOLD_SWEEP_TELEMETRY_TOP10_CSV,
    primary_bundle=config.PRODUCTION_FULL_BUNDLE,
    secondary_bundle=config.PRODUCTION_TOP10_BUNDLE,
    run_tuning=False,
    run_ablation=False,
)


def _roc_figures(results, prefix: str) -> None:
    """write the two ROC comparison figures for one grid"""
    plots_mpl.sampling_comparison(results, plot=False).savefig(
        config.IMG_DIR / f"{prefix}roc_sampling_comparison.png"
    )
    plots_mpl.model_comparison(results, plot=False).savefig(
        config.IMG_DIR / f"{prefix}roc_model_comparison.png"
    )


def _pagevalues_ablation(X_train, X_test, y_train, y_test, winner) -> dict:
    with_pv, without_pv = run_ablation(X_train, X_test, y_train, y_test, 
                                       model_name=winner["Model"], regime=winner["SamplingTechnique"])
    labels = ["Model", "Regime", "Accuracy", "Precision", "Recall", "F1", "AUC"]

    print("\nPageValues ablation (" + ", ".join(labels) + "):")
    for entry in (with_pv, without_pv):
        metrics = [entry[0], entry[1]] + [f"{v:.4f}" for v in entry[2:6]] + [
            f"{entry[8]:.4f}"
        ]
        print("  " + ", ".join(metrics))
    print(f"  AUC delta: {without_pv[8] - with_pv[8]:+.4f}")

    return {
        "dropped": list(config.ABLATION_DROP),
        "auc_with": float(with_pv[8]),
        "auc_without": float(without_pv[8]),
        "f1_with": float(with_pv[5]),
        "f1_without": float(without_pv[5]),
        "interpretation": (
            "large degradation (AUC drop > 0.10): the model is substantially a "
            "PageValues proxy, and PageValues is derived from transaction "
            "completion by definition. Treat 0.919 as a ceiling obtained with an "
            "outcome-derived feature; the telemetry-only arm "
            "(results_no_pagevalues.csv) is the defensible result."
        ),
    }


def _report_selection(results, arm: Arm, feature_set: str) -> tuple:
    winner = best_configuration(results)
    legacy = legacy_best_configuration(results)
    moved = (
        winner["Model"],
        winner["SamplingTechnique"],
    ) != (legacy["Model"], legacy["SamplingTechnique"])

    ceiling = results[config.SELECTION_METRIC].max()
    tied = int((results[config.SELECTION_METRIC] >= ceiling - config.AUC_TIE_BAND).sum())

    print(
        f"\nSelection [{arm.label} / {feature_set}]"
        f"({config.SELECTION_METRIC} + {config.AUC_TIE_BAND:g} tie band, "
        f"{tied} configurations tied):"
    )
    print(
        f"  winner : {winner['Model']} / {winner['SamplingTechnique']} "
        f"(AUC={winner['AUC']:.4f}, F1@predict={winner['F1-Score']:.4f})"
    )
    print(
        f"  legacy : {legacy['Model']} / {legacy['SamplingTechnique']} "
        f"(AUC={legacy['AUC']:.4f}, F1@predict={legacy['F1-Score']:.4f}) "
        f"— highest F1"
    )
    print(f"  headline configuration {'CHANGES' if moved else 'is unchanged'} under the policy")

    selection = {
        "policy": (
            f"{config.SELECTION_METRIC} with a {config.AUC_TIE_BAND:g} tie band; "
            f"ties broken by regime preference {config.REGIME_PREFERENCE}, then "
            f"{config.SELECTION_METRIC}"
        ),
        "tied_configurations": tied,
        "auc_ceiling": float(ceiling),
        "legacy_f1_winner": f"{legacy['Model']} / {legacy['SamplingTechnique']}",
        "changed_from_legacy": bool(moved),
    }
    return winner, selection


def _run_arm(X_train, X_test, y_train, y_test, arm: Arm):
    """grid -> figures -> tuning -> sweep -> ablation -> bundles, for a singular feature space
        - returns: (comparison rows, SMOTE row)
    """
    Xtr = X_train.drop(columns=list(arm.drop)) if arm.drop else X_train
    Xte = X_test.drop(columns=list(arm.drop)) if arm.drop else X_test

    print(f"\n{'=' * 72}\nARM: {arm.label} — {Xtr.shape[1]} input features [tier: {arm.tier}]\n{'=' * 72}")

    full_probas: dict = {}
    results = run_grid(Xtr, Xte, y_train, y_test, tag=arm.full_tag, csv_path=arm.results_csv, proba_sink=full_probas)
    _roc_figures(results, arm.roc_prefix)

    top10_probas: dict = {}
    top10results = run_grid(Xtr, Xte, y_train, y_test, select_k=config.K_BEST, 
                            tag=arm.top10_tag, csv_path=arm.results_top10_csv, proba_sink=top10_probas)
    _roc_figures(top10results, arm.roc_top10_prefix)

    # every config at 0.5 and at its own best threshold
    rows = comparison.comparison_rows(
        arm.label, arm.tier, "full", results, full_probas, y_test
    ) + comparison.comparison_rows(arm.label, arm.tier, "top10", top10results, top10_probas, y_test)

    # SMOTE fit
    smote_row = comparison.smote_check(Xtr, Xte, y_train, y_test, arm.label)

    # tuning
    if arm.run_tuning:
        tune_random_forest(Xtr, Xte, y_train, y_test)

    # select -> sweep winner
    winner, selection = _report_selection(results, arm, "full")
    primary_pipe, y_proba = _refit_winner(winner, Xtr, Xte, y_train, y_test)
    _, primary_points = thresholds.run_sweep(y_test, y_proba, csv_path=arm.sweep_csv, tier=arm.tier)

    if arm.run_ablation:
        ablation = _pagevalues_ablation(Xtr, Xte, y_train, y_test, winner)
    else:
        ablation = {
            "dropped": list(config.ABLATION_DROP),
            "interpretation": (
                "not applicable: this arm excludes PageValues by construction, so "
                "its metrics are already the telemetry-only figures."
            ),
        }

    _persist_bundles(primary_pipe, winner, selection, primary_points, 
                     top10results, ablation, Xtr, Xte, y_train, y_test, arm)
    return rows, smote_row


def _persist_bundles(primary_pipe, winner, selection, primary_points, top10results, 
                     ablation, X_train, X_test, y_train, y_test, arm: Arm):
    """serialize and report one arm's primary and secondary bundles"""
    constraints = thresholds.constraint_description(arm.tier)
    prevalence = thresholds.base_rate(y_test)

    primary = persistence.build_bundle(
        primary_pipe,
        model_name=winner["Model"],
        regime=winner["SamplingTechnique"],
        arm=arm.label,
        feature_set=f"{arm.label}, full ({X_train.shape[1]} features)",
        X_reference=X_test,
        metrics=_metrics_of(winner),
        operating_points=primary_points,
        constraints=constraints,
        threshold_source=str(arm.sweep_csv.name),
        base_rate=prevalence,
        selection=selection,
        ablation=ablation,
        notes={
            "arm": arm.label,
            "role": "primary; winner model, this arm's published operating points were derived from",
            "sampler_at_predict_time": "inert; imblearn resamples during fit only",
        },
    )
    persistence.save_bundle(primary, arm.primary_bundle)
    print(f"\nBundle -> {config.display_path(arm.primary_bundle)}")

    second, second_selection = _report_selection(top10results, arm, "top10")
    secondary_pipe, secondary_proba = _refit_winner(second, X_train, X_test, y_train, y_test, select_k=config.K_BEST)
    _, secondary_points = thresholds.run_sweep(y_test, secondary_proba, csv_path=arm.sweep_top10_csv, tier=arm.tier)

    secondary = persistence.build_bundle(
        secondary_pipe,
        model_name=second["Model"],
        regime=second["SamplingTechnique"],
        arm=arm.label,
        feature_set=f"{arm.label}, SelectKBest(f_classif, k={config.K_BEST})",
        X_reference=X_test,
        metrics=_metrics_of(second),
        operating_points=secondary_points,
        constraints=constraints,
        threshold_source=str(arm.sweep_top10_csv.name),
        base_rate=prevalence,
        selection=second_selection,
        selected_features=selected_features(secondary_pipe),
        ablation=ablation,
        notes={
            "arm": arm.label,
            "role": "secondary; wins this arm's top-10 grid, trains on all 9,864 rows, and is the real-time-scoring candidate",
            "input_contract": "still expects all encoded input columns of its arm: SelectKBest is a pipeline step, so selection happens inside predict, not before it",
        },
    )
    persistence.save_bundle(secondary, arm.secondary_bundle)
    print(f"\nBundle -> {config.display_path(arm.secondary_bundle)}")


if __name__ == "__main__":
    run_pipeline()
