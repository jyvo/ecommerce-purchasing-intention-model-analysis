import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

from . import config, plots_mpl
from .models import classifier_registry, positive_class_weight, sampler_registry
from .pipeline import assert_column_order_preserved, build_pipeline

import time

def fit_and_evaluate(pipe, clf_name, X_train, X_test, y_train, y_test, regime=""):
    """fit a pipeline and record metrics + ROC curve
        - returns (entry, y_pred, y_proba)
    """
    pipe.fit(X_train, y_train)
    assert_column_order_preserved(pipe, X_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    entry = [
        clf_name,
        regime,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        fpr,
        tpr,
        auc(fpr, tpr),
    ]
    return entry, y_pred, y_proba


def build_registries(y_train) -> dict:
    """regime -> {model name: unfitted classifier}"""
    scale_pos_weight = positive_class_weight(y_train)
    return {
        regime: classifier_registry(
            class_weighted=(regime == config.CLASS_WEIGHTED_REGIME),
            scale_pos_weight=scale_pos_weight,
        )
        for regime in config.SAMPLING_REGIMES
    }


def run_grid(X_train, X_test, y_train, y_test, select_k=None, tag=config.FULL_TAG, 
             csv_path=config.RESULTS_FULL_CSV, plot=False, proba_sink=None):
    """evaluate every classifier against every sampling regime and persist results"""
    registries = build_registries(y_train)
    samplers = sampler_registry()

    results = pd.DataFrame(columns=config.RESULTS_COLUMNS)
    selected = []

    for name in classifier_registry():
        predictions = []
        
        print(f"\n{name} evaluating...")
        start = time.time()
        for regime in config.SAMPLING_REGIMES:
            clf = registries[regime].get(name)
            if clf is None:
                # KNN under class_weighted (recorded N/A, not substituted)
                print(f"{name} / {regime}: N/A (no class-weight mechanism)")
                continue

            pipe = build_pipeline(clf, sampler=samplers[regime], select_k=select_k)
            entry, y_pred, y_proba = fit_and_evaluate(pipe, name, X_train, X_test, y_train, y_test, regime)
            results.loc[len(results)] = entry
            predictions.append((regime, y_pred, pipe.named_steps["clf"].classes_))
            if proba_sink is not None:
                proba_sink[(name, regime)] = y_proba

            if select_k and not selected:
                selected = list(pipe.named_steps["selector"].get_feature_names_out())

        plots_mpl.confusion_grid(y_test, predictions, name, tag, plot=plot)
        print(f"{name} finished in {time.time() - start:.2f} seconds")

    print("\nAll models finished.")
    print(results.drop(["TPR", "FPR"], axis=1).to_string())

    if selected:
        print(f"\nSelectKBest(k={select_k}) features: {selected}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    print(f"Wrote {config.display_path(csv_path)}")
    return results


def best_configuration(results: pd.DataFrame) -> pd.Series:
    """model selection by AUC, tie-breaker by sampling regime preference
        - accuracy never used
        - ranking by f1 at fixed 0.5 threshold
    """
    ceiling = results[config.SELECTION_METRIC].max()
    tied = results[results[config.SELECTION_METRIC] >= ceiling - config.AUC_TIE_BAND].copy()

    rank = {regime: i for i, regime in enumerate(config.REGIME_PREFERENCE)}
    tied["_regime_rank"] = tied["SamplingTechnique"].map(rank).fillna(len(rank))
    tied = tied.sort_values(
        ["_regime_rank", config.SELECTION_METRIC], ascending=[True, False]
    )
    return results.loc[tied.index[0]]


def legacy_best_configuration(results: pd.DataFrame) -> pd.Series:
    """historical selection by f1 at fixed 0.5 threshold, tie-breaker by sampling regime preference"""
    return results.loc[results[config.LEGACY_SELECTION_METRIC].idxmax()]


def run_ablation(X_train, X_test, y_train, y_test, model_name, regime, select_k=None, drop=None):
    """refit one configuration
        - drop removed
        - returns (entry_with, entry_without)
    """
    drop = list(drop if drop is not None else config.ABLATION_DROP)
    samplers = sampler_registry()

    def _one(Xtr, Xte, label):
        clf = build_registries(y_train)[regime][model_name]
        pipe = build_pipeline(clf, sampler=samplers[regime], select_k=select_k)
        entry, _, _ = fit_and_evaluate(pipe, model_name, Xtr, Xte, y_train, y_test, label)
        return entry

    with_feature = _one(X_train, X_test, regime)
    without_feature = _one(
        X_train.drop(columns=drop), X_test.drop(columns=drop), f"{regime}_ablated"
    )
    return with_feature, without_feature
