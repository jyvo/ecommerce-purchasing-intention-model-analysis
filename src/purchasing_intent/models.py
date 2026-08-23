from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config


def classifier_registry(class_weighted: bool = False, scale_pos_weight: float | None = None) -> dict:
    """The five classifiers used in the pipeline"""
    balanced = config.CLASS_WEIGHT_BALANCED if class_weighted else None

    # note the following registry, SVC.probability is depreciated, hence, CalibratedClassifierCV is used for probability=True
    registry = {
        "KNN": KNeighborsClassifier(),
        "SVM": CalibratedClassifierCV(
            estimator=SVC(
                random_state=config.RANDOM_STATE,
                class_weight=balanced,
            ),
            ensemble=False,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=config.LOGREG_MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight=balanced,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=config.RANDOM_STATE, class_weight=balanced
        ),
        "XGBoost": XGBClassifier(
            random_state=config.RANDOM_STATE,
            eval_metric=config.XGB_EVAL_METRIC,
            scale_pos_weight=scale_pos_weight if class_weighted else None,
        ),
    }

    if class_weighted:
        for name in config.CLASS_WEIGHT_UNSUPPORTED:
            registry.pop(name, None)
    return registry


def sampler_registry() -> dict:
    """resampler (base and class_weighted resample nothing)"""
    return {
        config.BASE_REGIME: None,
        config.OVERSAMPLING_REGIME: RandomOverSampler(random_state=config.RANDOM_STATE),
        config.UNDERSAMPLING_REGIME: RandomUnderSampler(random_state=config.RANDOM_STATE),
        config.CLASS_WEIGHTED_REGIME: None,
    }


def positive_class_weight(y_train) -> float:
    """n_neg / n_pos on training for XGBoost (compute from y_train only, not cross-validation folds)."""
    positives = int(y_train.sum())
    return float(len(y_train) - positives) / float(positives)
