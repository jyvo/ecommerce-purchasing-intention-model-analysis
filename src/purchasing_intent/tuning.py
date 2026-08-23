from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from . import config
from .pipeline import build_pipeline


def tune_random_forest(X_train, X_test, y_train, y_test, select_k=None):
    """randomized search over RF hyperparameter grid
        - parameter distribution is config.RF_PARAM_DIST (param args for pipeline clf)
        - returns (best_estimator, f1, cv)
    """

    pipe = build_pipeline(
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        sampler=RandomOverSampler(random_state=config.RANDOM_STATE),
        select_k=select_k,
    )
    param_distributions = {
        f"clf__{key}": values for key, values in config.RF_PARAM_DIST.items()
    }

    cv = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=config.TUNING_N_ITER,
        cv=config.TUNING_CV_FOLDS,
        scoring=config.TUNING_SCORING,
        random_state=config.RANDOM_STATE,
    )
    cv.fit(X_train, y_train)
    print(f"\nBest Random Forest Parameters: {cv.best_params_}")
    print(f"Best cross-validated F1: {cv.best_score_:.3f}")

    best = cv.best_estimator_
    rf_f1 = f1_score(y_test, best.predict(X_test))
    print("Best Random Forest F1 Score (on test set): {:.3f}".format(rf_f1))
    return best, rf_f1, cv
