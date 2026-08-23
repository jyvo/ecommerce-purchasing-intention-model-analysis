from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = PROJECT_ROOT / "img"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"
BASELINE_DIR = DATA_DIR / "baseline"

# dataset
UCI_DATASET_ID = 468
TARGET = "Revenue"

# seed and split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# feature engineering/selection
CORRELATION_THRESHOLD = 0.9
K_BEST = 10

CONTINUOUS_DTYPE_INCLUDE = [float]

CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
BOOLEAN_COLUMNS = ["Weekend", "Revenue"]

CORRELATED_DROP = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "ExitRates",
]

# models
LOGREG_MAX_ITER = 1000
XGB_EVAL_METRIC = "logloss"
SVM_PROBABILITY = True # to be removed when CalibratedClassifierCV is used for SVM

# sampling
BASE_REGIME = "base"
OVERSAMPLING_REGIME = "oversampling"
UNDERSAMPLING_REGIME = "undersampling"
CLASS_WEIGHTED_REGIME = "class_weighted"
SAMPLING_REGIMES = [
    BASE_REGIME,
    OVERSAMPLING_REGIME,
    UNDERSAMPLING_REGIME,
    CLASS_WEIGHTED_REGIME,
]

CLASS_WEIGHT_BALANCED = "balanced"
# KNN has no native class-weight mechanism
CLASS_WEIGHT_UNSUPPORTED = ["KNN"]

# tuning
RF_PARAM_DIST = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
TUNING_N_ITER = 10
TUNING_CV_FOLDS = 5
TUNING_SCORING = "f1"
