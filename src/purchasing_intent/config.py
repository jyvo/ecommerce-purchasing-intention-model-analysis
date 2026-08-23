from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = PROJECT_ROOT / "img"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"
BASELINE_DIR = DATA_DIR / "baseline"

RAW_CSV = DATA_DIR / "raw.csv"
DATASET_CACHE = CACHE_DIR / "uci_468.joblib"

def display_path(path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

RESULTS_FULL_CSV = DATA_DIR / "results.csv"
RESULTS_TOP10_CSV = DATA_DIR / "results_top10.csv"
THRESHOLD_SWEEP_CSV = DATA_DIR / "threshold_sweep.csv"
THRESHOLD_SWEEP_TOP10_CSV = DATA_DIR / "threshold_sweep_top10.csv"

REGIME_THRESHOLD_COMPARISON_CSV = DATA_DIR / "regime_threshold_comparison.csv"
SMOTE_CHECK_CSV = DATA_DIR / "smote_check.csv"

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

LEGACY_REGIMES = [BASE_REGIME, OVERSAMPLING_REGIME, UNDERSAMPLING_REGIME]

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

# selection policy
SELECTION_METRIC = "AUC"
AUC_TIE_BAND = 0.005
REGIME_PREFERENCE = [
    BASE_REGIME,
    CLASS_WEIGHTED_REGIME,
    OVERSAMPLING_REGIME,
    UNDERSAMPLING_REGIME,
]
LEGACY_SELECTION_METRIC = "F1-Score"

# results
RESULTS_COLUMNS = [
    "Model",
    "SamplingTechnique",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "FPR",
    "TPR",
    "AUC",
]
FULL_TAG = "result"
TOP10_TAG = "top10"

# page values ablation
ABLATION_DROP = ["PageValues"]

# matplotlib figures
HEATMAP_FIGSIZE = (14, 10)
HEATMAP_FIGSIZE_REDUCED = (12, 10)
HEATMAP_CMAP = "coolwarm"
HEATMAP_FMT = ".2f"
CM_FIGSIZE = (18, 6)
CM_CMAP = "Blues"
CM_TITLE_FONTSIZE = 16
ROC_GRID_COLS = 3
ROC_SAMPLING_FIGSIZE = (15, 10)
ROC_MODEL_FIGSIZE = (15, 10)
ROC_DIAGONAL_COLOR = "navy"
ROC_DIAGONAL_STYLE = "--"
ROC_AXIS_LIMITS = (0.0, 1.0)
ROC_LEGEND_LOC = "lower right"

# threshold sweep
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD = 0.5
THRESHOLD_DECIMALS = 2
COMPARISON_THRESHOLD_MIN = 0.01
COMPARISON_THRESHOLD_MAX = 0.99
COMPARISON_THRESHOLD_STEP = 0.01
COMPARISON_THRESHOLD_DECIMALS = 2

# operating point constraints
# analysis constraints
OPERATING_PRECISION_FLOOR = 0.45
OPERATING_RECALL_FLOOR = 0.50
OPERATING_MIN_RECALL = 0.50
OPERATING_MARGIN_SE = 1.0

# production constraints
OPERATING_LIFT_FLOOR = 1.5

# test partition (for lift calculation)
TEST_BASE_RATE = 0.1549

# arms and tiers
ARM_ANALYSIS = "with PageValues"
ARM_PRODUCTION = "telemetry only"
TIER_PRODUCTION = "production"
TIER_ANALYSIS = "analysis-ceiling"
ARM_TIER = {ARM_ANALYSIS: TIER_ANALYSIS, ARM_PRODUCTION: TIER_PRODUCTION}
ARM_FEATURE_COUNT = {ARM_ANALYSIS: 65, ARM_PRODUCTION: 64}

NOT_DEPLOYABLE_REASON = (
    "requires PageValues, a page-level aggregate over a reporting window derived from completed transactions. "
    "It cannot be computed for a session in progress, so this bundle cannot serve the mid-session scoring use case "
    "regardless of its metrics. Retained as a measured ceiling and as the evidence for the contamination finding."
)

# smote
SMOTE_REGIME = "smote"
SMOTE_CHECK_MODEL = "RandomForest"


# persisted bundles
BUNDLE_FORMAT_VERSION = 2
PRODUCTION_FULL_BUNDLE = MODELS_DIR / "production_full.joblib"
PRODUCTION_TOP10_BUNDLE = MODELS_DIR / "production_top10.joblib"
CEILING_FULL_BUNDLE = MODELS_DIR / "ceiling_full.joblib"
CEILING_TOP10_BUNDLE = MODELS_DIR / "ceiling_top10.joblib"
PRIMARY_BUNDLE = CEILING_FULL_BUNDLE
SECONDARY_BUNDLE = CEILING_TOP10_BUNDLE
TELEMETRY_PRIMARY_BUNDLE = PRODUCTION_FULL_BUNDLE
TELEMETRY_SECONDARY_BUNDLE = PRODUCTION_TOP10_BUNDLE
BUNDLE_TRACKED_LIBRARIES = [
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "numpy",
    "pandas",
    "scipy",
    "joblib",
]
BUNDLE_PROBA_TOLERANCE = 1e-9
BUNDLE_COMPRESS_LEVEL = 3
