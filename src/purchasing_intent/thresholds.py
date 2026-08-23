import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config

SWEEP_COLUMNS = [
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Lift",
    "PredictedPositive",
]


def threshold_grid(minimum=config.THRESHOLD_MIN, maximum=config.THRESHOLD_MAX,
                   step=config.THRESHOLD_STEP, decimals=config.THRESHOLD_DECIMALS,) -> list[float]:
    """decimal thresholds from minimum to maximum (inclusive)"""
    n = int(round((maximum - minimum) / step)) + 1
    return [round(minimum + i * step, decimals) for i in range(n)]


def comparison_grid() -> list[float]:
    return threshold_grid(
        config.COMPARISON_THRESHOLD_MIN,
        config.COMPARISON_THRESHOLD_MAX,
        config.COMPARISON_THRESHOLD_STEP,
        config.COMPARISON_THRESHOLD_DECIMALS,
    )


def base_rate(y_test) -> float:
    """positive rate of the evaluation partition (denominator for lift)"""
    return float(np.mean(np.asarray(y_test)))


def sweep(y_test, y_proba, grid=None) -> pd.DataFrame:
    """precision/recall/f1/lift at each threshold on grid"""
    grid = threshold_grid() if grid is None else grid
    prevalence = base_rate(y_test)

    rows = []
    for threshold in grid:
        # comparison uses the same value the row stores
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        rows.append(
            [
                threshold,
                accuracy_score(y_test, y_pred),
                precision,
                recall_score(y_test, y_pred, zero_division=0),
                f1_score(y_test, y_pred, zero_division=0),
                precision / prevalence if prevalence else 0.0,
                int(y_pred.sum()),
            ]
        )
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def best_f1(y_test, y_proba, grid=None) -> tuple[float, float]:
    """(best f1, threshold)
        - ties go to the lower threshold (the higher recall side of a flat optimum)
    """
    swept = sweep(y_test, y_proba, comparison_grid() if grid is None else grid)
    row = swept.loc[swept["F1-Score"].idxmax()]
    return float(row["F1-Score"]), float(row["Threshold"])


def _standard_error(proportion: float, n: int) -> float:
    """binomial standard error of a proportion measured over n trials
        - n: denominator the metric was actually computed over
        - returns inf for empty denominator
    """
    if n <= 0:
        return float("inf")
    return float(np.sqrt(max(proportion * (1.0 - proportion), 0.0) / n))


def _clears(value: float, floor: float, se: float) -> bool:
    """whether value clears floor with confidence allowance"""
    return bool(value - config.OPERATING_MARGIN_SE * se >= floor)


def _feasible(sweep_df: pd.DataFrame, n_positives: int, prevalence: float, constraints: list) -> pd.DataFrame:
    """rows satisfying every (column, floor, denominator) constraint with margin"""
    keep = pd.Series(True, index=sweep_df.index)
    for column, floor, denominator in constraints:
        for idx, row in sweep_df.iterrows():
            if not keep[idx]:
                continue
            flagged = int(row["PredictedPositive"])
            if column == "Lift":
                se = _standard_error(float(row["Precision"]), flagged) / prevalence
            elif denominator == "flagged":
                se = _standard_error(float(row[column]), flagged)
            else:
                se = _standard_error(float(row[column]), n_positives)
            if not _clears(float(row[column]), floor, se):
                keep[idx] = False
    return sweep_df[keep]


def _pick(candidates: pd.DataFrame, objective: str, secondary: str):
    """highest objective among candidates, ties broken by secondary"""
    if candidates.empty:
        return None
    return candidates.sort_values([objective, secondary], ascending=False).iloc[0]


def operating_points(sweep_df: pd.DataFrame, y_test, tier: str) -> dict:
    """the two published operating points for one tier
        - analysis tier: high_recall maximizes recall subject to precision >= 0.45,
                         high precision maximizes precision subject to recall >= 0.5
        - production tier: high_recall maximizes recall subject to lift >= 1.5x,
                           high_precision maximizes lift subject to recall >= 0.5
        - both tiers additionally require OPERATING_MIN_RECALL and clear every floor 
            by OPERATING_MARGIN_SE standard errors
    """
    n_positives = int(np.sum(np.asarray(y_test)))
    prevalence = base_rate(y_test)
    min_recall = ("Recall", config.OPERATING_MIN_RECALL, "positives")
    recall_floor = (
        "Recall",
        max(config.OPERATING_RECALL_FLOOR, config.OPERATING_MIN_RECALL),
        "positives",
    )

    if tier == config.TIER_PRODUCTION:
        reach_constraints = [("Lift", config.OPERATING_LIFT_FLOOR, "flagged"), min_recall]
        precision_objective = "Lift"
    else:
        reach_constraints = [
            ("Precision", config.OPERATING_PRECISION_FLOOR, "flagged"),
            min_recall,
        ]
        precision_objective = "Precision"
    precision_constraints = [recall_floor]

    return {
        "high_recall": _pick(
            _feasible(sweep_df, n_positives, prevalence, reach_constraints),
            "Recall",
            "Precision",
        ),
        "high_precision": _pick(
            _feasible(sweep_df, n_positives, prevalence, precision_constraints),
            precision_objective,
            "Recall",
        ),
    }


def constraint_description(tier: str) -> dict[str, str]:
    """statement of what each point was selected against"""
    margin = f"each floor cleared by ≥{config.OPERATING_MARGIN_SE:g} SE of its own estimate"
    useful = f"and recall ≥ {config.OPERATING_MIN_RECALL:.2f}"
    recall_floor = max(config.OPERATING_RECALL_FLOOR, config.OPERATING_MIN_RECALL)
    high_precision_floor = (
        f"recall ≥ {recall_floor:.2f}"
        f"minimum-useful-recall floor)"
    )
    if tier == config.TIER_PRODUCTION:
        return {
            "high_recall": (
                f"maximize recall subject to lift ≥ {config.OPERATING_LIFT_FLOOR:g}x over the base rate {useful}; {margin}"
            ),
            "high_precision": (
                f"maximize lift subject to {high_precision_floor}; {margin}"
            ),
        }
    return {
        "high_recall": (
            f"maximize recall subject to precision ≥ {config.OPERATING_PRECISION_FLOOR:.2f} "
            f"{useful}; {margin}"
        ),
        "high_precision": (
            f"maximize precision subject to {high_precision_floor}; {margin}"
        ),
    }


def run_sweep(y_test, y_proba, csv_path=config.THRESHOLD_SWEEP_CSV, tier=config.TIER_ANALYSIS):
    """sweep, persist to csv_path, and return (sweep_df, operating_points)"""
    sweep_df = sweep(y_test, y_proba)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(csv_path, index=False)
    print(f"\nThreshold sweep [{tier}] -> {config.display_path(csv_path)}")
    print(sweep_df.to_string(index=False))

    points = operating_points(sweep_df, y_test, tier)
    for label, row in points.items():
        if row is None:
            print(f"{label}: no threshold satisfies the constraint — None persisted")
            continue
        print(
            f"{label}: threshold={row['Threshold']:.2f} "
            f"precision={row['Precision']:.3f} lift={row['Lift']:.2f}x "
            f"recall={row['Recall']:.3f} F1={row['F1-Score']:.3f}"
        )
    return sweep_df, points
