import numpy as np
import pandas as pd

from . import config


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Numeric/boolean correlation matrix"""
    return data.select_dtypes(include=[np.number, bool]).corr()


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add session duration and three per-page intensity features

    Replace with 0 (for nan from div by zero, when the session had no page of that type)
    """
    data.insert(
        loc=0,
        column="SessionDuration",
        value=data.Administrative_Duration
        + data.Informational_Duration
        + data.ProductRelated_Duration,
    )
    data.insert(
        loc=1,
        column="AdministrativeAvgDuration",
        value=(data.Administrative_Duration / data.Administrative).replace(np.nan, 0),
    )
    data.insert(
        loc=4,
        column="InformationalAvgDuration",
        value=(data.Informational_Duration / data.Informational).replace(np.nan, 0),
    )
    data.insert(
        loc=7,
        column="ProductRelatedAvgDuration",
        value=(data.ProductRelated_Duration / data.ProductRelated).replace(np.nan, 0),
    )
    return data


def detect_correlated(corr: pd.DataFrame) -> list[str]:
    """Upper triangular scan for |r| > threshold

    Catch strong negative pairs with abs()
    """
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col].abs() > config.CORRELATION_THRESHOLD)]


def eliminate_correlated(data: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant raw count/duration pairs and ExitRates"""
    data.drop(config.CORRELATED_DROP, axis=1, inplace=True)
    return data


def encode(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categoricals (cast bool to int)"""
    data = pd.get_dummies(data, columns=config.CATEGORICAL_COLUMNS, dtype=int, drop_first=True)
    data[config.BOOLEAN_COLUMNS] = data[config.BOOLEAN_COLUMNS].astype(int)
    return data
