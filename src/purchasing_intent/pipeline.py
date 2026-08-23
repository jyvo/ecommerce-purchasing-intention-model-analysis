from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from . import config

PASSTHROUGH = "passthrough"


def continuous_scaler() -> ColumnTransformer:
    """standardize the continuous (float) columns, pass everything else through"""
    transformer = ColumnTransformer(
        [
            ("scale",
                StandardScaler(),
                make_column_selector(dtype_include=config.CONTINUOUS_DTYPE_INCLUDE),
            )
        ],
        remainder=PASSTHROUGH,
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")


def feature_selector(k: int) -> SelectKBest:
    """ANOVA f test top k selector, refitted per training partition"""
    return SelectKBest(f_classif, k=k).set_output(transform="pandas")


def build_pipeline(clf, sampler=None, select_k: int | None = None) -> Pipeline:
    """scaler -> selector -> sampler -> clf"""
    return Pipeline(
        [
            ("scaler", continuous_scaler()),
            ("selector", feature_selector(select_k) if select_k else PASSTHROUGH),
            ("sampler", sampler if sampler is not None else PASSTHROUGH),
            ("clf", clf),
        ]
    )


def assert_column_order_preserved(pipe: Pipeline, X) -> None:
    """preserve column order, fail if scaler reorders columns"""
    produced = list(pipe.named_steps["scaler"].get_feature_names_out())
    expected = list(X.columns)
    if produced != expected:
        raise AssertionError(f"scaler step reordered columns - Random Forest results would fault \nexpected: {expected}\nproduced: {produced}")


def selected_features(pipe: Pipeline) -> list[str]:
    selector = pipe.named_steps["selector"]
    if selector == PASSTHROUGH:
        return []
    return list(selector.get_feature_names_out())
