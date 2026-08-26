from __future__ import annotations

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .. import config
from .theme import apply_theme, series_style


def _as_curve(value, column: str, label: str) -> np.ndarray:
    """convert one stored ROC array to floats (raise if it is still a string)"""
    if isinstance(value, str):
        raise TypeError(f"{column} for {label} is still a string. ROC arrays round-trip must be parsed in dataload.parse_curve, not here.")
    return np.asarray(value, dtype=float)


def _ordered(values, preference) -> list:
    present = list(dict.fromkeys(values))
    known = [v for v in preference if v in present]
    return known + [v for v in present if v not in known]


def _facet_grid(n_panels: int, ncols: int | None) -> tuple[int, int]:
    cols = min(ncols or config.ROC_FACET_COLS, max(n_panels, 1))
    rows = -(-n_panels // cols)
    return rows, cols


def _roc_facets(results: pd.DataFrame, *, facet_col: str, curve_col: str, curve_kind: str, facet_title: str, 
                ncols: int | None, title: str | None, height: int | None) -> go.Figure:
    required = {"Model", "SamplingTechnique", "FPR", "TPR", "AUC"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"results is missing {sorted(missing)}")

    facet_pref = (config.SAMPLING_REGIMES if facet_col == "SamplingTechnique" else list(config.MODEL_COLORS))
    curve_pref = (config.SAMPLING_REGIMES if curve_col == "SamplingTechnique" else list(config.MODEL_COLORS))
    panels = _ordered(results[facet_col], facet_pref)
    curves = _ordered(results[curve_col], curve_pref)

    rows, cols = _facet_grid(len(panels), ncols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{facet_title}: {p}" for p in panels],
        horizontal_spacing=0.08,
        vertical_spacing=0.13 if rows > 1 else 0.0,
    )

    seen: set = set()
    for index, panel in enumerate(panels):
        row, col = index // cols + 1, index % cols + 1
        panel_rows = results[results[facet_col] == panel]

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": config.ROC_CHANCE_COLOR, "width": 1, **config.ROC_CHANCE_LINE_STYLE},
                name="Chance (AUC = 0.50)",
                legendgroup="_chance",
                showlegend="_chance" not in seen,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        seen.add("_chance")

        for curve in curves:
            entry = panel_rows[panel_rows[curve_col] == curve]
            if entry.empty:
                # KNN has no class-weighted configuration (cell does not exist)
                continue
            first = entry.iloc[0]
            label = f"{panel} / {curve}"
            style = series_style(curve, curve_kind)
            fig.add_trace(
                go.Scatter(
                    x=_as_curve(first["FPR"], "FPR", label),
                    y=_as_curve(first["TPR"], "TPR", label),
                    mode="lines",
                    name=curve,
                    legendgroup=curve,
                    showlegend=curve not in seen,
                    line={"width": 2.2, **style},
                    hovertemplate=(
                        f"<b>{curve}</b><br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}"
                        f"<br>AUC {first['AUC']:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            seen.add(curve)

    fig.update_xaxes(
        title_text="False positive rate",
        range=list(config.ROC_AXIS_LIMITS),
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="True positive rate",
        range=list(config.ROC_AXIS_LIMITS),
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = config.PLOTLY_FONT_SIZE

    panel_height = height or (300 * rows + 120)
    apply_theme(fig, title=title, height=panel_height)
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.16 / max(rows, 1) - 0.06,
                "xanchor": "left", "x": 0.0},
        hovermode="closest",
    )
    return fig


def roc_by_model(results: pd.DataFrame, *, ncols: int | None = None, title: str | None = None, height: int | None = None,) -> go.Figure:
    return _roc_facets(
        results,
        facet_col="SamplingTechnique",
        curve_col="Model",
        curve_kind="model",
        facet_title="Regime",
        ncols=ncols,
        title=title,
        height=height,
    )


def roc_by_sampling(results: pd.DataFrame, *, ncols: int | None = None, title: str | None = None, height: int | None = None,) -> go.Figure:
    return _roc_facets(
        results,
        facet_col="Model",
        curve_col="SamplingTechnique",
        curve_kind="regime",
        facet_title="Model",
        ncols=ncols,
        title=title,
        height=height,
    )
