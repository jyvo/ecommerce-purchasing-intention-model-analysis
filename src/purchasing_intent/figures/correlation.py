from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .. import config
from .theme import apply_theme


def correlation_heatmap(corr: pd.DataFrame, *, title: str | None = None, annotate: bool | None = None, height: int | None = None) -> go.Figure:
    """annotated correlation matrix heatmap"""
    labels = list(corr.columns)
    show_text = len(labels) <= config.CORRELATION_ANNOTATE_MAX if annotate is None else annotate
    low, high = config.CORRELATION_RANGE

    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=labels,
            y=labels,
            colorscale=config.CORRELATION_COLORSCALE,
            zmin=low,
            zmid=0.0,
            zmax=high,
            texttemplate="%{z:.2f}" if show_text else None,
            textfont={"size": max(8, config.PLOTLY_FONT_SIZE - 4)},
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
            colorbar={"title": {"text": "r"}},
        )
    )
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(tickangle=-45, automargin=True)
    apply_theme(fig, title=title, height=height or max(480, 22 * len(labels) + 220))
    return fig
