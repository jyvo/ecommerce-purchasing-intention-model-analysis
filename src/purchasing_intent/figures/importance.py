from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .. import config
from .theme import apply_theme, model_style


def feature_importance(frame: pd.DataFrame, *, kind: str, model: str | None = None, top_n: int | None = None, 
                       title: str | None = None, height: int | None = None) -> go.Figure:
    """horizontal bars for the strongest features"""
    missing = {"Feature", "Value"} - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing {sorted(missing)}")

    limit = top_n or config.IMPORTANCE_TOP_N
    ranked = (
        frame.assign(Magnitude=frame["Value"].abs())
        .sort_values("Magnitude", ascending=False)
        .head(limit)
        .sort_values("Magnitude", ascending=True)
    )

    style = model_style(model) if model else {"color": config.PLOTLY_FALLBACK_COLOR}
    fig = go.Figure(
        go.Bar(
            x=ranked["Magnitude"],
            y=ranked["Feature"],
            orientation="h",
            marker={"color": style["color"]},
            customdata=ranked["Value"],
            hovertemplate="<b>%{y}</b><br>magnitude %{x:.4f}<br>value %{customdata:+.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=kind)
    fig.update_yaxes(title_text=None, automargin=True)
    shown = len(ranked)
    apply_theme(
        fig,
        title=title or (f"{model}: {kind}" if model else kind),
        height=height or max(320, 26 * shown + 150),
    )
    return fig
