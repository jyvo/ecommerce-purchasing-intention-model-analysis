from __future__ import annotations

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .. import config
from .theme import apply_theme

COUNT_COLUMNS = ("TN", "FP", "FN", "TP")


def _counts(results: pd.DataFrame, model: str, regime: str) -> tuple[int, int, int, int]:
    missing = set(COUNT_COLUMNS) - set(results.columns)
    if missing:
        raise KeyError(f"results is missing {sorted(missing)}; derive the counts in dataload.attach_confusion_counts")
    
    entry = results[(results["Model"] == model) & (results["SamplingTechnique"] == regime)]

    if entry.empty:
        raise KeyError(f"no configuration {model} / {regime} in this grid")
    
    row = entry.iloc[0]
    return int(row["TN"]), int(row["FP"]), int(row["FN"]), int(row["TP"])


def _matrix_traces(tn: int, fp: int, fn: int, tp: int) -> tuple[list[list[int]], list[list[float]], list[list[str]]]:
    """counts, row-normalized shares and cell text (ordered)"""
    counts = [[tn, fp], [fn, tp]]
    shares = []
    text = []
    for row in counts:
        total = sum(row) or 1
        share_row = [value / total for value in row]
        shares.append(share_row)
        text.append([f"<b>{value:,}</b><br>{share:.1%} of row" for value, share in zip(row, share_row)])
    return counts, shares, text


def confusion_matrix(results: pd.DataFrame, *, model: str, regime: str, 
                     title: str | None = None, height: int | None = None) -> go.Figure:
    """one configuration's confusion matrix, annotated with count and row share"""
    tn, fp, fn, tp = _counts(results, model, regime)
    counts, shares, text = _matrix_traces(tn, fp, fn, tp)
    negative, positive = config.CONFUSION_CLASS_LABELS

    fig = go.Figure(
        go.Heatmap(
            z=shares,
            x=[f"Predicted: {negative}", f"Predicted: {positive}"],
            y=[f"Actual: {negative}", f"Actual: {positive}"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": config.PLOTLY_FONT_SIZE + 2},
            colorscale=config.CONFUSION_COLORSCALE,
            zmin=0.0,
            zmax=1.0,
            customdata=counts,
            hovertemplate="%{y}<br>%{x}<br>%{customdata:,} sessions (%{z:.1%} of row)<extra></extra>",
            colorbar={"title": {"text": "share of<br>actual class"}, "tickformat": ".0%"},
        )
    )
    fig.update_yaxes(autorange="reversed")
    apply_theme(fig, title=title or f"{model} / {regime}", height=height or 380)
    return fig


def confusion_grid(results: pd.DataFrame, *, model: str, regimes: list[str] | None = None, 
                   title: str | None = None, height: int | None = None) -> go.Figure:
    """one row of confusion matrices per sampling regime, for one classifier"""
    available = list(dict.fromkeys(results[results["Model"] == model]["SamplingTechnique"]))
    ordered = [r for r in (regimes or config.SAMPLING_REGIMES) if r in available]
    if not ordered:
        raise KeyError(f"no configurations for {model} in this grid")

    negative, positive = config.CONFUSION_CLASS_LABELS
    fig = make_subplots(
        rows=1,
        cols=len(ordered),
        subplot_titles=list(ordered),
        horizontal_spacing=0.06,
    )
    for index, regime in enumerate(ordered, start=1):
        tn, fp, fn, tp = _counts(results, model, regime)
        counts, shares, text = _matrix_traces(tn, fp, fn, tp)
        fig.add_trace(
            go.Heatmap(
                z=shares,
                x=[f"Pred: {negative}", f"Pred: {positive}"],
                y=[f"Actual: {negative}", f"Actual: {positive}"],
                text=text,
                texttemplate="%{text}",
                textfont={"size": config.PLOTLY_FONT_SIZE},
                colorscale=config.CONFUSION_COLORSCALE,
                zmin=0.0,
                zmax=1.0,
                customdata=counts,
                hovertemplate="%{y}<br>%{x}<br>%{customdata:,} sessions (%{z:.1%} of row)<extra></extra>",
                showscale=index == len(ordered),
                colorbar={"title": {"text": "share of<br>actual class"}, "tickformat": ".0%"},
            ),
            row=1,
            col=index,
        )
    fig.update_yaxes(autorange="reversed")

    # only display class labels on the leftmost panel to avoid overlapping within the next panels
    for index in range(2, len(ordered) + 1):
        fig.update_yaxes(showticklabels=False, row=1, col=index)
    for annotation in fig.layout.annotations:
        annotation.font.size = config.PLOTLY_FONT_SIZE
    apply_theme(fig, title=title or f"Confusion matrices — {model}", height=height or 400)
    return fig
