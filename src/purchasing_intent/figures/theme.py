from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from .. import config

TEMPLATE = config.PLOTLY_TEMPLATE_NAME


def _template() -> go.layout.Template:
    axis = {
        "gridcolor": config.PLOTLY_GRID_COLOR,
        "linecolor": config.PLOTLY_GRID_COLOR,
        "zerolinecolor": config.PLOTLY_GRID_COLOR,
        "ticks": "outside",
        "tickcolor": config.PLOTLY_GRID_COLOR,
        "title": {"font": {"size": config.PLOTLY_FONT_SIZE}},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "font": {
                "family": config.PLOTLY_FONT_FAMILY,
                "size": config.PLOTLY_FONT_SIZE,
                "color": config.PLOTLY_AXIS_COLOR,
            },
            "title": {
                "font": {"size": config.PLOTLY_TITLE_SIZE},
                "x": 0.0,
                "xanchor": "left",
                "yanchor": "top",
            },
            "paper_bgcolor": config.PLOTLY_PAPER_BG,
            "plot_bgcolor": config.PLOTLY_PLOT_BG,
            "margin": dict(config.PLOTLY_MARGIN),
            "height": config.PLOTLY_HEIGHT,
            "colorway": list(config.MODEL_COLORS.values()),
            "xaxis": dict(axis),
            "yaxis": dict(axis),
            "legend": {
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": config.PLOTLY_GRID_COLOR,
                "borderwidth": 1,
                "font": {"size": config.PLOTLY_FONT_SIZE - 1},
                "itemsizing": "constant",
            },
            "hoverlabel": {
                "font": {"family": config.PLOTLY_FONT_FAMILY, "size": config.PLOTLY_FONT_SIZE},
            },
            "colorscale": {"sequential": config.CONFUSION_COLORSCALE},
        }
    )


def register_theme() -> str:
    if TEMPLATE not in pio.templates:
        pio.templates[TEMPLATE] = _template()
    return TEMPLATE

register_theme()


def model_style(name: str) -> dict:
    return {
        "color": config.MODEL_COLORS.get(name, config.PLOTLY_FALLBACK_COLOR),
        **config.MODEL_LINE_STYLES.get(name, config.PLOTLY_FALLBACK_LINE_STYLE),
    }


def regime_style(name: str) -> dict:
    return {
        "color": config.REGIME_COLORS.get(name, config.PLOTLY_FALLBACK_COLOR),
        **config.REGIME_LINE_STYLES.get(name, config.PLOTLY_FALLBACK_LINE_STYLE),
    }


def series_style(name: str, kind: str) -> dict:
    """return the style dictionary for a series name, specified by kind (either model or regime)"""
    return model_style(name) if kind == "model" else regime_style(name)


def apply_theme(fig: go.Figure, *, title: str | None = None, height: int | None = None) -> go.Figure:
    fig.update_layout(template=TEMPLATE)
    if title is not None:
        fig.update_layout(title={"text": title})
    if height is not None:
        fig.update_layout(height=height)
    return fig
