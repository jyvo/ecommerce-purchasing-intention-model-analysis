import math
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from . import config


def correlation_heatmap(corr, path, figsize=config.HEATMAP_FIGSIZE):
    """annotated correlation heatmap"""
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap=config.HEATMAP_CMAP, fmt=config.HEATMAP_FMT, annot=True)
    fig.savefig(path)
    return fig


def confusion_grid(y_test, predictions, model_name, tag, plot=False):
    """one row of confusion matrices per sampling regime
        - predictions is a list of (regime_name, y_pred, class_labels)
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(predictions), figsize=config.CM_FIGSIZE)
    fig.suptitle(
        f"Confusion Matrices for {model_name}", fontsize=config.CM_TITLE_FONTSIZE
    )

    for i, (method, y_pred, labels) in enumerate(predictions):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        display.plot(ax=axs[i], cmap=config.CM_CMAP)
        axs[i].set_title(method)

    plt.tight_layout()
    if plot:
        plt.show()

    fig.savefig(config.IMG_DIR / f"{tag}_{model_name}_cm.png")
    return fig


def _roc_grid(n_panels):
    """shared subplot for the two ROC comparison figures"""
    cols = config.ROC_GRID_COLS
    rows = math.ceil(n_panels / cols)
    return rows, cols


def _finish_roc_axis(ax, title):
    ax.plot(
        [0, 1],
        [0, 1],
        color=config.ROC_DIAGONAL_COLOR,
        linestyle=config.ROC_DIAGONAL_STYLE,
    )
    ax.set_xlim(list(config.ROC_AXIS_LIMITS))
    ax.set_ylim(list(config.ROC_AXIS_LIMITS))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc=config.ROC_LEGEND_LOC)


def sampling_comparison(data, plot=True, figsize=config.ROC_SAMPLING_FIGSIZE):
    """one panel per model, one curve per sampling technique"""
    classifiers = data["Model"].unique()
    fig_rows, fig_cols = _roc_grid(len(classifiers))
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize)

    # remove axis for extra plots
    if fig_rows > 1:
        for col in range(len(classifiers) % fig_cols, fig_cols):
            axs[fig_rows - 1, col].axis("off")

    axs = axs.flatten()

    for ax, clf_name in zip(axs, classifiers):
        filtered = data[data["Model"] == clf_name]

        for technique in data["SamplingTechnique"].unique():
            entry = filtered[filtered["SamplingTechnique"] == technique]
            if entry.empty:
                # KNN has no class_weighted configuration
                continue
            ax.plot(
                entry["FPR"].iloc[0],
                entry["TPR"].iloc[0],
                label=f"{clf_name} {technique.capitalize()} (AUC = {entry['AUC'].iloc[0]:.2f})",
            )

        _finish_roc_axis(ax, f"ROC Curve Comparison for {clf_name}")

    plt.tight_layout()
    if plot:
        plt.show()
    return fig


def model_comparison(data, plot=True, figsize=config.ROC_MODEL_FIGSIZE):
    """one panel per sampling technique, one curve per model"""
    sampling_techs = data["SamplingTechnique"].unique()
    fig_rows, fig_cols = _roc_grid(len(sampling_techs))
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize)

    # remove axis for extra plots
    if fig_rows > 1:
        for col in range(len(sampling_techs) % fig_cols, fig_cols):
            axs[fig_rows - 1, col].axis("off")

    axs = axs.flatten()

    for ax, technique in zip(axs, sampling_techs):
        filtered = data[data["SamplingTechnique"] == technique]

        for clf_name in data["Model"].unique():
            entry = filtered[filtered["Model"] == clf_name]
            if entry.empty:
                # again, KNN has no class_weighted configuration
                continue
            ax.plot(
                entry["FPR"].iloc[0],
                entry["TPR"].iloc[0],
                label=f"{clf_name} (AUC = {entry['AUC'].iloc[0]:.2f})",
            )

        _finish_roc_axis(ax, f"ROC Curve Comparison ({technique.capitalize()})")

    plt.tight_layout()
    if plot:
        plt.show()
    return fig
