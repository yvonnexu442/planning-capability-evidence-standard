"""Paper-ready plotting helpers for manuscript figures.

The project uses these helpers to keep experiment figures visually consistent
with journal-style empirical papers: restrained typography, colorblind-safe
colors, clear axes, and PDF output for direct LaTeX inclusion.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - the plotting layer should degrade gracefully.
    sns = None


PathLike = Union[str, Path]

JOURNAL_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#999999",  # gray
]

STRATEGY_COLORS = {
    "global_best_model": "#0072B2",
    "family_best_model": "#56B4E9",
    "category_best_model": "#56B4E9",
    "simple_ensemble": "#E69F00",
    "operational_loss_ensemble": "#A6761D",
    "best_inventory_cost_model": "#D55E00",
    "best_stability_model": "#009E73",
    "stability_aware_selector": "#009E73",
    "feasibility_aware_selector": "#CC79A7",
    "greedy_feasibility_selector": "#CC79A7",
    "dp_feasibility_selector": "#7B3294",
    "budgeted_dp_feasibility_selector": "#008837",
    "feasibility_aware_smoothed_alpha_0_25": "#6A3D9A",
    "feasibility_aware_smoothed_alpha_0_50": "#8E63B6",
    "feasibility_aware_smoothed_alpha_0_75": "#B28AD1",
    "feasibility_aware_smoothed_scenario_alpha": "#7851A9",
    "feasibility_aware_smoothed_utility_alpha": "#4B0082",
    "feasibility_aware_ensemble_inverse_accuracy": "#A6761D",
    "feasibility_aware_ensemble_inverse_operational_loss": "#666600",
    "feasibility_aware_ensemble_constrained": "#1B9E77",
    "oracle_dp_feasibility_selector": "#222222",
    "full_outcome_oracle_dp_feasibility_selector": "#555555",
    "oracle_realized_demand": "#000000",
    "individual_global_lightgbm": "#D55E00",
    "individual_global_xgboost": "#E69F00",
    "individual_global_sklearn": "#E69F00",
    "individual_moving_average": "#777777",
    "moving_average_baseline": "#777777",
    "individual_naive_last_value": "#999999",
    "individual_seasonal_naive": "#666666",
    "individual_exponential_smoothing": "#333333",
}

STRATEGY_MARKERS = {
    "global_best_model": "o",
    "family_best_model": "s",
    "category_best_model": "s",
    "simple_ensemble": "p",
    "operational_loss_ensemble": "P",
    "best_inventory_cost_model": "*",
    "best_stability_model": "H",
    "stability_aware_selector": "^",
    "feasibility_aware_selector": "D",
    "greedy_feasibility_selector": "D",
    "dp_feasibility_selector": "X",
    "budgeted_dp_feasibility_selector": "P",
    "feasibility_aware_smoothed_alpha_0_25": "v",
    "feasibility_aware_smoothed_alpha_0_50": "^",
    "feasibility_aware_smoothed_alpha_0_75": "<",
    "feasibility_aware_smoothed_scenario_alpha": ">",
    "feasibility_aware_smoothed_utility_alpha": "D",
    "feasibility_aware_ensemble_inverse_accuracy": "P",
    "feasibility_aware_ensemble_inverse_operational_loss": "X",
    "feasibility_aware_ensemble_constrained": "s",
    "oracle_dp_feasibility_selector": "d",
    "full_outcome_oracle_dp_feasibility_selector": "*",
    "oracle_realized_demand": "o",
    "individual_global_lightgbm": "P",
    "individual_global_xgboost": "X",
    "individual_global_sklearn": "X",
    "individual_moving_average": "v",
    "moving_average_baseline": "v",
    "individual_naive_last_value": "h",
    "individual_seasonal_naive": "<",
    "individual_exponential_smoothing": ">",
}

STRATEGY_LINESTYLES = {
    "global_best_model": "-",
    "family_best_model": "-.",
    "category_best_model": "-.",
    "simple_ensemble": "-",
    "operational_loss_ensemble": "-.",
    "best_inventory_cost_model": "--",
    "best_stability_model": "-.",
    "stability_aware_selector": "--",
    "feasibility_aware_selector": "-",
    "greedy_feasibility_selector": "-",
    "dp_feasibility_selector": "--",
    "budgeted_dp_feasibility_selector": "-.",
    "feasibility_aware_smoothed_alpha_0_25": "--",
    "feasibility_aware_smoothed_alpha_0_50": "-.",
    "feasibility_aware_smoothed_alpha_0_75": ":",
    "feasibility_aware_smoothed_scenario_alpha": "--",
    "feasibility_aware_smoothed_utility_alpha": "-",
    "feasibility_aware_ensemble_inverse_accuracy": "--",
    "feasibility_aware_ensemble_inverse_operational_loss": "-.",
    "feasibility_aware_ensemble_constrained": "-",
    "oracle_dp_feasibility_selector": "--",
    "full_outcome_oracle_dp_feasibility_selector": "-.",
    "oracle_realized_demand": ":",
    "individual_global_lightgbm": ":",
    "individual_global_xgboost": ":",
    "individual_global_sklearn": ":",
    "individual_moving_average": "--",
    "moving_average_baseline": "--",
}


def apply_paper_style() -> None:
    """Apply a consistent LaTeX-friendly plotting style.

    The style uses seaborn when available and then sets Matplotlib parameters
    that matter for manuscript output, especially embedded fonts in PDF files.
    """
    rc_params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.75,
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
    }
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", palette=JOURNAL_PALETTE, rc=rc_params)
    matplotlib.rcParams.update(rc_params)


def save_placeholder_figure_note(output_path: PathLike, message: str) -> Path:
    """Save a text note where a future figure will be generated."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(message), encoding="utf-8")
    return path


def validate_plot_table(data: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that a table contains all columns required for plotting."""
    missing = set(required_columns).difference(set(data.columns))
    if missing:
        raise ValueError("Plot table is missing required columns: {}".format(sorted(missing)))


def strategy_color(strategy: str, fallback_index: int = 0) -> str:
    """Return a stable color for a strategy name."""
    if strategy in STRATEGY_COLORS:
        return STRATEGY_COLORS[strategy]
    return JOURNAL_PALETTE[int(fallback_index) % len(JOURNAL_PALETTE)]


def strategy_marker(strategy: str) -> str:
    """Return a stable marker for a strategy name."""
    return STRATEGY_MARKERS.get(strategy, "o")


def strategy_linestyle(strategy: str) -> str:
    """Return a stable line style for a strategy name."""
    return STRATEGY_LINESTYLES.get(strategy, "-")


def palette_for_strategies(strategies: Sequence[str]) -> Mapping[str, str]:
    """Return an ordered strategy-to-color mapping for seaborn plots."""
    return {strategy: strategy_color(strategy, index) for index, strategy in enumerate(strategies)}


def format_axis(
    ax,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    grid_axis: str = "y",
    compact_y: bool = True,
) -> None:
    """Apply common journal-style axis formatting."""
    if title:
        ax.set_title(title, pad=8)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.75)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#333333", width=0.8, length=3)
    if compact_y:
        ax.yaxis.set_major_formatter(FuncFormatter(compact_number_formatter))


def compact_number_formatter(value: float, _position: int) -> str:
    """Format large axis tick values without scientific notation."""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return "{:.1f}M".format(value / 1_000_000).replace(".0M", "M")
    if absolute >= 1_000:
        return "{:.1f}K".format(value / 1_000).replace(".0K", "K")
    if 0 < absolute < 0.01:
        return "{:.1e}".format(value)
    if absolute < 1:
        return "{:.3f}".format(value).rstrip("0").rstrip(".")
    if absolute < 10:
        return "{:.2f}".format(value).rstrip("0").rstrip(".")
    return "{:.0f}".format(value)


def place_legend(ax, columns: int = 2, location: str = "upper center") -> None:
    """Place a compact legend that does not cover the plotting area."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc=location,
        bbox_to_anchor=(0.5, 1.16),
        ncol=columns,
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def save_paper_figure(fig, png_path: PathLike, pdf_path: PathLike, dpi: int = 240) -> None:
    """Save a figure as PNG for review and PDF for LaTeX."""
    png_output = Path(png_path)
    pdf_output = Path(pdf_path)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_output, dpi=dpi)
    fig.savefig(pdf_output)
    plt.close(fig)
