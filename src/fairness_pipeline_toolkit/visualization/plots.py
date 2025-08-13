"""Plotting utilities for fairness metrics visualization.

Provides standardized plotting functions using Plotly with consistent styling
and configurable branding through dependency injection.
"""

from typing import Dict, List
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.config_parser import VisualizationConfig


def _apply_base_layout(
    fig: go.Figure,
    viz_config: VisualizationConfig,
    title: str,
    width: int = None,
    height: int = None,
) -> go.Figure:
    """Applies consistent base styling to eliminate repetitive layout code.

    Centralizes paper background, plot background, font configuration, and title
    formatting that's repeated across all plotting functions. Width defaults to
    config value if not specified.

    Args:
        fig: Plotly figure to style
        viz_config: Visualization configuration
        title: Chart title
        width: Override default width
        height: Override default height

    Returns:
        Styled Plotly figure
    """
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": viz_config.fonts.family,
            "color": "#1A1E21",
            "size": viz_config.fonts.axis_size,
        },
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": viz_config.fonts.title_size},
        },
        height=height or viz_config.layout.height,
        width=width,
        margin=viz_config.layout.margins,
    )

    # Add professional grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def _save_as_static(
    fig: go.Figure,
    filename: str,
    viz_config: VisualizationConfig,
    output_dir: str = "images",
) -> None:
    """Exports figure as static image for GitHub rendering and documentation.

    Uses SVG format by default for crisp vector graphics that scale well in
    documentation. Creates output directory if needed and applies configured
    export settings for consistent quality across all plots.

    Args:
        fig: Plotly figure to export
        filename: Output filename (without extension)
        viz_config: Visualization configuration
        output_dir: Output directory path
    """
    # Use project root instead of current working directory
    if not Path(output_dir).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        output_path = project_root / output_dir
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Export settings from config or defaults
    export_config = getattr(viz_config, "export", None)
    fmt = getattr(export_config, "default_format", "png") if export_config else "png"
    width = getattr(export_config, "width", 1200) if export_config else 1200
    height = getattr(export_config, "height", 800) if export_config else 800
    scale = getattr(export_config, "scale", 2) if export_config else 2

    fig.write_image(
        output_path / f"{filename}.{fmt}",
        format=fmt,
        width=width,
        height=height,
        scale=scale,
    )


def plot_success_rates_by_groups(
    data: pd.DataFrame,
    viz_config: VisualizationConfig,
    target_column: str = "target",
    sensitive_features: List[str] = ["race", "sex"],
    title: str = "Success Rates by Demographic Groups",
) -> go.Figure:
    """Creates side-by-side bar charts comparing success rates across demographic groups.

    Calculates mean success rates for each group within sensitive features and displays
    them as separate subplots. Colors cycle through the config palette to distinguish
    different demographic categories visually.

    Args:
        data: DataFrame containing the data
        viz_config: Visualization configuration
        target_column: Name of the target column
        sensitive_features: List of sensitive feature column names
        title: Main title for the plot

    Returns:
        Plotly figure with subplots for each sensitive feature
    """
    n_features = len(sensitive_features)
    fig = make_subplots(
        rows=1,
        cols=n_features,
        subplot_titles=[
            f"Success Rate by {feature.title()}" for feature in sensitive_features
        ],
        horizontal_spacing=0.15,
    )

    for idx, feature in enumerate(sensitive_features, 1):
        success_rates = (
            data.groupby(feature)[target_column].mean().sort_values(ascending=False)
        )

        group_colors = [
            viz_config.colors.accent,
            viz_config.colors.warning,
            viz_config.colors.success,
        ]
        color = group_colors[(idx - 1) % len(group_colors)]

        fig.add_trace(
            go.Bar(
                x=success_rates.index,
                y=success_rates.values,
                name=feature.title(),
                marker_color=color,
                text=[f"{val:.2f}" for val in success_rates.values],
                textposition="outside",
                textfont=dict(size=12, color="#1A1E21"),
                showlegend=False,
                hovertemplate=f"<b>{feature.title()}: %{{x}}</b><br>"
                + "Success Rate: %{y:.3f}<br>"
                + "<extra></extra>",
            ),
            row=1,
            col=idx,
        )

        fig.update_yaxes(
            title_text="Success Rate" if idx == 1 else "",
            range=[0, 1.0],
            tickformat=".2f",
            row=1,
            col=idx,
        )

        fig.update_xaxes(title_text=feature.title(), row=1, col=idx)

    fig = _apply_base_layout(fig, viz_config, title, width=400 * n_features)

    _save_as_static(fig, "success_rates_by_groups", viz_config)

    return fig


def plot_combined_success_rates(
    data: pd.DataFrame,
    viz_config: VisualizationConfig,
    target_column: str = "target",
    sensitive_features: List[str] = ["race", "sex"],
    title: str = "Success Rate by Combined Demographics",
) -> go.Figure:
    """Creates grouped bar chart showing success rates for intersectional demographics.

    Groups data by both sensitive features simultaneously to reveal disparities at
    intersections (e.g., race × gender combinations). Requires exactly two features
    to create a meaningful grouped visualization without overcrowding.

    Args:
        data: DataFrame containing the data
        viz_config: Visualization configuration
        target_column: Name of the target column
        sensitive_features: List of sensitive feature column names (should be exactly 2)
        title: Title for the plot

    Returns:
        Plotly figure with grouped bar chart
    """
    if len(sensitive_features) != 2:
        raise ValueError("Combined plot requires exactly 2 sensitive features")

    feature1, feature2 = sensitive_features
    combined_rates = data.groupby([feature1, feature2])[target_column].mean().unstack()

    fig = go.Figure()

    group_colors = [
        viz_config.colors.primary,
        viz_config.colors.secondary,
        viz_config.colors.accent,
    ]

    for idx, col in enumerate(combined_rates.columns):
        color = group_colors[idx % len(group_colors)]

        fig.add_trace(
            go.Bar(
                name=f"{feature2.title()}: {col}",
                x=combined_rates.index,
                y=combined_rates[col],
                marker_color=color,
                text=[
                    f"{val:.2f}" if not pd.isna(val) else ""
                    for val in combined_rates[col]
                ],
                textposition="outside",
                textfont=dict(size=11, color="#1A1E21"),
                hovertemplate=f"<b>{feature1.title()}: %{{x}}</b><br>"
                + f"<b>{feature2.title()}: {col}</b><br>"
                + "Success Rate: %{y:.3f}<br>"
                + "<extra></extra>",
            )
        )

    fig = _apply_base_layout(fig, viz_config, title, width=800)

    fig.update_layout(
        xaxis_title=feature1.title(),
        yaxis_title="Success Rate",
        yaxis=dict(range=[0, 1.0], tickformat=".2f"),
        barmode="group",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    _save_as_static(fig, "combined_success_rates", viz_config)

    return fig


def plot_fairness_comparison(
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    viz_config: VisualizationConfig,
    fairness_threshold: float = 0.1,
    title: str = "Fairness Metrics: Before vs After",
) -> go.Figure:
    """Compares fairness metrics before and after bias mitigation interventions.

    Displays side-by-side bars for demographic parity and equalized odds differences,
    with a threshold line indicating the fairness goal. Lower values indicate better
    fairness, helping evaluate intervention effectiveness.

    Args:
        baseline_metrics: Dictionary of baseline fairness metrics
        final_metrics: Dictionary of final fairness metrics
        viz_config: Visualization configuration
        fairness_threshold: Fairness threshold line to display
        title: Title for the plot

    Returns:
        Plotly figure comparing fairness metrics
    """
    fairness_metrics = ["demographic_parity_difference", "equalized_odds_difference"]

    available_metrics = [
        m for m in fairness_metrics if m in baseline_metrics and m in final_metrics
    ]

    if not available_metrics:
        raise ValueError(
            "No matching fairness metrics found in baseline and final results"
        )

    metric_labels = [m.replace("_", " ").title() for m in available_metrics]
    baseline_values = [baseline_metrics[m] for m in available_metrics]
    final_values = [final_metrics[m] for m in available_metrics]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Before",
            x=metric_labels,
            y=baseline_values,
            marker_color=viz_config.colors.danger,
            text=[f"{val:.3f}" for val in baseline_values],
            textposition="outside",
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>"
            + "Before: %{y:.3f}<br>"
            + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="After",
            x=metric_labels,
            y=final_values,
            marker_color=viz_config.colors.primary,
            text=[f"{val:.3f}" for val in final_values],
            textposition="outside",
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" + "After: %{y:.3f}<br>" + "<extra></extra>",
        )
    )

    fig.add_hline(
        y=fairness_threshold,
        line_dash="dash",
        line_color=viz_config.colors.secondary,
        annotation_text=f"Fairness Goal (≤ {fairness_threshold})",
        annotation_position="top left",
    )

    fig = _apply_base_layout(fig, viz_config, title, width=600)

    fig.update_layout(
        yaxis_title="Difference Score (Lower = More Fair)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    _save_as_static(fig, "fairness_comparison", viz_config)

    return fig


def plot_performance_comparison(
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    viz_config: VisualizationConfig,
    title: str = "Performance Metrics: Before vs After",
) -> go.Figure:
    """Compares model performance metrics before and after bias mitigation.

    Shows accuracy, precision, and recall side-by-side to assess whether fairness
    improvements come at the cost of predictive performance. Essential for evaluating
    the fairness-performance trade-off in bias mitigation strategies.

    Args:
        baseline_metrics: Dictionary of baseline performance metrics
        final_metrics: Dictionary of final performance metrics
        viz_config: Visualization configuration
        title: Title for the plot

    Returns:
        Plotly figure comparing performance metrics
    """
    performance_metrics = ["accuracy", "precision", "recall"]

    available_metrics = [
        m for m in performance_metrics if m in baseline_metrics and m in final_metrics
    ]

    if not available_metrics:
        raise ValueError(
            "No matching performance metrics found in baseline and final results"
        )

    metric_labels = [m.title() for m in available_metrics]
    baseline_values = [baseline_metrics[m] for m in available_metrics]
    final_values = [final_metrics[m] for m in available_metrics]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Before",
            x=metric_labels,
            y=baseline_values,
            marker_color=viz_config.colors.accent,
            text=[f"{val:.3f}" for val in baseline_values],
            textposition="outside",
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>"
            + "Before: %{y:.3f}<br>"
            + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="After",
            x=metric_labels,
            y=final_values,
            marker_color=viz_config.colors.primary,
            text=[f"{val:.3f}" for val in final_values],
            textposition="outside",
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" + "After: %{y:.3f}<br>" + "<extra></extra>",
        )
    )

    fig = _apply_base_layout(fig, viz_config, title, width=600)

    fig.update_layout(
        yaxis_title="Score (Higher = Better)",
        yaxis=dict(range=[0, 1.0], tickformat=".2f"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    _save_as_static(fig, "performance_comparison", viz_config)

    return fig
