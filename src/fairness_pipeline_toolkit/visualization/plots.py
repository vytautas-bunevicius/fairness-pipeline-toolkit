"""Plotting utilities for fairness metrics visualization.

This module provides standardized plotting functions for visualizing bias detection
and fairness metrics using Plotly with consistent branding and styling following
2025 Python best practices with proper dependency injection.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.config_parser import VisualizationConfig


def plot_success_rates_by_groups(
    data: pd.DataFrame, 
    viz_config: VisualizationConfig,
    target_column: str = 'target',
    sensitive_features: List[str] = ['race', 'sex'],
    title: str = "Success Rates by Demographic Groups"
) -> go.Figure:
    """Create subplots showing success rates across different demographic groups.
    
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
        subplot_titles=[f'Success Rate by {feature.title()}' for feature in sensitive_features],
        horizontal_spacing=0.15
    )
    
    for idx, feature in enumerate(sensitive_features, 1):
        success_rates = data.groupby(feature)[target_column].mean().sort_values(ascending=False)
        
        # Color palette for demographic groups
        group_colors = [viz_config.colors.accent, viz_config.colors.warning, viz_config.colors.success]
        color = group_colors[(idx-1) % len(group_colors)]
        
        fig.add_trace(
            go.Bar(
                x=success_rates.index,
                y=success_rates.values,
                name=feature.title(),
                marker_color=color,
                text=[f'{val:.2f}' for val in success_rates.values],
                textposition='outside',
                textfont=dict(size=12, color="#1A1E21"),
                showlegend=False,
                hovertemplate=f"<b>{feature.title()}: %{{x}}</b><br>" +
                             "Success Rate: %{y:.3f}<br>" +
                             "<extra></extra>"
            ),
            row=1, col=idx
        )
        
        # Update y-axis for this subplot
        fig.update_yaxes(
            title_text="Success Rate" if idx == 1 else "",
            range=[0, 1.0],
            tickformat='.2f',
            row=1, col=idx
        )
        
        # Update x-axis for this subplot
        fig.update_xaxes(
            title_text=feature.title(),
            row=1, col=idx
        )
    
    # Update overall layout using config
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": viz_config.fonts.family,
            "color": "#1A1E21",
            "size": viz_config.fonts.axis_size,
        },
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': viz_config.fonts.title_size}
        },
        height=viz_config.layout.height,
        width=400 * n_features,
        margin=viz_config.layout.margins
    )
    
    return fig


def plot_combined_success_rates(
    data: pd.DataFrame,
    viz_config: VisualizationConfig,
    target_column: str = 'target',
    sensitive_features: List[str] = ['race', 'sex'],
    title: str = "Success Rate by Combined Demographics"
) -> go.Figure:
    """Create a grouped bar chart showing success rates for combined demographic groups.
    
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
    
    # Add bars for each category of the second feature
    group_colors = [viz_config.colors.primary, viz_config.colors.secondary, viz_config.colors.accent]
    
    for idx, col in enumerate(combined_rates.columns):
        color = group_colors[idx % len(group_colors)]
        
        fig.add_trace(
            go.Bar(
                name=f'{feature2.title()}: {col}',
                x=combined_rates.index,
                y=combined_rates[col],
                marker_color=color,
                text=[f'{val:.2f}' if not pd.isna(val) else '' for val in combined_rates[col]],
                textposition='outside',
                textfont=dict(size=11, color="#1A1E21"),
                hovertemplate=f"<b>{feature1.title()}: %{{x}}</b><br>" +
                             f"<b>{feature2.title()}: {col}</b><br>" +
                             "Success Rate: %{y:.3f}<br>" +
                             "<extra></extra>"
            )
        )
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": viz_config.fonts.family,
            "color": "#1A1E21",
            "size": viz_config.fonts.axis_size,
        },
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': viz_config.fonts.title_size}
        },
        xaxis_title=feature1.title(),
        yaxis_title="Success Rate",
        yaxis=dict(range=[0, 1.0], tickformat='.2f'),
        barmode='group',
        height=viz_config.layout.height,
        width=800,
        margin=viz_config.layout.margins,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def plot_fairness_comparison(
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    viz_config: VisualizationConfig,
    fairness_threshold: float = 0.1,
    title: str = "Fairness Metrics: Before vs After"
) -> go.Figure:
    """Create a comparison plot of fairness metrics before and after intervention.
    
    Args:
        baseline_metrics: Dictionary of baseline fairness metrics
        final_metrics: Dictionary of final fairness metrics
        viz_config: Visualization configuration
        fairness_threshold: Fairness threshold line to display
        title: Title for the plot
        
    Returns:
        Plotly figure comparing fairness metrics
    """
    fairness_metrics = [
        'demographic_parity_difference',
        'equalized_odds_difference'
    ]
    
    # Filter to only fairness metrics that exist in both
    available_metrics = [m for m in fairness_metrics 
                        if m in baseline_metrics and m in final_metrics]
    
    if not available_metrics:
        raise ValueError("No matching fairness metrics found in baseline and final results")
    
    metric_labels = [m.replace('_', ' ').title() for m in available_metrics]
    baseline_values = [baseline_metrics[m] for m in available_metrics]
    final_values = [final_metrics[m] for m in available_metrics]
    
    fig = go.Figure()
    
    # Add baseline bars
    fig.add_trace(
        go.Bar(
            name='Before',
            x=metric_labels,
            y=baseline_values,
            marker_color=viz_config.colors.danger,
            text=[f'{val:.3f}' for val in baseline_values],
            textposition='outside',
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" +
                         "Before: %{y:.3f}<br>" +
                         "<extra></extra>"
        )
    )
    
    # Add final bars
    fig.add_trace(
        go.Bar(
            name='After',
            x=metric_labels,
            y=final_values,
            marker_color=viz_config.colors.primary,
            text=[f'{val:.3f}' for val in final_values],
            textposition='outside',
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" +
                         "After: %{y:.3f}<br>" +
                         "<extra></extra>"
        )
    )
    
    # Add fairness threshold line
    fig.add_hline(
        y=fairness_threshold,
        line_dash="dash",
        line_color=viz_config.colors.secondary,
        annotation_text=f"Fairness Goal (≤ {fairness_threshold})",
        annotation_position="top left"
    )
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": viz_config.fonts.family,
            "color": "#1A1E21",
            "size": viz_config.fonts.axis_size,
        },
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': viz_config.fonts.title_size}
        },
        yaxis_title="Difference Score (Lower = More Fair)",
        barmode='group',
        height=viz_config.layout.height,
        width=600,
        margin=viz_config.layout.margins,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def plot_performance_comparison(
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    viz_config: VisualizationConfig,
    title: str = "Performance Metrics: Before vs After"
) -> go.Figure:
    """Create a comparison plot of performance metrics before and after intervention.
    
    Args:
        baseline_metrics: Dictionary of baseline performance metrics
        final_metrics: Dictionary of final performance metrics
        viz_config: Visualization configuration
        title: Title for the plot
        
    Returns:
        Plotly figure comparing performance metrics
    """
    performance_metrics = ['accuracy', 'precision', 'recall']
    
    # Filter to only performance metrics that exist in both
    available_metrics = [m for m in performance_metrics 
                        if m in baseline_metrics and m in final_metrics]
    
    if not available_metrics:
        raise ValueError("No matching performance metrics found in baseline and final results")
    
    metric_labels = [m.title() for m in available_metrics]
    baseline_values = [baseline_metrics[m] for m in available_metrics]
    final_values = [final_metrics[m] for m in available_metrics]
    
    fig = go.Figure()
    
    # Add baseline bars
    fig.add_trace(
        go.Bar(
            name='Before',
            x=metric_labels,
            y=baseline_values,
            marker_color=viz_config.colors.accent,
            text=[f'{val:.3f}' for val in baseline_values],
            textposition='outside',
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" +
                         "Before: %{y:.3f}<br>" +
                         "<extra></extra>"
        )
    )
    
    # Add final bars
    fig.add_trace(
        go.Bar(
            name='After',
            x=metric_labels,
            y=final_values,
            marker_color=viz_config.colors.primary,
            text=[f'{val:.3f}' for val in final_values],
            textposition='outside',
            textfont=dict(size=12, color="#1A1E21"),
            hovertemplate="<b>%{x}</b><br>" +
                         "After: %{y:.3f}<br>" +
                         "<extra></extra>"
        )
    )
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": viz_config.fonts.family,
            "color": "#1A1E21",
            "size": viz_config.fonts.axis_size,
        },
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': viz_config.fonts.title_size}
        },
        yaxis_title="Score (Higher = Better)",
        yaxis=dict(range=[0, 1.0], tickformat='.2f'),
        barmode='group',
        height=viz_config.layout.height,
        width=600,
        margin=viz_config.layout.margins,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig