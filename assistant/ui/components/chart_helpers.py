"""Helper functions for creating charts in the analytics panel.

This module provides reusable chart creation functions to reduce
code duplication and ensure consistent styling.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any, Optional
from .analytics_config import CHART_COLORS, CHART_CONFIG


def create_pie_chart(values: List[float], names: List[str], title: str) -> go.Figure:
    """Create a consistent pie chart."""
    if not values or not names:
        return None

    fig = px.pie(
        values=values,
        names=names,
        title=title,
        height=CHART_CONFIG['pie_chart']['height']
    )
    return fig


def create_trend_chart(df: pd.DataFrame, title: str, traces: List[Dict[str, Any]]) -> go.Figure:
    """Create a consistent trend line chart with multiple traces.

    Args:
        df: DataFrame with 'date' column
        title: Chart title
        traces: List of trace configurations with keys:
            - 'column': column name in df
            - 'name': display name
            - 'color': line color (optional, uses default if not provided)
    """
    if df.empty:
        return None

    fig = go.Figure()

    for i, trace in enumerate(traces):
        color = trace.get('color', list(CHART_COLORS.values())[i % len(CHART_COLORS)])

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[trace['column']],
            mode=CHART_CONFIG['trend_chart']['mode'],
            name=trace['name'],
            line=dict(color=color)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sessions",
        hovermode=CHART_CONFIG['trend_chart']['hovermode'],
        height=CHART_CONFIG['height']
    )

    return fig


def create_heatmap(pivot_data: pd.DataFrame, title: str) -> go.Figure:
    """Create a consistent heatmap chart."""
    if pivot_data.empty:
        return None

    fig = px.imshow(
        pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        aspect=CHART_CONFIG['heatmap']['aspect'],
        title=title
    )

    return fig


def format_performance_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format performance metrics for display.

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Dictionary with formatted metric strings
    """
    from .analytics_config import METRIC_FORMATS

    formatted = {}

    for key, value in metrics.items():
        if 'time' in key.lower() and isinstance(value, (int, float)):
            formatted[key] = METRIC_FORMATS['seconds'].format(value)
        elif 'rate' in key.lower() or 'accuracy' in key.lower():
            formatted[key] = METRIC_FORMATS['percentage'].format(value)
        elif 'duration' in key.lower():
            formatted[key] = METRIC_FORMATS['minutes'].format(value)
        elif isinstance(value, (int, float)) and value > 1:
            formatted[key] = METRIC_FORMATS['count'].format(int(value))
        else:
            formatted[key] = str(value)

    return formatted


def create_performance_data_table(performance_by_role: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    """Create formatted data for performance tables."""
    perf_data = []

    for role, metrics in performance_by_role.items():
        formatted_metrics = format_performance_metrics(metrics)
        perf_data.append({
            'Role': role,
            'Success Rate': formatted_metrics.get('success_rate', 'N/A'),
            'Avg Response Time': formatted_metrics.get('avg_response_time', 'N/A')
        })

    return perf_data


def create_content_performance_table(performance_by_type: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Create formatted data for content performance tables."""
    perf_data = []

    for content_type, metrics in performance_by_type.items():
        perf_data.append({
            'Content Type': content_type,
            'Unique Items': f"{metrics['unique_items']:,}",
            'Total Accesses': f"{metrics['total_accesses']:,}",
            'Avg Relevance': f"{metrics['avg_relevance']:.2f}"
        })

    return perf_data


def prepare_heatmap_data(query_patterns_by_role: Dict[str, Dict[str, int]]) -> Optional[pd.DataFrame]:
    """Prepare data for query patterns heatmap."""
    if not query_patterns_by_role:
        return None

    pattern_data = []
    for role, patterns in query_patterns_by_role.items():
        for query_type, count in patterns.items():
            pattern_data.append({
                'Role': role,
                'Query Type': query_type,
                'Count': count
            })

    if not pattern_data:
        return None

    df_patterns = pd.DataFrame(pattern_data)
    pivot_table = df_patterns.pivot(index='Role', columns='Query Type', values='Count').fillna(0)

    return pivot_table
