"""Configuration constants for analytics panel components.

This module centralizes all configuration values used in the analytics panel
to improve maintainability and consistency.
"""

# Chart colors for consistent theming
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'hiring_managers': '#1f77b4',
    'developers': '#ff7f0e',
    'casual_visitors': '#2ca02c',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545'
}

# Time period options
TIME_PERIODS = [7, 14, 30, 90]
DEFAULT_TIME_PERIOD_INDEX = 2  # 30 days

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'max_query_time_warning': 5.0,  # seconds
    'max_query_time_critical': 10.0,  # seconds
    'min_citation_accuracy': 0.8,  # 80%
    'max_response_size_warning': 25000,  # characters
    'max_response_size_critical': 50000,  # characters
}

# UI formatting
METRIC_FORMATS = {
    'percentage': '{:.1%}',
    'seconds': '{:.2f}s',
    'minutes': '{:.1f} min',
    'count': '{:,}',
    'float_2': '{:.2f}'
}

# Tab configuration
TAB_CONFIG = [
    {
        'emoji': '🎯',
        'title': 'User Behavior',
        'key': 'user_behavior'
    },
    {
        'emoji': '💡',
        'title': 'Content Insights',
        'key': 'content_insights'
    },
    {
        'emoji': '📈',
        'title': 'Business Intelligence',
        'key': 'business_intelligence'
    },
    {
        'emoji': '⚡',
        'title': 'System Performance',
        'key': 'system_performance'
    }
]

# Chart configuration
CHART_CONFIG = {
    'use_container_width': True,
    'height': 400,
    'trend_chart': {
        'mode': 'lines+markers',
        'hovermode': 'x unified'
    },
    'pie_chart': {
        'height': 300
    },
    'heatmap': {
        'aspect': 'auto'
    }
}

# Messages
MESSAGES = {
    'no_data': 'No data available for this time period.',
    'no_interactions': 'No user interaction data available for this period.',
    'no_content_access': 'No content access data available.',
    'no_performance_data_24h': 'No 24-hour performance data available.',
    'no_performance_data_7d': 'No 7-day performance data available.',
    'system_healthy': '✅ System is healthy',
    'system_unhealthy': '❌ System unhealthy: {error}'
}
