import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from assistant.analytics.comprehensive_analytics import ComprehensiveAnalytics
from assistant.analytics.code_display_monitor import CodeDisplayMonitor
from .analytics_config import (
    TIME_PERIODS, DEFAULT_TIME_PERIOD_INDEX, TAB_CONFIG,
    CHART_COLORS, MESSAGES, PERFORMANCE_THRESHOLDS
)
from .chart_helpers import (
    create_pie_chart, create_trend_chart, create_heatmap,
    create_performance_data_table, create_content_performance_table,
    prepare_heatmap_data, format_performance_metrics
)

class AnalyticsPanel:
    """Enhanced analytics dashboard for Noah's AI Assistant."""

    def __init__(self):
        self.analytics = ComprehensiveAnalytics()
        self.code_monitor = CodeDisplayMonitor()

    def display_metrics(self):
        st.header("📊 Noah's AI Assistant Analytics Dashboard")

        # Time period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            days = st.selectbox("Time Period", TIME_PERIODS, index=DEFAULT_TIME_PERIOD_INDEX)

        # Main metrics tabs using configuration
        tab_labels = [f"{tab['emoji']} {tab['title']}" for tab in TAB_CONFIG]
        tabs = st.tabs(tab_labels)

        with tabs[0]:  # User Behavior
            self._display_user_behavior(days)

        with tabs[1]:  # Content Insights
            self._display_content_insights()

        with tabs[2]:  # Business Intelligence
            self._display_business_intelligence(days)

        with tabs[3]:  # System Performance
            self._display_system_performance()

    def _display_user_behavior(self, days: int):
        """Display user behavior analytics."""
        st.subheader("🎭 Role Distribution & Query Patterns")

        try:
            insights = self.analytics.get_user_behavior_insights(days)

            if insights.get('total_interactions', 0) == 0:
                st.info(MESSAGES['no_interactions'])
                return

            col1, col2 = st.columns(2)

            with col1:
                # Role distribution pie chart using helper
                if insights['role_distribution']:
                    fig_roles = create_pie_chart(
                        values=list(insights['role_distribution'].values()),
                        names=list(insights['role_distribution'].keys()),
                        title="User Role Distribution"
                    )
                    if fig_roles:
                        st.plotly_chart(fig_roles, use_container_width=True)

            with col2:
                # Performance metrics by role using helper
                if insights['performance_by_role']:
                    perf_data = create_performance_data_table(insights['performance_by_role'])
                    st.subheader("Performance by Role")
                    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

            # Query patterns heatmap using helper
            if insights['query_patterns_by_role']:
                st.subheader("🔍 Query Patterns by Role")

                pivot_table = prepare_heatmap_data(insights['query_patterns_by_role'])
                if pivot_table is not None:
                    fig_heatmap = create_heatmap(pivot_table, "Query Type Distribution by Role")
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading user behavior data: {e}")

    def _display_content_insights(self):
        """Display content effectiveness metrics."""
        st.subheader("📚 Content Performance & Utilization")

        try:
            content_report = self.analytics.get_content_effectiveness_report()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Top Accessed Content")
                if content_report['top_content']:
                    top_content_df = pd.DataFrame(content_report['top_content'])
                    st.dataframe(top_content_df, use_container_width=True)
                else:
                    st.info(MESSAGES['no_content_access'])

            with col2:
                st.subheader("📊 Content Type Performance")
                if content_report['performance_by_type']:
                    perf_data = create_content_performance_table(content_report['performance_by_type'])
                    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading content insights: {e}")

    def _display_business_intelligence(self, days: int):
        """Display business intelligence dashboard."""
        st.subheader("💼 Business Intelligence & Trends")

        try:
            bi_data = self.analytics.get_business_intelligence_dashboard(days)

            if 'error' in bi_data:
                st.warning(bi_data['error'])
                return

            # Key metrics cards
            self._display_bi_metrics(bi_data)

            # Trends chart using helper
            if bi_data['daily_trends']:
                st.subheader("📈 Daily Trends")

                df_trends = pd.DataFrame(bi_data['daily_trends'])
                df_trends['date'] = pd.to_datetime(df_trends['date'])

                # Configure trend traces
                trend_traces = [
                    {
                        'column': 'hiring_manager_sessions',
                        'name': 'Hiring Managers',
                        'color': CHART_COLORS['hiring_managers']
                    },
                    {
                        'column': 'developer_sessions',
                        'name': 'Developers',
                        'color': CHART_COLORS['developers']
                    },
                    {
                        'column': 'casual_visitor_sessions',
                        'name': 'Casual Visitors',
                        'color': CHART_COLORS['casual_visitors']
                    }
                ]

                fig_trends = create_trend_chart(df_trends, "Session Trends by User Type", trend_traces)
                if fig_trends:
                    st.plotly_chart(fig_trends, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading business intelligence data: {e}")

    def _display_bi_metrics(self, bi_data: dict):
        """Display business intelligence metrics cards."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Hiring Manager Sessions",
                bi_data['total_hiring_manager_sessions']
            )

        with col2:
            st.metric(
                "Developer Sessions",
                bi_data['total_developer_sessions']
            )

        with col3:
            st.metric(
                "Avg Session Duration",
                f"{bi_data['avg_session_duration']:.1f} min"
            )

        with col4:
            st.metric(
                "Code Engagement Rate",
                f"{bi_data['avg_code_engagement']:.1%}"
            )

    def _display_system_performance(self):
        """Display system performance metrics."""
        st.subheader("⚡ System Performance & Health")

        try:
            # Get performance summary from code monitor
            performance_24h = self.code_monitor.get_performance_summary(24)
            performance_7d = self.code_monitor.get_performance_summary(24 * 7)

            col1, col2 = st.columns(2)

            with col1:
                self._display_24h_performance(performance_24h)

            with col2:
                self._display_7d_performance(performance_7d)

            # System health check
            self._display_system_health()

        except Exception as e:
            st.error(f"Error loading system performance data: {e}")

    def _display_24h_performance(self, performance_24h: dict):
        """Display 24-hour performance metrics."""
        st.subheader("📊 24 Hour Performance")

        if 'error' not in performance_24h:
            metrics_24h = {
                'Total Queries': performance_24h['total_queries'],
                'Avg Query Time': f"{performance_24h['avg_query_time']:.2f}s",
                'Max Query Time': f"{performance_24h['max_query_time']:.2f}s",
                'Citation Accuracy': f"{performance_24h['citation_accuracy_rate']:.1%}",
                'Performance Alerts': performance_24h['performance_alerts']
            }

            self._display_performance_metrics(metrics_24h)
        else:
            st.info(MESSAGES['no_performance_data_24h'])

    def _display_7d_performance(self, performance_7d: dict):
        """Display 7-day performance metrics."""
        st.subheader("📈 7 Day Trends")

        if 'error' not in performance_7d:
            metrics_7d = {
                'Total Queries': performance_7d['total_queries'],
                'Avg Query Time': f"{performance_7d['avg_query_time']:.2f}s",
                'Avg Code Snippets': f"{performance_7d['avg_code_snippets']:.1f}",
                'Citation Accuracy': f"{performance_7d['citation_accuracy_rate']:.1%}"
            }

            for metric, value in metrics_7d.items():
                st.metric(metric, value)
        else:
            st.info(MESSAGES['no_performance_data_7d'])

    def _display_performance_metrics(self, metrics: dict):
        """Display performance metrics with appropriate styling."""
        for metric, value in metrics.items():
            if metric == 'Performance Alerts':
                if value > 0:
                    st.error(f"{metric}: {value}")
                else:
                    st.success(f"{metric}: {value}")
            else:
                st.metric(metric, value)

    def _display_system_health(self):
        """Display system health check results."""
        st.subheader("🏥 System Health Check")

        from assistant.analytics.code_display_monitor import health_check
        health = health_check()

        if health['status'] == 'healthy':
            st.success(MESSAGES['system_healthy'])

            health_metrics = {
                'Response Time': f"{health['response_time']:.2f}s",
                'Code Index Version': health['code_index_version'],
                'pgvector Active': health.get('has_pgvector', False),
                'Degraded Mode': health.get('degraded_mode', False)
            }

            self._display_health_metrics(health_metrics)
        else:
            error_msg = MESSAGES['system_unhealthy'].format(error=health.get('error', 'Unknown error'))
            st.error(error_msg)

    def _display_health_metrics(self, health_metrics: dict):
        """Display health metrics with appropriate styling."""
        for metric, value in health_metrics.items():
            if metric == 'Degraded Mode':
                if value:
                    st.warning(f"{metric}: {value}")
                else:
                    st.success(f"{metric}: {value}")
            else:
                st.info(f"{metric}: {value}")

    def run(self):
        """Run the analytics dashboard."""
        self.display_metrics()
