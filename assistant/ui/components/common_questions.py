"""Common Questions Display Component for Noah's AI Assistant.

This component displays frequently asked questions to help users get started
and discover the most valuable queries for their role.
"""
import streamlit as st
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class CommonQuestionsDisplay:
    """Display component for most common questions by role."""

    def __init__(self, analytics_system=None):
        """Initialize with analytics system for data retrieval."""
        self.analytics = analytics_system

        # Fallback questions if analytics data not available
        self.fallback_questions = {
            'Hiring Manager (technical)': [
                "What's Noah's technical background and experience?",
                "Show me examples of Noah's code architecture",
                "How does Noah approach system design?",
                "What AI and ML projects has Noah worked on?",
                "Tell me about Noah's Python and LangChain experience"
            ],
            'Hiring Manager (nontechnical)': [
                "What's Noah's professional background?",
                "What are Noah's key achievements?",
                "How does Noah work with teams?",
                "What industries has Noah worked in?",
                "Tell me about Noah's leadership experience"
            ],
            'Software Developer': [
                "How does the RAG engine work?",
                "Show me the code architecture",
                "How is the vector store implemented?",
                "What's the role routing logic?",
                "How does the memory system work?"
            ],
            'Just looking around': [
                "Tell me something interesting about Noah",
                "What's Noah's background?",
                "Any fun facts about Noah?",
                "What does Noah do for work?",
                "Noah MMA fight"
            ]
        }

    def display_for_role(self, role: str, show_analytics: bool = True) -> Optional[str]:
        """Display common questions for a specific role.

        Returns:
            Selected question if user clicks on one, None otherwise
        """
        selected_question = None

        st.subheader("💡 Common Questions")

        if show_analytics and self.analytics:
            try:
                # Get data-driven common questions
                common_questions = self.analytics.get_most_common_questions(
                    role=role, days=30, limit=5
                )

                if common_questions:
                    st.caption(f"Based on popular questions from other {role.lower()}s:")

                    for i, q_data in enumerate(common_questions):
                        question = q_data['question']
                        frequency = q_data['frequency']
                        success_rate = q_data['success_rate']

                        # Create button with question and metadata
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            if st.button(
                                f"❓ {question}",
                                key=f"common_q_{role}_{i}",
                                help=f"Asked {frequency} times • {success_rate:.1%} success rate"
                            ):
                                selected_question = question

                        with col2:
                            # Show popularity indicator
                            if frequency >= 10:
                                st.write("🔥")
                            elif frequency >= 5:
                                st.write("📈")
                            else:
                                st.write("💬")

                    return selected_question

            except Exception as e:
                logger.warning(f"Failed to load analytics data: {e}")
                # Fall through to fallback questions

        # Use fallback questions
        if role in self.fallback_questions:
            st.caption("Popular questions to get you started:")

            for i, question in enumerate(self.fallback_questions[role]):
                if st.button(
                    f"❓ {question}",
                    key=f"fallback_q_{role}_{i}"
                ):
                    selected_question = question

        return selected_question

    def display_trending_questions(self, days: int = 7) -> Optional[str]:
        """Display trending questions across all roles."""
        selected_question = None

        if not self.analytics:
            return None

        try:
            trending = self.analytics.get_most_common_questions(days=days, limit=3)

            if trending:
                st.subheader("🔥 Trending This Week")

                for i, q_data in enumerate(trending):
                    question = q_data['question']
                    frequency = q_data['frequency']
                    roles = q_data.get('roles', [])

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        if st.button(
                            f"🔥 {question}",
                            key=f"trending_{i}",
                            help=f"Popular across: {', '.join(roles[:2])}"
                        ):
                            selected_question = question

                    with col2:
                        st.caption(f"{frequency}x")

        except Exception as e:
            logger.warning(f"Failed to load trending questions: {e}")

        return selected_question

    def display_role_comparison(self) -> Dict[str, Any]:
        """Display comparison of question patterns across roles."""
        if not self.analytics:
            return {}

        try:
            role_questions = self.analytics.get_common_questions_by_role(days=30, limit_per_role=3)

            st.subheader("🎭 Questions by Role")

            tabs = st.tabs(list(role_questions.keys()))

            for tab, (role, questions) in zip(tabs, role_questions.items()):
                with tab:
                    if questions:
                        for q_data in questions:
                            question = q_data['question']
                            frequency = q_data['frequency']
                            success_rate = q_data['success_rate']

                            with st.container():
                                st.write(f"**{question}**")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Frequency", frequency)
                                with col2:
                                    st.metric("Success Rate", f"{success_rate:.1%}")
                                with col3:
                                    st.metric("Type", q_data.get('query_type', 'general'))

                                st.divider()
                    else:
                        st.info(f"No data available for {role}")

            return role_questions

        except Exception as e:
            logger.error(f"Failed to display role comparison: {e}")
            return {}

    def display_question_analytics(self) -> Dict[str, Any]:
        """Display detailed analytics about question patterns."""
        if not self.analytics:
            return {}

        try:
            # Get overall question statistics
            insights = self.analytics.get_user_behavior_insights(days=30)

            st.subheader("📊 Question Analytics")

            # Display query patterns by role
            if 'query_patterns_by_role' in insights:
                st.write("**Query Type Distribution by Role:**")

                for role, patterns in insights['query_patterns_by_role'].items():
                    with st.expander(f"{role} Query Patterns"):
                        for query_type, count in patterns.items():
                            st.write(f"• {query_type.title()}: {count} queries")

            # Display performance metrics
            if 'performance_by_role' in insights:
                st.write("**Performance by Role:**")

                for role, metrics in insights['performance_by_role'].items():
                    if 'error' not in metrics:
                        display_metrics = {
                            'Success Rate': f"{metrics['success_rate']:.1%}",
                            'Avg Response Time': f"{metrics['avg_response_time']:.2f}s",
                        }
                        for metric, value in display_metrics.items():
                            st.metric(metric, value)
                    else:
                        st.info("No data available.")

            # Hard-coded time periods and formatting
            performance_7d = self.code_monitor.get_performance_summary(24 * 7)
            metrics_7d = {
                'Avg Query Time': f"{performance_7d['avg_query_time']:.2f}s",
                'Citation Accuracy': f"{performance_7d['citation_accuracy_rate']:.1%}"
            }

            return insights

        except Exception as e:
            logger.error(f"Failed to display question analytics: {e}")
            return {}

    def _display_system_performance(self):
        """Display system performance metrics."""
        st.subheader("⚡ System Performance & Health")

        # Delegate to smaller, focused methods
        self._display_performance_metrics()
        self._display_system_health()

    def _display_performance_metrics(self):
        """Display 24h and 7d performance data."""
        # Focused implementation

    def _display_system_health(self):
        """Display system health check results."""
        # Focused implementation

    def _validate_analytics_data(self, data: dict, required_keys: list) -> bool:
        """Validate analytics data structure."""
        if not data or 'error' in data:
            return False
        return all(key in data for key in required_keys)

    def _create_trend_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create standardized trend chart."""
        fig = go.Figure()

        for column, color in self.config.CHART_COLORS.items():
            if column in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[column],
                    mode='lines+markers',
                    name=column.replace('_', ' ').title(),
                    line=dict(color=color)
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sessions",
            hovermode='x unified'
        )

        return fig


class AnalyticsPanelConfig:
    CHART_COLORS = {
        'hiring_managers': '#1f77b4',
        'developers': '#ff7f0e',
        'casual_visitors': '#2ca02c'
    }

    TIME_PERIODS = [7, 14, 30, 90]
    DEFAULT_PERIOD = 30

    METRIC_FORMATS = {
        'response_time': 'time',
        'success_rate': 'percentage',
        'avg_snippets': 'decimal'
    }


def display_suggested_questions_sidebar(role: str, analytics_system=None) -> Optional[str]:
    """Display suggested questions in sidebar for quick access."""
    with st.sidebar:
        st.subheader("💡 Quick Questions")

        if analytics_system:
            try:
                suggestions = analytics_system.get_suggested_questions_for_role(role, limit=3)

                if suggestions:
                    st.caption("Based on successful queries:")

                    for i, question in enumerate(suggestions):
                        if st.button(
                            question,
                            key=f"sidebar_q_{i}",
                            use_container_width=True
                        ):
                            return question

                    return None

            except Exception as e:
                logger.warning(f"Failed to load sidebar suggestions: {e}")

        # Fallback suggestions
        fallback_map = {
            'Hiring Manager (technical)': [
                "Show me Noah's code",
                "Technical background?",
                "AI/ML experience?"
            ],
            'Hiring Manager (nontechnical)': [
                "Professional background?",
                "Key achievements?",
                "Team experience?"
            ],
            'Software Developer': [
                "How does this work?",
                "Show me the architecture",
                "Code examples?"
            ],
            'Just looking around': [
                "Tell me about Noah",
                "Fun facts?",
                "MMA fight link?"
            ]
        }

        if role in fallback_map:
            for i, question in enumerate(fallback_map[role]):
                if st.button(
                    question,
                    key=f"sidebar_fallback_{i}",
                    use_container_width=True
                ):
                    return question

    return None


# Helper function for integration with main app
def get_question_suggestions(role: str, analytics_system=None, limit: int = 5) -> List[str]:
    """Get question suggestions for a role (for programmatic use)."""
    if analytics_system:
        try:
            return analytics_system.get_suggested_questions_for_role(role, limit=limit)
        except Exception as e:
            logger.warning(f"Failed to get analytics suggestions: {e}")

    # Fallback to static suggestions
    fallback = {
        'Hiring Manager (technical)': [
            "What's Noah's technical background?",
            "Show me examples of his code",
            "How does he approach system design?",
            "What AI projects has he worked on?",
            "Tell me about his Python experience"
        ],
        'Hiring Manager (nontechnical)': [
            "What's Noah's professional background?",
            "What are his key achievements?",
            "How does he work with teams?",
            "What industries has he worked in?",
            "Tell me about his career progression"
        ],
        'Software Developer': [
            "How does the RAG engine work?",
            "Show me the architecture",
            "How is the vector store implemented?",
            "What's the role routing logic?",
            "How does the memory system work?"
        ],
        'Just looking around': [
            "Tell me about Noah",
            "Any fun facts?",
            "What does he do for work?",
            "Noah MMA fight",
            "What's interesting about him?"
        ]
    }

    return fallback.get(role, [])[:limit]
