"""Integration module for common questions with Noah's AI Assistant.

This module provides seamless integration between the common questions system
and the main application, including analytics tracking and UI integration.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import streamlit as st

from assistant.analytics.comprehensive_analytics import (
    ComprehensiveAnalytics, UserInteraction, create_interaction_from_rag_result
)
from assistant.ui.components.common_questions import (
    CommonQuestionsDisplay, display_suggested_questions_sidebar, get_question_suggestions
)

logger = logging.getLogger(__name__)


class CommonQuestionsIntegration:
    """Integration layer for common questions functionality."""

    def __init__(self, rag_engine=None, memory_system=None, role_router=None):
        """Initialize integration with core systems."""
        self.rag_engine = rag_engine
        self.memory_system = memory_system
        self.role_router = role_router

        # Initialize analytics system
        try:
            self.analytics = ComprehensiveAnalytics()
            self.analytics_enabled = True
        except Exception as e:
            logger.warning(f"Analytics system unavailable: {e}")
            self.analytics = None
            self.analytics_enabled = False

        # Initialize display component
        self.questions_display = CommonQuestionsDisplay(analytics_system=self.analytics)

    def track_user_interaction(self, session_id: str, user_role: str, query: str,
                             query_type: str, response_time: float, rag_result: Dict[str, Any],
                             conversation_turn: int = 1, follow_up_query: bool = False) -> bool:
        """Track a user interaction for analytics."""
        if not self.analytics_enabled:
            return False

        try:
            interaction = create_interaction_from_rag_result(
                session_id=session_id,
                user_role=user_role,
                query=query,
                query_type=query_type,
                response_time=response_time,
                rag_result=rag_result
            )

            # Add conversation context
            interaction.conversation_turn = conversation_turn
            interaction.follow_up_query = follow_up_query

            self.analytics.log_interaction(interaction)

            # Update content analytics if applicable
            if rag_result.get('code_snippets'):
                for snippet in rag_result['code_snippets']:
                    content_id = snippet.get('citation', 'unknown')
                    self.analytics.update_content_analytics(
                        content_type='code_snippet',
                        content_id=content_id,
                        relevance_score=0.8,  # Could be calculated based on user feedback
                        user_role=user_role
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
            return False

    def display_welcome_questions(self, role: str) -> Optional[str]:
        """Display welcome questions for a user role.

        Returns:
            Selected question if user clicks one, None otherwise
        """
        st.markdown("---")

        # Display role-specific common questions
        selected_question = self.questions_display.display_for_role(
            role=role, show_analytics=self.analytics_enabled
        )

        if selected_question:
            st.success(f"Selected: {selected_question}")
            return selected_question

        return None

    def display_sidebar_suggestions(self, role: str) -> Optional[str]:
        """Display quick question suggestions in sidebar.

        Returns:
            Selected question if user clicks one, None otherwise
        """
        return display_suggested_questions_sidebar(
            role=role, analytics_system=self.analytics
        )

    def get_personalized_suggestions(self, role: str, session_history: List[str] = None) -> List[str]:
        """Get personalized question suggestions based on role and history."""
        base_suggestions = get_question_suggestions(
            role=role, analytics_system=self.analytics, limit=5
        )

        if not session_history:
            return base_suggestions

        # Filter out questions already asked in this session
        history_lower = [q.lower().strip() for q in session_history]
        filtered_suggestions = [
            q for q in base_suggestions
            if q.lower().strip() not in history_lower
        ]

        return filtered_suggestions[:3]  # Return top 3 new suggestions

    def display_analytics_dashboard(self) -> Dict[str, Any]:
        """Display analytics dashboard for admin users."""
        if not self.analytics_enabled:
            st.error("Analytics system not available")
            return {}

        try:
            st.header("📊 Question Analytics Dashboard")

            # Overall insights
            insights = self.analytics.get_user_behavior_insights(days=30)

            if insights and 'total_interactions' in insights:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Interactions (30d)",
                        insights['total_interactions']
                    )

                with col2:
                    role_count = len(insights.get('role_distribution', {}))
                    st.metric("Active Roles", role_count)

                with col3:
                    avg_success = sum(
                        perf['success_rate']
                        for perf in insights.get('performance_by_role', {}).values()
                    ) / max(len(insights.get('performance_by_role', {})), 1)
                    st.metric("Avg Success Rate", f"{avg_success:.1%}")

            # Display detailed analytics
            self.questions_display.display_question_analytics()

            # Role comparison
            self.questions_display.display_role_comparison()

            # Content effectiveness
            content_report = self.analytics.get_content_effectiveness_report()

            if content_report.get('top_content'):
                st.subheader("🏆 Most Accessed Content")

                for item in content_report['top_content'][:5]:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.write(f"**{item['type']}**: {item['id']}")

                        with col2:
                            st.metric("Access Count", item['access_count'])

                        with col3:
                            st.metric("Relevance", f"{item['relevance_score']:.1f}")

            return insights

        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
            return {}

    def process_question_with_tracking(self, session_id: str, user_role: str,
                                     query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a question with full tracking and analytics."""
        if not self.rag_engine or not self.role_router:
            raise ValueError("RAG engine and role router required for processing")

        import time
        start_time = time.time()

        try:
            # Determine conversation context
            conversation_turn = len(chat_history) // 2 + 1 if chat_history else 1
            follow_up_query = conversation_turn > 1

            # Classify query type
            query_type = self._classify_query_type(query)

            # Process through role router
            response = self.role_router.route(
                role=user_role,
                query=query,
                memory=self.memory_system,
                rag_engine=self.rag_engine,
                chat_history=chat_history
            )

            response_time = time.time() - start_time

            # Track the interaction
            self.track_user_interaction(
                session_id=session_id,
                user_role=user_role,
                query=query,
                query_type=query_type,
                response_time=response_time,
                rag_result=response,
                conversation_turn=conversation_turn,
                follow_up_query=follow_up_query
            )

            # Add metadata to response
            response['_metadata'] = {
                'response_time': response_time,
                'conversation_turn': conversation_turn,
                'query_type': query_type,
                'tracked': self.analytics_enabled
            }

            return response

        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            # Still track failed attempts
            if self.analytics_enabled:
                try:
                    failed_result = {'response': '', 'error': str(e)}
                    self.track_user_interaction(
                        session_id=session_id,
                        user_role=user_role,
                        query=query,
                        query_type=self._classify_query_type(query),
                        response_time=time.time() - start_time,
                        rag_result=failed_result
                    )
                except:
                    pass  # Don't let tracking errors break the main flow

            raise

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for analytics tracking."""
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ['code', 'architecture', 'function', 'class', 'method']):
            return 'technical'
        elif any(keyword in query_lower for keyword in ['career', 'background', 'experience', 'achievement']):
            return 'career'
        elif any(keyword in query_lower for keyword in ['mma', 'fight', 'ufc']):
            return 'mma'
        elif any(keyword in query_lower for keyword in ['fun', 'interesting', 'hobby']):
            return 'fun'
        else:
            return 'general'

    def get_trending_questions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get trending questions for display."""
        if not self.analytics_enabled:
            return []

        try:
            return self.analytics.get_most_common_questions(days=days, limit=5)
        except Exception as e:
            logger.error(f"Failed to get trending questions: {e}")
            return []

    def cleanup(self):
        """Clean up resources."""
        if self.analytics:
            try:
                self.analytics.close()
            except Exception as e:
                logger.warning(f"Failed to close analytics: {e}")


# Streamlit integration helpers
def init_common_questions_session_state():
    """Initialize session state for common questions functionality."""
    if 'questions_integration' not in st.session_state:
        st.session_state.questions_integration = None

    if 'selected_questions_history' not in st.session_state:
        st.session_state.selected_questions_history = []

    if 'last_question_selection_time' not in st.session_state:
        st.session_state.last_question_selection_time = None


def setup_questions_integration(rag_engine=None, memory_system=None, role_router=None) -> CommonQuestionsIntegration:
    """Set up common questions integration in Streamlit app."""
    if st.session_state.questions_integration is None:
        st.session_state.questions_integration = CommonQuestionsIntegration(
            rag_engine=rag_engine,
            memory_system=memory_system,
            role_router=role_router
        )

    return st.session_state.questions_integration


def display_question_metrics_sidebar():
    """Display question metrics in sidebar for power users."""
    integration = st.session_state.get('questions_integration')

    if integration and integration.analytics_enabled:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📈 Quick Stats")

            try:
                # Get basic stats
                insights = integration.analytics.get_user_behavior_insights(days=7)

                if insights and 'total_interactions' in insights:
                    st.metric("Questions This Week", insights['total_interactions'])

                    # Show top question
                    questions = integration.analytics.get_most_common_questions(days=7, limit=1)
                    if questions:
                        st.caption(f"Top: {questions[0]['question'][:30]}...")

            except Exception as e:
                st.caption("Stats unavailable")


# Example usage function for documentation
def example_integration_usage():
    """Example of how to use the common questions integration."""

    # In your main Streamlit app:
    init_common_questions_session_state()

    # Set up integration (typically done once per session)
    # integration = setup_questions_integration(rag_engine, memory_system, role_router)

    # Display welcome questions
    # selected_question = integration.display_welcome_questions(user_role)

    # Show sidebar suggestions
    # sidebar_question = integration.display_sidebar_suggestions(user_role)

    # Process question with tracking
    # if selected_question or sidebar_question:
    #     question = selected_question or sidebar_question
    #     response = integration.process_question_with_tracking(
    #         session_id=st.session_state.session_id,
    #         user_role=user_role,
    #         query=question,
    #         chat_history=st.session_state.chat_history
    #     )

    pass
