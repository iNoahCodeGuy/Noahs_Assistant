"""Supabase-based analytics system for Noah's AI Assistant.

This module replaces the Google Cloud SQL + Pub/Sub analytics system with
a simpler Supabase Postgres implementation. All events are logged directly
to the database without the complexity of event streaming.

Key differences from GCP version:
- No Pub/Sub → Direct database writes (simpler, cheaper)
- No Secret Manager → Environment variables
- pgvector → Integrated vector search instead of separate Vertex AI
- Supabase RLS → Built-in security instead of IAM policies

Cost savings: ~$100-200/month (GCP) → ~$25-50/month (Supabase)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import uuid

from assistant.config.supabase_config import get_supabase_client, supabase_settings

logger = logging.getLogger(__name__)


@dataclass
class UserInteractionData:
    """Data structure for logging user interactions.

    This replaces the UserInteraction model from the GCP version.
    Maps directly to the 'messages' table in Supabase.

    Why this structure:
    - session_id: Track conversation flows
    - role_mode: Analyze behavior by user type
    - query/answer: Store the actual conversation
    - latency_ms: Monitor performance
    - tokens_*: Track OpenAI usage for cost optimization
    """
    session_id: str
    role_mode: str
    query: str
    answer: str
    query_type: str  # technical, career, mma, fun, general
    latency_ms: int
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    success: bool = True
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class RetrievalLogData:
    """Data structure for logging RAG retrieval events.

    This helps us understand:
    - Which KB chunks are most useful
    - If similarity scores are good enough
    - Whether responses are properly grounded in retrieved context
    """
    message_id: int
    topk_ids: List[int]  # IDs of retrieved kb_chunks
    scores: List[float]  # Similarity scores
    grounded: bool  # Did the response cite sources?


@dataclass
class QualityEventData:
    """Data structure for logging quality validation events.

    This helps us understand:
    - How often quality warnings are triggered
    - Whether warnings are more common for RAG or template responses
    - At which conversation turn warnings occur most
    - What specific warning types are most frequent

    Used for data-driven decisions about whether proofreading is needed.
    """
    session_id: str
    conversation_turn: int
    warning_type: Optional[str]  # None if no warning, else "answer_relevance_low_0.25" etc
    response_path: str  # "rag" or "template"
    query_preview: str  # First 200 chars of query
    answer_preview: Optional[str] = None  # Only for flagged responses (first 500 chars)
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None  # role_mode, menu_branch, engagement_score, etc.


class SupabaseAnalytics:
    """Analytics system using Supabase Postgres.

    This replaces CloudAnalytics from the GCP implementation.
    Much simpler because:
    1. No connection pooling needed (Supabase handles it)
    2. No Pub/Sub → direct writes
    3. No manual table creation → handled by migrations
    4. Built-in RLS for security

    Example usage:
        from analytics.supabase_analytics import supabase_analytics

        # Log a chat interaction
        interaction = UserInteractionData(
            session_id="abc-123",
            role_mode="Software Developer",
            query="How does RAG work?",
            answer="RAG combines retrieval with generation...",
            query_type="technical",
            latency_ms=1500,
            tokens_prompt=50,
            tokens_completion=200
        )

        message_id = supabase_analytics.log_interaction(interaction)
    """

    def __init__(self):
        """Initialize Supabase client.

        Why lazy initialization:
        - Client creation happens on first use
        - Tests can mock get_supabase_client easily
        - Allows app to start even if Supabase is temporarily down
        """
        self._client = None

    @property
    def client(self):
        """Get Supabase client with lazy initialization."""
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    def log_interaction(self, interaction: UserInteractionData) -> Optional[int]:
        """Log a user interaction to the messages table.

        Args:
            interaction: User interaction data

        Returns:
            Message ID if successful, None if failed

        Why this approach:
        - Returns ID so we can log retrieval info later
        - Fails gracefully (logs error but doesn't crash app)
        - Simple direct write (no Pub/Sub complexity)
        """
        try:
            result = self.client.table('messages').insert({
                'session_id': interaction.session_id,
                'role_mode': interaction.role_mode,
                'query': interaction.query,
                'answer': interaction.answer,
                'query_type': interaction.query_type,
                'latency_ms': interaction.latency_ms,
                'tokens_prompt': interaction.tokens_prompt,
                'tokens_completion': interaction.tokens_completion,
                'success': interaction.success,
                'created_at': interaction.timestamp.isoformat()
            }).execute()

            message_id = result.data[0]['id'] if result.data else None
            logger.info(f"Logged interaction for session {interaction.session_id}, message_id: {message_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            return None

    def log_retrieval(self, retrieval_log: RetrievalLogData):
        """Log retrieval information for a message.

        This helps us analyze:
        - Which KB chunks are most relevant
        - If our similarity threshold is good
        - Whether responses are properly grounded

        Args:
            retrieval_log: Retrieval event data
        """
        try:
            self.client.table('retrieval_logs').insert({
                'message_id': retrieval_log.message_id,
                'topk_ids': retrieval_log.topk_ids,
                'scores': retrieval_log.scores,
                'grounded': retrieval_log.grounded
            }).execute()

            logger.debug(f"Logged retrieval for message {retrieval_log.message_id}")

        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")

    def log_feedback(self, message_id: int, rating: int, comment: str = "",
                    contact_requested: bool = False, user_email: str = "",
                    user_name: str = "", user_phone: str = ""):
        """Log user feedback.

        Args:
            message_id: ID of the message being rated
            rating: 1-5 star rating (or 0 if not rated)
            comment: Optional feedback text
            contact_requested: Whether user wants to be contacted
            user_email: User's email (if contact requested)
            user_name: User's name (if provided)
            user_phone: User's phone (if provided)

        Returns:
            Feedback ID if successful

        Side effects:
        - If contact_requested=True, triggers Twilio SMS notification
          (handled by a separate background job or API route)
        """
        try:
            result = self.client.table('feedback').insert({
                'message_id': message_id,
                'rating': rating,
                'comment': comment,
                'contact_requested': contact_requested,
                'user_email': user_email,
                'user_name': user_name,
                'user_phone': user_phone
            }).execute()

            feedback_id = result.data[0]['id'] if result.data else None
            logger.info(f"Logged feedback for message {message_id}, feedback_id: {feedback_id}")

            # If contact requested, we should send notification
            # This is handled by the /api/feedback API route
            if contact_requested:
                logger.info(f"Contact requested for message {message_id}, email: {user_email}")

            return feedback_id

        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return None

    def get_user_behavior_insights(self, days: int = 30) -> Dict[str, Any]:
        """Generate user behavior insights for the last N days.

        This replaces the complex analytics queries from GCP version
        with simpler Supabase queries using the analytics_by_role view.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with insights: {
                'period_days': 30,
                'total_messages': 150,
                'by_role': [...]
                'avg_latency_ms': 1234,
                ...
            }
        """
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get messages from the last N days
            messages = self.client.table('messages')\
                .select('role_mode, latency_ms, success, created_at')\
                .gte('created_at', cutoff_date)\
                .execute()

            if not messages.data:
                return {
                    'period_days': days,
                    'total_messages': 0,
                    'message': 'No data for this period'
                }

            # Analyze by role
            role_stats = {}
            for msg in messages.data:
                role = msg['role_mode']
                if role not in role_stats:
                    role_stats[role] = {
                        'count': 0,
                        'total_latency': 0,
                        'successes': 0
                    }

                role_stats[role]['count'] += 1
                role_stats[role]['total_latency'] += msg.get('latency_ms', 0)
                if msg.get('success'):
                    role_stats[role]['successes'] += 1

            # Calculate aggregates
            by_role = []
            for role, stats in role_stats.items():
                by_role.append({
                    'role': role,
                    'count': stats['count'],
                    'avg_latency_ms': stats['total_latency'] // stats['count'] if stats['count'] > 0 else 0,
                    'success_rate': stats['successes'] / stats['count'] if stats['count'] > 0 else 0
                })

            return {
                'period_days': days,
                'total_messages': len(messages.data),
                'by_role': by_role,
                'avg_latency_ms': sum(m.get('latency_ms', 0) for m in messages.data) // len(messages.data)
            }

        except Exception as e:
            logger.error(f"Failed to get user behavior insights: {e}")
            return {'error': str(e)}

    def log_quality_event(self, event: QualityEventData) -> Optional[int]:
        """Log a quality validation event to the quality_events table.

        This helps track:
        - How often quality warnings are triggered
        - Whether warnings are more common for RAG or template responses
        - At which conversation turn warnings occur most
        - What specific warning types are most frequent

        Args:
            event: Quality event data

        Returns:
            Event ID if successful, None if failed

        Design decisions:
        - Only store answer_preview if warning_type is set (flagged responses)
        - This saves storage and respects privacy for unflagged responses
        - Fails gracefully (logs error but doesn't crash app)
        """
        try:
            # Only store answer_preview if warning_type is set (flagged)
            answer_preview = event.answer_preview if event.warning_type else None

            result = self.client.table('quality_events').insert({
                'session_id': event.session_id,
                'conversation_turn': event.conversation_turn,
                'warning_type': event.warning_type,
                'response_path': event.response_path,
                'query_preview': event.query_preview[:200] if event.query_preview else None,  # Limit to 200 chars
                'answer_preview': answer_preview[:500] if answer_preview else None,  # Limit to 500 chars
                'retry_count': event.retry_count,
                'metadata': event.metadata or {}
            }).execute()

            event_id = result.data[0]['id'] if result.data else None
            logger.debug(f"Logged quality event for session {event.session_id}, event_id: {event_id}")
            return event_id

        except Exception as e:
            logger.error(f"Failed to log quality event: {e}")
            return None

    def get_quality_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get quality metrics for the last N days.

        Returns aggregated statistics about quality warnings:
        - Total events and warning rate
        - Breakdown by warning type
        - Breakdown by response path (RAG vs template)
        - Average conversation turn when warnings occur

        Args:
            days: Number of days to analyze (default: 7)

        Returns:
            Dictionary with quality metrics:
            {
                'period_days': 7,
                'total_events': 150,
                'events_with_warnings': 12,
                'warning_rate': 0.08,
                'by_warning_type': [...],
                'by_response_path': [...],
                'avg_turn_with_warning': 2.5
            }
        """
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get all events from the last N days
            events = self.client.table('quality_events')\
                .select('warning_type, response_path, conversation_turn')\
                .gte('created_at', cutoff_date)\
                .execute()

            if not events.data:
                return {
                    'period_days': days,
                    'total_events': 0,
                    'message': 'No data for this period'
                }

            total_events = len(events.data)
            events_with_warnings = sum(1 for e in events.data if e.get('warning_type'))

            # Breakdown by warning type
            warning_type_counts = {}
            warning_turns = []
            for event in events.data:
                warning_type = event.get('warning_type')
                if warning_type:
                    warning_type_counts[warning_type] = warning_type_counts.get(warning_type, 0) + 1
                    turn = event.get('conversation_turn')
                    if turn:
                        warning_turns.append(turn)

            by_warning_type = [
                {'warning_type': wt, 'count': count}
                for wt, count in sorted(warning_type_counts.items(), key=lambda x: x[1], reverse=True)
            ]

            # Breakdown by response path
            path_counts = {}
            path_warning_counts = {}
            for event in events.data:
                path = event.get('response_path', 'unknown')
                path_counts[path] = path_counts.get(path, 0) + 1
                if event.get('warning_type'):
                    path_warning_counts[path] = path_warning_counts.get(path, 0) + 1

            by_response_path = [
                {
                    'response_path': path,
                    'total_events': count,
                    'events_with_warnings': path_warning_counts.get(path, 0),
                    'warning_rate': path_warning_counts.get(path, 0) / count if count > 0 else 0
                }
                for path, count in sorted(path_counts.items())
            ]

            return {
                'period_days': days,
                'total_events': total_events,
                'events_with_warnings': events_with_warnings,
                'warning_rate': events_with_warnings / total_events if total_events > 0 else 0,
                'by_warning_type': by_warning_type,
                'by_response_path': by_response_path,
                'avg_turn_with_warning': sum(warning_turns) / len(warning_turns) if warning_turns else None
            }

        except Exception as e:
            logger.error(f"Failed to get quality summary: {e}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the analytics system.

        Returns:
            Dictionary with status and metrics
        """
        try:
            # Simple query to test database connection
            result = self.client.table('messages')\
                .select('id', count='exact')\
                .limit(1)\
                .execute()

            # Get recent message count
            recent_cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            recent = self.client.table('messages')\
                .select('id', count='exact')\
                .gte('created_at', recent_cutoff)\
                .execute()

            return {
                "status": "healthy",
                "database_connected": True,
                "total_messages": result.count if hasattr(result, 'count') else 0,
                "recent_messages_24h": recent.count if hasattr(recent, 'count') else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "database_connected": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global analytics instance
# This replaces 'cloud_analytics' from the GCP implementation
supabase_analytics = SupabaseAnalytics()
