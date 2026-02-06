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
    - Whether responses are grounded in retrieved context
    """
    message_id: int
    topk_ids: List[int]  # IDs of retrieved kb_chunks
    scores: List[float]  # Similarity scores
    grounded: bool  # Did the response cite sources?


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
