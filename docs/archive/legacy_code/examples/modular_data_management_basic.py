"""
Example: Basic usage of the modular data management system.
This shows how the refactored system maintains the same API.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime
from src.analytics.data_management import AnalyticsDataManager
from src.analytics.comprehensive_analytics import UserInteraction


def basic_usage_example():
    """Demonstrate basic usage - same as before refactoring."""

    print("🔧 Basic Modular Data Management Example")
    print("=" * 50)

    # Initialize the system (same API as before)
    manager = AnalyticsDataManager("analytics_example.db", enable_privacy=True)

    # Create a sample interaction
    interaction = UserInteraction(
        session_id="example-session-001",
        timestamp=datetime.now(),
        user_role="developer",
        query="How do I implement user authentication?",
        query_type="technical",
        response_time=3.2,
        response_length=450,
        code_snippets_shown=2,
        citations_provided=3,
        success=True,
        user_rating=4.5,
        follow_up_query=False,
        conversation_turn=1
    )

    # Record the interaction (same method as before)
    success = manager.record_interaction(interaction)
    print(f"✅ Interaction recorded: {success}")

    # Get analytics summary (same method as before)
    summary = manager.get_analytics_summary(days_back=7)
    print(f"📊 Analytics summary generated:")
    print(f"   - Total interactions: {summary['interaction_metrics']['total_interactions']}")
    print(f"   - Success rate: {summary['interaction_metrics']['success_rate']}%")
    print(f"   - Average response time: {summary['interaction_metrics']['avg_response_time']}s")

    # Get system health (enhanced feature)
    health = manager.get_system_health()
    print(f"🏥 System health score: {health['overall_health_score']:.2f}")

    # Current Retention Policies (from your system):
    print("\n📂 Retention Policies:")
    print("user_interactions: 365 days      # ✅ Good")
    print("session_analytics: 365 days      # ✅ Good")
    print("performance_data: 30-90 days     # ✅ Good")
    print("business_metrics: 365 days       # ✅ Good")
    print("content_analytics: 180 days      # ✅ Good")
    print("session_memory: INDEFINITE       # ❌ NEEDS FIX")

    # Clean up
    manager.close()

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    basic_usage_example()
