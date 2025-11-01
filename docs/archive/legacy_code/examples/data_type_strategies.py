"""
Example: Data Management Strategies by Type
Demonstrates how different data types should be managed differently.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.analytics.data_management import AnalyticsDataManager
from src.analytics.comprehensive_analytics import UserInteraction, ContentAnalytics, BusinessMetrics
from src.core.memory import Memory
import json
import os


def demonstrate_data_type_strategies():
    """Show how different data types require different management strategies."""

    print("🗂️ Data Management Strategies by Type")
    print("=" * 60)

    # Initialize components
    manager = AnalyticsDataManager("strategy_demo.db", enable_privacy=True)
    memory = Memory("data/demo_session_memory.json")

    print("\n1️⃣ USER INTERACTIONS - High Value, Long Retention")
    print("-" * 50)

    # User interactions: Keep for long-term analytics
    interaction = UserInteraction(
        session_id="demo-session-001",
        timestamp=datetime.now(),
        user_role="Hiring Manager (technical)",
        query="What's Noah's experience with microservices?",
        query_type="technical",
        response_time=2.1,
        response_length=890,
        code_snippets_shown=1,
        citations_provided=4,
        success=True,
        user_rating=5.0,
        follow_up_query=True,
        conversation_turn=1
    )

    manager.record_interaction(interaction)
    print("✅ User interaction stored (365-day retention)")
    print(f"   - High business value: User behavior patterns")
    print(f"   - Privacy protected: PII anonymized")
    print(f"   - Analytics ready: Trend analysis, A/B testing")

    print("\n2️⃣ SESSION MEMORY - Temporary, Fast Access")
    print("-" * 50)

    # Session memory: Short-term conversation context
    chat_history = [
        {"role": "user", "content": "Tell me about Noah's background"},
        {"role": "assistant", "content": "Noah is a senior software engineer..."},
        {"role": "user", "content": "What about his technical skills?"}
    ]

    memory.store_session_context("demo-session-001", "Hiring Manager (technical)", chat_history)
    print("✅ Session memory stored (JSON file)")
    print(f"   - Fast access: In-memory + file persistence")
    print(f"   - Short retention: Should cleanup after 7-30 days")
    print(f"   - Growing concern: No automatic expiration")

    # Show current memory size issue
    memory_file = "data/demo_session_memory.json"
    if os.path.exists(memory_file):
        size_kb = os.path.getsize(memory_file) / 1024
        print(f"   - Current size: {size_kb:.1f} KB (grows indefinitely)")

    print("\n3️⃣ PERFORMANCE METRICS - High Volume, Short Retention")
    print("-" * 50)

    # Performance data: High frequency, operational value
    from src.analytics.data_management.performance import PerformanceMonitor
    perf_monitor = PerformanceMonitor(manager.connection)

    # Simulate recording performance metrics using the actual core database
    with manager.connection_lock:
        cursor = manager.connection.cursor()

        # Record query performance
        cursor.execute("""
            INSERT INTO query_performance (query_hash, execution_time, result_count, timestamp)
            VALUES (?, ?, ?, ?)
        """, ("query_hash_123", 0.45, 12, datetime.now()))

        # Record system performance
        cursor.execute("""
            INSERT INTO system_performance (metric_name, metric_value, timestamp)
            VALUES (?, ?, ?)
        """, ("cpu_usage", 23.5, datetime.now()))

        cursor.execute("""
            INSERT INTO system_performance (metric_name, metric_value, timestamp)
            VALUES (?, ?, ?)
        """, ("memory_usage", 67.2, datetime.now()))

        manager.connection.commit()

    print("✅ Performance metrics recorded (30-day retention)")
    print(f"   - High volume: Every query + system check")
    print(f"   - Operational value: Recent trends matter most")
    print(f"   - Storage efficient: Auto-cleanup prevents bloat")

    print("\n4️⃣ BUSINESS METRICS - Strategic Value, Long Retention")
    print("-" * 50)

    # Business metrics: Strategic decisions, keep longer
    # Note: The core system expects individual metric records, not the BusinessMetrics model
    with manager.connection_lock:
        cursor = manager.connection.cursor()

        # Record individual business metrics directly
        cursor.execute("""
            INSERT INTO business_metrics (metric_name, metric_value, metric_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("conversion_rate", 15.8, "percentage", datetime.now(), json.dumps({"source": "hiring_manager_interactions", "period": "weekly"})))

        cursor.execute("""
            INSERT INTO business_metrics (metric_name, metric_value, metric_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("avg_session_duration", 8.5, "minutes", datetime.now(), json.dumps({"source": "session_analytics", "period": "daily"})))

        cursor.execute("""
            INSERT INTO business_metrics (metric_name, metric_value, metric_type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("code_engagement_rate", 73.2, "percentage", datetime.now(), json.dumps({"source": "developer_interactions", "period": "weekly"})))

        manager.connection.commit()

    print("✅ Business metrics stored (365-day retention)")
    print(f"   - Strategic value: Business decisions, ROI analysis")
    print(f"   - Long-term trends: Year-over-year comparisons")
    print(f"   - Executive reporting: Board meetings, planning")

    print("\n5️⃣ CONTENT ANALYTICS - Medium Retention")
    print("-" * 50)

    # Content analytics: Understanding what works
    # Note: The core system expects different fields than the comprehensive_analytics model
    with manager.connection_lock:
        cursor = manager.connection.cursor()
        cursor.execute("""
            INSERT INTO content_analytics
            (content_type, engagement_score, effectiveness_score, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "code_examples",
            0.85,
            0.92,
            datetime.now(),
            json.dumps({"language": "python", "complexity": "intermediate"})
        ))

        cursor.execute("""
            INSERT INTO content_analytics
            (content_type, engagement_score, effectiveness_score, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "career_content",
            0.78,
            0.88,
            datetime.now(),
            json.dumps({"topic": "technical_background", "user_role": "hiring_manager"})
        ))

        manager.connection.commit()

    print("✅ Content analytics stored (180-day retention)")
    print(f"   - Content optimization: What resonates with users")
    print(f"   - Medium retention: Seasonal patterns, content lifecycle")
    print(f"   - Product insights: Feature usage, content gaps")

    print("\n📊 CURRENT RETENTION POLICIES")
    print("-" * 50)

    # Show current retention policies
    retention_info = manager.get_system_health()
    print("Current data retention strategies:")
    print(f"   • User Interactions: 365 days (High business value)")
    print(f"   • Session Analytics: 365 days (Aggregated summaries)")
    print(f"   • Performance Data: 30-90 days (Operational metrics)")
    print(f"   • Business Metrics: 365 days (Strategic decisions)")
    print(f"   • Content Analytics: 180 days (Product optimization)")
    print(f"   • Session Memory: ⚠️ No expiration (NEEDS FIX)")

    print("\n🚨 IDENTIFIED ISSUES")
    print("-" * 50)

    print("1. Session Memory Growth:")
    print("   - No automatic cleanup")
    print("   - All sessions loaded into RAM")
    print("   - Privacy risk from old conversations")

    print("\n2. Performance Data Volume:")
    print("   - Raw metrics stored without aggregation")
    print("   - Could benefit from hourly/daily summaries")

    print("\n3. Missing Data Lifecycle:")
    print("   - No hot/warm/cold storage tiers")
    print("   - No automated archival process")

    print("\n💡 RECOMMENDED IMPROVEMENTS")
    print("-" * 50)

    print("Priority 1 - Session Memory Management:")
    print("   • Add 7-day expiration for session memory")
    print("   • Implement tiered storage (RAM → JSON → Archive)")
    print("   • Add privacy-compliant session cleanup")

    print("\nPriority 2 - Performance Data Aggregation:")
    print("   • Aggregate raw metrics into hourly/daily summaries")
    print("   • Keep raw data for 7 days, aggregates for 90 days")
    print("   • Implement real-time dashboard caching")

    print("\nPriority 3 - Data Lifecycle Management:")
    print("   • Hot data (0-7 days): Fast access, full detail")
    print("   • Warm data (7-90 days): Compressed, indexed")
    print("   • Cold data (90+ days): Archived, summarized")

    # Cleanup
    manager.close()

    print("\n✅ Data strategy demonstration complete!")
    print("\nNext steps: Implement session memory expiration and performance aggregation")


if __name__ == "__main__":
    demonstrate_data_type_strategies()
