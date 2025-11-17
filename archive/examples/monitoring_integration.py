"""Example of integrating monitoring into your application."""

from src.core.rag_engine import RagEngine
from src.analytics.code_display_monitor import MonitoredRagEngine, CodeDisplayMonitor
from src.config.settings import Settings

def create_monitored_rag_engine():
    """Create a RagEngine with production monitoring."""
    settings = Settings()
    base_engine = RagEngine(settings=settings)
    monitor = CodeDisplayMonitor(metrics_file="logs/production_metrics.jsonl")

    return MonitoredRagEngine(base_engine, monitor)

# In your Streamlit app or API endpoints:
def handle_user_query(query: str, role: str):
    """Example of how to handle queries with monitoring."""
    engine = create_monitored_rag_engine()

    # This will automatically record performance metrics
    result = engine.retrieve_with_code(query, role)

    return result

# Dashboard endpoint for monitoring
def get_performance_dashboard():
    """Get performance metrics for admin dashboard."""
    monitor = CodeDisplayMonitor()

    return {
        "last_24h": monitor.get_performance_summary(24),
        "last_7d": monitor.get_performance_summary(24 * 7),
        "health_check": health_check()
    }
