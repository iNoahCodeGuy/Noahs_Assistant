"""Production monitoring for code display functionality.

This module provides monitoring capabilities to track code display
performance and accuracy in production environments.
"""
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CodeDisplayMetrics:
    """Metrics for code display operations."""
    timestamp: datetime
    query_time: float
    code_snippets_found: int
    citation_accuracy: bool
    role: str
    query_type: str
    response_size: int
    error_occurred: bool = False
    error_message: str = ""


class CodeDisplayMonitor:
    """Monitor code display performance and accuracy in production."""

    def __init__(self, metrics_file: str = "logs/code_display_metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(exist_ok=True)

        # Performance baselines (from testing)
        self.baselines = {
            'max_query_time': 10.0,  # seconds
            'min_code_snippets': 1,   # for technical roles
            'max_response_size': 50000,  # characters
        }

    def record_query(self, query: str, role: str, start_time: float, result: Dict[str, Any]) -> CodeDisplayMetrics:
        """Record metrics for a code display query."""
        end_time = time.time()
        query_time = end_time - start_time

        # Analyze result quality
        code_snippets_found = len(result.get('code_snippets', []))
        citation_accuracy = self._check_citation_accuracy(result)

        # Determine query type
        query_type = 'technical' if role in ['Software Developer', 'Hiring Manager (technical)'] else 'career'

        # Calculate response size
        response_content = str(result.get('response', ''))
        response_size = len(response_content)

        metrics = CodeDisplayMetrics(
            timestamp=datetime.now(),
            query_time=query_time,
            code_snippets_found=code_snippets_found,
            citation_accuracy=citation_accuracy,
            role=role,
            query_type=query_type,
            response_size=response_size
        )

        # Check for performance issues
        self._check_performance_alerts(metrics)

        # Log metrics
        self._log_metrics(metrics)

        return metrics

    def _check_citation_accuracy(self, result: Dict[str, Any]) -> bool:
        """Check if citations follow expected format."""
        code_snippets = result.get('code_snippets', [])

        for snippet in code_snippets:
            citation = snippet.get('citation', '')
            if not citation or ':' not in citation:
                return False

            # Check for proper file:line format
            parts = citation.split(':')
            if len(parts) < 2:
                return False

            filename = parts[0]
            if not filename.endswith('.py'):
                return False

        return True

    def _check_performance_alerts(self, metrics: CodeDisplayMetrics):
        """Check metrics against baselines and alert if needed."""
        alerts = []

        if metrics.query_time > self.baselines['max_query_time']:
            alerts.append(f"Slow query: {metrics.query_time:.2f}s > {self.baselines['max_query_time']}s")

        if (metrics.query_type == 'technical' and
            metrics.code_snippets_found < self.baselines['min_code_snippets']):
            alerts.append(f"Low code results: {metrics.code_snippets_found} snippets for technical query")

        if metrics.response_size > self.baselines['max_response_size']:
            alerts.append(f"Large response: {metrics.response_size} chars")

        if not metrics.citation_accuracy:
            alerts.append("Citation format issues detected")

        if alerts:
            logger.warning(f"Code display performance alerts: {'; '.join(alerts)}")
            # In production, you might send these to monitoring systems like:
            # - Datadog, New Relic, CloudWatch
            # - Slack/Discord notifications
            # - PagerDuty for critical issues

    def _log_metrics(self, metrics: CodeDisplayMetrics):
        """Log metrics to file for analysis."""
        with open(self.metrics_file, 'a') as f:
            # Convert datetime to ISO string for JSON serialization
            metrics_dict = asdict(metrics)
            metrics_dict['timestamp'] = metrics.timestamp.isoformat()
            f.write(json.dumps(metrics_dict) + '\n')

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if not self.metrics_file.exists():
            return {"error": "No metrics data available"}

        recent_metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp > cutoff_time:
                        recent_metrics.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue

        if not recent_metrics:
            return {"error": f"No metrics data in last {hours} hours"}

        # Calculate summary statistics
        query_times = [m['query_time'] for m in recent_metrics]
        code_results = [m['code_snippets_found'] for m in recent_metrics if m['query_type'] == 'technical']
        citation_accuracy = [m['citation_accuracy'] for m in recent_metrics]

        return {
            "period_hours": hours,
            "total_queries": len(recent_metrics),
            "avg_query_time": sum(query_times) / len(query_times),
            "max_query_time": max(query_times),
            "avg_code_snippets": sum(code_results) / len(code_results) if code_results else 0,
            "citation_accuracy_rate": sum(citation_accuracy) / len(citation_accuracy),
            "performance_alerts": sum(1 for qt in query_times if qt > self.baselines['max_query_time']),
            "timestamp": datetime.now().isoformat()
        }


# Integration with RagEngine
class MonitoredRagEngine:
    """Wrapper for RagEngine that adds monitoring."""

    def __init__(self, rag_engine, monitor: CodeDisplayMonitor = None):
        self.rag_engine = rag_engine
        self.monitor = monitor or CodeDisplayMonitor()

    def retrieve_with_code(self, query: str, role: str = None, **kwargs):
        """Monitored version of retrieve_with_code."""
        start_time = time.time()

        try:
            result = self.rag_engine.retrieve_with_code(query, role, **kwargs)
            self.monitor.record_query(query, role, start_time, result)
            return result
        except Exception as e:
            # Record error metrics
            error_result = {"error": str(e), "code_snippets": []}
            metrics = self.monitor.record_query(query, role, start_time, error_result)
            metrics.error_occurred = True
            metrics.error_message = str(e)
            raise

    def __getattr__(self, name):
        """Delegate other methods to the wrapped RagEngine."""
        return getattr(self.rag_engine, name)


# Health check endpoint for monitoring systems
def health_check() -> Dict[str, Any]:
    """Health check for code display functionality."""
    try:
        from assistant.core.rag_engine import RagEngine
        from assistant.config.settings import Settings

        settings = Settings()
        engine = RagEngine(settings=settings)

        # Quick functionality test
        start_time = time.time()
        result = engine.retrieve_with_code("test health check", role="Software Developer")
        response_time = time.time() - start_time

        return {
            "status": "healthy",
            "response_time": response_time,
            "code_index_version": engine.code_index_version(),
            "has_pgvector": bool(engine.pgvector_retriever),
            "degraded_mode": getattr(engine, 'degraded_mode', False),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    monitor = CodeDisplayMonitor()
    summary = monitor.get_performance_summary(24)
    print(f"Performance summary: {summary}")

    health = health_check()
    print(f"Health check: {health}")
