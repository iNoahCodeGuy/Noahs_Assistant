"""Data reporting and analytics display utilities.

This module handles the generation of comprehensive data reports for technical
stakeholders who request to see collected analytics. It fetches all datasets
from Supabase and formats them into analyst-grade markdown tables.
"""

import logging
from typing import Any, Dict, List, Optional

from src.analytics.supabase_analytics import supabase_analytics

logger = logging.getLogger(__name__)


def normalize_value(value: Any) -> str:
    """Normalize any value for safe display in markdown tables.

    Args:
        value: Any Python value (None, list, dict, str, int, etc.)

    Returns:
        String representation safe for markdown tables with escaped pipes.
        Long values are truncated to 120 characters.
    """
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(normalize_value(item) for item in value)
    if isinstance(value, dict):
        items = [f"{k}: {normalize_value(v)}" for k, v in value.items()]
        return ", ".join(items)
    text = str(value).replace("\n", " ").strip()
    text = text.replace("|", "\\|")  # Escape markdown pipes
    return text if len(text) <= 120 else text[:117] + "..."


def format_table(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    """Format data rows into a markdown table.

    Args:
        headers: Column names for the table
        rows: List of dictionaries containing row data

    Returns:
        Markdown-formatted table string. Returns "No records found." if rows is empty.
    """
    if not rows:
        return "No records found."

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator]

    for row in rows:
        line = "| " + " | ".join(normalize_value(row.get(header)) for header in headers) + " |"
        lines.append(line)

    return "\n".join(lines)


def render_full_data_report() -> str:
    """Generate comprehensive data report across all Supabase tables.

    This function:
    1. Fetches all records from messages, retrieval_logs, feedback, confessions, sms_logs
    2. Summarizes knowledge base coverage (kb_chunks)
    3. Calculates key performance metrics and insights
    4. Formats everything into analyst-grade markdown tables
    5. Includes dataset inventory with record counts and last entry timestamps

    Returns:
        Markdown-formatted report with multiple sections and tables.
        Returns error message if Supabase connection fails.
    """
    try:
        client = supabase_analytics.client
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Unable to initialize Supabase client for data report: %s", exc)
        return "There was an issue retrieving the stored datasets. Please verify Supabase credentials."

    # Define which columns to fetch for each table
    table_configs = {
        "messages": ["id", "role_mode", "query_type", "latency_ms", "success", "created_at"],
        "retrieval_logs": ["id", "message_id", "grounded", "topk_ids", "scores", "created_at"],
        "feedback": ["id", "rating", "comment", "user_name", "user_email", "user_phone", "contact_requested", "created_at"],
        "confessions": ["id", "is_anonymous", "name", "email", "phone", "created_at", "message"],
        "sms_logs": ["id", "event", "status", "is_urgent", "twilio_sid", "created_at", "message_preview"],
    }

    dataset_rows: Dict[str, List[Dict[str, Any]]] = {}
    inventory: List[Dict[str, Any]] = []

    # Fetch data from each table
    for table_name, columns in table_configs.items():
        try:
            selection = ",".join(columns)
            result = client.table(table_name).select(selection).order(
                "created_at" if "created_at" in columns else "id", desc=True
            ).execute()
            rows = result.data or []
            dataset_rows[table_name] = rows

            # Calculate last entry timestamp
            last_timestamp: Optional[str] = None
            if rows and "created_at" in columns:
                # Already ordered desc, so first is latest
                last_timestamp = rows[0].get("created_at") if rows else None

            inventory.append({
                "Dataset": table_name,
                "Records": len(rows),
                "Last Entry": last_timestamp or "—",
            })
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to fetch %s data: %s", table_name, exc)
            dataset_rows[table_name] = []
            inventory.append({
                "Dataset": table_name,
                "Records": "Error",
                "Last Entry": str(exc),
            })

    # Knowledge base summary (excluding embeddings for readability)
    # Aggregate by SOURCE only (not every individual section entry)
    kb_summary_rows: List[Dict[str, Any]] = []
    try:
        kb_result = client.table("kb_chunks").select("doc_id, section").execute()
        kb_data = kb_result.data or []
        inventory.append({
            "Dataset": "kb_chunks",
            "Records": len(kb_data),
            "Last Entry": "—",
        })

        # Aggregate chunks by SOURCE (doc_id) only - professional summary
        source_aggregation: Dict[str, int] = {}
        for row in kb_data:
            doc_id = row.get("doc_id", "unknown")
            source_aggregation[doc_id] = source_aggregation.get(doc_id, 0) + 1

        # Create clean summary: 1 row per knowledge source
        for doc_id, total_chunks in sorted(source_aggregation.items()):
            kb_summary_rows.append({
                "Knowledge Source": doc_id,
                "Total Chunks": total_chunks,
            })
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to summarize kb_chunks: %s", exc)
        kb_summary_rows = [{"Knowledge Source": "kb_chunks", "Total Chunks": f"Error: {exc}"}]

    # Build report sections
    report_sections: List[str] = []

    # 1. Dataset Inventory (overview of all tables)
    report_sections.append(
        "#### Dataset Inventory\n" + format_table(["Dataset", "Records", "Last Entry"], inventory)
    )

    # 2. Knowledge Base Coverage
    if kb_summary_rows:
        report_sections.append(
            "#### Knowledge Base Coverage\n" + format_table(["Knowledge Source", "Total Chunks"], kb_summary_rows)
        )

    # 3. Key Performance Metrics (Analytics Insights)
    messages = dataset_rows.get("messages", [])
    if messages:
        total_messages = len(messages)
        successful = sum(1 for m in messages if m.get("success"))
        avg_latency = sum(m.get("latency_ms", 0) for m in messages) / total_messages if total_messages > 0 else 0

        # Role distribution
        role_counts: Dict[str, int] = {}
        for m in messages:
            role = m.get("role_mode", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        top_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Query type distribution
        query_counts: Dict[str, int] = {}
        for m in messages:
            qtype = m.get("query_type", "unknown")
            query_counts[qtype] = query_counts.get(qtype, 0) + 1
        top_query_types = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        metrics_rows = [
            {"Metric": "Total Conversations", "Value": total_messages},
            {"Metric": "Success Rate", "Value": f"{(successful/total_messages*100):.1f}%" if total_messages > 0 else "0%"},
            {"Metric": "Avg Response Time", "Value": f"{avg_latency:.0f}ms"},
            {"Metric": "Top Role", "Value": f"{top_roles[0][0]} ({top_roles[0][1]} queries)" if top_roles else "—"},
            {"Metric": "Top Query Type", "Value": f"{top_query_types[0][0]} ({top_query_types[0][1]} queries)" if top_query_types else "—"},
        ]

        report_sections.append(
            "#### Key Performance Metrics\n" + format_table(["Metric", "Value"], metrics_rows)
        )

    # 4. Recent Activity (limit to last 10 messages for readability)
    if messages:
        recent_messages = messages[:10]  # Already sorted desc by created_at
        headers = table_configs["messages"]
        report_sections.append(
            "#### Recent Conversations (Last 10)\n" + format_table(headers, recent_messages)
        )

    # 5. Other detailed tables (only if they have data, and limit confessions for privacy)
    for table_name in ["retrieval_logs", "feedback", "sms_logs"]:
        rows = dataset_rows.get(table_name, [])
        if rows:
            headers = table_configs[table_name]
            # Limit to 10 most recent for readability
            display_rows = rows[:10]
            section_title = f"#### {table_name.replace('_', ' ').title()} (Recent)"
            report_sections.append(section_title + "\n" + format_table(headers, display_rows))

    # Confessions: Show count only for privacy, no details
    confessions = dataset_rows.get("confessions", [])
    if confessions:
        report_sections.append(
            f"#### Confessions\n**Total Received**: {len(confessions)} (details withheld for privacy)"
        )

    return "\n\n".join(report_sections)
