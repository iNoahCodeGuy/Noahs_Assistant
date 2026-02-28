"""Analytics display utilities for live data rendering.

This module handles the formatting of live analytics data from Supabase
into analyst-quality markdown tables with smart follow-up suggestions.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_value(value: Any) -> str:
    """Normalize any value for safe display in markdown tables.

    Args:
        value: Any Python value (None, list, dict, str, int, etc.)

    Returns:
        String representation safe for markdown tables with escaped pipes.
        Long values are truncated to 140 characters.
    """
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        items = [f"{k}: {v}" for k, v in value.items()]
        return ", ".join(items)
    text = str(value).replace("\n", " ").strip()
    text = text.replace("|", "\\|")  # Escape markdown pipes
    return text if len(text) <= 140 else text[:137] + "..."


def format_table(headers: List[str], rows: List[Dict[str, Any]], table_name: str = "") -> str:
    """Format data rows into a markdown table.

    Args:
        headers: Column names for the table
        rows: List of dictionaries containing row data
        table_name: Optional table name for empty state messaging

    Returns:
        Markdown-formatted table string with proper handling of empty states.
    """
    if not rows:
        return f"No {table_name} records found."

    # Header row
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator]

    # Data rows
    for row in rows:
        values = []
        for header in headers:
            # Handle snake_case to column names
            key = header.lower().replace(" ", "_")
            value = row.get(key, row.get(header, "—"))
            values.append(normalize_value(value))
        line = "| " + " | ".join(values) + " |"
        lines.append(line)

    return "\n".join(lines)


def render_inventory_table(inventory: Dict[str, int]) -> str:
    """Render dataset inventory as a summary table.

    Args:
        inventory: Dict with table names as keys and counts as values

    Returns:
        Markdown table showing all datasets and their row counts
    """
    rows = []
    for table, count in inventory.items():
        rows.append({
            "Dataset": table.replace("_", " ").title(),
            "Records": f"{count:,}"
        })

    return format_table(["Dataset", "Records"], rows, "inventory")


def render_analytics_response(data: Dict[str, Any], role: str = "Hiring Manager (technical)") -> str:
    """Render full analytics response with all tables.

    Args:
        data: Analytics data from /api/analytics endpoint
        role: User role for determining detail level

    Returns:
        Complete markdown report with tables and CTAs
    """
    components = []

    # Title
    components.append("# 📊 Live Analytics Dashboard\n")
    components.append(f"*Generated at: {data.get('generated_at', 'N/A')}*\n")

    # 1. Dataset Inventory
    if "inventory" in data:
        components.append("\n## Dataset Inventory\n")
        components.append(render_inventory_table(data["inventory"]))

    # 2. Messages (last 50)
    if "messages" in data and data["messages"].get("data"):
        components.append("\n## Recent Messages (Last 50)\n")
        messages_table = format_table(
            ["ID", "Role Mode", "User Query", "Latency (ms)", "Token Count", "Created At"],
            data["messages"]["data"],
            "messages"
        )
        components.append(messages_table)
    elif "messages" in data and data["messages"].get("error"):
        components.append(f"\n## Recent Messages\n*Could not load due to {data['messages']['error']}; try again.*\n")

    # 3. Retrieval Logs (last 50)
    if "retrieval_logs" in data and data["retrieval_logs"].get("data"):
        components.append("\n## Retrieval Logs (Last 50)\n")
        retrieval_table = format_table(
            ["Message ID", "Chunk ID", "Similarity Score", "Grounded", "Created At"],
            data["retrieval_logs"]["data"],
            "retrieval_logs"
        )
        components.append(retrieval_table)
    elif "retrieval_logs" in data and data["retrieval_logs"].get("error"):
        components.append(f"\n## Retrieval Logs\n*Could not load due to {data['retrieval_logs']['error']}; try again.*\n")

    # 4. Feedback (last 50) - with PII redaction note
    if "feedback" in data and data["feedback"].get("data"):
        components.append("\n## User Feedback (Last 50)\n")
        components.append("*Note: Email addresses and phone numbers are redacted for privacy.*\n\n")
        feedback_table = format_table(
            ["Message ID", "Rating", "Comment", "Contact Requested", "Created At"],
            data["feedback"]["data"],
            "feedback"
        )
        components.append(feedback_table)
    elif "feedback" in data and data["feedback"].get("error"):
        components.append(f"\n## User Feedback\n*Could not load due to {data['feedback']['error']}; try again.*\n")

    # 5. Confessions (last 5 only, privacy-focused)
    if role in ["Hiring Manager (technical)", "Software Developer"]:
        if "confessions" in data and data["confessions"].get("data"):
            components.append("\n## Confessions (Last 5 - Privacy Protected)\n")
            components.append("*Note: Message content, names, and contact info are never displayed.*\n\n")
            confessions_table = format_table(
                ["ID", "Is Anonymous", "Created At"],
                data["confessions"]["data"],
                "confessions"
            )
            components.append(confessions_table)

    # 6. KB Chunks (sample)
    if "kb_chunks" in data and data["kb_chunks"].get("data"):
        components.append("\n## Knowledge Base Chunks (Sample)\n")
        kb_table = format_table(
            ["ID", "Section", "Created At"],
            data["kb_chunks"]["data"][:20],  # First 20 only
            "kb_chunks"
        )
        components.append(kb_table)

    # 7. KB Coverage Summary (if RPC available)
    if "kb_coverage" in data and data["kb_coverage"]:
        components.append("\n## Knowledge Base Coverage\n")
        coverage_table = format_table(
            ["Source", "Chunk Count"],
            [{"source": item.get("source", ""), "chunk_count": item.get("count", 0)}
             for item in data["kb_coverage"]],
            "coverage"
        )
        components.append(coverage_table)

    return "\n".join(components)


def analytics_cta(focus: Optional[str] = None) -> str:
    """Generate smart follow-up CTA after analytics display.

    Args:
        focus: Detected focus area (e.g., "messages", "retrieval quality")

    Returns:
        Markdown-formatted follow-up question with multiple options
    """
    target = focus if focus else "a dataset above"

    return (
        f"\n\n💡 **Would you like me to:**\n"
        f"- Explain the importance of **{target}**\n"
        f"- Display the **data pipeline** architecture\n"
        f"- Show **other datasets** (tool invocations, cost by role, low-similarity queries)\n"
        f"- Generate a **7-day performance report** with aggregated metrics"
    )


def render_live_analytics(analytics_data: Dict[str, Any], role: str, focus: Optional[str] = None) -> str:
    """Main entry point for rendering live analytics.

    Args:
        analytics_data: Raw data from /api/analytics
        role: User's current role
        focus: Optional detected focus for smart CTA

    Returns:
        Complete analytics response with tables and follow-up CTA
    """
    report = render_analytics_response(analytics_data, role)
    cta = analytics_cta(focus)

    return report + cta
