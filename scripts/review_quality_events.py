#!/usr/bin/env python3
"""Review quality monitoring events from Supabase.

This script helps analyze quality validation events to make data-driven decisions
about whether proofreading is needed.

Usage:
    # Show summary for last 7 days
    python scripts/review_quality_events.py

    # Show summary for last 30 days
    python scripts/review_quality_events.py --days 30

    # Export flagged responses for manual review
    python scripts/review_quality_events.py --export flagged_responses.csv

    # Show detailed breakdown by warning type
    python scripts/review_quality_events.py --detailed

Output:
    - Summary statistics (warning rate, breakdown by type/path)
    - Sample flagged responses for manual review
    - Patterns (e.g., "most warnings on turn 2")
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import csv
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from assistant.analytics.supabase_analytics import supabase_analytics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_summary(summary: Dict[str, Any], detailed: bool = False) -> None:
    """Print quality summary in a readable format.

    Args:
        summary: Quality summary from get_quality_summary()
        detailed: If True, show detailed breakdown
    """
    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
        return

    if summary.get('total_events') == 0:
        print(f"📊 No quality events found for the last {summary['period_days']} days")
        return

    print(f"\n{'='*70}")
    print(f"📊 Quality Summary ({summary['period_days']} days)")
    print(f"{'='*70}\n")

    total = summary['total_events']
    with_warnings = summary['events_with_warnings']
    warning_rate = summary['warning_rate']

    print(f"Total Events: {total:,}")
    print(f"Events with Warnings: {with_warnings:,} ({warning_rate*100:.1f}%)")
    print(f"Events without Warnings: {total - with_warnings:,} ({(1-warning_rate)*100:.1f}%)")

    if summary.get('avg_turn_with_warning'):
        print(f"Avg Turn with Warning: {summary['avg_turn_with_warning']:.1f}")

    print(f"\n{'─'*70}")
    print("Breakdown by Response Path:")
    print(f"{'─'*70}")
    for path_data in summary.get('by_response_path', []):
        path = path_data['response_path']
        total_path = path_data['total_events']
        warnings_path = path_data['events_with_warnings']
        rate_path = path_data['warning_rate']
        print(f"  {path.upper():12} | Total: {total_path:6,} | Warnings: {warnings_path:4,} | Rate: {rate_path*100:5.1f}%")

    if detailed and summary.get('by_warning_type'):
        print(f"\n{'─'*70}")
        print("Breakdown by Warning Type:")
        print(f"{'─'*70}")
        for wt_data in summary['by_warning_type']:
            wt = wt_data['warning_type']
            count = wt_data['count']
            print(f"  {wt:40} | {count:4,} occurrences")

    print(f"\n{'='*70}\n")


def export_flagged_responses(days: int, output_file: str) -> None:
    """Export flagged responses to CSV for manual review.

    Args:
        days: Number of days to look back
        output_file: Path to output CSV file
    """
    try:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get flagged events (those with warning_type)
        events = supabase_analytics.client.table('quality_events')\
            .select('*')\
            .not_.is_('warning_type', 'null')\
            .gte('created_at', cutoff_date)\
            .order('created_at', desc=True)\
            .limit(1000)\
            .execute()

        if not events.data:
            print(f"📭 No flagged responses found for the last {days} days")
            return

        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'session_id', 'conversation_turn', 'warning_type',
                'response_path', 'query_preview', 'answer_preview',
                'retry_count', 'created_at'
            ])

            for event in events.data:
                writer.writerow([
                    event.get('id'),
                    event.get('session_id'),
                    event.get('conversation_turn'),
                    event.get('warning_type'),
                    event.get('response_path'),
                    event.get('query_preview', '')[:200],
                    event.get('answer_preview', '')[:500],
                    event.get('retry_count', 0),
                    event.get('created_at')
                ])

        print(f"✅ Exported {len(events.data)} flagged responses to {output_file}")
        print(f"   Review these to identify patterns and determine if proofreading is needed")

    except Exception as e:
        logger.error(f"Failed to export flagged responses: {e}")
        print(f"❌ Error exporting: {e}")


def identify_patterns(summary: Dict[str, Any]) -> None:
    """Identify patterns in quality events.

    Args:
        summary: Quality summary from get_quality_summary()
    """
    if summary.get('total_events') == 0:
        return

    print(f"\n{'='*70}")
    print("🔍 Pattern Analysis")
    print(f"{'='*70}\n")

    warning_rate = summary['warning_rate']

    # Pattern 1: Overall warning rate
    if warning_rate < 0.02:
        print("✅ LOW warning rate (<2%) - Current system is performing well")
    elif warning_rate < 0.05:
        print("⚠️  MODERATE warning rate (2-5%) - Monitor closely, consider improvements")
    else:
        print("🚨 HIGH warning rate (>5%) - Consider adding proofreading or improving prompts")

    # Pattern 2: RAG vs Template
    rag_data = next((p for p in summary.get('by_response_path', []) if p['response_path'] == 'rag'), None)
    template_data = next((p for p in summary.get('by_response_path', []) if p['response_path'] == 'template'), None)

    if rag_data and template_data:
        rag_rate = rag_data['warning_rate']
        template_rate = template_data['warning_rate']

        if rag_rate > template_rate * 2:
            print(f"⚠️  RAG responses have {rag_rate/template_rate:.1f}x higher warning rate than templates")
            print("   → Consider adding proofreading specifically for RAG responses")
        elif template_rate > rag_rate * 2:
            print(f"⚠️  Template responses have {template_rate/rag_rate:.1f}x higher warning rate than RAG")
            print("   → Review template handlers for issues")
        else:
            print("✅ Warning rates are similar between RAG and template responses")

    # Pattern 3: Most common warning type
    if summary.get('by_warning_type'):
        most_common = summary['by_warning_type'][0]
        print(f"\n📌 Most common warning: {most_common['warning_type']} ({most_common['count']} occurrences)")

        if 'relevance_low' in most_common['warning_type']:
            print("   → Consider improving prompt relevance or RAG retrieval")
        elif 'too_similar' in most_common['warning_type']:
            print("   → Consider adding variety to responses or better context tracking")

    # Pattern 4: Turn analysis
    if summary.get('avg_turn_with_warning'):
        avg_turn = summary['avg_turn_with_warning']
        if avg_turn <= 3:
            print(f"\n⚠️  Warnings occur early (avg turn {avg_turn:.1f}) - First impressions matter!")
            print("   → Consider proofreading for first 3 turns")
        elif avg_turn >= 8:
            print(f"\n✅ Warnings occur late (avg turn {avg_turn:.1f}) - Less critical")
        else:
            print(f"\n📊 Warnings occur mid-conversation (avg turn {avg_turn:.1f})")

    print(f"\n{'='*70}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Review quality monitoring events from Supabase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed breakdown by warning type'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export flagged responses to CSV file'
    )

    args = parser.parse_args()

    # Get summary
    print(f"📊 Fetching quality events for the last {args.days} days...")
    summary = supabase_analytics.get_quality_summary(days=args.days)

    # Print summary
    print_summary(summary, detailed=args.detailed)

    # Identify patterns
    identify_patterns(summary)

    # Export if requested
    if args.export:
        export_flagged_responses(args.days, args.export)


if __name__ == '__main__':
    main()
