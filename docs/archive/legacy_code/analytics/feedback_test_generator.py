"""Framework for expanding test coverage based on user feedback.

This module provides tools to:
1. Collect user feedback and error reports
2. Analyze usage patterns
3. Generate new test cases automatically
4. Track test coverage evolution
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class UserFeedbackReport:
    """Structure for user feedback reports."""
    timestamp: datetime
    user_role: str
    query: str
    issue_type: str  # 'incorrect_code', 'missing_citation', 'slow_response', 'wrong_format'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    response_received: str
    expected_behavior: str


class FeedbackAnalyzer:
    """Analyze user feedback to identify test case needs."""

    def __init__(self, feedback_file: str = "logs/user_feedback.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(exist_ok=True)

    def record_feedback(self, report: UserFeedbackReport):
        """Record a user feedback report."""
        with open(self.feedback_file, 'a') as f:
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'user_role': report.user_role,
                'query': report.query,
                'issue_type': report.issue_type,
                'description': report.description,
                'severity': report.severity,
                'response_received': report.response_received,
                'expected_behavior': report.expected_behavior
            }
            f.write(json.dumps(report_dict) + '\n')

    def analyze_feedback_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze feedback patterns to identify test gaps."""
        if not self.feedback_file.exists():
            return {"error": "No feedback data available"}

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = []

        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp > cutoff_date:
                        recent_feedback.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue

        if not recent_feedback:
            return {"error": f"No feedback in last {days} days"}

        # Analyze patterns
        issue_types = {}
        role_issues = {}
        query_patterns = []

        for feedback in recent_feedback:
            # Count issue types
            issue_type = feedback['issue_type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

            # Count issues by role
            role = feedback['user_role']
            if role not in role_issues:
                role_issues[role] = {}
            role_issues[role][issue_type] = role_issues[role].get(issue_type, 0) + 1

            # Collect query patterns
            query_patterns.append({
                'query': feedback['query'],
                'issue': issue_type,
                'role': role
            })

        return {
            'analysis_period_days': days,
            'total_reports': len(recent_feedback),
            'issue_type_distribution': issue_types,
            'role_specific_issues': role_issues,
            'query_patterns': query_patterns,
            'high_priority_issues': [f for f in recent_feedback if f['severity'] in ['high', 'critical']],
            'suggested_test_cases': self._generate_test_suggestions(recent_feedback)
        }

    def _generate_test_suggestions(self, feedback_data: List[Dict]) -> List[Dict]:
        """Generate test case suggestions based on feedback."""
        suggestions = []

        # Group by issue type
        issue_groups = {}
        for feedback in feedback_data:
            issue_type = feedback['issue_type']
            if issue_type not in issue_groups:
                issue_groups[issue_type] = []
            issue_groups[issue_type].append(feedback)

        # Generate suggestions for each issue type
        for issue_type, reports in issue_groups.items():
            if len(reports) >= 2:  # Multiple reports suggest a pattern
                test_case = self._create_test_case_suggestion(issue_type, reports)
                suggestions.append(test_case)

        return suggestions

    def _create_test_case_suggestion(self, issue_type: str, reports: List[Dict]) -> Dict:
        """Create a specific test case suggestion."""
        common_queries = [r['query'] for r in reports]
        common_roles = [r['user_role'] for r in reports]

        if issue_type == 'incorrect_code':
            return {
                'test_name': f'test_{issue_type}_pattern_{len(common_queries)}',
                'category': 'accuracy',
                'description': f'Test code accuracy for queries like: {common_queries[:3]}',
                'suggested_implementation': f'''
def test_code_accuracy_pattern_{len(common_queries)}(self):
    """Test code accuracy based on user feedback."""
    test_queries = {common_queries[:5]}
    roles = {list(set(common_roles))}

    for query in test_queries:
        for role in roles:
            result = self.rag_engine.retrieve_with_code(query, role)

            # Validate code snippets are relevant
            if result.get('has_code'):
                for snippet in result['code_snippets']:
                    assert 'content' in snippet
                    assert len(snippet['content']) > 0
                    assert snippet['citation'].endswith('.py')
                    # Add specific validation based on user reports
'''
            }

        elif issue_type == 'missing_citation':
            return {
                'test_name': f'test_citation_completeness_{len(common_queries)}',
                'category': 'format_validation',
                'description': f'Ensure citations are complete for queries: {common_queries[:3]}',
                'suggested_implementation': f'''
def test_citation_completeness_feedback_{len(common_queries)}(self):
    """Test citation completeness based on user feedback."""
    problematic_queries = {common_queries[:5]}

    for query in problematic_queries:
        result = self.rag_engine.retrieve_with_code(query, "Software Developer")

        for snippet in result.get('code_snippets', []):
            citation = snippet.get('citation', '')
            assert ':' in citation, f"Missing line info in citation: {{citation}}"
            assert citation.endswith('.py'), f"Citation should reference Python file: {{citation}}"
            assert snippet.get('github_url', '').startswith('https://github.com')
'''
            }

        else:
            return {
                'test_name': f'test_{issue_type}_general',
                'category': 'edge_cases',
                'description': f'General test for {issue_type} issues',
                'suggested_implementation': f'# Add test for {issue_type} with queries: {common_queries[:3]}'
            }


class TestCaseGenerator:
    """Generate new test cases from user feedback analysis."""

    def __init__(self, analyzer: FeedbackAnalyzer, test_dir: str = "tests"):
        self.analyzer = analyzer
        self.test_dir = Path(test_dir)

    def generate_feedback_based_tests(self) -> str:
        """Generate a new test file based on feedback analysis."""
        analysis = self.analyzer.analyze_feedback_patterns()

        if 'error' in analysis:
            return f"Cannot generate tests: {analysis['error']}"

        suggestions = analysis.get('suggested_test_cases', [])

        if not suggestions:
            return "No test case suggestions generated from current feedback"

        # Generate test file content
        test_content = self._generate_test_file_content(suggestions, analysis)

        # Write to file
        output_file = self.test_dir / f"test_user_feedback_{datetime.now().strftime('%Y%m%d')}.py"
        with open(output_file, 'w') as f:
            f.write(test_content)

        return f"Generated test file: {output_file}"

    def _generate_test_file_content(self, suggestions: List[Dict], analysis: Dict) -> str:
        """Generate the actual test file content."""
        header = f'''"""
Test cases generated from user feedback analysis.

Generated on: {datetime.now().isoformat()}
Based on {analysis['total_reports']} user reports over {analysis['analysis_period_days']} days.

Top issues identified:
{chr(10).join(f"- {issue}: {count} reports" for issue, count in analysis['issue_type_distribution'].items())}
"""
import pytest
from src.core.rag_engine import RagEngine
from src.config.settings import Settings
from src.agents.role_router import RoleRouter


class TestUserFeedbackIssues:
    """Tests generated from user feedback patterns."""

    @pytest.fixture
    def rag_engine(self):
        settings = Settings()
        settings.disable_auto_rebuild = True
        return RagEngine(settings=settings)
'''

        # Add suggested test methods
        for i, suggestion in enumerate(suggestions):
            header += f"\n    {suggestion.get('suggested_implementation', '# Test method placeholder')}\n"

        # Add helper methods
        header += '''
    def _validate_code_snippet_quality(self, snippet):
        """Helper to validate code snippet quality."""
        required_fields = ['file', 'citation', 'content', 'github_url']
        for field in required_fields:
            assert field in snippet, f"Missing required field: {field}"

        assert len(snippet['content'].strip()) > 0, "Code content should not be empty"
        assert '.py' in snippet['citation'], "Citation should reference Python file"
        assert snippet['github_url'].startswith('https://github.com'), "Invalid GitHub URL"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''

        return header


# Integration example for collecting feedback
def collect_user_feedback_example():
    """Example of how to collect user feedback in your app."""

    # In your Streamlit app, you might have:
    """
    if st.button("Report Issue"):
        issue_type = st.selectbox("Issue Type",
            ['incorrect_code', 'missing_citation', 'slow_response', 'wrong_format'])
        description = st.text_area("Describe the issue")

        if st.button("Submit"):
            analyzer = FeedbackAnalyzer()
            report = UserFeedbackReport(
                timestamp=datetime.now(),
                user_role=st.session_state.get('role', 'Unknown'),
                query=st.session_state.get('last_query', ''),
                issue_type=issue_type,
                description=description,
                severity='medium',  # Could be user-selectable
                response_received=st.session_state.get('last_response', ''),
                expected_behavior=st.text_input("What did you expect?")
            )
            analyzer.record_feedback(report)
            st.success("Thank you for your feedback!")
    """
    pass


if __name__ == "__main__":
    # Example usage
    analyzer = FeedbackAnalyzer()

    # Simulate some feedback
    example_report = UserFeedbackReport(
        timestamp=datetime.now(),
        user_role="Software Developer",
        query="Show me RagEngine implementation",
        issue_type="missing_citation",
        description="Code snippets don't have line numbers",
        severity="medium",
        response_received="Code shown but no line references",
        expected_behavior="Should show file:line citations"
    )

    analyzer.record_feedback(example_report)

    # Analyze patterns
    analysis = analyzer.analyze_feedback_patterns()
    print(f"Feedback analysis: {json.dumps(analysis, indent=2, default=str)}")

    # Generate tests
    generator = TestCaseGenerator(analyzer)
    result = generator.generate_feedback_based_tests()
    print(f"Test generation result: {result}")
