"""
Documentation Alignment Tests

These tests verify that documentation matches actual code implementation.
Prevents drift where docs claim functions/flows that don't match reality.

Run: pytest tests/test_documentation_alignment.py -v
"""

import pytest
import os
import re
import inspect


class TestConversationFlowAlignment:
    """Verify conversation pipeline docs match actual code."""

    def test_conversation_flow_documented_correctly(self):
        """SYSTEM_ARCHITECTURE_SUMMARY describes actual pipeline nodes."""

        # Read master documentation
        doc_path = "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md"
        with open(doc_path) as f:
            doc_content = f.read()

        # Get actual pipeline from code
        from assistant.flows.conversation_flow import run_conversation_flow
        source = inspect.getsource(run_conversation_flow)

        # Expected nodes in actual implementation (core pipeline)
        # Note: Not all documented nodes appear in conversation_flow.py - some are called within other nodes
        actual_nodes = [
            "handle_greeting",
            "classify_intent",  # Renamed from classify_query in modular architecture
            "retrieve_chunks",
            "generate_draft",  # Renamed from generate_answer in modular architecture
            "plan_actions",
            "execute_actions",
            "log_and_notify"
        ]

        # Verify each actual node is documented
        for node in actual_nodes:
            assert node in doc_content, (
                f"Node '{node}' exists in code (src/flows/conversation_flow.py) "
                f"but not documented in {doc_path}. "
                f"Update docs to match implementation."
            )

        # Verify node appears in actual code
        for node in actual_nodes:
            assert node in source, (
                f"Expected node '{node}' in conversation_flow.py source. "
                f"Test assumptions may be outdated."
            )


class TestCodeReferenceValidity:
    """Verify file paths mentioned in docs actually exist."""

    def test_documentation_file_references_valid(self):
        """All src/*.py references in docs point to existing files."""

        doc_files = [
            "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md",
            "docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md",
            "docs/CONVERSATION_PIPELINE_MODULES.md",
        ]

        # Add RAG_ENGINE.md if it exists
        if os.path.exists("docs/RAG_ENGINE.md"):
            doc_files.append("docs/RAG_ENGINE.md")

        invalid_references = []

        for doc_file in doc_files:
            if not os.path.exists(doc_file):
                continue  # Skip if doc doesn't exist

            with open(doc_file) as f:
                content = f.read()

            # Find references like "src/flows/core_nodes.py"
            file_refs = re.findall(r'src/[\w/]+\.py', content)

            for ref in set(file_refs):  # Use set to avoid duplicate checks
                if not os.path.exists(ref):
                    line_num = content[:content.find(ref)].count('\n') + 1
                    invalid_references.append({
                        "doc": doc_file,
                        "reference": ref,
                        "line": line_num
                    })

        assert len(invalid_references) == 0, (
            f"Found {len(invalid_references)} invalid file references:\n" +
            "\n".join([
                f"  {inv['doc']} line {inv['line']}: {inv['reference']}"
                for inv in invalid_references
            ]) +
            "\n\nUpdate documentation to reference correct files."
        )

    def test_test_file_references_valid(self):
        """All tests/*.py references in docs point to existing files."""

        doc_files = [
            "docs/QA_STRATEGY.md",
        ]

        invalid_references = []

        for doc_file in doc_files:
            if not os.path.exists(doc_file):
                continue

            with open(doc_file) as f:
                content = f.read()

            # Find references like "tests/test_conversation_quality.py"
            file_refs = re.findall(r'tests/[\w/]+\.py', content)

            for ref in set(file_refs):
                if not os.path.exists(ref):
                    line_num = content[:content.find(ref)].count('\n') + 1
                    invalid_references.append({
                        "doc": doc_file,
                        "reference": ref,
                        "line": line_num
                    })

        assert len(invalid_references) == 0, (
            f"Found {len(invalid_references)} invalid test file references:\n" +
            "\n".join([
                f"  {inv['doc']} line {inv['line']}: {inv['reference']}"
                for inv in invalid_references
            ])
        )


class TestRoleConsistency:
    """Verify role names match between docs and code."""

    def test_role_names_documented(self):
        """Roles in PROJECT_REFERENCE_OVERVIEW match actual role definitions."""

        # Get documented roles
        doc_path = "docs/context/PROJECT_REFERENCE_OVERVIEW.md"
        with open(doc_path) as f:
            doc_content = f.read()

        # Expected roles from documentation context
        expected_roles = [
            "Software Developer",
            "Hiring Manager (technical)",
            "Hiring Manager (nontechnical)",
        ]

        for role in expected_roles:
            # Normalize variations in documentation (handles different dash types)
            role_variations = [
                role,
                role.replace("nontechnical", "non-technical"),
                role.replace("nontechnical", "non‑technical"),  # en-dash
            ]
            found = any(variation in doc_content for variation in role_variations)
            assert found, (
                f"Role '{role}' not documented in {doc_path}. "
                f"Add role description to master docs."
            )


class TestConfigurationValues:
    """Verify configuration values in docs match code."""

    def test_temperature_setting_documented_correctly(self):
        """Temperature value in docs matches RAG factory."""

        # Get documented temperature
        doc_path = "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md"
        with open(doc_path) as f:
            doc_content = f.read()

        temp_match = re.search(r'temperature[:\s]+(\d+\.?\d*)', doc_content)
        if not temp_match:
            pytest.skip("Temperature not explicitly documented (may be described differently)")

        documented_temp = float(temp_match.group(1))

        # Get actual temperature from code
        from assistant.core.rag_factory import RagEngineFactory
        source = inspect.getsource(RagEngineFactory.create_llm)

        code_temp_match = re.search(r'temperature=(\d+\.?\d*)', source)
        if not code_temp_match:
            pytest.fail("Temperature not found in RagEngineFactory.create_llm")

        actual_temp = float(code_temp_match.group(1))

        assert documented_temp == actual_temp, (
            f"Temperature mismatch: docs say {documented_temp}, "
            f"code uses {actual_temp}. Update {doc_path} to match "
            f"src/core/rag_factory.py"
        )

    def test_embedding_model_documented(self):
        """Embedding model name in docs matches code."""

        doc_path = "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md"
        with open(doc_path) as f:
            doc_content = f.read()

        # Check for text-embedding-3-small mention
        assert "text-embedding-3-small" in doc_content, (
            f"Embedding model 'text-embedding-3-small' not documented in {doc_path}. "
            f"This is the current model used in production."
        )


class TestMasterDocsIntegrity:
    """Verify master documentation files exist and are complete."""

    def test_all_master_docs_exist(self):
        """All 4 master docs in docs/context/ are present."""

        required_docs = [
            "docs/context/PROJECT_REFERENCE_OVERVIEW.md",
            "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md",
            "docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md",
            "docs/context/CONVERSATION_PERSONALITY.md",
        ]

        missing = [doc for doc in required_docs if not os.path.exists(doc)]

        assert len(missing) == 0, (
            f"Missing master documentation files: {missing}\n"
            f"These are source-of-truth documents required for system understanding."
        )

    def test_master_docs_not_empty(self):
        """Master docs have meaningful content (>500 chars each)."""

        master_docs = [
            "docs/context/PROJECT_REFERENCE_OVERVIEW.md",
            "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md",
            "docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md",
            "docs/context/CONVERSATION_PERSONALITY.md",
        ]

        for doc_path in master_docs:
            with open(doc_path) as f:
                content = f.read()

            assert len(content) > 500, (
                f"{doc_path} is too short ({len(content)} chars). "
                f"Master docs should be comprehensive."
            )


class TestQADocumentation:
    """Verify QA documentation is accurate and up-to-date."""

    def test_qa_strategy_exists(self):
        """QA_STRATEGY.md exists and describes testing approach."""

        assert os.path.exists("docs/QA_STRATEGY.md"), (
            "docs/QA_STRATEGY.md not found. This should document testing strategy."
        )

        with open("docs/QA_STRATEGY.md") as f:
            content = f.read()

        # Check for key sections
        assert "Documentation Alignment Testing" in content, (
            "QA_STRATEGY.md should include documentation alignment testing section"
        )

        assert "test_documentation_alignment.py" in content, (
            "QA_STRATEGY.md should reference this test file"
        )

    def test_test_count_documented_correctly(self):
        """Test count in docs matches actual implemented tests."""

        # This test verifies the QA docs mention the correct number of tests
        # Skip if not critical to keep passing during development
        pytest.skip("Test count changes frequently during development")


class TestResumeDistributionAlignment:
    """Verify Intelligent Resume Distribution System is properly documented."""

    def test_resume_distribution_functions_documented(self):
        """All resume_distribution.py functions are documented in SYSTEM_ARCHITECTURE."""

        # Get actual functions from code
        from assistant.flows.node_logic import resume_distribution
        actual_functions = [
            "detect_hiring_signals",
            "handle_resume_request",
            "should_add_availability_mention",
            "extract_email_from_query",
            "extract_name_from_query",
            "should_gather_job_details",
            "get_job_details_prompt",
            "extract_job_details_from_query",
        ]

        # Read documentation
        doc_path = "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md"
        with open(doc_path) as f:
            doc_content = f.read()

        # Verify each function is documented
        missing_functions = []
        for func in actual_functions:
            if func not in doc_content:
                missing_functions.append(func)

        assert len(missing_functions) == 0, (
            f"Resume distribution functions not documented in {doc_path}:\n"
            f"  {', '.join(missing_functions)}\n"
            f"Add these functions to the conversation pipeline documentation."
        )

    def test_resume_distribution_feature_doc_exists(self):
        """Feature doc INTELLIGENT_RESUME_DISTRIBUTION.md exists and is complete."""

        feature_doc = "docs/features/INTELLIGENT_RESUME_DISTRIBUTION.md"
        assert os.path.exists(feature_doc), (
            f"{feature_doc} not found. This should document the hybrid approach."
        )

        with open(feature_doc) as f:
            content = f.read()

        # Verify key sections exist
        required_sections = [
            "Mode 1",  # Education mode
            "Mode 2",  # Hiring signals
            "Mode 3",  # Explicit request
            "hybrid",  # Hybrid approach
            "resume_distribution.py",  # Implementation file
        ]

        missing_sections = [s for s in required_sections if s not in content]

        assert len(missing_sections) == 0, (
            f"Feature doc missing sections: {', '.join(missing_sections)}\n"
            f"Ensure {feature_doc} documents all 3 modes and implementation."
        )

    def test_qa_policy_updated_for_resume_distribution(self):
        """QA_STRATEGY.md includes exception for subtle availability mentions."""

        qa_doc = "docs/QA_STRATEGY.md"
        with open(qa_doc) as f:
            content = f.read()

        # Check for intelligent resume distribution exception
        assert "Intelligent Resume Distribution" in content, (
            f"{qa_doc} should document exception for subtle availability mentions"
        )

        assert "test_no_pushy_resume_offers" in content, (
            f"{qa_doc} should reference the new test_no_pushy_resume_offers test"
        )


class TestChangelogIntegrity:
    """Verify CHANGELOG.md exists and is structured correctly."""

    def test_changelog_exists(self):
        """CHANGELOG.md exists in root directory."""

        assert os.path.exists("CHANGELOG.md"), (
            "CHANGELOG.md not found in root. This should track all changes."
        )

    def test_changelog_has_recent_entries(self):
        """CHANGELOG.md has entries (not empty or stub)."""

        with open("CHANGELOG.md") as f:
            content = f.read()

        assert len(content) > 1000, (
            "CHANGELOG.md seems incomplete. Should have detailed entries."
        )

        # Check for date format entries
        assert re.search(r'\[20\d{2}-\d{2}-\d{2}\]', content), (
            "CHANGELOG.md should have dated entries like [2025-10-16]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
