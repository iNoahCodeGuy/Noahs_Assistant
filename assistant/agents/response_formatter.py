from typing import Any, Dict, List
from assistant.core.langchain_compat import Document


FOLLOW_UP_LIBRARY: Dict[str, Dict[str, List[str]]] = {
    "architecture": {
        "default": [
            "Walk the full request lifecycle - input -> embeddings -> vector search -> LangGraph nodes - with latency callouts?",
            "Pull up the LangGraph node diagram plus the orchestrator code so you can see the control flow in context?",
            "Model a 10x traffic scenario and point out the exact scaling levers, bottlenecks, and mitigation plan?",
        ]
    },
    "data": {
        "default": [
            "Open the analytics dashboard with precision@k, similarity averages, and no-match rate trend lines?",
            "Show the Supabase schema + IVFFLAT index plan to unpack how pgvector powers the retrieval layer?",
            "Talk through how we monitor embedding drift and schedule refreshes so the system stays sharp over time?",
        ]
    },
    "retrieval": {
        "default": [
            "Trace a live query through the retrieval pipeline and overlay the similarity scores for each chunk?",
            "Visualize the embedding space (t-SNE projection) so you can see how career, technical, and architecture content cluster?",
            "Compare fallback strategies when similarity drops under threshold - how we guardrail hallucinations and offer clarifying paths?",
        ]
    },
    "testing": {
        "default": [
            "Walk the test pyramid - unit, integration, E2E - and show real pytest cases that keep the pipeline honest?",
            "Review the failure injection playbook and how we rehearse degraded-mode responses for external API hiccups?",
            "Pull up monitoring hooks (LangSmith traces, Supabase logs) so you can see how we debug production incidents fast?",
        ]
    },
    "career": {
        "default": [
            "Explore the architecture decision records Noah authored to understand his senior-level judgment calls?",
            "Quantify the business impact of key projects - revenue protected, latency reduced, teams unblocked - with concrete metrics?",
            "Walk through how this assistant fits into broader GenAI roadmaps Noah can deliver for an enterprise like yours?",
        ]
    },
    "fun": {
        "default": [
            "Peek behind the scenes at how I keep conversations grounded without losing the cinematic vibe?",
            "Hear a quick story about Noah shipping production AI systems at Tesla level scale?",
            "Jump into the MMA Easter eggs hidden in my knowledge base for a just for fun detour?",
        ]
    },
    "general": {
        "default": [
            "See the system architecture diagram with component interactions and error handling paths annotated?",
            "Dig into the RAG performance dashboard - latency percentiles, precision@k, and cost per query?",
            "Talk through scalability trade offs and what changes when you aim for 100K monthly conversations?",
        ]
    },
}

class ResponseFormatter:
    """Formats responses based on role and content type."""

    def format(self, response_data: Dict[str, Any]) -> str:
        response = response_data.get("response", "")
        rtype = response_data.get("type", "general")
        context = response_data.get("context", [])

        # Convert context dict to list if needed
        if isinstance(context, dict):
            # Extract matches from the dict returned by retrieve()
            context = context.get("matches", [])
        elif context is None:
            context = []

        if rtype == "technical":
            return self._format_technical_response(response_data, response, context)
        if rtype == "mma":
            return self._format_mma_response(response_data)
        if rtype == "career":
            return self._format_career_response(response_data, response, context)
        if rtype == "fun":
            return self._format_fun_response(response)
        if rtype == "confession":
            return "💌 " + response
        return self._format_general_response(response_data, response)

    def _format_technical_response(self, response_data: Dict[str, Any], response: str, context: List[Document]) -> str:
        """Enhanced technical formatting with code snippets and proper citations."""
        sections = []

        # Engineer Detail Section
        sections.append("## 🔧 Engineer Detail")
        sections.append(response)

        # Ensure context is a list
        if not isinstance(context, list):
            context = []

        # Code Examples Section (if context contains code snippets)
        code_snippets = [doc for doc in context if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'code']
        if code_snippets:
            sections.append("\n## 💻 Code Examples")
            for snippet in code_snippets[:3]:
                metadata = snippet.metadata
                sections.append(f"\n### {metadata.get('name', 'Code Snippet')}")
                sections.append(f"**File:** `{metadata.get('citation', 'unknown')}`")
                sections.append(f"```python\n{snippet.page_content}\n```")
                if metadata.get('github_url'):
                    sections.append(f"[View on GitHub]({metadata['github_url']})")

        # Plain-English Summary
        sections.append("\n## 📋 Plain-English Summary")
        sections.append(self._generate_summary(response))

        # Citations Section
        if context and isinstance(context, list):
            sections.append("\n## 📚 Citations")
            for i, doc in enumerate(context[:5], 1):
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get("source") or doc.metadata.get("file_path", "unknown")
                    line = doc.metadata.get("start_line", "")
                    line_info = f":{line}" if line else ""
                    sections.append(f"{i}. `{source}{line_info}`")
                else:
                    sections.append(f"{i}. {str(doc)[:100]}...")

        follow_up = self._build_follow_up_section(response_data, heading="## 🔭 Where We Can Go Next")
        if follow_up:
            sections.append("\n" + follow_up)

        return "\n".join(sections)

    def _generate_summary(self, technical_text: str) -> str:
        """Generate plain-English summary from technical content."""
        # Simple heuristic - take first 2 sentences and simplify
        sentences = technical_text.split('. ')[:2]
        summary = '. '.join(sentences)

        # Replace technical terms with simpler alternatives
        replacements = {
            'FAISS': 'a search system',
            'vector store': 'database',
            'embeddings': 'text representations',
            'RAG': 'information retrieval',
            'LangChain': 'AI framework',
            'AST': 'code analysis',
            'API': 'programming interface'
        }

        for tech_term, simple_term in replacements.items():
            summary = summary.replace(tech_term, simple_term)

        return summary

    def _format_mma_response(self, data: Dict[str, Any]) -> str:
        base = data.get("response", "")
        link = data.get("youtube_link")
        if link:
            return f"{base}\n\nWatch: {link}"
        return base

    def _format_career_response(self, response_data: Dict[str, Any], response: str, context: List[Document]) -> str:
        """Format career/technical responses with sources."""
        # Ensure context is a list
        if not isinstance(context, list):
            context = []

        # Build better source citations
        if context:
            sources_list = []
            for i, doc in enumerate(context[:3], 1):
                # Handle both Document objects and dicts
                if hasattr(doc, 'metadata'):
                    # Document object - extract from metadata
                    metadata = doc.metadata
                    source = metadata.get('source', 'unknown')
                    doc_id = metadata.get('doc_id', '')
                    section = metadata.get('section', '')[:80] if metadata.get('section') else ''
                    similarity = metadata.get('similarity', 0)
                    if not similarity and hasattr(doc, 'similarity'):
                        similarity = doc.similarity
                elif isinstance(doc, dict):
                    # Dict from pgvector_retriever - fields are at top level
                    source = doc.get('source', 'unknown')
                    doc_id = doc.get('doc_id', '')
                    section = doc.get('section', '')[:80] if doc.get('section') else ''
                    similarity = doc.get('similarity', 0)
                else:
                    continue

                # Format source with more detail
                if section and doc_id:
                    sources_list.append(f"{i}. **{doc_id}** - {section} (similarity: {similarity:.2f})")
                elif doc_id:
                    sources_list.append(f"{i}. **{doc_id}** (similarity: {similarity:.2f})")
                else:
                    sources_list.append(f"{i}. **{source}** (similarity: {similarity:.2f})")

            sources_text = "\n".join(sources_list) if sources_list else "- (no sources)"
        else:
            sources_text = "- (no sources)"

        base = f"{response}\n\n---\n\n### 📚 Sources\n{sources_text}"
        follow_up = self._build_follow_up_section(response_data)
        if follow_up:
            base += f"\n\n{follow_up}"
        return base

    def _format_fun_response(self, response: str) -> str:
        return f"### Fun Facts\n{response}"

    def _format_general_response(self, response_data: Dict[str, Any], response: str) -> str:
        follow_up = self._build_follow_up_section(response_data)
        if follow_up:
            return f"{response}\n\n{follow_up}"
        return response

    def _build_follow_up_section(self, response_data: Dict[str, Any], heading: str = "### 🔭 Where We Can Go Next") -> str:
        meta = response_data.get("meta") or {}
        if meta.get("suppress_follow_up"):
            return ""

        # Avoid duplicating follow-up blocks if already appended upstream
        response = response_data.get("response", "")
        if "Where We Can Go Next" in response:
            return ""

        topic_focus = meta.get("topic_focus") or response_data.get("topic_focus") or "general"
        role = meta.get("role", "")
        emotional_mode = meta.get("emotional_mode")
        if not emotional_mode:
            turn_index = meta.get("turn_index")
            if isinstance(turn_index, int) and turn_index > 0 and turn_index % 2 == 0:
                emotional_mode = "reflective"
            else:
                emotional_mode = "energetic"

        suggestions = self._get_follow_up_suggestions(topic_focus, role)
        if not suggestions:
            return ""

        lead_in = "Still hungry? Here are a few high-signal next steps:" if emotional_mode == "energetic" else "If you'd like to keep exploring, we can dive into:"
        bullets = "\n".join(f"- {suggestion}" for suggestion in suggestions)
        return f"{heading}\n{lead_in}\n{bullets}"

    def _get_follow_up_suggestions(self, topic_focus: str, role: str) -> List[str]:
        topic = topic_focus if topic_focus in FOLLOW_UP_LIBRARY else "general"
        topic_map = FOLLOW_UP_LIBRARY.get(topic, FOLLOW_UP_LIBRARY["general"])
        base_options = topic_map.get("default", [])

        if role == "Just looking around":
            # Keep approachable but still educational
            return [
                "Peek at the cinematic architecture tour so you can literally see how each component hands off to the next?",
                "Get a plain English walkthrough of the RAG pipeline with a quick story about why enterprises love this pattern?",
                "Jump into a live metric snapshot - latency, precision, cost per query - to feel how production ready this is?",
            ]

        if role == "Hiring Manager (nontechnical)":
            return [
                "Map the architecture to ROI - where accuracy, latency, and governance create enterprise value?",
                "See the leadership stories behind these systems so you can gauge Noah's operating cadence?",
                "Preview the implementation roadmap for bringing a GenAI assistant like this into your org without disruption?",
            ]

        if role == "Hiring Manager (technical)":
            return [
                "Inspect the LangGraph node execution order with commentary on guardrails and failure paths?",
                "Zoom into the retrieval metrics, precision curves, and the Supabase SQL that keeps responses grounded?",
                "Stress test the scalability plan - connection pooling, batching, and cost modeling for 100K monthly queries?",
            ]

        if role == "Software Developer":
            return [
                "Step through the orchestrator code with inline comments so you can see exactly how state flows between nodes?",
                "Pull live performance stats (latency, similarity, precision@k) and compare them against the theoretical limits?",
                "Review the automated testing harness - pytest suites, fixtures, and how we simulate degraded modes?",
            ]

        return base_options[:3]
