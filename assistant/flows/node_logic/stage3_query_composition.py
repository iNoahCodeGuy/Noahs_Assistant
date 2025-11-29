"""Compose retrieval-ready queries that respect role and entity context."""

from __future__ import annotations
import logging

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)


def _correct_common_typos(query: str) -> str:
    """Correct common typos that affect retrieval quality.

    Args:
        query: Original user query

    Returns:
        Query with typos corrected
    """
    corrections = {
        # Architecture typos
        "archexture": "architecture",
        "archetecture": "architecture",
        "architechture": "architecture",
        "architecure": "architecture",
        "architectxure": "architecture",  # 'x' typo
        "architexture": "architecture",   # 'x' typo variant
        "archtecture": "architecture",    # missing 'i'
        "architeture": "architecture",    # missing 'c'
        "architectre": "architecture",    # missing 'u'
        # Customer typos
        "custmer": "customer",
        "cusotmer": "customer",           # transposition
        "customr": "customer",            # missing 'e'
        # Support typos
        "suport": "support",
        "supprot": "support",             # transposition
        "supoprt": "support",             # transposition
        # Enterprise typos
        "enterpise": "enterprise",
        "enterprize": "enterprise",       # 'z' variant
        "enterpris": "enterprise",        # missing 'e'
        # Adaptation typos
        "adaptaion": "adaptation",
        "adaption": "adaptation",         # missing 'a'
        "addaptation": "adaptation",      # double 'd'
    }
    original = query
    query_lower = query.lower()
    for typo, correct in corrections.items():
        if typo in query_lower:
            # Preserve original case
            import re
            query = re.sub(re.escape(typo), correct, query, flags=re.IGNORECASE)
            logger.info(f"Corrected typo: '{typo}' → '{correct}' in query")
    return query


def _expand_menu_selection(menu_choice: str, role_mode: str) -> str:
    """Expand menu number to full retrieval query based on role.

    Maps menu selections to rich queries that retrieve relevant content.
    Different roles have different menu options, so we check role_mode first.

    Args:
        menu_choice: The menu number (1, 2, 3, or 4)
        role_mode: The user's role (hiring_manager_technical, etc.)

    Returns:
        Expanded query optimized for vector search retrieval
    """

    # Technical Hiring Manager menu (from stage2_role_routing.py welcome message)
    if role_mode == "hiring_manager_technical":
        menu_map = {
            "1": "full technology stack architecture frontend backend data pipeline observability LangGraph Supabase pgvector deployment infrastructure",
            "2": "LangGraph orchestration layer nodes states safeguards conversation flow pipeline stage routing error handling",
            "3": "enterprise adaptation patterns large-scale deployment customization scalability reliability production best practices",
            "4": "Noah technical background certifications GitHub projects engineering foundation credentials proof skills"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Nontechnical Hiring Manager menu
    elif role_mode == "hiring_manager_nontechnical":
        menu_map = {
            "1": "career progression achievements key accomplishments milestones growth",
            "2": "business impact ROI value delivered results outcomes",
            "3": "leadership teamwork collaboration communication soft skills",
            "4": "resume download contact information availability"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Software Developer menu
    elif role_mode == "software_developer":
        menu_map = {
            "1": "code architecture implementation technical decisions patterns",
            "2": "system design infrastructure deployment production engineering",
            "3": "debugging troubleshooting problem-solving technical challenges",
            "4": "open source contributions GitHub projects technical portfolio"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Explorer menu
    elif role_mode == "explorer":
        menu_map = {
            "1": "career stories professional journey highlights interesting projects",
            "2": "technical projects how they work architecture overview",
            "3": "behind the scenes how assistant built RAG pipeline",
            "4": "fun facts personal interests hobbies MMA fights"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Fallback: return original if role not recognized
    return menu_choice


def compose_query(state: ConversationState) -> ConversationState:
    """Blend the user question with role and entity hints for retrieval."""
    with create_custom_span(
        name="compose_query",
        inputs={"query": state.get("query", "")[:120], "role_mode": state.get("role_mode")}
    ):
        # PRIORITY: Handle menu selections first
        if state.get("query_type") == "menu_selection":
            menu_choice = state.get("menu_choice", state.get("query", ""))
            role_mode = state.get("role_mode", "")

            # Expand menu selection to full retrieval query
            expanded_query = _expand_menu_selection(menu_choice, role_mode)

            # FALLBACK: Check if the ORIGINAL query (not menu_choice) has enterprise keywords
            # This handles cases where follow-up selections contain enterprise terms
            original_query = state.get("query", "").lower()
            enterprise_keywords = ["adapt", "customer support", "enterprise", "use case", "chatbot"]
            if any(kw in original_query for kw in enterprise_keywords):
                # Override menu expansion with enterprise-focused query
                expanded_query = f"{original_query} enterprise adaptation guide customer support ROI what to change"
                logger.info(f"Enterprise fallback: overriding menu expansion with enterprise terms")

            # Add role context for better retrieval
            if role_mode:
                composed = f"[{role_mode}] {expanded_query}"
            else:
                composed = expanded_query

            # ENHANCEMENT: Include accumulated topics for progressive inference
            session_memory = state.get("session_memory", {})
            topics = session_memory.get("topics", [])
            if topics:
                # Include last 2-3 topics to connect menu selection to conversation history
                recent_topics = topics[-3:] if len(topics) > 3 else topics
                topic_context = " ".join(recent_topics)
                composed = f"{composed} {topic_context}"
                logger.debug(f"Enhanced menu selection with topics: {topic_context}")

            state["composed_query"] = composed.strip()
            state["menu_expanded"] = True  # Flag for debugging
            return state

        # Regular query composition for non-menu queries
        base_query = state.get("expanded_query") or state.get("query", "")
        # Fix typos before composition to improve retrieval quality
        base_query = _correct_common_typos(base_query)
        role_hint = state.get("role_mode", "")
        entity_hint = state.get("entities", {})

        # Get accumulated topics from session memory for progressive inference
        session_memory = state.get("session_memory", {})
        topics = session_memory.get("topics", [])

        entity_fragments = []
        if company := entity_hint.get("company"):
            entity_fragments.append(f"company={company}")
        if position := entity_hint.get("position"):
            entity_fragments.append(f"position={position}")
        if timeline := entity_hint.get("timeline"):
            entity_fragments.append(f"timeline={timeline}")

        composed = base_query
        if role_hint:
            composed = f"[{role_hint}] {composed}"

        # Use conversation flow analysis to enhance query composition
        chat_history = state.get("chat_history", [])
        if chat_history and len(chat_history) >= 2:
            # Import pattern analysis (avoid circular import)
            from assistant.flows.node_logic.stage6_formatting_nodes import _analyze_conversation_flow

            flow_analysis = _analyze_conversation_flow(chat_history, session_memory)
            detected_pattern = flow_analysis.get("pattern")

            if detected_pattern == "orchestration_to_enterprise":
                # User is asking about enterprise relevance of orchestration
                if "enterprise" not in base_query.lower() and "orchestration" in topics:
                    composed = f"{composed} enterprise orchestration relevance"
                    logger.debug("Enhanced query with pattern: orchestration_to_enterprise")
            elif detected_pattern == "architecture_to_implementation":
                # User wants implementation details, not high-level architecture
                if "code" not in base_query.lower() and "implementation" not in base_query.lower():
                    composed = f"{composed} code implementation details"
                    logger.debug("Enhanced query with pattern: architecture_to_implementation")
            elif detected_pattern == "general_to_specific":
                # User is diving deeper, enhance with previous topics for specificity
                if topics:
                    composed = f"{composed} {' '.join(topics[-2:])} detailed"
                    logger.debug("Enhanced query with pattern: general_to_specific")

        # Special handling for enterprise adaptation queries
        # Detect queries about adapting architecture to use cases (customer support, enterprise, etc.)
        base_lower = base_query.lower()
        enterprise_keywords = ["adapt", "adapts", "adaptation", "customer support", "enterprise",
                               "use case", "chatbot", "internal docs", "sales enablement"]
        is_enterprise_query = any(kw in base_lower for kw in enterprise_keywords)

        if is_enterprise_query:
            # PRESERVE original terms + add enhancement (don't replace)
            # Extract key terms from original query to ensure they're preserved
            original_key_terms = []
            if "customer support" in base_lower or "support" in base_lower:
                original_key_terms.extend(["customer", "support", "adapt"])
                if "customer support" not in composed.lower():
                    # Add specific terms that match ENTERPRISE_ADAPTATION_GUIDE.md
                    composed = f"{composed} customer support chatbot enterprise adaptation guide what to change knowledge base roles actions expected ROI"
                    logger.debug("Enhanced query with customer support adaptation guide terms")
            elif "enterprise" in base_lower:
                original_key_terms.extend(["enterprise", "adapt"])
                if "enterprise adaptation" not in composed.lower():
                    composed = f"{composed} enterprise adaptation guide patterns deployment customization"
                    logger.debug("Enhanced query with enterprise adaptation pattern")
            elif "architecture" in base_lower and ("support" in base_lower or "adapt" in base_lower):
                # Handle "architecture adapts to customer support" type queries
                original_key_terms.extend(["architecture", "adapt"])
                composed = f"{composed} enterprise adaptation guide customer support chatbot what to change ROI"
                logger.debug("Enhanced architecture adaptation query")
            else:
                # General enterprise/use case query
                composed = f"{composed} enterprise adaptation guide use cases"
                logger.debug("Enhanced query with general enterprise terms")

            # VALIDATION: Ensure original key terms are preserved
            if original_key_terms:
                missing_terms = [term for term in original_key_terms
                                if term not in composed.lower()]
                if missing_terms:
                    logger.warning(f"Composed query missing original terms: {missing_terms}. Adding them back.")
                    composed = f"{composed} {' '.join(missing_terms)}"

        # Enhance query with previous topics for progressive inference
        # Include last 2-3 topics to maintain context without query bloat
        # SCALABILITY: Only use last 3 topics (bounded memory for indefinite conversations)
        if topics:
            # Use sliding window: last 3 topics only (scalable for 100+ turns)
            recent_topics = topics[-3:] if len(topics) > 3 else topics
            topic_context = " ".join(recent_topics)
            # Only add topics if they're not already in composed query
            for topic in recent_topics:
                if topic not in composed.lower():
                    composed = f"{composed} {topic}"

        if entity_fragments:
            fragments = " | ".join(entity_fragments)
            composed = f"{composed} :: {fragments}"

        state["composed_query"] = composed.strip()

        # Validate query enhancement
        role_mode = state.get("role_mode", "")
        topics = session_memory.get("topics", [])
        composed = state.get("composed_query", "")

        # Check role is included (if role_mode exists)
        if role_mode and len(composed) > 10:
            if role_mode not in composed.lower() and not any(alias in composed.lower() for alias in ["hiring_manager", "technical", "developer"]):
                logger.warning(
                    f"Query enhancement missing role: role={role_mode}, query={composed[:50]}. "
                    f"This may indicate role context was not applied."
                )

        # Check topics are included (if topics exist and query length allows)
        if topics and len(composed) > 50:
            recent_topics = topics[-3:]
            included_topics = sum(1 for topic in recent_topics if topic.lower() in composed.lower())
            if included_topics == 0 and len(recent_topics) > 0:
                logger.warning(
                    f"Query enhancement missing topics: topics={recent_topics}, query={composed[:50]}. "
                    f"This may indicate topic context was not applied for progressive inference."
                )

    return state
