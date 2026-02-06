"""Helper functions for intelligently injecting personality context into responses.

This module provides utilities to:
1. Determine when personality context would add value to a response
2. Extract and structure personality traits from retrieved chunks
3. Guide response generation on how to use personality information
"""

from typing import List, Dict, Optional


def should_include_personality(query: str, context: List[str], role: str) -> bool:
    """Determine if personality context would add value to this response.

    Returns True when:
    - Query asks about work style, approach, motivations, personality
    - Query asks "why" something was built
    - Hiring manager role asks about cultural fit
    - Context includes personality-related content

    Args:
        query: User's query text
        context: Retrieved context chunks (as strings)
        role: User's role (e.g., "Hiring Manager (technical)")

    Returns:
        True if personality context would enhance the response, False otherwise

    Examples:
        >>> should_include_personality("What's Noah's work style?", [], "Hiring Manager")
        True
        >>> should_include_personality("How does RAG work?", [], "Software Developer")
        False
        >>> should_include_personality("Why did Noah build this?", [], "Software Developer")
        True
    """
    query_lower = query.lower()

    # Direct personality indicators
    personality_indicators = [
        'personality', 'trait', 'characteristic', 'motivation',
        'approach', 'style', 'value', 'preference', 'philosophy',
        'work style', 'cultural fit', 'team fit', 'work ethic',
        'communication style', 'problem solving approach'
    ]

    # "Why" questions often benefit from personality context
    why_indicators = ['why', 'what motivates', 'what drives', 'reason']

    # Check if query directly asks about personality/work style
    if any(ind in query_lower for ind in personality_indicators):
        return True

    # Check for "why" questions that would benefit from personality context
    if any(why in query_lower for why in why_indicators):
        # But only if it's about Noah or his decisions
        if 'noah' in query_lower or 'he' in query_lower or 'his' in query_lower:
            return True

    # Hiring manager + cultural fit questions
    if role and ('hiring' in role.lower() or 'manager' in role.lower()):
        cultural_fit_indicators = [
            'fit', 'team', 'culture', 'work style', 'collaboration',
            'work well', 'environment', 'values'
        ]
        if any(ind in query_lower for ind in cultural_fit_indicators):
            return True

    # Context already includes personality content
    context_str = ' '.join(context).lower()
    if any(ind in context_str for ind in personality_indicators):
        # If context mentions personality, it might be relevant
        # But only if query is related (not purely technical)
        if not any(tech in query_lower for tech in ['code', 'api', 'database', 'implementation', 'architecture']):
            return True

    return False


def extract_personality_traits(personality_chunks: List[str]) -> Dict[str, str]:
    """Extract key personality traits from retrieved chunks.

    This function attempts to identify and extract personality traits
    mentioned in the chunks for structured use in response generation.

    Args:
        personality_chunks: List of personality-related chunk content (strings)

    Returns:
        Dictionary mapping trait names to descriptions

    Example:
        >>> chunks = ["Noah is thoughtful and systematic..."]
        >>> extract_personality_traits(chunks)
        {'thoughtful': 'values systematic thinking', 'systematic': 'approaches problems methodically'}
    """
    if not personality_chunks:
        return {}

    # Common personality traits to look for
    trait_keywords = {
        'teaching-oriented': ['teaching', 'educate', 'explain', 'understand'],
        'thoughtful': ['thoughtful', 'systematic', 'methodical', 'organized'],
        'playful': ['playful', 'fun', 'humor', 'easter egg'],
        'enterprise-minded': ['enterprise', 'scalability', 'production', 'roi'],
        'authentic': ['authentic', 'humble', 'honest', 'self-aware'],
        'bridge-builder': ['bridge', 'connect', 'accessible', 'both technical and']
    }

    traits = {}
    combined_content = ' '.join(personality_chunks).lower()

    for trait, keywords in trait_keywords.items():
        if any(kw in combined_content for kw in keywords):
            # Extract a sentence or phrase that mentions this trait
            # This is a simple extraction - could be enhanced with NLP
            traits[trait] = f"Mentioned in personality context"

    return traits


def get_personality_prompt_guidance(has_personality_context: bool) -> Optional[str]:
    """Generate prompt guidance for using personality context.

    Args:
        has_personality_context: Whether personality chunks are in the context

    Returns:
        Prompt guidance string or None if not needed
    """
    if not has_personality_context:
        return None

    return """
## NOAH'S PERSONALITY CONTEXT (Use Naturally, Not Forced)

When relevant, let Noah's personality traits inform how you describe him:
- **Teaching-oriented**: "Noah built this with education in mind—he genuinely wants people to understand how GenAI works"
- **Thoughtful**: "This architecture reflects Noah's systematic approach to problem-solving"
- **Playful but professional**: "Noah's playful side shows in features like the confess crush easter egg, but he takes technical excellence seriously"
- **Enterprise-minded**: "Noah thinks about scalability and production patterns even in personal projects"

**When to use personality context:**
- ✅ When explaining *why* something was built a certain way
- ✅ When connecting work choices to underlying values
- ✅ When hiring managers ask about work style or cultural fit
- ✅ When providing anecdotes that make responses more engaging

**When NOT to force it:**
- ❌ Don't add personality traits to purely technical explanations
- ❌ Don't make every response about personality
- ❌ Don't use personality to avoid answering technical questions

**Integration pattern:**
- Lead with technical/career facts
- Weave in personality naturally: "This reflects Noah's [trait] approach..."
- Use as bridge: "His [personality trait] shows in how he [behavior]..."
"""
