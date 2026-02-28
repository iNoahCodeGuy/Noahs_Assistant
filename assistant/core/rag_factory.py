"""RagEngine Component Factory

Handles the complex initialization logic for RagEngine components.
Separates construction concerns from core RAG orchestration.

**Note**: FAISS support has been removed. This factory now only supports
pgvector-based retrieval through Supabase.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Any

from .langchain_compat import OpenAIEmbeddings, ChatAnthropic
from .document_processor import DocumentProcessor
from .response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

class RagEngineFactory:
    """Factory for creating RagEngine components.

    **Architecture**: Creates components for pgvector-based RAG system.
    No longer supports FAISS fallback - all operations use Supabase.
    """
    def __init__(self, settings=None):
        self.settings = settings
        self.degraded_mode = False

    def create_embeddings(self) -> Tuple[Any, bool]:
        """Create embeddings with fallback. Returns (embeddings, is_degraded)."""
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=getattr(self.settings, "openai_api_key", None),
                model=getattr(self.settings, "embedding_model", "text-embedding-ada-002")
            )
            return embeddings, False
        except Exception as e:
            logger.warning(f"Embedding initialization failed, entering degraded mode: {e}")
            class _FallbackEmb:
                def embed_query(self, text: str):
                    return [float((hash(text) >> i) & 0xFF) / 255.0 for i in range(0, 32)]
            return _FallbackEmb(), True

    def create_llm(self, model_name: Optional[str] = None) -> Tuple[Any, bool]:
        """Create LLM with fallback and LangSmith wrapping. Returns (llm, is_degraded).

        Args:
            model_name: Optional model override.
                       If None, uses claude-sonnet-4-5-20250929.
        """
        try:
            # Use provided model or fall back to Anthropic Sonnet
            selected_model = model_name or getattr(self.settings, "anthropic_model", "claude-sonnet-4-5-20250929")

            # Use ChatAnthropic with the specified model
            llm = ChatAnthropic(
                anthropic_api_key=getattr(self.settings, "anthropic_api_key", None),
                model_name=selected_model,
                temperature=0.7,  # Sweet spot: conversational + creative, but still follows structure (0.9 was too high)
                max_tokens=4096  # Allow full analytics dashboard (11,772 chars ≈ 3,000 tokens)
            )
            logger.debug(f"LLM initialized with Anthropic model={selected_model}")

            return llm, False
        except Exception as e:
            logger.warning(f"LLM initialization failed, degraded mode responses will be used: {e}")
            class _FallbackLLM:
                """Fallback LLM that implements both legacy and modern LangChain interfaces."""
                def predict(self, prompt: str) -> str:
                    words = prompt.strip().split()
                    tail = " ".join(words[-40:])
                    return f"[DEGRADED MODE SYNTHESIS]\n{tail}"

                def invoke(self, input_data, **kwargs):
                    """Modern LangChain interface - invoke method."""
                    from langchain_core.messages import AIMessage
                    # Handle different input types
                    if isinstance(input_data, str):
                        prompt = input_data
                    elif isinstance(input_data, list):
                        # List of messages - extract content
                        prompt = " ".join(
                            getattr(m, 'content', str(m)) for m in input_data
                        )
                    else:
                        prompt = str(input_data)

                    result = self.predict(prompt)
                    return AIMessage(content=result)

                def __call__(self, *args, **kwargs):
                    """Make callable for legacy compatibility."""
                    return self.invoke(*args, **kwargs)
            return _FallbackLLM(), True

    def create_career_kb(self, provided_kb=None):
        """Create or use provided career knowledge base."""
        if provided_kb is not None:
            return provided_kb

        try:
            from assistant.retrieval.career_kb import CareerKnowledgeBase
            kb_path = getattr(self.settings, "career_kb_path")
            return CareerKnowledgeBase(kb_path)
        except Exception as e:
            logger.warning(f"Failed to create career KB: {e}")
            return None

    def create_code_index(self, provided_index=None):
        """Create or use provided code index."""
        if provided_index is not None:
            return provided_index

        try:
            from assistant.retrieval.code_index import CodeIndex
            index_path = getattr(self.settings, "code_index_path", "vector_stores/code_index")
            return CodeIndex(index_path)
        except Exception as e:
            logger.warning(f"Failed to create code index: {e}")
            return None

    def load_documents(self, career_kb, provided_career_kb=None):
        """Load and process documents."""
        processor = DocumentProcessor(chunk_size=600, chunk_overlap=60)

        if provided_career_kb is not None:
            return processor.load_from_career_kb(provided_career_kb)

        kb_path = getattr(self.settings, "career_kb_path")
        return processor.load_from_csv(kb_path, source_column="Question")
