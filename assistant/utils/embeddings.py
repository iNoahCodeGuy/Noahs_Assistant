"""Embedding generation utility (legacy - not currently used in production).

Note: This file is kept for reference but is not imported by the main application.
Production embedding generation uses src/retrieval/pgvector_retriever.py.
"""

import logging
from openai.embeddings_utils import get_embedding

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_embedding(self, text: str) -> list:
        embedding = get_embedding(text, model=self.model_name)
        return embedding


def main():
    """Example usage (CLI tool - run with: python -m src.utils.embeddings)"""
    generator = EmbeddingGenerator(model_name="text-embedding-ada-002")
    sample_text = "This is a sample text for embedding."
    embedding = generator.generate_embedding(sample_text)

    # Use logger instead of print for consistency
    logger.info(f"Generated embedding with {len(embedding)} dimensions")
    logger.debug(f"Embedding preview: {embedding[:5]}...")


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    main()
