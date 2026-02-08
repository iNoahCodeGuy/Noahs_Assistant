"""Supabase configuration for Noah's AI Assistant.

This module replaces Google Cloud Platform services with Supabase:
- Cloud SQL PostgreSQL → Supabase Postgres (with pgvector extension)
- Cloud Storage → Supabase Storage
- Secret Manager → Environment variables
- Pub/Sub → Direct database writes (simpler architecture)

Environment Variables Required:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SERVICE_ROLE_KEY: Service role key (for server-side operations)
- SUPABASE_ANON_KEY: Anonymous key (for client-side operations, optional)
- OPENAI_API_KEY: OpenAI API key for embeddings and chat
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()


# ============================================================================
# LOGGING CONFIGURATION
# Set noisy HTTP/API loggers to WARNING to reduce debug noise
# Keep assistant.* loggers at DEBUG for actual application debugging
# ============================================================================
def configure_logging():
    """Configure logging levels to reduce noise from third-party libraries.

    Sets these noisy loggers to WARNING:
    - hpack: HTTP/2 header compression (very verbose)
    - urllib3: HTTP client (connection details)
    - anthropic._base_client: Anthropic API client (HTTP noise)
    - langsmith: LangSmith tracing (verbose trace data)
    - httpcore: HTTP core library
    - httpx: HTTP client
    - openai: OpenAI API client

    Keeps assistant.* loggers at their default level for debugging.
    """
    # List of noisy loggers to silence
    noisy_loggers = [
        "hpack",
        "urllib3",
        "httpcore",
        "httpx",
        "anthropic",
        "anthropic._base_client",
        "langsmith",
        "langsmith.client",
        "langsmith.run_trees",
        "openai",
        "openai._base_client",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Apply logging configuration on import
configure_logging()


@dataclass
class SupabaseConfig:
    """Supabase configuration settings.

    This replaces the CloudConfig from GCP implementation.
    All configuration is driven by environment variables for
    compatibility with both local development and Vercel deployment.
    """

    # Supabase connection
    url: str
    service_role_key: str
    anon_key: Optional[str] = None

    # Storage buckets
    public_bucket: str = "public"  # For images, public assets
    private_bucket: str = "private"  # For resumes, CSVs, KB data

    # Database settings
    database_name: str = "postgres"  # Default Supabase database

    # Vector search settings
    vector_dimensions: int = 1536  # OpenAI ada-002 embedding size
    similarity_threshold: float = 0.7  # Minimum similarity for retrieval
    top_k: int = 3  # Number of KB chunks to retrieve

    def __post_init__(self):
        """Validate required configuration."""
        if not self.url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.service_role_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")


class SupabaseSettings:
    """Global settings manager for Supabase integration.

    Why this approach:
    - Centralized configuration management
    - Easy environment detection (local vs production)
    - Backward compatible with existing Settings pattern
    - Clear separation between Supabase config and application config
    """

    def __init__(self):
        """Initialize settings from environment variables.

        This works seamlessly in:
        - Local development (.env file)
        - Vercel deployment (environment variables panel)
        - Testing (mocked environment variables)
        """
        # Detect environment
        self.is_production = os.getenv("VERCEL_ENV") == "production"
        self.is_vercel = os.getenv("VERCEL") == "1"

        # Supabase configuration (strip whitespace to prevent header issues)
        self.supabase_config = SupabaseConfig(
            url=os.getenv("SUPABASE_URL", "").strip(),
            service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
            anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip() if os.getenv("SUPABASE_ANON_KEY") else None
        )

        # Anthropic configuration (primary LLM provider)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929").strip()

        # OpenAI configuration (for embeddings only)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002").strip()

        # Legacy OpenAI model settings (deprecated — generation now uses Anthropic Claude)
        self.openai_model = os.getenv("OPENAI_MODEL", "claude-sonnet-4-5-20250929").strip()
        self.openai_reasoning_model = os.getenv("OPENAI_REASONING_MODEL", "claude-sonnet-4-5-20250929").strip()
        self.openai_fast_model = os.getenv("OPENAI_FAST_MODEL", "claude-haiku-4-5-20251001").strip()

        # External services (for Next.js API routes)
        self.resend_api_key = os.getenv("RESEND_API_KEY", "").strip()
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self.twilio_from_number = os.getenv("TWILIO_FROM", "").strip()

        # Application settings
        self.youtube_fight_link = os.getenv(
            "YOUTUBE_FIGHT_LINK",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

        # Legacy compatibility (for existing code that expects these attributes)
        self.api_key = self.openai_api_key
        self.career_kb_path = "data/career_kb.csv"  # Still used for initial data load
        self.vector_store_path = "vector_stores/"  # Deprecated but kept for compatibility
        self.confessions_path = "data/confessions.csv"  # Confession storage path

    def validate_api_key(self):
        """Validate that OpenAI API key is present and properly formatted.

        This replaces cloud_settings.validate_api_key() from GCP version.
        Also checks for common issues like trailing newlines that cause HTTP header errors.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
            )

        # Check for newlines that cause "Illegal header value" errors
        if '\n' in self.openai_api_key or '\r' in self.openai_api_key:
            raise ValueError(
                "OpenAI API key contains newline characters. "
                "This causes 'Illegal header value' errors in HTTP requests. "
                "Check your environment variables for trailing newlines."
            )

        return True

    def validate_supabase(self):
        """Validate that Supabase configuration is complete.

        Call this before any Supabase operations to ensure proper setup.
        Also checks for common issues like trailing newlines.
        """
        try:
            # Check if config exists and is valid
            self.supabase_config  # This will raise ValueError if invalid

            # Check for newlines or other non-printable characters
            if '\n' in self.supabase_config.url or '\r' in self.supabase_config.url:
                raise ValueError(
                    "Supabase URL contains newline characters. "
                    "Check your environment variables for trailing newlines."
                )
            if '\n' in self.supabase_config.service_role_key or '\r' in self.supabase_config.service_role_key:
                raise ValueError(
                    "Supabase service role key contains newline characters. "
                    "Check your environment variables for trailing newlines."
                )

            return True
        except ValueError as e:
            raise ValueError(
                f"Supabase configuration incomplete: {e}. "
                "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your .env file."
            )

    def validate_configuration(self) -> bool:
        """Validate all required configuration.

        This replaces cloud_settings.validate_configuration() from GCP version.
        """
        self.validate_api_key()
        self.validate_supabase()
        return True


def configure_logging():
    """Configure production-ready logging for the application.

    Sets up logging with appropriate levels for different environments:
    - Production (Vercel): INFO level (clean, essential logs only)
    - Development (local): DEBUG level (verbose, helpful for debugging)

    Log format includes timestamp, module name, level, and message for
    easy troubleshooting in both local terminal and Vercel function logs.

    Why this approach:
    - Uses Python's built-in logging (zero dependencies)
    - Environment-aware (auto-detects production vs development)
    - Compatible with Vercel serverless stdout logs
    - Works with LangSmith observability (logs appear in traces)

    Usage:
        ```python
        import logging
        logger = logging.getLogger(__name__)

        logger.info("User query processed successfully")
        logger.debug(f"Retrieved {len(chunks)} chunks")  # Only in dev
        logger.error("Failed to connect to Supabase")
        ```
    """
    # Detect environment (use is_production from settings if available, else check env)
    try:
        is_production = supabase_settings.is_production
    except NameError:
        # Settings not initialized yet, check env directly
        is_production = os.getenv("VERCEL_ENV") == "production"

    # Set log level based on environment
    level = logging.INFO if is_production else logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Reduce noise from verbose third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("supabase").setLevel(logging.WARNING)

    # Silence extremely verbose HTTP/2 and networking libraries
    logging.getLogger("hpack").setLevel(logging.ERROR)  # HTTP/2 header compression (very verbose)
    logging.getLogger("hpack.hpack").setLevel(logging.ERROR)
    logging.getLogger("hpack.table").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # HTTP connection pooling
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    # Reduce LangSmith tracing noise in production
    logging.getLogger("langsmith").setLevel(logging.ERROR if is_production else logging.WARNING)
    logging.getLogger("langsmith.client").setLevel(logging.ERROR if is_production else logging.WARNING)

    # Keep assistant.* loggers at DEBUG in development for detailed debugging
    # In production, they'll inherit the root INFO level
    if not is_production:
        logging.getLogger("assistant").setLevel(logging.DEBUG)

    # Log configuration complete (but only in debug mode to avoid spam)
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={logging.getLevelName(level)}, environment={'production' if is_production else 'development'}")


# Singleton instance
# This replaces 'cloud_settings' from the GCP implementation
supabase_settings = SupabaseSettings()


# Helper function to get Supabase client (lazy initialization)
_supabase_client = None

def get_supabase_client():
    """Get or create Supabase client instance.

    Why lazy initialization:
    - Client creation might fail if environment vars are missing
    - Tests can mock this function easily
    - Allows for client recycling/pooling in the future

    Returns:
        supabase.Client: Authenticated Supabase client

    Example:
        from config.supabase_config import get_supabase_client

        supabase = get_supabase_client()
        result = supabase.table('messages').select('*').execute()
    """
    global _supabase_client

    if _supabase_client is None:
        try:
            from supabase import create_client, Client

            config = supabase_settings.supabase_config
            _supabase_client = create_client(
                config.url,
                config.service_role_key
            )
        except ImportError:
            raise ImportError(
                "Supabase Python client not installed. "
                "Run: pip install supabase"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Supabase client: {e}. "
                "Check your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )

    return _supabase_client


# Initialize logging when module is imported
configure_logging()
