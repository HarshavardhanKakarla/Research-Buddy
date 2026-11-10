"""
config.py
---------
Centralized configuration for the Smart Research Assistant.
All environment variables and tunable constants live here so the
rest of the codebase never touches os.environ directly.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process env.
load_dotenv()


def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: '{key}'. "
            f"Add it to your .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # --- API keys ---
    openai_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)

    # --- Pinecone ---
    pinecone_index_name: str = "research-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Embeddings & LLM ---
    embedding_model: str = "text-embedding-3-small"   # 1536 dims, cost-effective
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0.2

    # --- Chunking ---
    chunk_size: int = 1000          # characters per chunk
    chunk_overlap: int = 150        # overlap between chunks (~15%)

    # --- Retrieval ---
    top_k: int = 8                  # candidates pulled from Pinecone
    top_k_after_rerank: int = 4     # chunks actually sent to the LLM

    # --- Storage ---
    upload_dir: str = "data/uploads"
    manifest_path: str = "data/document_manifest.json"


def get_settings(require_openai: bool = False) -> Settings:
    """Build settings, optionally requiring a server-provided OpenAI key."""
    openai_api_key = _require_env("OPENAI_API_KEY") if require_openai else ""
    return Settings(
        openai_api_key=openai_api_key,
        pinecone_api_key=_require_env("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "research-assistant"),
    )
