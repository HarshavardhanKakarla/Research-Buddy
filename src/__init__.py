"""Smart Research Assistant - core package."""

from .config import Settings, get_settings
from .ingestion import ingest_directory, ingest_file, list_ingested_documents
from .retrieval import retrieve, semantic_search
from .schemas import StructuredAnswer
from .summarizer import answer_query

__all__ = [
    "Settings",
    "get_settings",
    "ingest_file",
    "ingest_directory",
    "list_ingested_documents",
    "semantic_search",
    "retrieve",
    "answer_query",
    "StructuredAnswer",
]
