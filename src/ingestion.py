"""
ingestion.py
------------
Handles document ingestion:
  1. Load PDFs (or plain-text articles).
  2. Split into overlapping semantic chunks.
  3. Embed chunks and upsert them into a Pinecone index.
  4. Track ingested documents in a local manifest (for the UI's document list).

Each vector's metadata carries enough info (source file, page, section guess)
to let the retrieval/summarization layers cite and filter results.
"""

import hashlib
import json
import os
import re
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import Settings

# Cheap heuristic used to tag chunks with a likely paper section.
# This powers "summarize the Methods section" style queries later on.
SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "methods": r"\b(methods?|methodology|materials and methods)\b",
    "results": r"\bresults?\b",
    "discussion": r"\bdiscussion\b",
    "comparison": r"\b(comparison|related work|baselines?)\b",
    "conclusion": r"\bconclusions?\b",
}


def guess_section(text: str) -> str:
    """Best-effort classification of a chunk into a paper section."""
    head = text[:120].lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, head):
            return section
    return "body"


def _file_hash(path: str) -> str:
    """Stable short hash of a file's contents, used as a dedup key."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_document(path: str) -> List[Document]:
    """Load a single file (PDF or .txt/.md) into LangChain Documents."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def chunk_documents(
    docs: List[Document], settings: Settings, source_name: str
) -> List[Document]:
    """Split raw documents into overlapping chunks with rich metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": source_name,
                "chunk_id": f"{source_name}-{i}",
                "page": chunk.metadata.get("page", 0),
                "section": guess_section(chunk.page_content),
            }
        )
    return chunks


def ensure_index(settings: Settings) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )


def get_vector_store(
    settings: Settings, namespace: Optional[str] = None
) -> PineconeVectorStore:
    """Return a LangChain-wrapped handle to the Pinecone index."""
    ensure_index(settings)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
        namespace=namespace,
    )


def _load_manifest(settings: Settings) -> Dict[str, dict]:
    if os.path.exists(settings.manifest_path):
        with open(settings.manifest_path, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(settings: Settings, manifest: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(settings.manifest_path), exist_ok=True)
    with open(settings.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def ingest_file(
    path: str,
    settings: Settings,
    namespace: Optional[str] = None,
    manifest: Optional[Dict[str, dict]] = None,
    source_name: Optional[str] = None,
) -> dict:
    """
    Full ingestion pipeline for one file: load -> chunk -> embed -> upsert.
    Returns a small status dict for UI feedback. Skips files already ingested
    (based on content hash) to avoid duplicate vectors.
    """
    source_name = source_name or os.path.basename(path)
    file_hash = _file_hash(path)

    session_manifest = manifest is not None
    current_manifest = manifest if session_manifest else _load_manifest(settings)
    previous_entry = current_manifest.get(source_name, {})
    if previous_entry.get("hash") == file_hash:
        return {"source": source_name, "status": "skipped (already ingested)", "chunks": 0}

    raw_docs = load_document(path)
    chunks = chunk_documents(raw_docs, settings, source_name)
    if not chunks:
        raise ValueError(f"No extractable text found in {source_name}")

    vector_store = get_vector_store(settings, namespace=namespace)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    vector_store.add_documents(documents=chunks, ids=ids)

    old_ids = previous_entry.get("chunk_ids")
    if old_ids is None:
        old_ids = [
            f"{source_name}-{index}"
            for index in range(previous_entry.get("chunks", 0))
        ]
    stale_ids = [chunk_id for chunk_id in old_ids if chunk_id not in ids]
    if stale_ids:
        vector_store.delete(ids=stale_ids)

    current_manifest[source_name] = {
        "hash": file_hash,
        "chunks": len(chunks),
        "sections": sorted({c.metadata["section"] for c in chunks}),
        "chunk_ids": ids,
    }
    if not session_manifest:
        _save_manifest(settings, current_manifest)

    return {"source": source_name, "status": "ingested", "chunks": len(chunks)}


def ingest_directory(directory: str, settings: Settings) -> List[dict]:
    """Ingest every supported file in a directory."""
    results = []
    for fname in os.listdir(directory):
        if fname.lower().endswith((".pdf", ".txt", ".md")):
            results.append(ingest_file(os.path.join(directory, fname), settings))
    return results


def list_ingested_documents(
    settings: Settings, manifest: Optional[Dict[str, dict]] = None
) -> Dict[str, dict]:
    """Expose the manifest for the UI's sidebar document list."""
    return manifest if manifest is not None else _load_manifest(settings)


def delete_namespace(settings: Settings, namespace: str) -> None:
    """Delete vectors belonging to an expired browser session."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    index.delete(delete_all=True, namespace=namespace)
