"""
retrieval.py
------------
Semantic search over the Pinecone index, with:
  - optional metadata filters (restrict to a source file and/or section)
  - a lightweight LLM-based rerank step to trim noisy vector-similarity hits
    down to the most relevant chunks before they reach the summarizer.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings
from .ingestion import get_vector_store

RERANK_PROMPT = ChatPromptTemplate.from_template(
    """You are grading document chunks for relevance to a research question.
Question: {question}

For each chunk below, respond with ONLY its number if it is relevant enough
to help answer the question, one number per line. Skip irrelevant chunks.

{numbered_chunks}
"""
)


def semantic_search(
    query: str,
    settings: Settings,
    source: Optional[str] = None,
    section: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[Document]:
    """Query Pinecone for the top_k most similar chunks, with optional filters."""
    vector_store = get_vector_store(settings, namespace=namespace)

    metadata_filter = {}
    if source:
        metadata_filter["source"] = source
    if section:
        metadata_filter["section"] = section

    return vector_store.similarity_search(
        query, k=settings.top_k, filter=metadata_filter or None
    )


def rerank(query: str, docs: List[Document], settings: Settings) -> List[Document]:
    """
    Cheap LLM-based reranker: ask the chat model which candidate chunks are
    actually relevant, then keep only those (capped at top_k_after_rerank).
    This avoids pulling in a separate cross-encoder dependency while still
    filtering out embedding-similarity false positives.
    """
    if not docs:
        return []

    numbered = "\n\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs))
    llm = ChatOpenAI(
        model=settings.chat_model, temperature=0, api_key=settings.openai_api_key
    )
    chain = RERANK_PROMPT | llm
    response = chain.invoke({"question": query, "numbered_chunks": numbered})

    kept_indices = []
    for line in response.content.splitlines():
        line = line.strip()
        if line.isdigit():
            index = int(line)
            if 0 <= index < len(docs) and index not in kept_indices:
                kept_indices.append(index)

    if not kept_indices:
        # Fallback: if the reranker returned nothing parseable, trust the
        # original vector-similarity order rather than returning empty.
        return docs[: settings.top_k_after_rerank]

    reranked = [docs[i] for i in kept_indices if i < len(docs)]
    return reranked[: settings.top_k_after_rerank]


def retrieve(
    query: str,
    settings: Settings,
    source: Optional[str] = None,
    section: Optional[str] = None,
    use_rerank: bool = True,
    namespace: Optional[str] = None,
) -> List[Document]:
    """End-to-end retrieval: vector search then optional rerank."""
    candidates = semantic_search(
        query, settings, source=source, section=section, namespace=namespace
    )
    if use_rerank:
        return rerank(query, candidates, settings)
    return candidates[: settings.top_k_after_rerank]
