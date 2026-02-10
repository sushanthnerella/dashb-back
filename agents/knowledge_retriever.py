"""Agent 2 (Part 2): Runtime knowledge retriever.

Loads the persisted ChromaDB created by knowledge_ingestor.py and returns
relevant textbook/guideline chunks for a query.

Usage:
  from agents.knowledge_retriever import search_knowledge
  chunks = search_knowledge("Interpret IOP and vision trends in glaucoma")
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = REPO_ROOT / "vector_db" / "book_chunks"
COLLECTION_NAME = "ophthal_book"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR")
EMBED_OFFLINE = os.getenv("EMBED_OFFLINE", "0") == "1"


_embedder: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        if EMBED_OFFLINE:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        kwargs = {}
        if EMBED_CACHE_DIR:
            kwargs["cache_folder"] = EMBED_CACHE_DIR

        _embedder = SentenceTransformer(MODEL_NAME, **kwargs)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def search_knowledge(query: str, top_k: int = 3) -> list[str]:
    if not query or not query.strip():
        return []

    embedding = _get_embedder().encode(query.strip()).tolist()
    results = _get_collection().query(query_embeddings=[embedding], n_results=top_k)
    return results.get("documents", [[]])[0]
