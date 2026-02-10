"""Agent 2 (Part 2): Runtime knowledge retriever.

Loads the persisted ChromaDB created by knowledge_ingestor.py and returns
relevant textbook/guideline chunks for a query.

Usage:
  from agents.knowledge_retriever import search_knowledge
  chunks = search_knowledge("Interpret IOP and vision trends in glaucoma")
"""

from __future__ import annotations

import os
import re
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


_STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "into",
    "about",
}


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.split(r"\W+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = a & b
    if not intersection:
        return 0.0
    union = a | b
    return len(intersection) / len(union)


def _normalize_scores(candidates: list[dict]) -> list[float]:
    scores = [float(c.get("score", 0.0) or 0.0) for c in candidates]
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s <= 0:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _extract_query_points(query: str, max_points: int) -> list[str]:
    if not query:
        return []
    tokens = re.split(r"\W+", query.lower())
    points = []
    seen = set()
    for token in tokens:
        if len(token) < 3:
            continue
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        points.append(token)
        if len(points) >= max_points:
            break
    return points


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


def select_context(
    query: str,
    candidates: list[dict],
    k: int,
    *,
    lambda_param: float = 0.7,
    sim_threshold: float = 0.5,
    max_points: int = 5,
) -> dict:
    """Select K diverse, relevant chunks from cross-encoder-scored candidates.

    Expected candidate shape:
      {"chunk_id": str, "page": ..., "section_type": ..., "score": float, "text": str}
    """
    query = query or ""
    candidates = candidates or []
    points = _extract_query_points(query, max_points)

    if k <= 0 or not candidates:
        return {
            "selected": [],
            "coverage_check": {
                "covered_points": [],
                "missing_points": points,
            },
            "redundancy_notes": [],
        }

    norm_scores = _normalize_scores(candidates)
    token_sets = [_tokenize(c.get("text", "") or "") for c in candidates]

    remaining = list(range(len(candidates)))
    selected = []

    first_idx = max(remaining, key=lambda i: norm_scores[i])
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            max_sim = 0.0
            for sel_idx in selected:
                sim = _jaccard(token_sets[idx], token_sets[sel_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_param * norm_scores[idx] - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    selected_output = []
    seen_tokens: set[str] = set()
    for rank, idx in enumerate(selected, start=1):
        chunk = candidates[idx]
        tokens = token_sets[idx]
        novel = [t for t in tokens if t not in seen_tokens]
        for t in tokens:
            seen_tokens.add(t)
        if rank == 1:
            why = "Highest score; strong overall relevance."
        else:
            if novel:
                sample = ", ".join(sorted(novel)[:3])
                why = f"High score and adds novel terms: {sample}."
            else:
                why = "High score and adds non-overlapping content."
        selected_output.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "rank": rank,
                "why": why,
            }
        )

    covered_points = []
    missing_points = []
    selected_text = " ".join((candidates[i].get("text", "") or "") for i in selected).lower()
    for point in points:
        if point in selected_text:
            covered_points.append(point)
        else:
            missing_points.append(point)

    redundancy_notes = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            idx_a = selected[i]
            idx_b = selected[j]
            sim = _jaccard(token_sets[idx_a], token_sets[idx_b])
            if sim >= sim_threshold:
                overlap = sorted(token_sets[idx_a] & token_sets[idx_b])[:3]
                overlap_str = ", ".join(overlap) if overlap else "shared terms"
                redundancy_notes.append(
                    f"chunk_id {candidates[idx_a].get('chunk_id','')} overlaps with "
                    f"chunk_id {candidates[idx_b].get('chunk_id','')} on tokens: {overlap_str}"
                )

    return {
        "selected": selected_output,
        "coverage_check": {
            "covered_points": covered_points,
            "missing_points": missing_points,
        },
        "redundancy_notes": redundancy_notes,
    }


def search_knowledge(query: str, top_k: int = 3) -> list[str]:
    if not query or not query.strip():
        return []

    embedding = _get_embedder().encode(query.strip()).tolist()
    results = _get_collection().query(query_embeddings=[embedding], n_results=top_k)
    return results.get("documents", [[]])[0]
