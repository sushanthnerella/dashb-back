"""Multi-book runtime knowledge retrieval and semantic planning."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = REPO_ROOT / "vector_db" / "book_chunks"
REGISTRY_PATH = REPO_ROOT / "vector_db" / "books_registry.json"
COLLECTION_PREFIX = "book__"
LEGACY_COLLECTION = "ophthal_book"

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR")
EMBED_OFFLINE = os.getenv("EMBED_OFFLINE", "0") == "1"

_embedder: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None
_collections: dict[str, object] = {}

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
    "can",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
}

_INTENT_PATTERNS = {
    "definition": ["what is", "what are", "define", "definition", "meaning"],
    "procedure": ["how to", "steps", "procedure", "method", "workflow", "implement", "perform"],
    "compare": ["compare", "difference", "versus", "vs", "contrast"],
    "formula": ["formula", "equation", "calculate", "compute", "derive"],
    "troubleshooting": ["error", "issue", "problem", "fails", "failure", "not working", "fix"],
}

_TERM_EXPANSIONS = {
    "definition": ["meaning", "description", "concept"],
    "procedure": ["steps", "workflow", "protocol"],
    "compare": ["difference", "contrast", "distinguish"],
    "formula": ["equation", "calculation", "expression"],
    "troubleshooting": ["error", "root cause", "resolution"],
    "diagnosis": ["assessment", "finding", "clinical impression"],
    "treatment": ["therapy", "management", "intervention"],
    "symptom": ["sign", "presentation", "manifestation"],
    "glaucoma": ["optic neuropathy", "visual field loss", "ocular hypertension"],
    "iop": ["intraocular pressure", "eye pressure", "tonometry"],
    "retina": ["retinal tissue", "fundus", "posterior segment"],
}

_EXPECTED_COMPONENTS = {
    "definition": ["definition", "key facts", "constraints"],
    "procedure": ["definition", "steps", "constraints"],
    "compare": ["definition", "comparison criteria", "constraints"],
    "explain": ["definition", "supporting facts", "constraints"],
    "formula": ["definition", "formula", "constraints"],
    "troubleshooting": ["problem", "diagnosis steps", "constraints"],
}


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


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_collection(collection_name: str):
    if collection_name in _collections:
        return _collections[collection_name]
    collection = _get_client().get_or_create_collection(name=collection_name)
    _collections[collection_name] = collection
    return collection


def load_books_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_available_books() -> list[dict]:
    registry = load_books_registry()
    books = []
    for book_id, meta in registry.items():
        books.append(
            {
                "book_id": book_id,
                "title": meta.get("title", book_id),
                "collection_name": meta.get("collection_name", f"{COLLECTION_PREFIX}{book_id}"),
                "pdf_path": meta.get("pdf_path", ""),
            }
        )
    return books


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.split(r"\W+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / len(a | b)


def _normalize_scores(candidates: list[dict]) -> list[float]:
    scores = [float(c.get("score", 0.0) or 0.0) for c in candidates]
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn <= 0:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _extract_query_points(query: str, max_points: int = 6) -> list[str]:
    tokens = re.split(r"\W+", (query or "").lower())
    points = []
    seen = set()
    for t in tokens:
        if len(t) < 3 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        points.append(t)
        if len(points) >= max_points:
            break
    return points


def _classify_intent(query: str) -> str:
    q = (query or "").lower().strip()
    scores = {intent: 0 for intent in _INTENT_PATTERNS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for p in patterns:
            if p in q:
                scores[intent] += 2

    if q.startswith("how"):
        scores["procedure"] += 1
    elif q.startswith("what"):
        scores["definition"] += 1
    elif q.startswith("why"):
        scores["troubleshooting"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "explain"


def _expand_terms(intent: str, points: list[str]) -> list[str]:
    expanded = []
    seen = set()

    def add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            expanded.append(term)

    add(intent)
    for p in points:
        add(p)
        for syn in _TERM_EXPANSIONS.get(p, []):
            add(syn)
    for syn in _TERM_EXPANSIONS.get(intent, []):
        add(syn)
    return expanded[:12]


def _resolve_books(book_ids: list[str] | None = None, book_titles: list[str] | None = None) -> list[dict]:
    books = get_available_books()
    if not books:
        return []

    if not book_ids and not book_titles:
        return books

    wanted_ids = set(book_ids or [])
    wanted_titles = {t.lower() for t in (book_titles or [])}
    out = []
    for b in books:
        if b["book_id"] in wanted_ids or b["title"].lower() in wanted_titles:
            out.append(b)
    return out


def _query_single_collection(collection_name: str, query_embedding: list[float], top_k: int, book_meta: dict) -> list[dict]:
    collection = _get_collection(collection_name)
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0] if "distances" in result else []

    rows = []
    for idx, chunk_id in enumerate(ids):
        metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
        distance = distances[idx] if idx < len(distances) else None
        score = 1.0 - float(distance) if distance is not None else 0.0
        rows.append(
            {
                "chunk_id": metadata.get("chunk_id", chunk_id),
                "book_id": metadata.get("book_id", book_meta.get("book_id", "")),
                "book_title": metadata.get("book_title", book_meta.get("title", "")),
                "page": metadata.get("page", 0),
                "section_type": metadata.get("section_type", ""),
                "score": score,
                "text": docs[idx] if idx < len(docs) else "",
            }
        )
    return rows


def search_knowledge_multi(
    query: str,
    *,
    book_ids: list[str] | None = None,
    book_titles: list[str] | None = None,
    top_k_per_book: int = 8,
) -> list[dict]:
    if not query or not query.strip():
        return []

    books = _resolve_books(book_ids=book_ids, book_titles=book_titles)
    if not books:
        # Backward-compatible fallback for pre-registry setup.
        legacy = _get_collection(LEGACY_COLLECTION)
        emb = _get_embedder().encode(query.strip()).tolist()
        result = legacy.query(query_embeddings=[emb], n_results=top_k_per_book)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        rows = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            rows.append(
                {
                    "chunk_id": meta.get("chunk_id", ids[i] if i < len(ids) else ""),
                    "book_id": meta.get("book_id", ""),
                    "book_title": meta.get("book_title", ""),
                    "page": meta.get("page", 0),
                    "section_type": meta.get("section_type", ""),
                    "score": 0.0,
                    "text": doc,
                }
            )
        return rows

    emb = _get_embedder().encode(query.strip()).tolist()
    merged = []
    with ThreadPoolExecutor(max_workers=min(8, len(books))) as ex:
        futures = []
        for book in books:
            futures.append(
                ex.submit(
                    _query_single_collection,
                    book.get("collection_name", f"{COLLECTION_PREFIX}{book['book_id']}"),
                    emb,
                    top_k_per_book,
                    book,
                )
            )
        for fut in as_completed(futures):
            merged.extend(fut.result())

    by_chunk = {}
    for row in merged:
        cid = row.get("chunk_id", "")
        if not cid:
            continue
        prev = by_chunk.get(cid)
        if prev is None or row.get("score", 0.0) > prev.get("score", 0.0):
            by_chunk[cid] = row

    return sorted(by_chunk.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)


def search_knowledge(query: str, top_k: int = 3) -> list[str]:
    rows = search_knowledge_multi(query, top_k_per_book=max(3, top_k))
    return [r.get("text", "") for r in rows[:top_k] if r.get("text")]


def plan_multi_book_search(query: str, book_titles: list[str]) -> dict:
    intent = _classify_intent(query)
    points = _extract_query_points(query, max_points=6)
    expansions = _expand_terms(intent, points)
    core = " ".join(points[:4]).strip() or query.strip()
    syn = " ".join(expansions[:6]).strip()

    if intent == "definition":
        direct = f"{query} definition key terms core fact"
        support = f"{core} characteristics diagnostic context constraints"
    elif intent == "procedure":
        direct = f"{query} step by step procedure"
        support = f"{core} prerequisites steps constraints exceptions"
    elif intent == "compare":
        direct = f"{query} differences similarities compare"
        support = f"{core} criteria limitations indications"
    elif intent == "formula":
        direct = f"{query} formula equation variables"
        support = f"{core} derivation assumptions constraints"
    elif intent == "troubleshooting":
        direct = f"{query} root cause diagnosis fix"
        support = f"{core} failure patterns checks constraints"
    else:
        direct = f"{query} concise explanation key concepts"
        support = f"{core} mechanisms constraints important details"

    disambiguation = f"{core} synonyms related terminology {syn}".strip()
    if len(points) <= 2:
        # For short/ambiguous query, make disambiguation richer.
        disambiguation = f"{core} alternate terms synonyms related concepts definitions {syn}".strip()

    return {
        "intent": intent,
        "direct_query": " ".join(direct.split()),
        "support_query": " ".join(support.split()),
        "disambiguation_query": " ".join(disambiguation.split()),
        "expected_components": _EXPECTED_COMPONENTS.get(intent, _EXPECTED_COMPONENTS["explain"]),
    }


def select_multi_book_context(
    query: str,
    candidates_json: list[dict],
    *,
    k: int = 5,
    lambda_param: float = 0.7,
    sim_threshold: float = 0.5,
) -> dict:
    candidates = candidates_json or []
    if len(candidates) < k:
        raise ValueError(f"Need at least {k} candidates to select exactly {k} chunk_ids.")

    norm_scores = _normalize_scores(candidates)
    token_sets = [_tokenize(c.get("text", "") or "") for c in candidates]
    selected_idx = []
    remaining = list(range(len(candidates)))

    first = max(remaining, key=lambda i: norm_scores[i])
    selected_idx.append(first)
    remaining.remove(first)

    while len(selected_idx) < k:
        best = None
        best_val = -float("inf")
        selected_books = {candidates[i].get("book_title", "") for i in selected_idx}
        for idx in remaining:
            max_sim = 0.0
            for s in selected_idx:
                sim = _jaccard(token_sets[idx], token_sets[s])
                if sim > max_sim:
                    max_sim = sim

            book_bonus = 0.08 if candidates[idx].get("book_title", "") not in selected_books else 0.0
            val = lambda_param * norm_scores[idx] - (1 - lambda_param) * max_sim + book_bonus
            if val > best_val:
                best_val = val
                best = idx
        selected_idx.append(best)
        remaining.remove(best)

    selected_chunk_ids = [candidates[i].get("chunk_id", "") for i in selected_idx]
    books_used = []
    seen_books = set()
    for i in selected_idx:
        b = candidates[i].get("book_title", "")
        if b and b not in seen_books:
            seen_books.add(b)
            books_used.append(b)

    redundancy_notes = []
    for i in range(len(selected_idx)):
        for j in range(i + 1, len(selected_idx)):
            a = selected_idx[i]
            b = selected_idx[j]
            sim = _jaccard(token_sets[a], token_sets[b])
            if sim >= sim_threshold:
                overlap = sorted(token_sets[a] & token_sets[b])[:3]
                overlap_txt = ", ".join(overlap) if overlap else "shared terms"
                redundancy_notes.append(
                    f"{candidates[a].get('chunk_id','')} overlaps {candidates[b].get('chunk_id','')} on {overlap_txt}"
                )

    query_points = _extract_query_points(query, max_points=8)
    selected_text = " ".join((candidates[i].get("text", "") or "") for i in selected_idx).lower()
    missing_topics = [p for p in query_points if p not in selected_text]

    return {
        "selected_chunk_ids": selected_chunk_ids,
        "books_used": books_used,
        "redundancy_notes": redundancy_notes,
        "missing_topics": missing_topics,
    }


def build_final_structured_json_prompt(query: str, evidence_cards_json: list[dict]) -> dict:
    system_prompt = (
        "You are a grounded synthesis engine. "
        "Use ONLY provided Evidence Cards. "
        "No external knowledge. "
        "Return ONLY valid JSON. "
        "one_line <= 25 words. "
        "Each claim <= 18 words. "
        "Each claim must cite chunk_id, book_title, and page. "
        "If insufficient evidence, fill missing_info."
    )
    user_payload = {
        "query": query,
        "evidence_cards": evidence_cards_json,
        "required_schema": {
            "query": "...",
            "answer": {
                "one_line": "...",
                "key_points": [
                    {
                        "claim": "...",
                        "type": "definition|fact|step|constraint",
                        "evidence": [{"chunk_id": "...", "book_title": "...", "page": 12}],
                    }
                ],
            },
            "evidence_used": [{"chunk_id": "...", "book_title": "...", "page": 12}],
            "missing_info": [],
        },
        "generation_settings": {"temperature": 0.0, "max_new_tokens": 200},
    }
    return {"system": system_prompt, "user": user_payload}
