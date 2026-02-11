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
    "can",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
    "does",
    "did",
    "done",
    "should",
    "would",
    "could",
    "using",
    "use",
}

_INTENT_PATTERNS = {
    "definition": [
        "what is",
        "what are",
        "define",
        "definition",
        "meaning",
        "refers to",
    ],
    "procedure": [
        "how to",
        "steps",
        "procedure",
        "method",
        "workflow",
        "implement",
        "perform",
    ],
    "compare": [
        "compare",
        "difference",
        "versus",
        "vs",
        "contrast",
    ],
    "formula": [
        "formula",
        "equation",
        "calculate",
        "compute",
        "derive",
    ],
    "troubleshooting": [
        "error",
        "issue",
        "problem",
        "fails",
        "failure",
        "not working",
        "fix",
        "troubleshoot",
    ],
}

_PREDICATE_TERMS = {
    "define",
    "explain",
    "compare",
    "calculate",
    "compute",
    "derive",
    "apply",
    "diagnose",
    "treat",
    "manage",
    "implement",
    "perform",
    "prevent",
    "classify",
    "identify",
    "interpret",
    "evaluate",
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

_INTENT_DEFAULT_COMPONENTS = {
    "definition": ["definition", "key characteristics", "conditions", "exceptions"],
    "procedure": ["goal", "steps", "conditions", "exceptions"],
    "explain": ["definition", "mechanism", "conditions", "exceptions"],
    "compare": ["comparison criteria", "similarities", "differences", "when to use each"],
    "formula": ["formula", "variable definitions", "application conditions", "worked example"],
    "troubleshooting": ["symptom", "root cause", "diagnostic steps", "fixes"],
}

_INTENT_STRUCTURE_PREFS = {
    "definition": ["definition section", "summary section", "key terms"],
    "procedure": ["procedural steps", "summary section", "key terms"],
    "explain": ["summary section", "definition section", "key terms"],
    "compare": ["summary section", "key terms", "definition section"],
    "formula": ["definition section", "key terms", "procedural steps"],
    "troubleshooting": ["procedural steps", "summary section", "key terms"],
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


def _classify_intent(query: str) -> str:
    query_lower = (query or "").lower()
    scores = {intent: 0 for intent in _INTENT_PATTERNS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                scores[intent] += 2

    if query_lower.startswith("how"):
        scores["procedure"] += 1
    elif query_lower.startswith("what"):
        scores["definition"] += 1
    elif query_lower.startswith("why"):
        scores["troubleshooting"] += 1

    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return "explain"
    return best_intent


def _extract_entities_and_predicates(query: str) -> tuple[list[str], list[str]]:
    ordered_tokens = re.split(r"\W+", (query or "").lower())
    entities = []
    predicates = []
    seen_entities = set()
    seen_predicates = set()

    for token in ordered_tokens:
        if not token or len(token) < 3:
            continue
        if token in _PREDICATE_TERMS:
            if token not in seen_predicates:
                predicates.append(token)
                seen_predicates.add(token)
            continue
        if token in _STOPWORDS:
            continue
        if token not in seen_entities:
            entities.append(token)
            seen_entities.add(token)

    return entities[:6], predicates[:5]


def _expand_keywords(intent: str, entities: list[str], predicates: list[str], domain: str) -> list[str]:
    expanded = []
    seen = set()

    def add_term(term: str) -> None:
        if term and term not in seen:
            expanded.append(term)
            seen.add(term)

    add_term(intent)
    for term in entities + predicates:
        add_term(term)
        for synonym in _TERM_EXPANSIONS.get(term, []):
            add_term(synonym)

    for synonym in _TERM_EXPANSIONS.get(intent, []):
        add_term(synonym)

    domain_lower = (domain or "").lower()
    if "ophthalm" in domain_lower or "eye" in domain_lower:
        for extra in ["ocular", "visual acuity", "optic nerve", "clinical guideline"]:
            add_term(extra)

    return expanded[:15]


def _build_subqueries(
    intent: str,
    original_query: str,
    entities: list[str],
    predicates: list[str],
    keyword_expansion: list[str],
) -> dict:
    entity_phrase = " ".join(entities[:4]).strip()
    predicate_phrase = " ".join(predicates[:2]).strip()
    expansion_phrase = " ".join(keyword_expansion[:5]).strip()

    if intent == "definition":
        direct = f"{original_query} precise definition key terms"
        support = f"{entity_phrase} characteristics diagnostic criteria context"
        disambig = f"{entity_phrase} synonyms alternate terminology {expansion_phrase}".strip()
    elif intent == "procedure":
        direct = f"{original_query} step by step procedure"
        support = f"{entity_phrase} prerequisites sequence constraints safety considerations"
        disambig = f"{entity_phrase} workflow protocol method {expansion_phrase}".strip()
    elif intent == "compare":
        direct = f"{original_query} differences similarities comparison table"
        support = f"{entity_phrase} criteria indications limitations"
        disambig = f"{entity_phrase} versus contrast distinguish {expansion_phrase}".strip()
    elif intent == "formula":
        direct = f"{original_query} formula equation variable definitions"
        support = f"{entity_phrase} derivation assumptions units examples"
        disambig = f"{entity_phrase} compute calculate derive {expansion_phrase}".strip()
    elif intent == "troubleshooting":
        direct = f"{original_query} root cause diagnosis fix"
        support = f"{entity_phrase} failure conditions warning signs corrective actions"
        disambig = f"{entity_phrase} common errors mitigation {expansion_phrase}".strip()
    else:
        direct = f"{original_query} explanation mechanism key concepts"
        support = f"{entity_phrase} causes effects conditions exceptions"
        disambig = f"{entity_phrase} related terms {predicate_phrase} {expansion_phrase}".strip()

    return {
        "direct_answer_query": " ".join(direct.split()),
        "support_query": " ".join(support.split()),
        "disambiguation_query": " ".join(disambig.split()),
    }


def optimize_semantic_search_query(user_query: str, domain: str = "") -> dict:
    """Create optimized semantic sub-queries for high-precision retrieval."""
    query = (user_query or "").strip()
    intent = _classify_intent(query)
    entities, predicates = _extract_entities_and_predicates(query)
    keyword_expansion = _expand_keywords(intent, entities, predicates, domain)
    subqueries = _build_subqueries(intent, query, entities, predicates, keyword_expansion)

    return {
        "intent_classification": intent,
        "core_entities": entities,
        "core_predicates": predicates,
        "semantic_subqueries": subqueries,
        "keyword_expansion": keyword_expansion,
        "structure_preferences": _INTENT_STRUCTURE_PREFS.get(intent, _INTENT_STRUCTURE_PREFS["explain"]),
        "expected_answer_components": _INTENT_DEFAULT_COMPONENTS.get(
            intent, _INTENT_DEFAULT_COMPONENTS["explain"]
        ),
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
