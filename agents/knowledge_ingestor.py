"""Multi-book knowledge ingestion (Option B: one collection per textbook).

Loads one or more reference PDFs (or .txt/.md), chunks and embeds them, and
stores each textbook in its own Chroma collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --- Paths / constants ---
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]

CHROMA_DIR = REPO_ROOT / "vector_db" / "book_chunks"
REGISTRY_PATH = REPO_ROOT / "vector_db" / "books_registry.json"
COLLECTION_PREFIX = "book__"

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR")
EMBED_OFFLINE = os.getenv("EMBED_OFFLINE", "0") == "1"

DEFAULT_BOOKS = [
    Path(r"c:\Users\DELL\Downloads\ophthalmology-reference1.pdf"),
    Path(r"c:\Users\DELL\Downloads\Ophthalmology-reference2.pdf"),
    REPO_ROOT / "data" / "books" / "ophthalmology-reference1.pdf",
    REPO_ROOT / "data" / "books" / "Ophthalmology-reference2.pdf",
    BACKEND_DIR / "data" / "books" / "ophthalmology-reference1.pdf",
    BACKEND_DIR / "data" / "books" / "Ophthalmology-reference2.pdf",
]


# --- Lazy singletons ---
_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collections: dict[str, object] = {}


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
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _collection_name(book_id: str) -> str:
    return f"{COLLECTION_PREFIX}{book_id}"


def _get_collection(book_id: str):
    if book_id in _collections:
        return _collections[book_id]
    collection = _get_client().get_or_create_collection(name=_collection_name(book_id))
    _collections[book_id] = collection
    return collection


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "book"


def _book_title_from_path(source_path: Path) -> str:
    name = source_path.stem
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name or source_path.name


def _ensure_book_entry(source_path: Path, *, title: str | None = None, book_id: str | None = None) -> dict:
    registry = load_registry()
    source_key = source_path.resolve().as_posix()

    for existing_id, meta in registry.items():
        if meta.get("pdf_path") == source_key:
            return {"book_id": existing_id, **meta}

    if book_id is None:
        base = _slug(title or _book_title_from_path(source_path))
        book_id = f"TBK_{base[:20]}"
        suffix = 2
        while book_id in registry:
            book_id = f"TBK_{base[:16]}_{suffix}"
            suffix += 1

    book_title = title or _book_title_from_path(source_path)
    entry = {
        "title": book_title,
        "pdf_path": source_key,
        "collection_name": _collection_name(book_id),
    }
    registry[book_id] = entry
    save_registry(registry)
    return {"book_id": book_id, **entry}


def _get_chroma_max_batch_size(collection) -> int:
    candidates = []
    candidates.append(lambda: collection._client.get_max_batch_size())  # type: ignore[attr-defined]
    candidates.append(lambda: collection._client._client.get_max_batch_size())  # type: ignore[attr-defined]
    candidates.append(lambda: collection._client._server.get_max_batch_size())  # type: ignore[attr-defined]
    candidates.append(lambda: collection._client._api.get_max_batch_size())  # type: ignore[attr-defined]

    for fn in candidates:
        try:
            value = fn()
            if isinstance(value, int) and value > 0:
                return value
        except Exception:
            continue
    return 5000


def _batched_upsert(collection, *, documents, metadatas, embeddings, ids) -> None:
    max_batch = _get_chroma_max_batch_size(collection)
    total = len(ids)
    if total == 0:
        return

    print(f"Chroma max batch size: {max_batch}. Upserting {total} chunks...")
    for start in range(0, total, max_batch):
        end = min(start + max_batch, total)
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
        )
        print(f"  Upserted {end}/{total}")


def load_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((idx, text))
    return pages


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = end - chunk_overlap
    return chunks


def _classify_section_type(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["defined as", "refers to", "is called", "definition"]):
        return "definition"
    if any(k in lower for k in ["step", "procedure", "algorithm", "protocol"]):
        return "procedure"
    if any(k in lower for k in ["summary", "key points", "overview"]):
        return "summary"
    if any(k in lower for k in ["contraindication", "limitation", "exception", "risk"]):
        return "constraint"
    return "body"


def _chunk_id(book_id: str, page: int, chunk_index: int) -> str:
    return f"{book_id}_p{page}_c{chunk_index}"


def _iter_chunks(source_path: Path, *, book_id: str, book_title: str) -> Iterable[Tuple[str, dict, str]]:
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        for page_num, page_text in load_pdf_pages(source_path):
            for i, chunk in enumerate(chunk_text(page_text), start=0):
                metadata = {
                    "book_id": book_id,
                    "book_title": book_title,
                    "pdf_path": source_path.resolve().as_posix(),
                    "source_file": source_path.name,
                    "page": page_num,
                    "section_title": "",
                    "section_type": _classify_section_type(chunk),
                    "chunk_index": i,
                    "chunk_id": _chunk_id(book_id, page_num, i),
                }
                yield chunk, metadata, metadata["chunk_id"]
        return

    if suffix in {".txt", ".md"}:
        text = load_text_file(source_path)
        for i, chunk in enumerate(chunk_text(text), start=0):
            metadata = {
                "book_id": book_id,
                "book_title": book_title,
                "pdf_path": source_path.resolve().as_posix(),
                "source_file": source_path.name,
                "page": 0,
                "section_title": "",
                "section_type": _classify_section_type(chunk),
                "chunk_index": i,
                "chunk_id": _chunk_id(book_id, 0, i),
            }
            yield chunk, metadata, metadata["chunk_id"]
        return

    raise ValueError(f"Unsupported file type: {source_path.suffix} (use .pdf/.txt/.md)")


def ingest_book(
    source_path: str | Path,
    *,
    title: str | None = None,
    book_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Reference file not found: {source_path}")

    entry = _ensure_book_entry(source_path, title=title, book_id=book_id)
    resolved_book_id = entry["book_id"]
    collection_name = entry["collection_name"]
    book_title = entry["title"]

    print(f"Reading: {source_path} -> {collection_name}")

    client = _get_client()
    if overwrite:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        _collections.pop(resolved_book_id, None)

    chunks: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    for chunk, metadata, chunk_id in _iter_chunks(source_path, book_id=resolved_book_id, book_title=book_title):
        chunks.append(chunk)
        metadatas.append(metadata)
        ids.append(chunk_id)

    if not chunks:
        print("No text extracted; nothing to ingest.")
        return {"book_id": resolved_book_id, "chunks_ingested": 0}

    print(f"Chunked into {len(chunks)} segments. Embedding...")
    embeddings = _get_embedder().encode(chunks, show_progress_bar=True)
    embeddings_list = [e.tolist() for e in embeddings]

    print("Storing in Chroma...")
    collection = _get_collection(resolved_book_id)
    _batched_upsert(
        collection,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings_list,
        ids=ids,
    )

    print(f"Done. Ingested {len(chunks)} chunks into: {collection_name}")
    return {"book_id": resolved_book_id, "collection_name": collection_name, "chunks_ingested": len(chunks)}


def discover_books(book_dir: str | Path) -> list[Path]:
    path = Path(book_dir)
    if not path.exists():
        return []
    return sorted([*path.glob("*.pdf"), *path.glob("*.txt"), *path.glob("*.md")])


def ingest_many_books(
    book_paths: list[str | Path],
    *,
    overwrite: bool = False,
) -> list[dict]:
    results = []
    for book_path in book_paths:
        results.append(ingest_book(book_path, overwrite=overwrite))
    return results


def _resolve_default_books() -> list[Path]:
    seen = set()
    books = []
    for candidate in DEFAULT_BOOKS:
        candidate = Path(candidate)
        if candidate.exists():
            key = candidate.resolve().as_posix()
            if key not in seen:
                seen.add(key)
                books.append(candidate)
    return books


if __name__ == "__main__":
    defaults = _resolve_default_books()
    if defaults:
        ingest_many_books(defaults)
    else:
        print("No default books found. Add PDFs then call ingest_book(...) or ingest_many_books(...).")
