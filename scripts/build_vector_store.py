#!/usr/bin/env python3
"""
Build a persistent Chroma vector store from PDFs in the repo.

Defaults:
  - scans for *.pdf under repo root
  - stores index under data/vector_store/
  - uses sentence-transformers embeddings (local)

Optional:
  - set EMBEDDING_PROVIDER=openai and OPENAI_API_KEY to use OpenAI embeddings
"""
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import gc


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_pdfs(root: Path, exclude: Sequence[str]) -> Iterable[Path]:
    for path in root.rglob("*.pdf"):
        rel = _normalize_path(path, root)
        if any(rel.startswith(prefix) for prefix in exclude):
            continue
        yield path


def _iter_pdf_pages(path: Path) -> Iterable[str]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error
        raise SystemExit(
            "Missing dependency: PyMuPDF. Install with `pip install pymupdf`."
        ) from exc

    doc = fitz.open(path)
    try:
        for page in doc:
            yield page.get_text("text")
    finally:
        doc.close()


def _batched_upsert(
    collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    batch_size: int,
) -> int:
    if batch_size <= 0:
        batch_size = len(ids) or 1
    total = 0
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        total += end - start
    return total


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start == end:
            break
    return chunks


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingConfig:
    provider: str
    model: str


class EmbeddingFunction:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        if config.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency error
                raise SystemExit(
                    "Missing dependency: sentence-transformers. Install with "
                    "`pip install sentence-transformers`."
                ) from exc
            self._model = SentenceTransformer(config.model)
        elif config.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency error
                raise SystemExit(
                    "Missing dependency: openai. Install with `pip install openai`."
                ) from exc
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is required for OpenAI embeddings.")
            self._client = OpenAI(api_key=api_key)
        else:
            raise SystemExit(f"Unknown embedding provider: {config.provider}")

    def name(self) -> str:
        return f"{self.config.provider}:{self.config.model}"

    def get_config(self) -> dict:
        return {"provider": self.config.provider, "model": self.config.model}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.config.provider == "sentence-transformers":
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        if self.config.provider == "openai":
            response = self._client.embeddings.create(model=self.config.model, input=texts)
            return [item.embedding for item in response.data]
        raise SystemExit(f"Unknown embedding provider: {self.config.provider}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)


def _get_embedding_config() -> EmbeddingConfig:
    provider = os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers").strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
    else:
        provider = "sentence-transformers"
        model = os.environ.get("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    return EmbeddingConfig(provider=provider, model=model)


def _ensure_chroma() -> None:
    try:
        import chromadb  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency error
        raise SystemExit(
            "Missing dependency: chromadb. Install with `pip install chromadb`."
        ) from exc


def build_index(args: argparse.Namespace) -> None:
    _ensure_chroma()
    from chromadb import PersistentClient  # type: ignore

    root = Path(args.root).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    embed_config = _get_embedding_config()
    embedding_fn = EmbeddingFunction(embed_config)

    client = PersistentClient(path=index_dir.as_posix())
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn,
        metadata={"embedding_provider": embed_config.provider, "embedding_model": embed_config.model},
    )

    exclude = list(args.exclude)
    if "data/vector_store" not in exclude:
        exclude.append("data/vector_store")

    pdf_paths = list(_iter_pdfs(root, exclude))
    if not pdf_paths:
        print("No PDFs found.")
        return

    total_chunks = 0
    batch_size = max(1, args.pdf_batch_size)
    for batch_start in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[batch_start : batch_start + batch_size]
        if args.max_files is not None:
            remaining = args.max_files - batch_start
            if remaining <= 0:
                break
            batch = batch[:remaining]
        for pdf_path in batch:
            rel_path = _normalize_path(pdf_path, root)
            for page_idx, page_text in enumerate(_iter_pdf_pages(pdf_path)):
                chunks = _chunk_text(page_text, args.chunk_size, args.chunk_overlap)
                if not chunks:
                    continue
                ids: List[str] = []
                documents: List[str] = []
                metadatas: List[dict] = []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{rel_path}:{page_idx}:{chunk_idx}:{_sha1(chunk)[:12]}"
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "source": rel_path,
                            "page": page_idx,
                            "chunk": chunk_idx,
                            "sha1": _sha1(chunk),
                        }
                    )
                total_chunks += _batched_upsert(
                    collection,
                    ids,
                    documents,
                    metadatas,
                    batch_size=args.batch_size,
                )
        gc.collect()

    print(
        "Indexed PDFs.",
        f"files={len(pdf_paths)}",
        f"chunks={total_chunks}",
        f"index_dir={index_dir}",
        f"collection={args.collection}",
        f"embedding={embed_config.provider}:{embed_config.model}",
    )


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Build Chroma vector store from PDFs.")
    parser.add_argument("--root", default=root.as_posix(), help="Root directory to scan.")
    parser.add_argument(
        "--index-dir",
        default=(root / "data" / "vector_store").as_posix(),
        help="Directory to store the Chroma index.",
    )
    parser.add_argument("--collection", default="pdfs", help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters.")
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap in characters."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to upsert per batch (lower to reduce RAM).",
    )
    parser.add_argument(
        "--pdf-batch-size",
        type=int,
        default=5,
        help="Number of PDFs to process per batch (lower to reduce RAM).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of PDFs to index.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[".git", "data/vector_store"],
        help="Paths (relative to root) to exclude from scanning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
