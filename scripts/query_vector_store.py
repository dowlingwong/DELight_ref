#!/usr/bin/env python3
"""
Query a persistent Chroma vector store and print top matching chunks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


class EmbeddingFunction:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        if provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency error
                raise SystemExit(
                    "Missing dependency: sentence-transformers. Install with "
                    "`pip install sentence-transformers`."
                ) from exc
            self._model = SentenceTransformer(model)
        elif provider == "openai":
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
            raise SystemExit(f"Unknown embedding provider: {provider}")

    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def get_config(self) -> dict:
        return {"provider": self.provider, "model": self.model}

    def __call__(self, input: List[str]) -> List[List[float]]:
        if self.provider == "sentence-transformers":
            return self._model.encode(input, normalize_embeddings=True).tolist()
        if self.provider == "openai":
            response = self._client.embeddings.create(model=self.model, input=input)
            return [item.embedding for item in response.data]
        raise SystemExit(f"Unknown embedding provider: {self.provider}")


def _embedding_config() -> tuple[str, str]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers").strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
    else:
        provider = "sentence-transformers"
        model = os.environ.get("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    return provider, model


def _ensure_chroma() -> None:
    try:
        import chromadb  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency error
        raise SystemExit(
            "Missing dependency: chromadb. Install with `pip install chromadb`."
        ) from exc


def query(args: argparse.Namespace) -> None:
    _ensure_chroma()
    from chromadb import PersistentClient  # type: ignore

    index_dir = Path(args.index_dir).resolve()
    provider, model = _embedding_config()
    embedding_fn = EmbeddingFunction(provider, model)

    client = PersistentClient(path=index_dir.as_posix())
    collection = client.get_collection(name=args.collection, embedding_function=embedding_fn)

    results = collection.query(
        query_texts=[args.query],
        n_results=args.k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        print("No results.")
        return

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk = meta.get("chunk", "?")
        print(f"[{idx}] {source} (page {page}, chunk {chunk}) dist={dist:.4f}")
        print(doc.strip())
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Query Chroma vector store.")
    parser.add_argument("--index-dir", default=(root / "data" / "vector_store").as_posix())
    parser.add_argument("--collection", default="pdfs")
    parser.add_argument("-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("query", help="Search query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query(args)


if __name__ == "__main__":
    main()
