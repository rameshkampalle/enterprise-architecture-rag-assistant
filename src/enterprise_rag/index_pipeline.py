from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from rich.console import Console

from .chunking import split_documents
from .config import Settings
from .document_loader import load_corpus
from .vector_store import ensure_index_exists, get_store, upsert_chunks

console = Console()


def run_ingest(settings: Settings) -> dict:
    console.log(f"[blue]Loading corpus from:[/blue] {settings.corpus_path}")
    documents: List[Document] = load_corpus(settings.corpus_path)
    if not documents:
        raise RuntimeError(f"No supported docs found in {settings.corpus_path}")
    console.log(f"[blue]Loaded documents:[/blue] {len(documents)}")

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    console.log(f"[blue]Chunks created:[/blue] {len(chunks)}")

    ensure_index_exists(
        index_name=settings.pinecone_index_name,
        dimension=settings.pinecone_index_dimension,
        api_key=settings.pinecone_api_key.get_secret_value(),
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    store = get_store(
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        dimension=settings.pinecone_index_dimension,
        embedding_model=settings.embedding_model,
        embedding_api_key=settings.gemini_api_key.get_secret_value(),
        pinecone_api_key=settings.pinecone_api_key.get_secret_value(),
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    inserted = upsert_chunks(
        store=store,
        chunks=chunks,
        namespace=settings.pinecone_namespace,
        batch_size=max(1, settings.max_ingest_batch_size),
    )
    return {"documents": len(documents), "chunks": len(chunks), "inserted": inserted, "index_name": settings.pinecone_index_name}
