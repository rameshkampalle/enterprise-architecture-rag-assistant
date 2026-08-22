from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from .config import Settings
from .vector_store import get_store, retrieve_chunks


def get_retriever(settings: Settings):
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
    return store


def retrieve_context(settings: Settings, question: str) -> List[tuple[Document, float]]:
    retriever = get_retriever(settings)
    results = retrieve_chunks(
        retriever,
        query=question,
        top_k=settings.top_k_retrieval,
        namespace=settings.pinecone_namespace,
        filter_expr=settings.pinecone_metadata_filter if settings.pinecone_metadata_filter else None,
    )
    return results
