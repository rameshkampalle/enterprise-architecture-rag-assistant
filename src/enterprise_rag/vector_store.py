from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


def get_embeddings(model_name: str, api_key: str):
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)


def get_store(
    index_name: str,
    namespace: str,
    dimension: int,
    embedding_model: str,
    embedding_api_key: str | None = None,
    pinecone_api_key: str | None = None,
    api_key: str | None = None,
    cloud: str | None = None,
    region: str | None = None,
) -> PineconeVectorStore:
    # Keep compatibility with older call sites that still pass `api_key` as only key.
    if pinecone_api_key is None:
        pinecone_api_key = api_key
    if embedding_api_key is None:
        embedding_api_key = api_key
    if pinecone_api_key is None or embedding_api_key is None:
        raise ValueError("Both embedding and pinecone keys must be provided to build the vector store.")

    embeddings = get_embeddings(embedding_model, embedding_api_key)
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=pinecone_api_key,
    )


def ensure_index_exists(index_name: str, dimension: int, api_key: str, cloud: str, region: str) -> None:
    pc = Pinecone(api_key=api_key)
    if index_name in [idx["name"] for idx in pc.list_indexes()]:
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec={"serverless": {"cloud": cloud, "region": region}},
    )


def upsert_chunks(store: PineconeVectorStore, chunks: Sequence[Document], namespace: str, batch_size: int = 10) -> int:
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = list(chunks[i : i + batch_size])
        store.add_documents(batch, namespace=namespace)
        total += len(batch)
    return total


def retrieve_chunks(store: PineconeVectorStore, query: str, top_k: int, namespace: str, filter_expr: dict | None = None) -> List[Document]:
    # Keep source docs and similarity score for downstream explanation.
    return store.similarity_search_with_score(query, k=top_k, filter=filter_expr or None, namespace=namespace)
