from __future__ import annotations

from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .config import Settings
from .retrieval import retrieve_context

SYSTEM_PROMPT = """You are an enterprise architecture assistant.
Use ONLY the provided "Context" below to answer.
- Do not use external or prior knowledge.
- If context is insufficient, say: "I couldn't find this in the provided enterprise architecture documents."
- Do not guess.

Return output in exactly:
Answer: <concise answer>
Sources: <comma-separated source filenames from context>"""


_AFC_WARNING = (
    "Direct use of automatic function calling (AFC) in Models.generate_content is not recommended."
)


def _to_text_block(response: Any) -> str:
    """Handle different response shapes from the LLM SDK safely."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _clean_context(hits: List[tuple[Document, float]]) -> List[str]:
    lines: List[str] = []
    for idx, (doc, score) in enumerate(hits, start=1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")
        lines.append(f"[{idx}] source={source} | score={score:.4f}\n{snippet}")
    return lines


def _build_prompt(question: str, docs: List[tuple[Document, float]]) -> str:
    if not docs:
        return (
            f"Question: {question}\n\nContext: No relevant chunks were retrieved.\n\n"
            "Answer:\nI couldn't find this in the provided enterprise architecture documents.\n"
            "Sources: []"
        )
    context = "\n\n".join(_clean_context(docs))
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"


def ask_question(question: str, settings: Settings) -> Dict[str, Any]:
    hits = retrieve_context(settings, question)
    context_text = _build_prompt(question, hits)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key.get_secret_value(),
    )
    chain = prompt | llm
    response = chain.invoke({"input": context_text})
    response_text = _to_text_block(response)
    response_text = response_text.replace(_AFC_WARNING, "").strip()
    response_text = response_text.replace("Answer:", "").strip()
    sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in hits})
    if "Sources:" not in response_text:
        source_text = ", ".join(sources) if sources else "[]"
        if not response_text:
            response_text = "I couldn't find this in the provided enterprise architecture documents."
        response_text = f"{response_text}\nSources: {source_text}"
    scores = [float(score) for _, score in hits]
    return {
        "question": question,
        "answer": response_text,
        "retrieved_sources": sources,
        "retrieval_count": len(hits),
        "retrieval_top_score": max(scores) if scores else 0.0,
        "raw_hits": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
                "content_preview": doc.page_content[:180],
            }
            for doc, score in hits
        ],
    }
