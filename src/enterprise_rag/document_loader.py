from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PDFMinerLoader, TextLoader


def _load_pdf(path: Path) -> List[Document]:
    return PDFMinerLoader(str(path)).load()


def _load_docx(path: Path) -> List[Document]:
    # Keep docx load lightweight and deterministic by using local docx parser.
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path)).load()


def load_document(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = _load_pdf(path)
    elif ext in [".txt", ".md"]:
        docs = TextLoader(str(path), encoding="utf-8").load()
    elif ext == ".docx":
        docs = _load_docx(path)
    else:
        # Keep behavior strict so unsupported formats are explicit.
        raise ValueError(f"Unsupported file extension: {ext}")

    if not docs:
        return []

    for doc in docs:
        doc.metadata.update(
            {
                "source": path.name,
                "source_file": path.name,
                "source_url": "",
                "document_id": path.stem,
                "path": str(path),
                "file_type": path.suffix.lower().lstrip("."),
            }
        )
    return docs


def load_corpus(corpus_dir: Path) -> List[Document]:
    files = sorted([p for p in corpus_dir.glob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}])
    all_docs: List[Document] = []
    for file in files:
        try:
            all_docs.extend(load_document(file))
        except Exception:
            # Fail fast in dev and surface broken files clearly.
            raise
    return all_docs
