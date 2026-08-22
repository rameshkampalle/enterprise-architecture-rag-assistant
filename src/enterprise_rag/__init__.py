"""Core package for enterprise architecture RAG assistant."""

__all__ = [
    "build_rag_chain",
    "run_evaluation",
]

from .config import Settings
from .answer import ask_question as build_rag_chain
from .evaluator import evaluate_all as run_evaluation
