from __future__ import annotations

from pathlib import Path
import typer

from .config import Settings
from .index_pipeline import run_ingest
from .answer import ask_question
from .evaluator import evaluate_all

app = typer.Typer(help="Enterprise Architecture RAG Assistant")


@app.command()
def index(
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Path containing the corpus files"),
    index_name: str = typer.Option(None, "--index-name", help="Target Pinecone index"),
) -> None:
    settings = Settings()
    if index_name:
        settings.pinecone_index_name = index_name
    settings.corpus_path = corpus
    result = run_ingest(settings)
    typer.echo(f"Ingest summary: {result}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="User question"),
) -> None:
    settings = Settings()
    result = ask_question(question, settings)
    typer.echo(result["answer"])
    typer.echo(f"Sources: {', '.join(result['retrieved_sources'])}")
    typer.echo(f"Top score: {result['retrieval_top_score']:.4f}")


@app.command()
def evaluate(
    questions: Path = typer.Option(..., "--questions", "--q"),
    ground_truth: Path = typer.Option(..., "--ground-truth", "--g"),
    results_dir: Path = typer.Option("evaluation/results", "--results-dir", "-r"),
) -> None:
    settings = Settings()
    payload = evaluate_all(settings, questions, ground_truth, results_dir)
    typer.echo(payload["summary"])


if __name__ == "__main__":
    app()
