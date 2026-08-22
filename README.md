# Enterprise Architecture RAG Assistant (Python + LangChain + LangGraph)

This is the Python implementation of the same use case built in the n8n prototype:

- 8 enterprise knowledge documents in `data/corpus`.
- One shared Gemini embedding model for indexing and retrieval.
- One Gemini model for answer generation.
- Pinecone vector store for retrieval.
- Deterministic CSV/JSON evaluation output for a 15-question benchmark suite.

## What this project does

- `index` ingests local documents, chunks them, computes embeddings, and upserts to Pinecone.
- `ask` answers questions using retrieval + generation.
- `evaluate` runs the 15-question suite and writes per-question and aggregate scores.

## Setup

1. Create a virtual environment and install:

```
pip install -e .
```

2. Copy `.env.example` to `.env` and fill keys.
3. Ensure your 8 corpus files exist under `data/corpus`.
4. Ingest documents:

```
enterprise-rag index --corpus data\corpus --index-name corp-enterprise-arch-python
```

5. Ask questions:

```
enterprise-rag ask --question "Which authentication methods are allowed for internet-facing APIs?"
```

6. Evaluate:

```
enterprise-rag evaluate --questions evaluation/questions.json --ground-truth evaluation/ground_truth.json
```

## Outputs

Evaluation outputs are written to `evaluation/results/`:

- `<timestamp>-run.json`: detailed per-question report.
- `evaluation-template.csv`: importable template for external scoring dashboards.

## Project files

```
src/enterprise_rag/
  __init__.py
  config.py
  document_loader.py
  chunking.py
  vector_store.py
  index_pipeline.py
  retrieval.py
  answer.py
  evaluator.py
  cli.py
```

scripts/

- `run_ingest.py`
- `run_evaluate.py`
- `run_chat.py`
*** End Patch
