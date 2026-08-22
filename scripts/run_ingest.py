from pathlib import Path
from enterprise_rag.index_pipeline import run_ingest
from enterprise_rag.config import Settings


def main():
    settings = Settings()
    settings.corpus_path = Path("data/corpus")
    run_summary = run_ingest(settings)
    print(run_summary)


if __name__ == "__main__":
    main()
