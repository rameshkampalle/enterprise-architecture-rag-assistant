from enterprise_rag.config import Settings
from enterprise_rag.evaluator import evaluate_all
from pathlib import Path


def main():
    settings = Settings()
    output = evaluate_all(
        settings=settings,
        questions_path=Path("evaluation/questions.json"),
        ground_truth_path=Path("evaluation/ground_truth.json"),
        results_out_dir=Path("evaluation/results"),
    )
    print(output["summary"])


if __name__ == "__main__":
    main()
