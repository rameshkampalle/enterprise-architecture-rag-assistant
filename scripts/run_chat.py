from enterprise_rag.answer import ask_question
from enterprise_rag.config import Settings


def main():
    settings = Settings()
    print("Type a question and press Enter. Empty input to exit.")
    while True:
        query = input("question> ").strip()
        if not query:
            break
        result = ask_question(query, settings)
        print("\nAnswer:\n", result["answer"])
        print("Sources:", ", ".join(result["retrieved_sources"]) or "<none>")
        print("Top score:", result["retrieval_top_score"])


if __name__ == "__main__":
    main()
