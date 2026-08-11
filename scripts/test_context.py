from app.retrieval.retriever import Retriever
from app.generation.context_builder import build_context


def main():
    query = "What are the main stages of a RAG system?"

    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    retriever = Retriever(top_k=3)

    results = retriever.retrieve(query)

    context = build_context(results)

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)
    print(context)


if __name__ == "__main__":
    main()
