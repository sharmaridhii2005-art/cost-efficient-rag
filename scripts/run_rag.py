
from app.retrieval.retriever import Retriever
from app.generation.local_generator import LocalGenerator


def main():
    print("=" * 70)
    print("COST-EFFICIENT RAG - END-TO-END PIPELINE")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nQuestion:")
    print(question)

    # ---------------------------------------------------------
    # 1. RETRIEVAL
    # ---------------------------------------------------------
    retriever = Retriever()

    retrieved = retriever.retrieve(question)

    documents = retrieved.get("documents", [[]])[0]
    metadatas = retrieved.get("metadatas", [[]])[0]
    distances = retrieved.get("distances", [[]])[0]

    contexts = []

    for document, metadata in zip(documents, metadatas):
        contexts.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
            }
        )

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    if contexts:
        for i, (context, distance) in enumerate(
            zip(contexts, distances)
        ):
            print(f"\nResult {i + 1}")
            print(f"Source: {context['source']}")
            print(f"Chunk: {context['chunk_index']}")
            print(f"Distance: {distance}")
            print(f"Text: {context['text']}")
    else:
        print("\nNo relevant context found.")

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(
            "I don't have enough information in the provided documents."
        )

        return

    # ---------------------------------------------------------
    # 2. LOCAL GENERATION
    # ---------------------------------------------------------
    generator = LocalGenerator()

    result = generator.generate(
        question,
        contexts
    )

    # ---------------------------------------------------------
    # 3. FINAL ANSWER
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result["answer"])

    # ---------------------------------------------------------
    # 4. METRICS
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("RAG METRICS")
    print("=" * 70)

    print(f"Model: {result['model']}")
    print(f"TOP_K: {retriever.top_k}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Cost: ${result['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
