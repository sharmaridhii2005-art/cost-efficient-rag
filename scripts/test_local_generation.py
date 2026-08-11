from app.retrieval.retriever import Retriever
from app.generation.local_generator import LocalGenerator


def main():
    question = "What are the main stages of a RAG system?"

    print("=" * 70)
    print("COST-EFFICIENT RAG - LOCAL GENERATION")
    print("=" * 70)

    print(f"\nQuestion:\n{question}")

    retriever = Retriever(top_k=3)
    retrieved = retriever.retrieve(question)

    documents = retrieved.get("documents", [[]])[0]
    metadatas = retrieved.get("metadatas", [[]])[0]

    contexts = []

    for document, metadata in zip(documents, metadatas):
        contexts.append({
            "text": document,
            "source": metadata.get("source"),
            "chunk_index": metadata.get("chunk_index")
        })

    print("\nRetrieved contexts:")

    for context in contexts:
        print(
            f"- {context['source']} "
            f"(chunk {context['chunk_index']})"
        )

    generator = LocalGenerator()

    result = generator.generate(
        question,
        contexts
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("LOCAL GENERATION METRICS")
    print("=" * 70)

    print(f"Model: {result['model']}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Cost: ${result['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
