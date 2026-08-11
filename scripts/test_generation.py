from app.retrieval.retriever import Retriever
from app.generation.generator import Generator


def main():
    question = "What are the main stages of a RAG system?"

    print("=" * 70)
    print("COST-EFFICIENT RAG")
    print("=" * 70)

    print(f"\nQuestion:\n{question}")

    # Retrieval
    retriever = Retriever(top_k=3)
    retrieved = retriever.retrieve(question)

    contexts = []

    documents = retrieved.get("documents", [[]])[0]
    metadatas = retrieved.get("metadatas", [[]])[0]

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

    # Generation
    generator = Generator()

    result = generator.generate(
        question,
        contexts
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("TOKEN USAGE")
    print("=" * 70)

    print(f"Model: {result['model']}")
    print(f"Prompt tokens: {result['usage']['prompt_tokens']}")
    print(f"Completion tokens: {result['usage']['completion_tokens']}")
    print(f"Total tokens: {result['usage']['total_tokens']}")


if __name__ == "__main__":
    main()
