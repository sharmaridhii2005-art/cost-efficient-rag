from app.retrieval.retriever import Retriever


def main():
    retriever = Retriever()

    queries = [
        "What is RAG?",
        "What are the main stages of a RAG system?",
        "What does a RAG system retrieve?"
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("=" * 60)

        results = retriever.retrieve(query)

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, document in enumerate(documents):
            print(f"\nResult {i + 1}")
            print("Source:", metadatas[i]["source"])
            print("Chunk:", metadatas[i]["chunk_index"])
            print("Distance:", distances[i])
            print("Text:")
            print(document)


if __name__ == "__main__":
    main()
