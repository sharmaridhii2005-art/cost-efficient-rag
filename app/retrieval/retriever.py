from app.retrieval.embedder import create_embeddings
from app.retrieval.chroma_store import ChromaStore
from app.config import TOP_K


class Retriever:
    """Retrieve the most relevant document chunks from ChromaDB."""

    def __init__(self, top_k=TOP_K):
        self.store = ChromaStore()
        self.top_k = top_k

    def retrieve(self, query):
        """Retrieve top-k chunks for a query."""

        query_embedding = create_embeddings([query])[0]

        return self.store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )


if __name__ == "__main__":
    retriever = Retriever()

    query = "What is this document about?"

    results = retriever.retrieve(query)

    print("\nRetrieved documents:")
    print(results)
