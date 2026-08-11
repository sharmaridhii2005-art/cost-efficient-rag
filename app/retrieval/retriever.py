from app.retrieval.embedder import create_embeddings
from app.retrieval.chroma_store import ChromaStore
from app.config import TOP_K, RETRIEVAL_DISTANCE_THRESHOLD


class Retriever:
    """Retrieve relevant document chunks from ChromaDB."""

    def __init__(
        self,
        top_k=TOP_K,
        distance_threshold=RETRIEVAL_DISTANCE_THRESHOLD
    ):
        self.store = ChromaStore()
        self.top_k = top_k
        self.distance_threshold = distance_threshold

    def retrieve(self, query):
        """Retrieve relevant chunks and reject weak matches."""

        query_embedding = create_embeddings([query])[0]

        results = self.store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        distances = results.get("distances", [[]])[0]

        if not distances:
            return results

        filtered_indices = [
            i
            for i, distance in enumerate(distances)
            if distance <= self.distance_threshold
        ]

        results["documents"][0] = [
            results["documents"][0][i]
            for i in filtered_indices
        ]

        results["metadatas"][0] = [
            results["metadatas"][0][i]
            for i in filtered_indices
        ]

        results["distances"][0] = [
            distances[i]
            for i in filtered_indices
        ]

        return results


if __name__ == "__main__":
    retriever = Retriever()

    query = "What is this document about?"

    results = retriever.retrieve(query)

    print("\nRetrieved documents:")
    print(results)