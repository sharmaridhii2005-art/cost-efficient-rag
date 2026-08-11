import chromadb
from app.config import CHROMA_DIR


class ChromaStore:
    """Persistent ChromaDB vector store."""

    def __init__(self, collection_name="documents"):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, embeddings):
        """Add document chunks and embeddings to ChromaDB."""

        ids = [
            f"{document['source']}_{document['chunk_index']}"
            for document in documents
        ]

        texts = [
            document["text"]
            for document in documents
        ]

        metadatas = [
            {
                "source": document["source"],
                "chunk_index": document["chunk_index"]
            }
            for document in documents
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        """Search for the most similar chunks."""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def count(self):
        """Return the number of stored chunks."""

        return self.collection.count()
