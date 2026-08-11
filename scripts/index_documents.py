from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.loader import load_documents
from app.retrieval.chunker import chunk_text
from app.retrieval.embedder import create_embeddings
from app.retrieval.chroma_store import ChromaStore


def main():
    print("Loading documents...")

    documents = load_documents(DOCUMENTS_DIR)

    print(f"Documents loaded: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    chunks = []

    for document in documents:
        document_chunks = chunk_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for index, chunk in enumerate(document_chunks):
            chunks.append({
                "source": document["source"],
                "chunk_index": index,
                "text": chunk
            })

    print(f"Chunks created: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = create_embeddings(texts)

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    print("Storing in ChromaDB...")

    store = ChromaStore()
    store.add_documents(chunks, embeddings)

    print(f"ChromaDB chunks: {store.count()}")
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()
