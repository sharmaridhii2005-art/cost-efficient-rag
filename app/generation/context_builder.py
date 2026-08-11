def build_context(results):
    """
    Convert ChromaDB retrieval results into a clean context string.
    """

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []

    for i, document in enumerate(documents):
        source = metadatas[i].get("source", "unknown")
        chunk_index = metadatas[i].get("chunk_index", i)

        context_parts.append(
            f"[Source: {source}, Chunk: {chunk_index}]\n"
            f"{document}"
        )

    return "\n\n".join(context_parts)
