def build_prompt(question, contexts):
    """
    Build a concise, grounded RAG prompt optimized for local generation.
    """

    context_text = "\n\n".join(
        f"[Source: {item['source']}, Chunk: {item['chunk_index']}]\n"
        f"{item['text']}"
        for item in contexts
    )

    prompt = f"""Answer the question using only the context below.

Rules:
- Give a direct answer to the question.
- Use information explicitly stated in the context.
- Ignore information that does not answer the question.
- Do not add outside knowledge.
- Keep the answer concise.
- If the answer is not in the context, say:
I don't have enough information in the provided documents.

Context:
{context_text}

Question:
{question}

Answer:"""

    return prompt