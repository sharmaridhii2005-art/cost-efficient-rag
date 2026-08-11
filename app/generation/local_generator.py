import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LocalGenerator:
    """Generate grounded RAG answers using a local CPU model."""

    def __init__(self, model_name="google/flan-t5-small"):
        print(f"Loading local model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model_name = model_name

    def generate(self, question, contexts):
        """Generate a grounded answer using only retrieved contexts."""

        context_text = "\n\n".join(
            f"[Source: {item['source']}, Chunk: {item['chunk_index']}]\n"
            f"{item['text']}"
            for item in contexts
        )

        prompt = f"""You are a question-answering assistant.

Answer the question using ONLY the information in the context.

Rules:
- Give a complete and direct answer.
- Include the important concepts needed to answer the question.
- Do not invent information.
- Keep the answer concise.
- If the answer is not present in the context, say:
"I don't have enough information in the provided context."

Context:
{context_text}

Question:
{question}

Answer:"""

        start = time.perf_counter()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(outputs.shape[1])
        return {
    "answer": answer,
    "model": self.model_name,
    "latency_ms": round(latency_ms, 2),
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "total_tokens": prompt_tokens + completion_tokens,
    "cost_usd": 0.0
}