import json
import time
from pathlib import Path

from app.retrieval.retriever import Retriever
from app.generation.local_generator import LocalGenerator


QUESTIONS_FILE = Path("evaluation/data/retrieval_questions.json")
OUTPUT_FILE = Path("evaluation/results/generation_topk_baseline.json")

TOP_K_VALUES = [1, 2, 3]


def build_contexts(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    contexts = []

    for document, metadata in zip(documents, metadatas):
        contexts.append({
            "text": document,
            "source": metadata.get("source"),
            "chunk_index": metadata.get("chunk_index")
        })

    return contexts


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    generator = LocalGenerator()

    all_results = []

    print("=" * 70)
    print("GENERATION TOP-K COST EFFICIENCY BENCHMARK")
    print("=" * 70)

    for top_k in TOP_K_VALUES:

        print("\n" + "=" * 70)
        print(f"TOP_K = {top_k}")
        print("=" * 70)

        retriever = Retriever(top_k=top_k)

        for item in questions:
            question = item["question"]

            print(f"\nQuestion: {question}")

            retrieval_start = time.perf_counter()

            retrieved = retriever.retrieve(question)

            retrieval_latency_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

            contexts = build_contexts(retrieved)

            result = generator.generate(
                question,
                contexts
            )

            record = {
                "question": question,
                "top_k": top_k,
                "retrieved_sources": [
                    context["source"]
                    for context in contexts
                ],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "retrieval_latency_ms": round(
                    retrieval_latency_ms, 2
                ),
                "generation_latency_ms": result["latency_ms"],
                "total_latency_ms": round(
                    retrieval_latency_ms +
                    result["latency_ms"],
                    2
                ),
                "cost_usd": result["cost_usd"],
                "answer": result["answer"]
            }

            all_results.append(record)

            print(
                f"Prompt tokens: {result['prompt_tokens']} | "
                f"Total tokens: {result['total_tokens']} | "
                f"Generation: {result['latency_ms']} ms"
            )

    summary = {}

    for top_k in TOP_K_VALUES:
        rows = [
            row for row in all_results
            if row["top_k"] == top_k
        ]

        summary[str(top_k)] = {
            "questions": len(rows),
            "average_prompt_tokens": round(
                sum(r["prompt_tokens"] for r in rows) /
                len(rows),
                2
            ),
            "average_completion_tokens": round(
                sum(r["completion_tokens"] for r in rows) /
                len(rows),
                2
            ),
            "average_total_tokens": round(
                sum(r["total_tokens"] for r in rows) /
                len(rows),
                2
            ),
            "average_generation_latency_ms": round(
                sum(r["generation_latency_ms"] for r in rows) /
                len(rows),
                2
            ),
            "average_total_latency_ms": round(
                sum(r["total_latency_ms"] for r in rows) /
                len(rows),
                2
            ),
            "total_cost_usd": round(
                sum(r["cost_usd"] for r in rows),
                6
            )
        }

    output = {
        "summary": summary,
        "results": all_results
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for top_k, metrics in summary.items():
        print(
            f"TOP_K={top_k} | "
            f"Prompt={metrics['average_prompt_tokens']} | "
            f"Total={metrics['average_total_tokens']} | "
            f"Generation={metrics['average_generation_latency_ms']} ms"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
