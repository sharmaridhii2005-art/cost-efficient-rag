import json
import time
from pathlib import Path

from app.retrieval.retriever import Retriever


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "evaluation" / "data" / "retrieval_questions.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "retrieval_baseline.json"


def evaluate():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retriever = Retriever(top_k=3)

    results = []

    hit_at_1 = 0
    hit_at_3 = 0
    total_latency = 0

    for item in questions:
        question = item["question"]
        expected_sources = set(item["expected_sources"])

        start = time.perf_counter()

        retrieved = retriever.retrieve(question)

        latency_ms = (time.perf_counter() - start) * 1000
        total_latency += latency_ms

        documents = retrieved.get("documents", [[]])[0]
        distances = retrieved.get("distances", [[]])[0]
        metadatas = retrieved.get("metadatas", [[]])[0]

        sources = [
            metadata.get("source")
            for metadata in metadatas
        ]

        top1_hit = bool(sources) and sources[0] in expected_sources

        top3_hit = any(
            source in expected_sources
            for source in sources[:3]
        )

        if top1_hit:
            hit_at_1 += 1

        if top3_hit:
            hit_at_3 += 1

        results.append({
            "question": question,
            "expected_sources": list(expected_sources),
            "retrieved_sources": sources,
            "distances": distances,
            "latency_ms": round(latency_ms, 2),
            "hit_at_1": top1_hit,
            "hit_at_3": top3_hit
        })

    total = len(questions)

    summary = {
        "questions": total,
        "hit_at_1": round(hit_at_1 / total, 4) if total else 0,
        "hit_at_3": round(hit_at_3 / total, 4) if total else 0,
        "average_latency_ms": round(
            total_latency / total, 2
        ) if total else 0
    }

    output = {
        "summary": summary,
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Questions: {total}")
    print(f"Hit@1: {summary['hit_at_1']:.2%}")
    print(f"Hit@3: {summary['hit_at_3']:.2%}")
    print(f"Average latency: {summary['average_latency_ms']:.2f} ms")

    print(f"\nResults saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    evaluate()
