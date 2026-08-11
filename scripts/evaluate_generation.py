import json
from pathlib import Path


INPUT_FILE = Path(
    "evaluation/results/generation_topk_baseline.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/generation_topk_evaluation.json"
)


EXPECTED_CONCEPTS = {
    "What is RAG?": [
        "retrieval",
        "generation"
    ],

    "What are the main stages of a RAG system?": [
        "document ingestion",
        "chunking",
        "embedding",
        "vector storage",
        "retrieval",
        "answer generation"
    ],

    "What are embeddings?": [
        "vector",
        "text"
    ],

    "Why is chunking important?": [
        "smaller",
        "retrieval"
    ],

    "How does a vector database work?": [
        "vector",
        "similarity"
    ],

    "How can RAG reduce cost?": [
        "cost",
        "retrieval"
    ]
}


def score_answer(question, answer):
    answer_lower = answer.lower()

    concepts = EXPECTED_CONCEPTS[question]

    matched = [
        concept
        for concept in concepts
        if concept.lower() in answer_lower
    ]

    score = len(matched) / len(concepts)

    return {
        "score": round(score, 3),
        "matched": matched,
        "expected": concepts
    }


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    evaluated_results = []

    for result in data["results"]:

        evaluation = score_answer(
            result["question"],
            result["answer"]
        )

        evaluated_results.append({
            **result,
            "quality_score": evaluation["score"],
            "matched_concepts": evaluation["matched"],
            "expected_concepts": evaluation["expected"]
        })

    summary = {}

    for top_k in sorted(
        set(r["top_k"] for r in evaluated_results)
    ):

        rows = [
            r for r in evaluated_results
            if r["top_k"] == top_k
        ]

        average_quality = (
            sum(r["quality_score"] for r in rows)
            / len(rows)
        )

        perfect_answers = sum(
            r["quality_score"] == 1.0
            for r in rows
        )

        summary[str(top_k)] = {
            "questions": len(rows),
            "average_quality_score": round(
                average_quality,
                3
            ),
            "perfect_answers": perfect_answers,
            "quality_rate": round(
                perfect_answers / len(rows),
                3
            )
        }

    output = {
        "summary": summary,
        "results": evaluated_results
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

    print("=" * 70)
    print("GENERATION QUALITY EVALUATION")
    print("=" * 70)

    for top_k, metrics in summary.items():

        print(
            f"TOP_K={top_k} | "
            f"Quality={metrics['average_quality_score']} | "
            f"Perfect={metrics['perfect_answers']}/"
            f"{metrics['questions']}"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
