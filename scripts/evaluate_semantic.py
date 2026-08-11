import json
from pathlib import Path

import numpy as np

from app.retrieval.embedder import create_embeddings


INPUT_FILE = Path(
    "evaluation/results/generation_topk_baseline.json"
)

QUESTIONS_FILE = Path(
    "evaluation/data/generation_questions.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/generation_semantic_evaluation.json"
)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        generation_data = json.load(f)

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        reference_data = json.load(f)

    references = {
        item["question"]: item["reference_answer"]
        for item in reference_data
    }

    reference_texts = list(references.values())

    print("Creating reference embeddings...")

    reference_embeddings = create_embeddings(
        reference_texts
    )

    reference_map = dict(
        zip(
            references.keys(),
            reference_embeddings
        )
    )

    evaluated_results = []

    for result in generation_data["results"]:

        question = result["question"]
        answer = result["answer"]

        answer_embedding = create_embeddings(
            [answer]
        )[0]

        reference_embedding = reference_map[
            question
        ]

        similarity = cosine_similarity(
            answer_embedding,
            reference_embedding
        )

        evaluated_results.append({
            **result,
            "semantic_similarity": round(
                similarity,
                4
            )
        })

    summary = {}

    top_k_values = sorted(
        set(
            result["top_k"]
            for result in evaluated_results
        )
    )

    for top_k in top_k_values:

        rows = [
            result
            for result in evaluated_results
            if result["top_k"] == top_k
        ]

        similarities = [
            result["semantic_similarity"]
            for result in rows
        ]

        summary[str(top_k)] = {
            "questions": len(rows),
            "average_semantic_similarity": round(
                sum(similarities) /
                len(similarities),
                4
            ),
            "minimum_semantic_similarity": round(
                min(similarities),
                4
            ),
            "maximum_semantic_similarity": round(
                max(similarities),
                4
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
    print("SEMANTIC GENERATION EVALUATION")
    print("=" * 70)

    for top_k, metrics in summary.items():

        print(
            f"TOP_K={top_k} | "
            f"Average similarity="
            f"{metrics['average_semantic_similarity']} | "
            f"Min="
            f"{metrics['minimum_semantic_similarity']} | "
            f"Max="
            f"{metrics['maximum_semantic_similarity']}"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
