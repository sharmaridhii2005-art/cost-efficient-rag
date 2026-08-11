import json
from app.config import BASE_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("COST-EFFICIENT RAG - CONFIGURATION BENCHMARK")
    print("=" * 70)

    results_dir = BASE_DIR / "evaluation" / "results"

    topk_file = results_dir / "generation_topk_baseline.json"
    quality_file = results_dir / "generation_topk_evaluation.json"
    semantic_file = results_dir / "generation_semantic_evaluation.json"

    topk_data = load_json(topk_file)
    quality_data = load_json(quality_file)
    semantic_data = load_json(semantic_file)

    configurations = []

    for top_k in ["1", "3"]:

        baseline = topk_data["summary"][top_k]
        quality = quality_data["summary"][top_k]

        # Semantic score is stored directly in summary
        semantic = semantic_data["summary"][top_k]

        configurations.append({
            "top_k": int(top_k),
            "prompt_tokens": baseline["average_prompt_tokens"],
            "total_tokens": baseline["average_total_tokens"],
            "generation_latency_ms": baseline[
                "average_generation_latency_ms"
            ],
            "total_latency_ms": baseline[
                "average_total_latency_ms"
            ],
            "quality": quality["average_quality_score"],
            "perfect_answers": quality["perfect_answers"],
            "quality_rate": quality["quality_rate"],
            "semantic_similarity": semantic[
                "average_semantic_similarity"
            ],
            "cost_usd": baseline["total_cost_usd"],
        })

    print("\n")
    print("FINAL CONFIGURATION COMPARISON")
    print("-" * 70)

    print(
        f"{'TOP_K':<8}"
        f"{'Prompt':<12}"
        f"{'Total':<12}"
        f"{'Latency(ms)':<15}"
        f"{'Quality':<10}"
        f"{'Semantic':<10}"
    )

    print("-" * 70)

    for config in configurations:
        print(
            f"{config['top_k']:<8}"
            f"{config['prompt_tokens']:<12.2f}"
            f"{config['total_tokens']:<12.2f}"
            f"{config['generation_latency_ms']:<15.2f}"
            f"{config['quality']:<10.3f}"
            f"{config['semantic_similarity']:<10.4f}"
        )

    # Choose best configuration.
    # Priority:
    # 1. Higher quality
    # 2. Higher semantic similarity
    # 3. Lower token usage

    best = max(
        configurations,
        key=lambda x: (
            x["quality"],
            x["semantic_similarity"],
            -x["total_tokens"],
        ),
    )

    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    print(f"TOP_K = {best['top_k']}")
    print(f"Quality Score = {best['quality']:.3f}")
    print(
        f"Semantic Similarity = "
        f"{best['semantic_similarity']:.4f}"
    )
    print(
        f"Average Prompt Tokens = "
        f"{best['prompt_tokens']:.2f}"
    )
    print(
        f"Average Total Tokens = "
        f"{best['total_tokens']:.2f}"
    )
    print(
        f"Generation Latency = "
        f"{best['generation_latency_ms']:.2f} ms"
    )

    output_dir = results_dir / "configuration_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "configuration_comparison.json"
    )

    output = {
        "configurations": configurations,
        "recommended_configuration": best,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()