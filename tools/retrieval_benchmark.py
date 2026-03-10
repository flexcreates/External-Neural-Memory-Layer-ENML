import argparse
import os
import sys
import statistics
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval import RetrievalPolicyEngine


def run_policy_benchmark(iterations: int):
    engine = RetrievalPolicyEngine()
    queries = [
        ("what is my age", "knowledge_collection", "small"),
        ("explain transformers", "research_collection", "medium"),
        ("open the readme file", "document_collection", "small"),
        ("how does this project architecture work", "project_collection", "medium"),
    ]

    timings = []
    for _ in range(iterations):
        for query, collection, profile in queries:
            start = time.perf_counter()
            engine.resolve(query, collection, profile)
            timings.append((time.perf_counter() - start) * 1000)

    print("Retrieval policy benchmark")
    print(f"iterations: {iterations}")
    print(f"calls: {len(timings)}")
    print(f"mean_ms: {statistics.mean(timings):.4f}")
    print(f"p95_ms: {statistics.quantiles(timings, n=20)[18]:.4f}" if len(timings) >= 20 else f"max_ms: {max(timings):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark lightweight ENML retrieval policy components")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    run_policy_benchmark(args.iterations)
