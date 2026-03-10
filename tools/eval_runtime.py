import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_entries(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(entries):
    total = len(entries)
    if total == 0:
        return {
            "total_requests": 0,
            "retrieval_hit_rate": 0.0,
            "citation_precision": 0.0,
            "strict_grounded_response_rate": 0.0,
            "mean_total_ms": 0.0,
            "p95_total_ms": 0.0,
        }

    retrieval_hits = 0
    strict_grounded = 0
    total_ms = []
    citation_precisions = []
    policy_counter = Counter()

    for entry in entries:
        evidence_count = entry.get("evidence_count", 0)
        if evidence_count > 0:
            retrieval_hits += 1
        if entry.get("strict_grounding"):
            strict_grounded += 1

        timings = entry.get("timings_ms", {})
        if "total" in timings:
            total_ms.append(timings["total"])

        citations = entry.get("citations", [])
        cited_count = len(citations)
        unsupported = entry.get("unsupported_claim_estimate", 0)
        if cited_count + unsupported > 0:
            citation_precisions.append(cited_count / (cited_count + unsupported))
        elif evidence_count == 0:
            citation_precisions.append(1.0)

        if entry.get("policy_name"):
            policy_counter[entry["policy_name"]] += 1

    result = {
        "total_requests": total,
        "retrieval_hit_rate": round(retrieval_hits / total, 4),
        "citation_precision": round(sum(citation_precisions) / len(citation_precisions), 4) if citation_precisions else 0.0,
        "strict_grounded_response_rate": round(strict_grounded / total, 4),
        "mean_total_ms": round(statistics.mean(total_ms), 3) if total_ms else 0.0,
        "p95_total_ms": round(statistics.quantiles(total_ms, n=20)[18], 3) if len(total_ms) >= 20 else (round(max(total_ms), 3) if total_ms else 0.0),
        "policy_distribution": dict(policy_counter),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ENML runtime replay logs")
    parser.add_argument("--input", default="logs/runtime_replay.jsonl")
    args = parser.parse_args()

    entries = load_entries(args.input)
    result = evaluate(entries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
