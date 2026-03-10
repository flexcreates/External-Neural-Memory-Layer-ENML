import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation usage logs")
    parser.add_argument("--input", default="logs/citations.jsonl")
    args = parser.parse_args()

    entries = load(args.input)
    total = len(entries)
    cited = sum(len(entry.get("citations", [])) for entry in entries)
    with_citations = sum(1 for entry in entries if entry.get("citations"))
    by_type = Counter()
    for entry in entries:
        for citation in entry.get("citations", []):
            by_type[citation.get("memory_type", "unknown")] += 1

    result = {
        "total_responses": total,
        "responses_with_citations": with_citations,
        "citation_coverage": round(with_citations / total, 4) if total else 0.0,
        "average_citations_per_response": round(cited / total, 4) if total else 0.0,
        "citations_by_type": dict(by_type),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
