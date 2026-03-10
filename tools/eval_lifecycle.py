import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.record_repository import MemoryRecordRepository


def main():
    parser = argparse.ArgumentParser(description="Summarize lifecycle state of memory records")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = MemoryRecordRepository()
    records = repo.all()
    status_counts = Counter(record.status for record in records)
    type_counts = Counter(record.memory_type for record in records)
    result = {
        "total_records": len(records),
        "status_counts": dict(status_counts),
        "type_counts": dict(type_counts),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Lifecycle summary")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
