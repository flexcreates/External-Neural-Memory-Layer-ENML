# Tools Module

This directory contains evaluation utilities and a sandboxed file helper.

## Main Files

| File | Purpose |
|---|---|
| `file_tool.py` | restricted file read/write/list operations inside `ALLOWED_PATHS` |
| `eval_runtime.py` | summarizes runtime replay metrics |
| `eval_citations.py` | summarizes citation coverage and citation type mix |
| `eval_lifecycle.py` | summarizes local memory lifecycle state |
| `retrieval_benchmark.py` | lightweight retrieval policy benchmark |

## Current Usage

```bash
python3 tools/eval_runtime.py
python3 tools/eval_citations.py
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 100
```

## File Tool

`file_tool.py` only allows operations inside configured `ALLOWED_PATHS`.

It provides:

- path validation
- file reads
- file writes
- directory listing
