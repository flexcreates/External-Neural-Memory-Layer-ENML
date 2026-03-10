# ENML User Guide

This guide covers first-time setup, normal usage, memory behavior, model startup, and common debugging.

## Quick Start

1. Run setup:

```bash
./setup.sh
```

2. Review `.env` and update:

- `MODELS_DIR`
- `LLAMA_SERVER`
- `ALLOWED_PATHS`
- `AI_NAME`

3. Start Qdrant:

```bash
./run_qdrant.sh
```

4. Start the model server:

```bash
./run_server.sh
```

5. Start ENML:

```bash
source .venv/bin/activate
python3 chat.py
```

Or run the web UI:

```bash
./run_web.sh
```

## What ENML Does

ENML learns from your messages, stores extracted facts and preferences, retrieves relevant evidence later, and grounds answers using that evidence.

It has three important memory layers:

- authority memory: deterministic profile in `memory/authority/profile.json`
- vector memory: Qdrant collections for knowledge, project, research, document, and episodic retrieval
- session history: JSON conversation logs in `memory/conversations/...`

## Teaching The System

Natural examples:

```text
my name is Flex
I am 22 years old
my graphics card has 6GB VRAM
I like creating art
I love working on my projects
I have a dog named Bruno
```

Useful notes:

- ENML extracts from your messages, not from the assistant's responses.
- `like` and `love` are now stored separately for new memories.
- corrections such as `actually my graphics card has 6GB vram` are intended to update memory.

You can also force-save a message:

```text
/remember I prefer concise responses
```

## Asking About Memory

Examples:

```text
what is my name?
what is my favorite color?
what do i like?
what do i love?
what are my system specs?
```

Expected behavior:

- if evidence exists, ENML should answer from memory
- if evidence is missing, ENML should avoid guessing personal facts
- responses should stay short for direct recall questions

## Sessions

New sessions are created automatically:

```text
session_YYYYMMDD_HHMMSS
```

Sessions are saved to:

```text
memory/conversations/YYYY/MM/session_....json
```

Resume a session:

```bash
python3 chat.py --session session_20260310_215921
```

## Documents And Large Input

Large pasted content is classified as document input and ingested separately.

CLI:

- normal chat stays in the conversation loop
- long pasted content is treated as document content

Web:

- the same behavior is exposed through the web UI

## Model Startup

`run_server.sh` scans your model directory and shows:

- model filename
- template support: `[ok]` or `[no]`
- detected family
- approximate parameter size

Example:

```text
[2] gemma-2-9b-it-Q4_K_M.gguf [ok] [gemma] [9B]
```

Important behavior:

- ENML detects the active server model from `/v1/models`
- prompt routing follows the active server model, not just static defaults
- internal helper models use the same template routing path

## Prompt Template Families

Current families:

- `llama3`
- `mistral`
- `qwen`
- `deepseek-coder`
- `deepseek`
- `gemma`
- `phi3`
- `openchat`
- `wizardcoder`
- `smollm3`
- `generic`

## Debugging

Useful checks:

```bash
curl http://localhost:6333/health
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:5000/api/health
```

Runtime metrics:

```bash
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 100
```

Main logs:

- `logs/memory_system.log`
- `logs/pipeline.log`
- `logs/audit.jsonl`
- `logs/runtime_replay.jsonl`
- `logs/citations.jsonl`

## Common Problems

### The assistant prints strange special tokens

Cause:

- prompt template mismatch between the active model and ENML formatting

Fix:

- restart `run_server.sh`
- restart ENML
- check `curl http://localhost:8080/v1/models`

### DeepSeek feels weak or slow

Possible reasons:

- coder-tuned model used for general chat
- partially offloaded layers due to VRAM limits
- degraded GGUF conversion
- long output budgets for short questions

### Memory answers are noisy

Check:

- `logs/pipeline.log`
- `memory/authority/profile.json`
- Qdrant health
- whether the fact was ever stored in the first place

### Setup failed on a fresh machine

Run:

```bash
./setup.sh
```

Then verify:

- `.venv` exists
- `.env` exists
- Docker is installed if you want Qdrant
- `llama-server` path in `.env` is correct
