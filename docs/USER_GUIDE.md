# ENML User Guide

This guide covers normal usage of the current ENML runtime.

## Quick Start

```bash
./setup.sh
./run_qdrant.sh
./run_server.sh
source .venv/bin/activate
python3 chat.py
```

Web UI:

```bash
./run_web.sh
```

## What ENML Learns

ENML is best at:

- identity facts
- preferences
- possessions and system specs
- project/document/research summaries
- recent episodic conversation context

Examples:

```text
my name is Flex
my favorite color is scion blue
my profession is vibecoding
my laptop is a Lenovo Loq
i like ubuntu linux
```

Corrections also matter:

```text
actually my laptop name is only Lenovo Loq
actually i deleted windows today
```

## What Happens To Large Pasted Text

Large pasted content is treated as a document instead of a normal chat turn.

The document path:

- splits content into sections
- summarizes sections with the active LLM
- stores summaries in document, project, or research memory depending on classification
- extracts factual claims where possible

This behavior exists in both the CLI and the web UI.

## Asking Memory Questions

Good recall prompts:

```text
what is my name?
what is my profession?
what are my laptop specs?
what do you know about me?
what is my favorite color?
```

Expected behavior:

- direct recall questions should be short and factual
- missing memories should produce an explicit unknown, not a guess
- ordinary conversation should not be forced through strict memory recall

## Sessions

Sessions are created automatically and saved under:

```text
memory/conversations/YYYY/MM/session_YYYYMMDD_HHMMSS.json
```

Resume one with:

```bash
python3 chat.py --session session_20260311_143847
```

## Useful CLI Flags

```bash
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
```

## Useful Commands During Chat

- `exit`
- `quit`
- `/remember <text>`

Example:

```text
/remember i prefer concise replies
```

## Startup Checks

Qdrant:

```bash
curl http://localhost:6333/readyz
```

Model server:

```bash
curl http://localhost:8080/v1/models
```

Web server:

```bash
curl http://localhost:5000/api/health
```

## Common Problems

### Qdrant is unavailable

Symptom:

- startup warning says ENML will run without vector memory

What it means:

- authority memory and local record memory still work
- vector retrieval and vector persistence are disabled until Qdrant returns

### The model responds with the wrong style or odd control tokens

Usually means:

- the running model family does not match the prompt format being used

Fix:

- restart `./run_server.sh`
- confirm `curl http://localhost:8080/v1/models`

### Mistral behaves strangely

The current prompt renderer intentionally does not prepend a manual BOS token for Mistral GGUFs. If you change prompt formatting, re-check Mistral specifically.

### Memory answers are missing

Check:

- `memory/authority/profile.json`
- `logs/pipeline.log`
- `logs/memory_system.log`
- Qdrant readiness with `/readyz`

### Web UI starts but chat fails

Check:

- `run_server.sh` is running
- `/v1/models` responds
- `run_web.sh` exported the active model into ENML’s runtime
