# ENML Resource Architecture

This document describes the practical resource model for ENML on typical local hardware.

## Main Resource Consumers

- `llama-server` for generation
- sentence-transformer embedding model
- reranker model
- Qdrant
- Python orchestration process

## Typical Behavior

`llama-server` is the dominant VRAM consumer. ENML’s embedding and reranking models are usually CPU-side, which keeps more GPU memory available for the active GGUF.

## Current Server Strategy

`run_server.sh` starts `llama-server` with:

- `--fit on`
- `--fit-target 300`
- `--flash-attn on`
- `--parallel 1`
- `--cache-ram 2048`
- `-c 4096`
- `-b 512`

That means:

- layer offload is automatic
- a 300 MB free-memory target is preserved
- prompt cache is enabled
- context size is capped at 4096 unless you change `.env`

## Why Some Models Feel Slow

Usually one or more of these:

- not all layers fit on GPU
- a large KV cache consumes VRAM
- the GGUF conversion is degraded
- the model is coder-tuned and weak for general chat
- response token budgets are too large for small tasks

## Practical Guidance

- 3B to 4B models: best when you want low latency and smaller evidence windows
- 7B class models: best balance for local chat and coding on limited VRAM
- 9B class models: often stronger, but they may offload fewer layers and slow down depending on quant and available GPU memory

## Related Scripts

- `run_server.sh`
- `run_web.sh`
- `setup.sh`
