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

`run_server.sh` plans a launch profile with `llama-fit-params`, then starts
`llama-server` with:

- live free VRAM from `nvidia-smi`
- the selected GGUF model's own `context_length`
- `FIT_TARGET_MB=512` by default
- `FIT_CONTEXT_MIN=1024` and `FIT_CONTEXT_STEP=256` by default
- explicit `-c <planned_context>`
- explicit `--gpu-layers <planned_layers>`
- `--flash-attn on`
- `--parallel 1`
- `--cache-ram 2048`
- `-b 512`

That means:

- context is computed from current GPU headroom at launch time
- layer offload is computed from the same launch-time GPU headroom
- a configurable free-memory target is preserved per GPU device
- prompt cache is enabled
- there is no model-family-specific context cap in `run_server.sh`
- if more VRAM is free later, the next launch can choose a larger context
- if VRAM is tighter later, the next launch can choose a smaller context

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
