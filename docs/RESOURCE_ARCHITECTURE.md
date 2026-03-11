# ENML Resource Architecture

This document describes how the current system uses CPU, GPU, and storage resources.

## GPU Ownership

In the normal local stack:

- `llama-server` is the primary GPU consumer
- embeddings and reranking run on CPU
- Qdrant runs in Docker and is not the main VRAM consumer

That split is intentional so the LLM server can keep most of the GPU.

## Current Server Planning Strategy

[run_server.sh](/home/flex/Projects/enml/run_server.sh) now does four things before launch:

1. reads live free VRAM from `nvidia-smi`
2. reads the selected model’s GGUF `context_length`
3. asks `llama-fit-params` what GPU fit is possible for candidate contexts
4. chooses the largest safe context for the current machine state, then launches `llama-server` with explicit `-c` and `--gpu-layers`

Relevant environment variables:

- `LLAMA_FIT_PARAMS`
- `FIT_TARGET_MB`
- `FIT_CONTEXT_MIN`
- `FIT_CONTEXT_STEP`
- `CACHE_TYPE_K`
- `CACHE_TYPE_V`

`CONTEXT_SIZE` remains only as a fallback if the planner cannot run.

## What Changes At Runtime

The planned context is dynamic across launches.

If more VRAM is free:

- ENML can choose a larger context
- more layers may still remain on GPU depending on the model

If less VRAM is free:

- ENML can choose a smaller context
- fewer layers may be offloaded

This is why the effective `n_ctx` can differ between models and between launches on the same machine.

## Current Retrieval Resource Split

### CPU

- dense embeddings via `SentenceTransformer`
- sparse BM25 embeddings
- cross-encoder reranking
- document summarization helper orchestration

### GPU

- active GGUF inference server only

### Disk

- authority profile JSON
- memory record repository JSON
- conversation session JSON files
- logs and metrics JSONL files
- Qdrant storage volume

## Practical Tradeoffs

Large context windows are not free.

Increasing context:

- increases KV cache size
- can reduce the number of layers offloaded to GPU
- usually increases latency

Reducing context:

- lowers KV pressure
- can increase GPU layer residency
- may break recall if prompts or retrieved evidence exceed the smaller window

The current planner prefers fitting the largest safe context available at launch time instead of using a fixed hard cap.

## Qdrant Resource Behavior

Qdrant is optional for degraded operation.

When Qdrant is down:

- ENML still starts
- authority memory still works
- local record fallback still works
- vector inserts and vector retrieval are skipped

Operational check:

```bash
curl http://localhost:6333/readyz
```
