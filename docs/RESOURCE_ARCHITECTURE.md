# ENML — System Resource Architecture

> **Target Machine**: RTX 3050 6GB · 16GB RAM · i5-12450HX (12 threads) · Ubuntu 24.04

## VRAM Strategy: Dynamic GPU-Process-Aware

The LLM server **dynamically delegates** GPU layer offloading directly to `llama.cpp` using the native `--fit` flag:

1. Evaluates **total** and **free** VRAM in real-time.
2. Senses **all active GPU processes** directly via free VRAM availability.
3. Sets a rigid **300MB strict buffer** (`--fit-target 300`) that must never be touched.
4. Auto-calculates safe layer capacity: Attempts to load maximum layers (`-ngl 999`), forcing the C++ engine to optimize and fit as many layers as possible into the remaining memory budget without exceeding the buffer.

### Example Scenarios

| GPU State | Free VRAM | Dedicated to LLM | Notes |
|---|---|---|---|
| Nothing running | ~5800MB | ~5500MB | Max layers offloaded; peak performance |
| Background Apps (~2000MB) | ~3800MB | ~3500MB | Auto-scales layers down softly |
| Heavy load (~4600MB) | ~1200MB | ~900MB | Critical fallback to CPU computation |

## RAM Allocation

| Component | RAM |
|---|---|
| LLM CPU layers | 2–3 GB |
| Memory system (Qdrant, embeddings) | 1–2 GB |
| OS + services | 3–4 GB |
| Free buffer | 4–5 GB |

## LLM Server Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Context size | 4096 tokens | Saves ~500MB KV cache vs 8192; sufficient for memory injection |
| Batch size | 512 | Reduces peak VRAM spikes |
| Prompt Cache | 2048 MB | Reduces RAM pressure from historical state retention |
| GPU layers | Auto (Target 999) | Adapts to available VRAM natively via `--fit` |
| Parallel slots | 1 | Single inference stream |
| Flash attention | On | Reduces KV cache memory |
| Memory lock | On | Prevents OS swapping model weights |

## Performance Philosophy

1. **Stability** — Never exceed safe VRAM bounds
2. **VRAM safety margin** — Always keep definitively 300MB free using `--fit-target`
3. **Sustained operation** — Designed for continuous uptime
4. **Then speed** — Token/s is a secondary concern

> This is a **hybrid compute node**, not a GPU-only server.
> CPU handles: remaining layers, tokenization, memory injection, RAG, orchestration.
