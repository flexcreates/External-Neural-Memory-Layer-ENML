# ENML — System Resource Architecture

> **Target Machine**: RTX 3050 6GB · 16GB RAM · i5-12450HX (12 threads) · Ubuntu 24.04

## VRAM Strategy: Dynamic GPU-Process-Aware

The LLM server **dynamically calculates** GPU layer offloading at startup:

1. Reads **total** and **free** VRAM via `nvidia-smi`
2. Detects **all active GPU processes** (Whisper, compositor, etc.)
3. Subtracts **500MB breathing room** from free VRAM
4. Calculates optimal layers: `(free - 500) / 140MB per layer`
5. Caps at **22 layers** maximum (safety ceiling)

### Example Scenarios

| GPU State | Free VRAM | Budget | Layers | Notes |
|---|---|---|---|---|
| Nothing running | ~5800MB | ~5300MB | 22 (capped) | Full power |
| Whisper (~2000MB) | ~3800MB | ~3300MB | 22 (capped) | Typical co-existence |
| Whisper (~3000MB) | ~2800MB | ~2300MB | 16 | Auto-reduced |
| Heavy load | ~1200MB | ~700MB | 5 | Graceful fallback |

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
| GPU layers | Dynamic (max 22) | Adapts to available VRAM |
| Parallel slots | 1 | Single inference stream |
| Flash attention | On | Reduces KV cache memory |
| Memory lock | On | Prevents OS swapping model weights |

## Performance Philosophy

1. **Stability** — Never exceed safe VRAM bounds
2. **VRAM safety margin** — Always keep 500MB free
3. **Sustained operation** — Designed for continuous uptime
4. **Then speed** — Token/s is a secondary concern

> This is a **hybrid compute node**, not a GPU-only server.
> CPU handles: remaining layers, tokenization, memory injection, RAG, orchestration.
