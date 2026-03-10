#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Llama.cpp Inference Server Startup
# ═══════════════════════════════════════════════════════════════════════
# Starts llama-server with optimized settings for ENML.
# All configuration is read from .env
# ═══════════════════════════════════════════════════════════════════════

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration from .env
MODELS_DIR="${MODELS_DIR:-/home/flex/ai-models}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/flex/Tools/llama.cpp/build/bin/llama-server}"
LLAMA_URL="${LLAMA_SERVER_URL:-http://localhost:8080}"
PORT=$(echo "$LLAMA_URL" | grep -oP ':\K[0-9]+$' || echo "8080")
HOST="0.0.0.0"
CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
BATCH_SIZE=512

# Scan for models and prompt user
if [ ! -d "$MODELS_DIR" ]; then
    echo "⚠ MODELS_DIR not found: $MODELS_DIR"
    echo "  Edit .env and set MODELS_DIR to your models folder."
    exit 1
fi

echo "Scanning for models in $MODELS_DIR..."
# Find all .gguf files
mapfile -t models < <(find "$MODELS_DIR" -type f -name "*.gguf")

if [ ${#models[@]} -eq 0 ]; then
    echo "✗ No .gguf models found in $MODELS_DIR"
    exit 1
fi

echo "Available Models:"
for i in "${!models[@]}"; do
    echo "  [$((i+1))] $(basename "${models[$i]}")"
done

read -p "Select a model (1-${#models[@]}): " model_idx
if ! [[ "$model_idx" =~ ^[0-9]+$ ]] || [ "$model_idx" -lt 1 ] || [ "$model_idx" -gt "${#models[@]}" ]; then
    echo "Invalid selection."
    exit 1
fi

MODEL_PATH="${models[$((model_idx-1))]}"
echo "Selected: $(basename "$MODEL_PATH")"

if [[ "$LLAMA_SERVER" == "/path/to/"* ]]; then
    echo "⚠ LLAMA_SERVER is not configured!"
    echo "  Edit .env and set LLAMA_SERVER to your llama-server binary."
    exit 1
fi

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "✗ llama-server not found or not executable: $LLAMA_SERVER"
    exit 1
fi

# ── Dynamic VRAM Layer Offloading (GPU-Process-Aware) ────────────────
# Reads actual free VRAM, detects all GPU processes, and calculates
# optimal layers dynamically. Leaves BREATHING_ROOM + AUTOMATION_RESERVE for stability.
BREATHING_ROOM=500           # MB — always keep free for stability
AUTOMATION_RESERVE=2048      # MB — reserve for future automation system
LAYER_SIZE_MB=140            # Approx VRAM per offloaded layer
MAX_LAYERS=22                # Safety ceiling

if command -v nvidia-smi &>/dev/null; then
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    USED_VRAM=$((TOTAL_VRAM - FREE_VRAM))
    AVAILABLE=$((FREE_VRAM - BREATHING_ROOM - AUTOMATION_RESERVE))
    if [ "$AVAILABLE" -lt 0 ]; then AVAILABLE=0; fi

    OPTIMAL=$((AVAILABLE / LAYER_SIZE_MB))
    FINAL_NGL=$((OPTIMAL < MAX_LAYERS ? OPTIMAL : MAX_LAYERS))
    if [ "$FINAL_NGL" -lt 0 ]; then FINAL_NGL=0; fi

    # Detect GPU processes for the startup report
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")
else
    TOTAL_VRAM="N/A"
    FREE_VRAM="N/A"
    USED_VRAM="N/A"
    AVAILABLE="N/A"
    GPU_PROCS=""
    FINAL_NGL=0
fi

# ── Startup Banner ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Llama.cpp Server                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Model:     $(basename "$MODEL_PATH")"
echo "  URL:       http://localhost:$PORT"
echo "  Context:   ${CONTEXT_SIZE} tokens | Batch: ${BATCH_SIZE}"
echo ""
echo "  ── GPU Resource Allocation ──"
if [ "$TOTAL_VRAM" != "N/A" ]; then
    echo "  Total:     ${TOTAL_VRAM}MB"
    echo "  Used:      ${USED_VRAM}MB (by other processes)"
    echo "  Free:      ${FREE_VRAM}MB"
    echo "  Reserved:  $((BREATHING_ROOM + AUTOMATION_RESERVE))MB (${BREATHING_ROOM}MB buffer + ${AUTOMATION_RESERVE}MB automation)"
    echo "  Budget:    ${AVAILABLE}MB available for LLM"
    echo "  Layers:    ${FINAL_NGL} GPU layers (dynamic, max ${MAX_LAYERS})"
    echo ""
    if [ -n "$GPU_PROCS" ]; then
        echo "  ── Active GPU Processes ──"
        while IFS=',' read -r pid pname mem; do
            pid=$(echo "$pid" | xargs)
            pname=$(echo "$pname" | xargs)
            mem=$(echo "$mem" | xargs)
            echo "    • ${pname} (PID ${pid}) — ${mem} MiB"
        done <<< "$GPU_PROCS"
    else
        echo "  ── No other GPU processes detected ──"
    fi
else
    echo "  NVIDIA GPU not detected — running CPU only (0 layers)"
fi
echo ""

# Export LD_LIBRARY_PATH so llama-server can find its shared libraries (.so files)
LLAMA_DIR="$(dirname "$LLAMA_SERVER")"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"

"$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -c "$CONTEXT_SIZE" \
    -ngl "$FINAL_NGL" \
    -b "$BATCH_SIZE" \
    --cache-ram 2048 \
    --parallel 1 \
    --mlock \
    --flash-attn on \
    --defrag-thold 0.1 \
    --metrics \
    --port "$PORT" \
    --host "$HOST" \
    --temp 0.6 \
    --top-k 40 \
    --top-p 0.9
