#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Llama.cpp Inference Server Startup
# ═══════════════════════════════════════════════════════════════════════
# Starts llama-server with optimized settings for ENML.
# Use alongside Qdrant and the ENML CLI/Web processes.
# All configuration is read from .env
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

read_env_value() {
    local key="$1"
    local env_file="$SCRIPT_DIR/.env"
    if [ ! -f "$env_file" ]; then
        return 1
    fi

    local raw
    raw=$(grep -E "^${key}=" "$env_file" | tail -n 1 || true)
    if [ -z "$raw" ]; then
        return 1
    fi

    raw="${raw#*=}"
    raw="${raw%\"}"
    raw="${raw#\"}"
    raw="${raw%\'}"
    raw="${raw#\'}"
    printf '%s\n' "$raw"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
fi

# Configuration from .env
MODELS_DIR="${MODELS_DIR:-$(read_env_value MODELS_DIR || printf '/home/flex/ai-models')}"
LLAMA_SERVER="${LLAMA_SERVER:-$(read_env_value LLAMA_SERVER || printf '/home/flex/Tools/llama.cpp/build/bin/llama-server')}"
LLAMA_URL="${LLAMA_SERVER_URL:-$(read_env_value LLAMA_SERVER_URL || printf 'http://localhost:8080')}"
PORT="${LLAMA_URL##*:}"
PORT="${PORT%%/*}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    PORT="8080"
fi
HOST="0.0.0.0"
CONTEXT_SIZE="${CONTEXT_SIZE:-$(read_env_value CONTEXT_SIZE || printf '4096')}"
BATCH_SIZE=512

describe_model_template() {
    local model_name="$1"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf '[no] [generic] [unknown]'
        return
    fi

    "$PYTHON_BIN" -c 'from core.prompt_templates import get_model_template_info; import sys
info = get_model_template_info(sys.argv[1])
mark = "ok" if info.supported else "no"
print(f"[{mark}] [{info.family}] [{info.size_label}]", end="")' "$model_name"
}

# Scan for models and prompt user
if [ ! -d "$MODELS_DIR" ]; then
    echo "⚠ MODELS_DIR not found: $MODELS_DIR"
    echo "  Edit .env and set MODELS_DIR to your models folder."
    exit 1
fi

echo "Scanning for models in $MODELS_DIR..."
# Find all .gguf files
mapfile -t models < <(find "$MODELS_DIR" -type f -name "*.gguf" | sort -f)

if [ ${#models[@]} -eq 0 ]; then
    echo "✗ No .gguf models found in $MODELS_DIR"
    exit 1
fi

echo "Available Models:"
for i in "${!models[@]}"; do
    model_file="$(basename "${models[$i]}")"
    echo "  [$((i+1))] ${model_file} $(describe_model_template "$model_file")"
done

read -p "Select a model (1-${#models[@]}): " model_idx
if ! [[ "$model_idx" =~ ^[0-9]+$ ]] || [ "$model_idx" -lt 1 ] || [ "$model_idx" -gt "${#models[@]}" ]; then
    echo "Invalid selection."
    exit 1
fi

MODEL_PATH="${models[$((model_idx-1))]}"
MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_ALIAS="${MODEL_BASENAME%.gguf}"
MODEL_TEMPLATE_INFO="$(describe_model_template "$MODEL_BASENAME")"
echo "Selected: ${MODEL_BASENAME} ${MODEL_TEMPLATE_INFO}"

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
# Delegates exact layer fitting to llama.cpp using native --fit flags.
# Leaves exactly BREATHING_ROOM free regardless of model size or processes.
BREATHING_ROOM=300           # MB — strict buffer to always keep free

if command -v nvidia-smi &>/dev/null; then
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    USED_VRAM=$((TOTAL_VRAM - FREE_VRAM))
    
    # Max theoretical layers are implicitly handled by --fit on
    
    # Detect GPU processes for the startup report
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")
else
    TOTAL_VRAM="N/A"
    FREE_VRAM="N/A"
    USED_VRAM="N/A"
    AVAILABLE="N/A"
    GPU_PROCS=""
fi

# ── Startup Banner ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Llama.cpp Server                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Model:     ${MODEL_BASENAME}"
echo "  Alias:     ${MODEL_ALIAS}"
echo "  Template:  ${MODEL_TEMPLATE_INFO}"
echo "  URL:       http://localhost:$PORT"
echo "  Context:   ${CONTEXT_SIZE} tokens | Batch: ${BATCH_SIZE}"
echo "  Metrics:   llama.cpp metrics enabled"
echo ""
echo "  ── GPU Resource Allocation ──"
if [ "$TOTAL_VRAM" != "N/A" ]; then
    echo "  Total:     ${TOTAL_VRAM}MB"
    echo "  Used:      ${USED_VRAM}MB (by other processes)"
    echo "  Free:      ${FREE_VRAM}MB"
    echo "  Reserved:  ${BREATHING_ROOM}MB (strict buffer via llama.cpp)"
    echo "  Budget:    Dynamic (managed precisely by llama.cpp)"
    echo "  Layers:    Auto-managed by llama.cpp (--fit on)"
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
export LD_LIBRARY_PATH="${LLAMA_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Auto-detect local Python venv CUDA libs if present to support user-compiled llama.cpp
NVIDIA_VENV_DIR=$(find "$(pwd)/.venv" -type d -name "nvidia" -path "*/site-packages/nvidia" -print -quit 2>/dev/null)
if [ -n "$NVIDIA_VENV_DIR" ]; then
    for dir in "$NVIDIA_VENV_DIR"/*/lib; do
        export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

"$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --alias "$MODEL_ALIAS" \
    -c "$CONTEXT_SIZE" \
    --fit on \
    --fit-target "$BREATHING_ROOM" \
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
