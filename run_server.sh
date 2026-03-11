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
CACHE_TYPE_K="${CACHE_TYPE_K:-$(read_env_value CACHE_TYPE_K || printf 'q8_0')}"
CACHE_TYPE_V="${CACHE_TYPE_V:-$(read_env_value CACHE_TYPE_V || printf 'q8_0')}"
MIN_CONTEXT_SIZE="${MIN_CONTEXT_SIZE:-$(read_env_value MIN_CONTEXT_SIZE || printf '1024')}"
CONTEXT_STEP="${CONTEXT_STEP:-$(read_env_value CONTEXT_STEP || printf '256')}"
KV_VRAM_FRACTION="${KV_VRAM_FRACTION:-$(read_env_value KV_VRAM_FRACTION || printf '0.06')}"

get_model_size_b() {
    local model_name="$1"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf 'unknown'
        return
    fi

    "$PYTHON_BIN" -c 'from core.prompt_templates import get_model_template_info; import sys
info = get_model_template_info(sys.argv[1])
print(info.size_b if info.size_b is not None else "unknown", end="")' "$model_name"
}

get_cache_quant_factor() {
    local cache_type="$1"
    case "${cache_type,,}" in
        q4* ) printf '0.50' ;;
        q5* ) printf '0.625' ;;
        q6* ) printf '0.75' ;;
        q8* ) printf '1.00' ;;
        f16|bf16 ) printf '2.00' ;;
        f32 ) printf '4.00' ;;
        * ) printf '1.00' ;;
    esac
}

estimate_kv_mb_per_token() {
    local model_size_b="$1"
    local cache_factor_k="$2"
    local cache_factor_v="$3"

    awk -v size="$model_size_b" -v kf="$cache_factor_k" -v vf="$cache_factor_v" '
        BEGIN {
            if (size == "unknown" || size == "") {
                base = 0.07;
            } else if (size <= 4.5) {
                base = 0.035;
            } else if (size <= 8.5) {
                base = 0.070;
            } else if (size <= 14.0) {
                base = 0.110;
            } else if (size <= 24.0) {
                base = 0.180;
            } else {
                base = 0.260;
            }
            factor = (kf + vf) / 2.0;
            printf "%.6f\n", base * factor;
        }
    '
}

estimate_model_footprint_mb() {
    local model_size_b="$1"

    awk -v size="$model_size_b" '
        BEGIN {
            if (size == "unknown" || size == "") {
                print 2600;
            } else {
                printf "%d\n", int((size * 340.0) + 0.5);
            }
        }
    '
}

estimate_runtime_overhead_mb() {
    local model_size_b="$1"
    local batch_size="$2"

    awk -v size="$model_size_b" -v batch="$batch_size" '
        BEGIN {
            if (size == "unknown" || size == "") {
                base = 900;
            } else if (size <= 4.5) {
                base = 550;
            } else if (size <= 8.5) {
                base = 900;
            } else if (size <= 14.0) {
                base = 1250;
            } else if (size <= 24.0) {
                base = 1800;
            } else {
                base = 2400;
            }
            batch_overhead = batch * 0.50;
            printf "%d\n", int(base + batch_overhead + 0.5);
        }
    '
}

get_effective_context_size() {
    local fallback_context="$1"
    local free_vram_mb="$2"
    local reserve_mb="$3"
    local kv_vram_fraction="$4"
    local kv_mb_per_token="$5"
    local min_context_size="$6"
    local context_step="$7"
    local model_footprint_mb="$8"
    local runtime_overhead_mb="$9"

    if [ -z "$free_vram_mb" ] || [ "$free_vram_mb" = "N/A" ]; then
        printf '%s\n' "$fallback_context"
        return
    fi

    awk \
        -v fallback_ctx="$fallback_context" \
        -v free_mb="$free_vram_mb" \
        -v reserve_mb="$reserve_mb" \
        -v frac="$kv_vram_fraction" \
        -v kv_per_token="$kv_mb_per_token" \
        -v min_ctx="$min_context_size" \
        -v step="$context_step" \
        -v model_mb="$model_footprint_mb" \
        -v runtime_mb="$runtime_overhead_mb" '
        BEGIN {
            usable = free_mb - reserve_mb;
            if (usable < 0) usable = 0;
            kv_budget_fraction = usable * frac;
            kv_budget_runtime = usable - model_mb - runtime_mb;
            kv_budget = kv_budget_fraction;
            if (kv_budget_runtime < kv_budget) kv_budget = kv_budget_runtime;
            if (kv_budget < 96) {
                ctx = min_ctx;
            } else {
                ctx = int(kv_budget / kv_per_token);
            }
            if (ctx < min_ctx) ctx = min_ctx;
            if (step > 0) {
                ctx = int(ctx / step) * step;
                if (ctx < min_ctx) ctx = min_ctx;
            }
            printf "%d\n", ctx;
        }
    '
}

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
MODEL_SIZE_B="$(get_model_size_b "$MODEL_BASENAME")"
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
KV_CACHE_FACTOR_K="$(get_cache_quant_factor "$CACHE_TYPE_K")"
KV_CACHE_FACTOR_V="$(get_cache_quant_factor "$CACHE_TYPE_V")"
KV_MB_PER_TOKEN="$(estimate_kv_mb_per_token "$MODEL_SIZE_B" "$KV_CACHE_FACTOR_K" "$KV_CACHE_FACTOR_V")"
MODEL_FOOTPRINT_MB="$(estimate_model_footprint_mb "$MODEL_SIZE_B")"
RUNTIME_OVERHEAD_MB="$(estimate_runtime_overhead_mb "$MODEL_SIZE_B" "$BATCH_SIZE")"

if command -v nvidia-smi &>/dev/null; then
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    USED_VRAM=$((TOTAL_VRAM - FREE_VRAM))
    EFFECTIVE_CONTEXT_SIZE="$(get_effective_context_size "$CONTEXT_SIZE" "$FREE_VRAM" "$BREATHING_ROOM" "$KV_VRAM_FRACTION" "$KV_MB_PER_TOKEN" "$MIN_CONTEXT_SIZE" "$CONTEXT_STEP" "$MODEL_FOOTPRINT_MB" "$RUNTIME_OVERHEAD_MB")"
    
    # Max theoretical layers are implicitly handled by --fit on
    
    # Detect GPU processes for the startup report
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")
else
    TOTAL_VRAM="N/A"
    FREE_VRAM="N/A"
    USED_VRAM="N/A"
    AVAILABLE="N/A"
    GPU_PROCS=""
    EFFECTIVE_CONTEXT_SIZE="$CONTEXT_SIZE"
fi

# ── Startup Banner ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Llama.cpp Server                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Model:     ${MODEL_BASENAME}"
echo "  Alias:     ${MODEL_ALIAS}"
echo "  Template:  ${MODEL_TEMPLATE_INFO}"
echo "  URL:       http://localhost:$PORT"
echo "  Context:   ${EFFECTIVE_CONTEXT_SIZE} tokens | Batch: ${BATCH_SIZE}"
echo "  KV Cache:  K=${CACHE_TYPE_K} | V=${CACHE_TYPE_V}"
echo "  KV Policy: min(${KV_VRAM_FRACTION} * free VRAM, free - reserve - model - runtime) | ~${KV_MB_PER_TOKEN} MiB/token"
echo "  Metrics:   llama.cpp metrics enabled"
echo ""
echo "  ── GPU Resource Allocation ──"
if [ "$TOTAL_VRAM" != "N/A" ]; then
    echo "  Total:     ${TOTAL_VRAM}MB"
    echo "  Used:      ${USED_VRAM}MB (by other processes)"
    echo "  Free:      ${FREE_VRAM}MB"
    echo "  Reserved:  ${BREATHING_ROOM}MB (strict buffer via llama.cpp)"
    echo "  Model Est: ${MODEL_FOOTPRINT_MB}MB"
    echo "  Runtime:   ${RUNTIME_OVERHEAD_MB}MB"
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
    -c "$EFFECTIVE_CONTEXT_SIZE" \
    --fit on \
    --fit-target "$BREATHING_ROOM" \
    -b "$BATCH_SIZE" \
    --cache-ram 2048 \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
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
