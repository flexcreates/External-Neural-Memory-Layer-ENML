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
FIT_CONTEXT_MIN="${FIT_CONTEXT_MIN:-$(read_env_value FIT_CONTEXT_MIN || printf '1024')}"
FIT_CONTEXT_STEP="${FIT_CONTEXT_STEP:-$(read_env_value FIT_CONTEXT_STEP || printf '256')}"
FIT_TARGET_MB="${FIT_TARGET_MB:-$(read_env_value FIT_TARGET_MB || printf '512')}"
BATCH_SIZE=512
CACHE_TYPE_K="${CACHE_TYPE_K:-$(read_env_value CACHE_TYPE_K || printf 'q8_0')}"
CACHE_TYPE_V="${CACHE_TYPE_V:-$(read_env_value CACHE_TYPE_V || printf 'q8_0')}"
LLAMA_FIT_PARAMS="${LLAMA_FIT_PARAMS:-$(read_env_value LLAMA_FIT_PARAMS || printf '%s/llama-fit-params' "$(dirname "$LLAMA_SERVER")")}"

setup_llama_runtime_libs() {
    local llama_dir
    llama_dir="$(dirname "$LLAMA_SERVER")"
    export LD_LIBRARY_PATH="${llama_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    local nvidia_venv_dir
    nvidia_venv_dir=$(find "$SCRIPT_DIR/.venv" -type d -name "nvidia" -path "*/site-packages/nvidia" -print -quit 2>/dev/null || true)
    if [ -n "$nvidia_venv_dir" ]; then
        local dir
        for dir in "$nvidia_venv_dir"/*/lib; do
            export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        done
    fi
}

get_model_max_context() {
    "$PYTHON_BIN" - "$MODEL_PATH" <<'PY'
import struct
import sys

TYPE_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of file")
    return data


def read_scalar(handle, fmt):
    return struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))[0]


def read_string(handle):
    length = read_scalar(handle, "<Q")
    return read_exact(handle, length).decode("utf-8", errors="ignore")


def read_value(handle, value_type):
    if value_type in TYPE_FMT:
        return read_scalar(handle, TYPE_FMT[value_type])
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        inner_type = read_scalar(handle, "<I")
        length = read_scalar(handle, "<Q")
        for _ in range(length):
            read_value(handle, inner_type)
        return None
    raise ValueError(f"unsupported GGUF value type: {value_type}")


path = sys.argv[1]
with open(path, "rb") as handle:
    magic = read_exact(handle, 4)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")

    version = read_scalar(handle, "<I")
    if version >= 2:
        _tensor_count = read_scalar(handle, "<Q")
        metadata_count = read_scalar(handle, "<Q")
    else:
        _tensor_count = read_scalar(handle, "<I")
        metadata_count = read_scalar(handle, "<I")

    for _ in range(metadata_count):
        key = read_string(handle)
        value_type = read_scalar(handle, "<I")
        value = read_value(handle, value_type)
        if key.endswith("context_length"):
            print(int(value))
            raise SystemExit(0)

raise SystemExit(1)
PY
}

fit_candidate_context() {
    local candidate_context="$1"
    local planner_output planner_line fitted_context fitted_layers

    if ! planner_output=$(
        "$LLAMA_FIT_PARAMS" \
            -m "$MODEL_PATH" \
            -c "$candidate_context" \
            --fit-target "$FIT_TARGET_MB" \
            -b "$BATCH_SIZE" \
            --cache-type-k "$CACHE_TYPE_K" \
            --cache-type-v "$CACHE_TYPE_V" \
            --flash-attn on 2>&1
    ); then
        return 1
    fi

    planner_line=$(printf '%s\n' "$planner_output" | awk '/^-c /{line=$0} END{print line}')
    if [ -z "$planner_line" ]; then
        return 1
    fi

    fitted_context=$(printf '%s\n' "$planner_line" | awk '{for (i = 1; i <= NF; i++) if ($i == "-c") {print $(i+1); exit}}')
    fitted_layers=$(printf '%s\n' "$planner_line" | awk '{for (i = 1; i <= NF; i++) if ($i == "-ngl") {print $(i+1); exit}}')
    if [ -z "$fitted_context" ] || [ -z "$fitted_layers" ]; then
        return 1
    fi

    printf '%s %s\n' "$fitted_context" "$fitted_layers"
}

select_launch_profile() {
    local model_max_context min_context step_size low_units high_units mid_units candidate
    local fit_result fitted_context fitted_layers

    if [ ! -x "$LLAMA_FIT_PARAMS" ]; then
        return 1
    fi

    model_max_context="$(get_model_max_context)"
    if ! [[ "$model_max_context" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    min_context="$FIT_CONTEXT_MIN"
    if ! [[ "$min_context" =~ ^[0-9]+$ ]] || [ "$min_context" -lt 1 ]; then
        min_context=1024
    fi
    if [ "$min_context" -gt "$model_max_context" ]; then
        min_context="$model_max_context"
    fi

    step_size="$FIT_CONTEXT_STEP"
    if ! [[ "$step_size" =~ ^[0-9]+$ ]] || [ "$step_size" -lt 1 ]; then
        step_size=256
    fi

    low_units=$(( (min_context + step_size - 1) / step_size ))
    high_units=$(( model_max_context / step_size ))
    if [ "$high_units" -lt "$low_units" ]; then
        high_units="$low_units"
    fi

    EFFECTIVE_CONTEXT_SIZE=""
    EFFECTIVE_GPU_LAYERS=""
    MODEL_MAX_CONTEXT="$model_max_context"

    while [ "$low_units" -le "$high_units" ]; do
        mid_units=$(( (low_units + high_units + 1) / 2 ))
        candidate=$(( mid_units * step_size ))

        if fit_result="$(fit_candidate_context "$candidate")"; then
            read -r fitted_context fitted_layers <<< "$fit_result"
            EFFECTIVE_CONTEXT_SIZE="$fitted_context"
            EFFECTIVE_GPU_LAYERS="$fitted_layers"
            low_units=$(( mid_units + 1 ))
        else
            high_units=$(( mid_units - 1 ))
        fi
    done

    [ -n "$EFFECTIVE_CONTEXT_SIZE" ] && [ -n "$EFFECTIVE_GPU_LAYERS" ]
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

BREATHING_ROOM="${FIT_TARGET_MB}"

setup_llama_runtime_libs

if command -v nvidia-smi &>/dev/null; then
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    USED_VRAM=$((TOTAL_VRAM - FREE_VRAM))
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")
else
    TOTAL_VRAM="N/A"
    FREE_VRAM="N/A"
    USED_VRAM="N/A"
    GPU_PROCS=""
fi

USE_PLANNED_PROFILE=0
MODEL_MAX_CONTEXT=""
EFFECTIVE_CONTEXT_SIZE=""
EFFECTIVE_GPU_LAYERS=""
if [ "$TOTAL_VRAM" != "N/A" ] && select_launch_profile; then
    USE_PLANNED_PROFILE=1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Llama.cpp Server                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Model:     ${MODEL_BASENAME}"
echo "  Alias:     ${MODEL_ALIAS}"
echo "  Template:  ${MODEL_TEMPLATE_INFO}"
echo "  URL:       http://localhost:$PORT"
if [ "$USE_PLANNED_PROFILE" -eq 1 ]; then
    echo "  Context:   ${EFFECTIVE_CONTEXT_SIZE} tokens"
    echo "  GPU Fit:   ${EFFECTIVE_GPU_LAYERS} layers on GPU"
    echo "  Planner:   llama-fit-params (max model ctx ${MODEL_MAX_CONTEXT})"
else
    echo "  Context:   fallback ${CONTEXT_SIZE} tokens"
    echo "  GPU Fit:   auto via llama.cpp --fit"
    echo "  Planner:   fallback mode"
fi
echo "  Ctx Floor: ${FIT_CONTEXT_MIN} tokens | Step: ${FIT_CONTEXT_STEP}"
echo "  KV Cache:  K=${CACHE_TYPE_K} | V=${CACHE_TYPE_V}"
echo "  Metrics:   llama.cpp metrics enabled"
echo ""
echo "  ── GPU Resource Allocation ──"
if [ "$TOTAL_VRAM" != "N/A" ]; then
    echo "  Total:     ${TOTAL_VRAM}MB"
    echo "  Used:      ${USED_VRAM}MB (by other processes)"
    echo "  Free:      ${FREE_VRAM}MB"
    echo "  Reserved:  ${BREATHING_ROOM}MB target free VRAM"
    echo "  Context:   recalculated from current GPU headroom at launch"
    echo "  Layers:    recomputed from current GPU headroom at launch"
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
    echo "  NVIDIA GPU not detected — llama.cpp will use default context loading behavior"
fi
echo ""

LLAMA_SERVER_ARGS=(
    -m "$MODEL_PATH"
    --alias "$MODEL_ALIAS"
    -b "$BATCH_SIZE"
    --cache-ram 2048
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --parallel 1
    --mlock
    --flash-attn on
    --defrag-thold 0.1
    --metrics
    --port "$PORT"
    --host "$HOST"
    --temp 0.6
    --top-k 40
    --top-p 0.9
)

if [ "$USE_PLANNED_PROFILE" -eq 1 ]; then
    LLAMA_SERVER_ARGS+=(
        -c "$EFFECTIVE_CONTEXT_SIZE"
        --gpu-layers "$EFFECTIVE_GPU_LAYERS"
        --fit off
    )
else
    LLAMA_SERVER_ARGS+=(
        -c "$CONTEXT_SIZE"
        --fit on
        --fit-target "$BREATHING_ROOM"
        --fit-ctx "$FIT_CONTEXT_MIN"
    )
fi

"$LLAMA_SERVER" "${LLAMA_SERVER_ARGS[@]}"
