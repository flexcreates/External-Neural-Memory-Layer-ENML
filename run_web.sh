#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Web Chat UI Server Startup
# ═══════════════════════════════════════════════════════════════════════
# Starts the ENML web server with the full memory pipeline.
# Access at http://localhost:5000 (or WEB_SERVER_PORT in .env)
#
# Prerequisites:
#   - Qdrant running (./run_qdrant.sh)
#   - llama-server running (./run_server.sh)
#   - Python venv activated with Flask installed
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

PORT="${WEB_SERVER_PORT:-$(read_env_value WEB_SERVER_PORT || printf '5000')}"
QDRANT_URL="${QDRANT_URL:-$(read_env_value QDRANT_URL || printf 'http://localhost:6333')}"
LLAMA_URL="${LLAMA_SERVER_URL:-$(read_env_value LLAMA_SERVER_URL || printf 'http://localhost:8080')}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "✗ Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/web_server.py" ]; then
    echo "✗ web_server.py not found in $SCRIPT_DIR"
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf "${QDRANT_URL%/}/health" >/dev/null 2>&1; then
        echo "⚠ Qdrant does not appear reachable at ${QDRANT_URL%/}"
    fi

    if ! curl -sf "${LLAMA_URL%/}/health" >/dev/null 2>&1 && ! curl -sf "${LLAMA_URL%/}/v1/models" >/dev/null 2>&1; then
        echo "⚠ llama-server does not appear reachable at ${LLAMA_URL%/}"
        echo "  ENML expects an OpenAI-compatible API at ${LLAMA_URL%/}/v1"
    fi
fi

ACTIVE_MODEL=""
if command -v curl >/dev/null 2>&1; then
    models_json="$(curl -sf "${LLAMA_URL%/}/v1/models" 2>/dev/null || true)"
    if [ -n "$models_json" ]; then
        ACTIVE_MODEL="$("$PYTHON_BIN" -c 'import json,sys
try:
    payload = json.loads(sys.stdin.read())
    data = payload.get("data") or []
    model_id = data[0].get("id", "") if data else ""
    print(model_id, end="")
except Exception:
    pass' <<< "$models_json")"
    fi
fi

if [ -n "$ACTIVE_MODEL" ]; then
    if [ "${DEFAULT_CHAT_MODEL:-}" != "$ACTIVE_MODEL" ] || \
       [ "${FAST_CHAT_MODEL:-}" != "$ACTIVE_MODEL" ] || \
       [ "${CODING_CHAT_MODEL:-}" != "$ACTIVE_MODEL" ] || \
       [ "${REASONING_CHAT_MODEL:-}" != "$ACTIVE_MODEL" ]; then
        echo "ℹ Syncing ENML chat model routing to active server model: $ACTIVE_MODEL"
    fi
    export DEFAULT_CHAT_MODEL="$ACTIVE_MODEL"
    export FAST_CHAT_MODEL="$ACTIVE_MODEL"
    export CODING_CHAT_MODEL="$ACTIVE_MODEL"
    export REASONING_CHAT_MODEL="$ACTIVE_MODEL"
fi

export PYTHONUNBUFFERED=1

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Web Chat Server                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  URL:      http://localhost:$PORT"
echo "  Python:   $PYTHON_BIN"
echo "  LLM API:  ${LLAMA_URL%/}/v1"
if [ -n "$ACTIVE_MODEL" ]; then
    echo "  Model:    $ACTIVE_MODEL"
fi
echo "  Qdrant:   ${QDRANT_URL%/}"
echo "  Pipeline: ENML memory + evidence grounding + runtime metrics"
echo "  Debug:    /api/debug/retrieve /api/debug/runtime-metrics /api/debug/citation-metrics"
echo ""

"$PYTHON_BIN" web_server.py
