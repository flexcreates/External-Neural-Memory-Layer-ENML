#!/bin/bash
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

set -e

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

PORT="${WEB_SERVER_PORT:-5000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv if available
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf "http://localhost:6333/health" >/dev/null 2>&1; then
        echo "⚠ Qdrant does not appear reachable at http://localhost:6333"
    fi
    if ! curl -sf "${LLAMA_SERVER_URL:-http://localhost:8080}/health" >/dev/null 2>&1; then
        echo "⚠ llama-server does not appear reachable at ${LLAMA_SERVER_URL:-http://localhost:8080}"
    fi
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Web Chat Server                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  URL:      http://localhost:$PORT"
echo "  Pipeline: ENML memory + evidence grounding + runtime metrics"
echo "  Debug:    /api/debug/retrieve /api/debug/runtime-metrics /api/debug/citation-metrics"
echo ""

cd "$SCRIPT_DIR"
python3 web_server.py
