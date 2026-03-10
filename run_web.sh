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

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENML — Web Chat Server                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  URL:      http://localhost:$PORT"
echo "  Pipeline: Full ENML (extraction + memory + knowledge graph)"
echo ""

cd "$SCRIPT_DIR"
python web_server.py
