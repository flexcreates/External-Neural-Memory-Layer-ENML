#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Qdrant Vector Database Startup Script
# ═══════════════════════════════════════════════════════════════════════
# Manages the Qdrant Docker container lifecycle.
# All configuration is read from .env
# ═══════════════════════════════════════════════════════════════════════

set -e

# Load .env configuration
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Parse port from QDRANT_URL
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
PORT=$(echo "$QDRANT_URL" | grep -oP ':\K[0-9]+$' || echo "6333")

CONTAINER_NAME="enml-qdrant"
STORAGE_DIR="$(pwd)/qdrant_storage"

echo "=== ENML Qdrant Startup ==="

# 1. Create storage directory if missing
if [ ! -d "$STORAGE_DIR" ]; then
    echo "[Info] Creating storage at $STORAGE_DIR"
    mkdir -p "$STORAGE_DIR"
fi

# 2. Check if container already exists
if sudo docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    # Check if running
    if sudo docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
        echo "[✓] Qdrant already running on port $PORT"
        exit 0
    else
        echo "[Info] Container stopped. Starting..."
        sudo docker start "$CONTAINER_NAME"
        echo "[✓] Qdrant started on port $PORT"
        exit 0
    fi
fi

# 3. Handle port conflicts
if command -v lsof &>/dev/null && sudo lsof -i :"$PORT" > /dev/null 2>&1; then
    echo "[⚠] Port $PORT in use."
    CONFLICTING=$(sudo docker ps -q --filter "publish=$PORT" 2>/dev/null)
    if [ -n "$CONFLICTING" ]; then
        echo "[Info] Stopping conflicting container: $CONFLICTING"
        sudo docker stop "$CONFLICTING"
        sudo docker rm "$CONFLICTING"
    else
        echo "[✗] Port $PORT used by non-Docker process. Free it manually."
        exit 1
    fi
fi

# 4. Start fresh container
echo "[Info] Starting Qdrant container..."
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:6333" \
  -v "$STORAGE_DIR:/qdrant/storage" \
  --restart unless-stopped \
  qdrant/qdrant

echo "[✓] Qdrant running at $QDRANT_URL"
