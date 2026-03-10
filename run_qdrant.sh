#!/usr/bin/env bash

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

QDRANT_URL="${QDRANT_URL:-$(read_env_value QDRANT_URL || printf 'http://localhost:6333')}"
PORT="${QDRANT_URL##*:}"
PORT="${PORT%%/*}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    PORT="6333"
fi

CONTAINER_NAME="enml-qdrant"
STORAGE_DIR="$SCRIPT_DIR/qdrant_storage"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run Qdrant."
    exit 1
fi

DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    else
        echo "Docker is installed but not accessible for the current user."
        exit 1
    fi
fi

mkdir -p "$STORAGE_DIR"

echo "=== ENML Qdrant Startup ==="

if $DOCKER_CMD ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    if $DOCKER_CMD ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
        echo "Qdrant already running at ${QDRANT_URL}"
        exit 0
    fi
    echo "Starting existing Qdrant container..."
    $DOCKER_CMD start "$CONTAINER_NAME" >/dev/null
    echo "Qdrant started at ${QDRANT_URL}"
    exit 0
fi

if command -v lsof >/dev/null 2>&1 && lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "Port $PORT is already in use."
    exit 1
fi

echo "Starting new Qdrant container..."
$DOCKER_CMD run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:6333" \
  -v "$STORAGE_DIR:/qdrant/storage" \
  --restart unless-stopped \
  qdrant/qdrant >/dev/null

echo "Qdrant running at ${QDRANT_URL}"
