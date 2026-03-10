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

run_docker() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        docker "$@"
        return
    fi
    if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
        sudo docker "$@"
        return
    fi
    return 1
}

MEMORY_ROOT="${MEMORY_ROOT:-$(read_env_value MEMORY_ROOT || printf '%s' "$SCRIPT_DIR/memory")}"
AI_NAME="${AI_NAME:-$(read_env_value AI_NAME || printf 'ENML Assistant')}"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  ENML Memory Reset                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "This will permanently delete:"
echo "  - ${MEMORY_ROOT}/conversations/*"
echo "  - ${MEMORY_ROOT}/projects/*"
echo "  - ${MEMORY_ROOT}/research/*"
echo "  - ${MEMORY_ROOT}/graph/*"
echo "  - ${MEMORY_ROOT}/records/*"
echo "  - ${MEMORY_ROOT}/authority/profile.json"
echo "  - ./qdrant_storage/*"
echo "  - ./logs/*"
echo "  - local Python caches"
echo "  - the Qdrant Docker container named enml-qdrant, if present"
echo ""
read -r -p "Are you sure you want to proceed? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${BOLD}Clearing ENML state...${NC}"

clear_dir() {
    local dir="$1"
    local label="$2"
    if [ -d "$dir" ]; then
        find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        echo -e "  ${GREEN}${label}${NC}"
    fi
}

clear_dir "${MEMORY_ROOT}/conversations" "Cleared conversations"
clear_dir "${MEMORY_ROOT}/projects" "Cleared projects"
clear_dir "${MEMORY_ROOT}/research" "Cleared research data"
clear_dir "${MEMORY_ROOT}/graph" "Cleared graph data"
clear_dir "${MEMORY_ROOT}/records" "Cleared memory records"
clear_dir "$SCRIPT_DIR/logs" "Cleared logs"

AUTHORITY_DIR="${MEMORY_ROOT}/authority"
PROFILE_FILE="${AUTHORITY_DIR}/profile.json"
mkdir -p "$AUTHORITY_DIR"
printf '{\n  "user": {\n    "name": null,\n    "age": null,\n    "preferences": {}\n  },\n  "assistant": {\n    "name": "%s"\n  },\n  "system": {}\n}\n' "$AI_NAME" > "$PROFILE_FILE"
echo -e "  ${GREEN}Reset authority profile${NC}"

echo -e "\n${BOLD}Resetting Qdrant storage...${NC}"
CONTAINER_NAME="enml-qdrant"
if run_docker ps -a --format '{{.Names}}' 2>/dev/null | grep -Eq "^${CONTAINER_NAME}\$"; then
    run_docker stop "$CONTAINER_NAME" >/dev/null || true
    run_docker rm "$CONTAINER_NAME" >/dev/null || true
    echo -e "  ${GREEN}Removed Qdrant container${NC}"
else
    echo -e "  ${YELLOW}Qdrant container not present${NC}"
fi

clear_dir "$SCRIPT_DIR/qdrant_storage" "Deleted local Qdrant storage"

find "$SCRIPT_DIR" -not -path "*/.venv/*" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -not -path "*/.venv/*" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
echo -e "  ${GREEN}Cleared Python caches${NC}"

echo ""
echo -e "${GREEN}${BOLD}ENML memory reset complete.${NC}"
echo "The next run will start from a clean local state."
