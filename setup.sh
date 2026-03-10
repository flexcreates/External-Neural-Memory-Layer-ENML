#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

read_env_value() {
    local key="$1"
    local env_file="${2:-$SCRIPT_DIR/.env}"
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

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    ENML Setup                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BOLD}[1/7] Checking prerequisites...${NC}"

if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Python 3 is required.${NC}"
    exit 1
fi
echo -e "  ${GREEN}Python:${NC} $(python3 --version 2>&1)"

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo -e "${RED}python3-venv is required.${NC}"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo -e "${YELLOW}Git not found. This is not fatal, but recommended.${NC}"
fi

DOCKER_AVAILABLE=false
if command -v docker >/dev/null 2>&1; then
    DOCKER_AVAILABLE=true
    echo -e "  ${GREEN}Docker:${NC} $(docker --version 2>&1 | head -n 1)"
else
    echo -e "${YELLOW}Docker not found. Qdrant startup will be skipped.${NC}"
fi

echo -e "\n${BOLD}[2/7] Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  ${GREEN}Created .venv${NC}"
else
    echo -e "  ${GREEN}.venv already exists${NC}"
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
PIP_BIN="$SCRIPT_DIR/.venv/bin/pip"

echo -e "\n${BOLD}[3/7] Installing Python dependencies...${NC}"
"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install -r requirements.txt
echo -e "  ${GREEN}Dependencies installed${NC}"

echo -e "\n${BOLD}[4/7] Preparing environment file...${NC}"
if [ ! -f ".env" ]; then
    if [ ! -f ".env.example" ]; then
        echo -e "${RED}.env.example is missing.${NC}"
        exit 1
    fi
    cp .env.example .env
    echo -e "  ${GREEN}Created .env from .env.example${NC}"
else
    echo -e "  ${GREEN}.env already exists; keeping current values${NC}"
fi

MEMORY_ROOT="$(read_env_value MEMORY_ROOT "$SCRIPT_DIR/.env" || printf '%s' "$SCRIPT_DIR/memory")"
AI_NAME="$(read_env_value AI_NAME "$SCRIPT_DIR/.env" || printf 'ENML Assistant')"

echo -e "\n${BOLD}[5/7] Creating runtime directories...${NC}"
dirs=(
    "$MEMORY_ROOT"
    "$MEMORY_ROOT/conversations"
    "$MEMORY_ROOT/projects"
    "$MEMORY_ROOT/research"
    "$MEMORY_ROOT/authority"
    "$MEMORY_ROOT/graph"
    "$SCRIPT_DIR/logs"
    "$SCRIPT_DIR/qdrant_storage"
    "$SCRIPT_DIR/graph"
)

for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done
echo -e "  ${GREEN}Directory structure ready${NC}"

echo -e "\n${BOLD}[6/7] Initializing authority memory...${NC}"
PROFILE_FILE="$MEMORY_ROOT/authority/profile.json"
if [ ! -f "$PROFILE_FILE" ]; then
    printf '{\n  "user": {\n    "name": null,\n    "age": null,\n    "preferences": {}\n  },\n  "assistant": {\n    "name": "%s"\n  },\n  "system": {}\n}\n' "$AI_NAME" > "$PROFILE_FILE"
    echo -e "  ${GREEN}Created authority profile${NC}"
else
    echo -e "  ${GREEN}Authority profile already exists${NC}"
fi

echo -e "\n${BOLD}[7/7] Starting optional services...${NC}"
if [ "$DOCKER_AVAILABLE" = true ]; then
    chmod +x run_qdrant.sh 2>/dev/null || true
    if ./run_qdrant.sh; then
        echo -e "  ${GREEN}Qdrant ready${NC}"
    else
        echo -e "  ${YELLOW}Qdrant did not start automatically. You can run ./run_qdrant.sh later.${NC}"
    fi
else
    echo -e "  ${YELLOW}Skipped Qdrant startup because Docker is unavailable.${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}Setup complete.${NC}"
echo ""
echo "Next steps:"
echo "  1. Review .env"
echo "  2. Start the model server with ./run_server.sh"
echo "  3. Start ENML with source .venv/bin/activate && python3 chat.py"
echo "  4. Or start the web UI with ./run_web.sh"
