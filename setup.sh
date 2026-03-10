#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — External Neural Memory Layer · Setup Script
# ═══════════════════════════════════════════════════════════════════════
# This script initializes the complete ENML environment:
#   1. Creates Python virtual environment
#   2. Installs dependencies
#   3. Generates .env from template
#   4. Creates required directory structure
#   5. Initializes authority memory profile
#   6. Starts Qdrant vector database
#
# Usage: chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════════════════════════════════

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      ENML — External Neural Memory Layer · Setup          ║"
echo "║      Infinite Learning for Local AI Systems               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Check System Requirements ─────────────────────────────────
echo -e "${BOLD}[1/7] Checking system requirements...${NC}"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ Python 3 is required but not installed.${NC}"
    echo "  Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python 3 found: $(python3 --version)"

if ! command -v docker &>/dev/null; then
    echo -e "${YELLOW}⚠ Docker not found. Qdrant vector DB requires Docker.${NC}"
    echo "  Install: https://docs.docker.com/engine/install/"
    echo "  You can continue setup without Docker and install it later."
    DOCKER_AVAILABLE=false
else
    echo -e "  ${GREEN}✓${NC} Docker found: $(docker --version | head -c 40)"
    DOCKER_AVAILABLE=true
fi

# ── Step 2: Create Virtual Environment ─────────────────────────────────
echo -e "\n${BOLD}[2/7] Setting up Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  ${GREEN}✓${NC} Created .venv"
else
    echo -e "  ${GREEN}✓${NC} .venv already exists"
fi

# ── Step 3: Install Dependencies ───────────────────────────────────────
echo -e "\n${BOLD}[3/7] Installing Python dependencies...${NC}"
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
echo -e "  ${GREEN}✓${NC} All dependencies installed"

# ── Step 4: Generate .env Configuration ────────────────────────────────
echo -e "\n${BOLD}[4/7] Configuring environment...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "  ${GREEN}✓${NC} Created .env from .env.example"
    else
        echo -e "  ${RED}✗${NC} .env.example not found!"
        exit 1
    fi
else
    echo -e "  ${GREEN}✓${NC} .env already exists (keeping your current config)"
fi

# Source .env to use its values for directory creation
set -a
source .env
set +a

# ── Step 5: Create Directory Structure ─────────────────────────────────
echo -e "\n${BOLD}[5/7] Creating directory structure...${NC}"

MEMORY_ROOT="${MEMORY_ROOT:-./memory}"
dirs=(
    "${MEMORY_ROOT}/conversations"
    "${MEMORY_ROOT}/projects"
    "${MEMORY_ROOT}/research"
    "${MEMORY_ROOT}/authority"
    "${MEMORY_ROOT}/graph"
    "./logs"
    "./qdrant_storage"
    "./graph"
)

for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done
echo -e "  ${GREEN}✓${NC} All directories created"

# ── Step 6: Initialize Authority Memory Profile ────────────────────────
echo -e "\n${BOLD}[6/7] Initializing authority memory profile...${NC}"

AI_NAME="${AI_NAME:-ENML Assistant}"
PROFILE_FILE="${MEMORY_ROOT}/authority/profile.json"

if [ ! -f "$PROFILE_FILE" ]; then
    cat > "$PROFILE_FILE" << EOJSON
{
  "identity": {},
  "assistant": {
    "name": "${AI_NAME}"
  },
  "system": {}
}
EOJSON
    echo -e "  ${GREEN}✓${NC} Created profile.json (AI Name: ${AI_NAME})"
else
    echo -e "  ${GREEN}✓${NC} profile.json already exists"
fi

# ── Step 7: Start Qdrant ───────────────────────────────────────────────
echo -e "\n${BOLD}[7/7] Starting Qdrant vector database...${NC}"
if [ "$DOCKER_AVAILABLE" = true ]; then
    chmod +x run_qdrant.sh 2>/dev/null || true
    if ./run_qdrant.sh 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Qdrant is running"
    else
        echo -e "  ${YELLOW}⚠${NC} Qdrant startup had issues (you can start it manually later)"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Skipping (Docker not available)"
fi

# ── Setup Complete ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║              ✅  ENML Setup Complete!                      ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}${YELLOW}  ⚡ IMPORTANT: Update your .env file before first run!${NC}"
echo ""
echo -e "  ${CYAN}Required settings to configure in .env:${NC}"
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  MODELS_DIR      = /home/flex/ai-models                 │"
echo "  │  LLAMA_SERVER    = /home/flex/Tools/llama.cpp/build/bin/llama-server │"
echo "  │  ALLOWED_PATHS   = /your/project/dirs (comma-separated) │"
echo "  │  AI_NAME         = Your preferred AI assistant name      │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo "  1. Edit .env with your paths:     nano .env"
echo "  2. Start Llama server:            ./run_server.sh"
echo "  3. Start ENML chat:               source .venv/bin/activate && python3 chat.py"
echo ""
echo -e "  ${BOLD}Other Commands:${NC}"
echo "  • Reset all memory:               ./reset_memory.sh"
echo "  • Start Qdrant only:              ./run_qdrant.sh"
echo ""
