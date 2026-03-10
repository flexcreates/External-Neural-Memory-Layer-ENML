#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# ENML — Reset Memory Script
# ═══════════════════════════════════════════════════════════════════════
# Clears all ENML memory: conversations, authority profile, Qdrant
# collections, graph data, and caches.
#
# All paths and collection names are read from .env
# ═══════════════════════════════════════════════════════════════════════

set -e

# Load .env configuration
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration from .env (with defaults)
MEMORY_ROOT="${MEMORY_ROOT:-./memory}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
AI_NAME="${AI_NAME:-ENML Assistant}"

QDRANT_KNOWLEDGE="${QDRANT_KNOWLEDGE_COLLECTION:-knowledge_collection}"
QDRANT_CONVERSATION="${QDRANT_CONVERSATION_COLLECTION:-conversation_collection}"
QDRANT_RESEARCH="${QDRANT_RESEARCH_COLLECTION:-research_collection}"
QDRANT_PROJECT="${QDRANT_PROJECT_COLLECTION:-project_collection}"
QDRANT_PROFILE="${QDRANT_PROFILE_COLLECTION:-profile_collection}"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          ⚠️  ENML Memory Reset                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  This will permanently delete:"
echo "  • ${MEMORY_ROOT}/conversations/*"
echo "  • ${MEMORY_ROOT}/projects/*"
echo "  • ${MEMORY_ROOT}/research/*"
echo "  • ${MEMORY_ROOT}/authority/profile.json"
echo "  • ${MEMORY_ROOT}/graph/*"
echo "  • Qdrant Docker container and all local Qdrant storage files"
echo ""
read -p "Are you sure you want to proceed? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${BOLD}Clearing memory...${NC}"

# Clear conversation history
if [ -d "${MEMORY_ROOT}/conversations" ]; then
    rm -rf "${MEMORY_ROOT}/conversations"/*
    echo -e "  ${GREEN}✓${NC} Cleared conversations"
fi

# Clear projects
if [ -d "${MEMORY_ROOT}/projects" ]; then
    rm -rf "${MEMORY_ROOT}/projects"/*
    echo -e "  ${GREEN}✓${NC} Cleared projects"
fi

# Clear research
if [ -d "${MEMORY_ROOT}/research" ]; then
    rm -rf "${MEMORY_ROOT}/research"/*
    echo -e "  ${GREEN}✓${NC} Cleared research data"
fi

# Clear graph data
if [ -d "${MEMORY_ROOT}/graph" ]; then
    rm -rf "${MEMORY_ROOT}/graph"/*
    echo -e "  ${GREEN}✓${NC} Cleared graph data"
fi

# Reset authority profile (preserves AI name from .env)
AUTHORITY_DIR="${MEMORY_ROOT}/authority"
PROFILE_FILE="${AUTHORITY_DIR}/profile.json"
mkdir -p "$AUTHORITY_DIR"
cat > "$PROFILE_FILE" << EOJSON
{
  "user": {},
  "assistant": {
    "name": "${AI_NAME}"
  },
  "system": {}
}
EOJSON
echo -e "  ${GREEN}✓${NC} Reset authority profile (AI Name: ${AI_NAME})"

# Hard reset Qdrant storage
echo -e "\n${BOLD}Hard-resetting Qdrant vector database...${NC}"
CONTAINER_NAME="enml-qdrant"
if sudo docker ps -a --format '{{.Names}}' 2>/dev/null | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "  [Info] Stopping and removing Qdrant container..."
    sudo docker stop "$CONTAINER_NAME" > /dev/null
    sudo docker rm "$CONTAINER_NAME" > /dev/null
    echo -e "  ${GREEN}✓${NC} Removed Qdrant container"
fi

if [ -d "qdrant_storage" ]; then
    sudo rm -rf qdrant_storage/* 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Deleted all Qdrant local storage data"
fi

# Clean Python caches
find . -not -path "*/.venv/*" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Cleared Python caches"

echo ""
echo -e "${GREEN}${BOLD}✨ Memory reset complete. ENML is fresh and ready to learn.${NC}"
echo ""
