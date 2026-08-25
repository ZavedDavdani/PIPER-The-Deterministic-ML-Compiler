#!/usr/bin/env bash
# PIPER — Setup Script (Linux/macOS)
# Run once to prepare a fresh machine for local development.
#
# Usage:  bash setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC}  $1"; }
fail() { echo -e "  ${RED}✗${NC}  $1"; exit 1; }
warn() { echo -e "  ${YELLOW}!${NC}  $1"; }
info() { echo -e "     $1"; }

echo
echo "PIPER SETUP"
echo "────────────────────────────────────────────"

# ── 1. Python 3.11+ ──────────────────────────────────────────────────────────
PYTHON=$(command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null)
if [ -z "$PYTHON" ]; then
    fail "Python not found. Install Python 3.11+ from https://python.org"
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    fail "Python $PY_VERSION found, but 3.11+ is required."
fi
ok "Python $PY_VERSION"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ -d ".venv" ]; then
    ok ".venv already exists"
else
    info "Creating .venv ..."
    "$PYTHON" -m venv .venv
    ok ".venv created"
fi

# ── 3. Install Python dependencies ───────────────────────────────────────────
info "Installing Python dependencies (requirements.txt) ..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── 4. Create required directories ───────────────────────────────────────────
mkdir -p data artifacts
ok "Directories: data/, artifacts/"

# ── 5. Node.js / npm ─────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    fail "Node.js not found. Install Node.js 20+ from https://nodejs.org"
fi
NODE_VERSION=$(node --version)
ok "Node.js $NODE_VERSION"

if ! command -v npm &>/dev/null; then
    fail "npm not found. Install npm (bundled with Node.js)."
fi

# ── 6. Frontend dependencies ──────────────────────────────────────────────────
info "Installing frontend dependencies (npm install) ..."
(cd frontend && npm install --silent)
ok "Frontend dependencies installed"

# ── 7. Ollama check ──────────────────────────────────────────────────────────
OLLAMA_HOST="${PIPER_OLLAMA_HOST:-http://localhost:11434}"
LLM_MODEL="${PIPER_LLM_MODEL:-qwen3:4b}"

if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    ok "Ollama reachable at $OLLAMA_HOST"

    # Check if the model is pulled
    TAGS=$(curl -sf "$OLLAMA_HOST/api/tags" 2>/dev/null)
    if echo "$TAGS" | grep -q "\"name\":\"$LLM_MODEL\""; then
        ok "Model $LLM_MODEL is available"
    else
        warn "Model $LLM_MODEL is not pulled yet"
        info "Run:  ollama pull $LLM_MODEL"
    fi
else
    warn "Ollama not reachable at $OLLAMA_HOST"
    info "Install Ollama from https://ollama.com, start it, then run:"
    info "  ollama pull $LLM_MODEL"
fi

echo
echo -e "${GREEN}Setup complete.${NC}"
echo
echo "  Run the system check:    python check.py"
echo "  Start PIPER:             bash run.sh"
echo