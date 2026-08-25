#!/usr/bin/env bash
# PIPER — Launcher (Linux/macOS)
# Starts the backend and frontend together.
#
# Usage:  bash run.sh
# Stop:   Ctrl+C

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail() { echo -e "${RED}ERROR:${NC} $1"; exit 1; }

# Verify .venv exists
if [ ! -f ".venv/bin/python" ]; then
    fail ".venv not found. Run:  bash setup.sh"
fi

# Verify frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    fail "frontend/node_modules not found. Run:  bash setup.sh"
fi

# Create required directories
mkdir -p data artifacts

echo
echo "Starting PIPER..."
echo "  Backend:   http://localhost:8000  (API docs: http://localhost:8000/docs)"
echo "  Frontend:  http://localhost:5173"
echo "  Stop:      Ctrl+C"
echo

# Trap Ctrl+C to kill both processes
cleanup() {
    echo
    echo "Stopping PIPER..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# Start backend
(cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000) &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend
(cd frontend && npm run dev -- --host 127.0.0.1) &
FRONTEND_PID=$!

echo -e "${GREEN}PIPER running.${NC} Press Ctrl+C to stop."

wait "$BACKEND_PID" "$FRONTEND_PID"