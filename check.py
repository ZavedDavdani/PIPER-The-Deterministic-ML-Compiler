#!/usr/bin/env python3
# PIPER System Check -- preflight readiness verification.
#
# Run this before starting PIPER for the first time, or to diagnose issues:
#
#     python check.py
#
# Checks:
#   - Python version (3.11+)
#   - Required Python dependencies installed
#   - data/ and artifacts/ directories exist (creates them if missing)
#   - Ollama server reachable
#   - Configured planner model available in Ollama
#   - SQLite database path writable
#
# No arguments required. Reads the same environment variables as PIPER:
#   PIPER_OLLAMA_HOST, PIPER_LLM_MODEL, PIPER_SQLITE_PATH, PIPER_ARTIFACT_DIR.

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# -- Configuration (mirrors app defaults) -------------------------------------

OLLAMA_HOST  = os.environ.get("PIPER_OLLAMA_HOST",  "http://localhost:11434")
LLM_MODEL    = os.environ.get("PIPER_LLM_MODEL",    "qwen3:4b")
SQLITE_PATH  = os.environ.get("PIPER_SQLITE_PATH",  "data/piper_runs.sqlite")
ARTIFACT_DIR = os.environ.get("PIPER_ARTIFACT_DIR", "artifacts")

# -- Helpers ------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _ok(label, detail=""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {GREEN}OK{RESET}  {label:<28}{suffix}")

def _fail(label, detail=""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {RED}FAIL{RESET}  {label:<26}{suffix}")

def _warn(label, detail=""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {YELLOW}WARN{RESET}  {label:<26}{suffix}")

def _check_python():
    major, minor = sys.version_info.major, sys.version_info.minor
    v = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        _ok("Python version", v)
        return True
    _fail("Python version", f"{v}  (need 3.11+)")
    return False

def _check_deps():
    required = [
        ("pandas",    "pandas"),
        ("sklearn",   "scikit-learn"),
        ("pydantic",  "pydantic"),
        ("fastapi",   "fastapi"),
        ("uvicorn",   "uvicorn"),
        ("langgraph", "langgraph"),
        ("joblib",    "joblib"),
        ("openpyxl",  "openpyxl"),
        ("pyarrow",   "pyarrow"),
        ("httpx",     "httpx"),
    ]
    missing = []
    for module, pkg in required:
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)
    if missing:
        _fail("Python dependencies", "missing: " + ", ".join(missing))
        print(f"       Run:  pip install -r requirements.txt")
        return False
    _ok("Python dependencies", "all required packages found")
    return True

def _check_directories():
    for label, p in [("data/", Path("data")), ("artifacts/", Path(ARTIFACT_DIR))]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            _ok(f"Directory {label}")
        except Exception as exc:
            _fail(f"Directory {label}", str(exc))

def _check_ollama():
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        models = [item["name"] for item in (body.get("models") or []) if item.get("name")]
        _ok("Ollama server", OLLAMA_HOST)
        return True, models
    except urllib.error.URLError as e:
        _fail("Ollama server", f"Cannot reach {OLLAMA_HOST}: {e.reason}")
        print(f"       Fix:  install Ollama from https://ollama.com and run: ollama serve")
        return False, []
    except Exception as e:
        _fail("Ollama server", str(e))
        return False, []

def _check_model(models):
    def _base(m):
        return m.split(":")[0]
    wanted_base = _base(LLM_MODEL)
    matched = [m for m in models if m == LLM_MODEL or _base(m) == wanted_base]
    if matched:
        _ok("Planner model", f"{matched[0]}  (PIPER_LLM_MODEL={LLM_MODEL})")
        return True
    _fail("Planner model", f"{LLM_MODEL} not found in Ollama")
    available = ", ".join(models[:6]) or "(none)"
    print(f"       Available:  {available}")
    print(f"       Fix:        ollama pull {LLM_MODEL}")
    return False

def _check_sqlite():
    path = Path(SQLITE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / ".piper_write_test"
        tmp.write_text("ok")
        tmp.unlink()
        _ok("SQLite database", str(path))
        return True
    except Exception as exc:
        _fail("SQLite database", str(exc))
        return False

# -- Main ---------------------------------------------------------------------

def main():
    print()
    print(f"{BOLD}PIPER SYSTEM CHECK{RESET}")
    print("-" * 48)

    failures = 0

    if not _check_python():
        failures += 1
    if not _check_deps():
        failures += 1

    _check_directories()

    ollama_ok, models = _check_ollama()
    if not ollama_ok:
        failures += 1
        _warn("Planner model", "skipped (Ollama unreachable)")
    else:
        if not _check_model(models):
            failures += 1

    if not _check_sqlite():
        failures += 1

    print()
    if failures == 0:
        print(f"{BOLD}{GREEN}PIPER READY{RESET}")
        print()
        print("  Start:  bash run.sh        (Linux/macOS)")
        print("          .\\run.ps1          (Windows)")
        print()
        return 0
    else:
        print(f"{BOLD}{RED}PIPER NOT READY -- {failures} issue(s) above must be resolved.{RESET}")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
