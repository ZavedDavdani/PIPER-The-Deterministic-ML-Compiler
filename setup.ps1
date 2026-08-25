# PIPER — Setup Script (Windows PowerShell)
# Run once to prepare a fresh Windows machine for local development.
#
# Usage:  .\setup.ps1
# If you see an execution-policy error:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"

function ok   { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green }
function fail { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red; exit 1 }
function warn { param($msg) Write-Host "  [!]   $msg" -ForegroundColor Yellow }
function info { param($msg) Write-Host "        $msg" }

Write-Host ""
Write-Host "PIPER SETUP" -ForegroundColor Cyan
Write-Host ("─" * 48)

# ── 1. Python 3.11+ ───────────────────────────────────────────────────────────
$python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $python) {
    fail "Python not found. Install Python 3.11+ from https://python.org"
}

$pyver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>&1
$parts = $pyver.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 11)) {
    fail "Python $pyver found, but 3.11+ is required."
}
ok "Python $pyver"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if (Test-Path ".venv\Scripts\python.exe") {
    ok ".venv already exists"
} else {
    info "Creating .venv ..."
    python -m venv .venv
    ok ".venv created"
}

# ── 3. Install Python dependencies ───────────────────────────────────────────
info "Installing Python dependencies (requirements.txt) ..."
& ".venv\Scripts\pip.exe" install --quiet --upgrade pip
& ".venv\Scripts\pip.exe" install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── 4. Create required directories ───────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "data"     | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null
ok "Directories: data\, artifacts\"

# ── 5. Node.js / npm ──────────────────────────────────────────────────────────
$node = (Get-Command node -ErrorAction SilentlyContinue)?.Source
if (-not $node) {
    fail "Node.js not found. Install Node.js 20+ from https://nodejs.org"
}
$nodeVer = & node --version
ok "Node.js $nodeVer"

# ── 6. Frontend dependencies ──────────────────────────────────────────────────
info "Installing frontend dependencies (npm install) ..."
Push-Location frontend
npm install --silent
Pop-Location
ok "Frontend dependencies installed"

# ── 7. Ollama check ──────────────────────────────────────────────────────────
$ollamaHost = $env:PIPER_OLLAMA_HOST
if (-not $ollamaHost) { $ollamaHost = "http://localhost:11434" }
$llmModel = $env:PIPER_LLM_MODEL
if (-not $llmModel) { $llmModel = "qwen3:4b" }

try {
    $response = Invoke-RestMethod -Uri "$ollamaHost/api/tags" -TimeoutSec 5 -ErrorAction Stop
    ok "Ollama reachable at $ollamaHost"

    $modelNames = $response.models | ForEach-Object { $_.name }
    if ($modelNames -contains $llmModel) {
        ok "Model $llmModel is available"
    } else {
        warn "Model $llmModel is not pulled yet"
        info "Run:  ollama pull $llmModel"
    }
} catch {
    warn "Ollama not reachable at $ollamaHost"
    info "Install Ollama from https://ollama.com, start it, then run:"
    info "  ollama pull $llmModel"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Run the system check:    python check.py"
Write-Host "  Start PIPER:             .\run.ps1"
Write-Host ""
