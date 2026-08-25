# PIPER — Launcher (Windows PowerShell)
# Starts the backend and frontend together.
#
# Usage:  .\run.ps1
# Stop:   Ctrl+C in this window, then close the opened terminal windows.
# If you see an execution-policy error:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"

function fail { param($msg) Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    fail ".venv not found. Run:  .\setup.ps1"
}
if (-not (Test-Path "frontend\node_modules")) {
    fail "frontend\node_modules not found. Run:  .\setup.ps1"
}

New-Item -ItemType Directory -Force -Path "data"     | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

Write-Host ""
Write-Host "Starting PIPER..." -ForegroundColor Cyan
Write-Host "  Backend:   http://localhost:8000  (API docs: http://localhost:8000/docs)"
Write-Host "  Frontend:  http://localhost:5173"
Write-Host "  Stop:      close the opened terminal windows, or press Ctrl+C here."
Write-Host ""

# Launch backend in a new terminal window
$backendCmd = "Set-Location '$PWD\backend'; ..\\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000; Read-Host 'Press Enter to close'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal

Start-Sleep -Seconds 2

# Launch frontend in a new terminal window
$frontendCmd = "Set-Location '$PWD\frontend'; npm run dev -- --host 127.0.0.1; Read-Host 'Press Enter to close'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal

Write-Host "PIPER is starting in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "  Open http://localhost:5173 in your browser once both windows show they are ready."
Write-Host ""
