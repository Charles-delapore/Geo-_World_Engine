$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root 'backend'
$frontendDir = Join-Path $root 'frontend\Geo-world'
$pythonExe = Join-Path $root '.venv313\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing Python environment: $pythonExe"
}

Write-Host "Starting Geo-WorldEngine Beta backend on http://127.0.0.1:8000"
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$backendDir'; `$env:RUN_MODE='local'; `$env:ARTIFACT_BACKEND='local'; `$env:DATABASE_URL='sqlite:///$($backendDir.Replace('\','/'))/data/geo_world_beta.db'; & '$pythonExe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
)

Write-Host "Starting Geo-WorldEngine Beta frontend on http://127.0.0.1:5173"
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$frontendDir'; npm run dev -- --host 127.0.0.1 --port 5173"
)

Write-Host "Beta services launched."
