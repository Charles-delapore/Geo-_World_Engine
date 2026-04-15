$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$composeFile = Join-Path $root "deployment\docker-compose.beta.yml"
$envFile = Join-Path $root ".env"

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $root ".env.example") -Destination $envFile
    Write-Host "Created .env from .env.example"
}

docker compose -f $composeFile --env-file $envFile up -d postgres redis minio
Write-Host "Infrastructure started. Run full stack with:"
Write-Host "docker compose -f $composeFile --env-file $envFile up --build"
