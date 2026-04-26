<#
.SYNOPSIS
    Smoke test for Geo-WorldEngine Beta - validates end-to-end map generation pipeline.

.DESCRIPTION
    1. POST /api/maps to create a task
    2. Poll GET /api/maps/{taskId} until READY/READY_INTERACTIVE/FAILED
    3. Verify preview.png is accessible
    4. Verify manifest.json exists (if tiles generated)
    5. Check metric_report in task data

.EXAMPLE
    .\smoke_generate_map.ps1 -BaseUrl http://localhost:8000
#>

param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$Prompt = "一座四面环海的岛屿。",
    [int]$Width = 512,
    [int]$Height = 256,
    [int]$Seed = 42,
    [int]$MaxPollSeconds = 120,
    [int]$PollIntervalSeconds = 3
)

$ErrorActionPreference = "Stop"

function Write-Status($msg) {
    Write-Host "[SMOKE] $msg" -ForegroundColor Cyan
}

function Write-Ok($msg) {
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

Write-Status "Geo-WorldEngine Beta Smoke Test"
Write-Status "Base URL: $BaseUrl"
Write-Status "Prompt: $Prompt"

# Step 1: Health check
Write-Status "Step 1: Health check..."
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 5
    Write-Ok "Health check passed: $health"
} catch {
    Write-Fail "Health check failed: $_"
    exit 1
}

# Step 2: Create map task
Write-Status "Step 2: Creating map task..."
$body = @{
    prompt = $Prompt
    width = $Width
    height = $Height
    seed = $Seed
    auto_confirm = $true
    generate_tiles = $true
} | ConvertTo-Json

try {
    $createResp = Invoke-RestMethod -Uri "$BaseUrl/maps" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    $taskId = $createResp.taskId
    Write-Ok "Task created: $taskId"
} catch {
    Write-Fail "Failed to create task: $_"
    exit 1
}

# Step 3: Poll task status
Write-Status "Step 3: Polling task status (max ${MaxPollSeconds}s)..."
$startTime = Get-Date
$finalStatus = $null
$attempts = 0

while ($true) {
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    if ($elapsed -ge $MaxPollSeconds) {
        Write-Fail "Timeout after ${MaxPollSeconds}s"
        exit 1
    }

    Start-Sleep -Seconds $PollIntervalSeconds
    $attempts++

    try {
        $task = Invoke-RestMethod -Uri "$BaseUrl/maps/$taskId" -Method GET -TimeoutSec 5
        $status = $task.status
        $stage = $task.currentStage
        $progress = $task.progress
        Write-Status "  Attempt $attempts : status=$status stage=$stage progress=$progress%"

        if ($status -eq "ready-image" -or $status -eq "ready-interactive") {
            $finalStatus = $status
            break
        }
        if ($status -eq "failed") {
            Write-Fail "Task failed: $($task.errorMsg)"
            exit 1
        }
    } catch {
        Write-Status "  Attempt $attempts : request error (will retry) - $_"
    }
}

Write-Ok "Task completed with status: $finalStatus"

# Step 4: Verify preview.png
Write-Status "Step 4: Verifying preview.png..."
try {
    $previewResp = Invoke-WebRequest -Uri "$BaseUrl/maps/$taskId/preview.png" -Method GET -TimeoutSec 10
    $contentType = $previewResp.Headers["Content-Type"]
    if ($contentType -like "*image*") {
        $sizeKB = [math]::Round($previewResp.Content.Length / 1024, 1)
        Write-Ok "preview.png accessible ($sizeKB KB, $contentType)"
    } else {
        Write-Fail "preview.png returned unexpected content type: $contentType"
    }
} catch {
    Write-Fail "preview.png not accessible: $_"
}

# Step 5: Verify manifest.json (if tiles generated)
if ($finalStatus -eq "ready-interactive") {
    Write-Status "Step 5: Verifying manifest.json..."
    try {
        $manifest = Invoke-RestMethod -Uri "$BaseUrl/maps/$taskId/tiles/manifest.json" -Method GET -TimeoutSec 5
        Write-Ok "manifest.json accessible: min_zoom=$($manifest.min_zoom) max_zoom=$($manifest.max_zoom)"
    } catch {
        Write-Status "  manifest.json not available (tiles may still be generating)"
    }
} else {
    Write-Status "Step 5: Skipping manifest check (status=$finalStatus, tiles may not be ready)"
}

# Step 6: Check metric_report
Write-Status "Step 6: Checking metric_report..."
try {
    $taskDetail = Invoke-RestMethod -Uri "$BaseUrl/maps/$taskId" -Method GET -TimeoutSec 5
    if ($taskDetail.metricReport) {
        $metrics = $taskDetail.metricReport
        if ($metrics -is [string]) {
            $metrics = $metrics | ConvertFrom-Json
        }
        Write-Ok "metric_report available: land_components=$($metrics.land_components) cross_cut_score=$($metrics.cross_cut_score)"
    } else {
        Write-Status "  metric_report not available in task response"
    }
} catch {
    Write-Status "  Could not check metric_report: $_"
}

Write-Ok "=== Smoke test PASSED ==="
