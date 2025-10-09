# run_pipeline.ps1
# Launch Kafka + Bronze + Silver + Trainer + Producer + Streamlit
# Uses direct executables (no nested PowerShell) to avoid VS Code EditorServices crashes.

$ErrorActionPreference = "Stop"

# Always run from this script's folder
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host "🚀 Starting Streaming ML Pipeline..." -ForegroundColor Cyan

# ---- Quick existence checks (helpful errors) ----
$need = @(
  "docker-compose.yml",
  "ingestion\producer.py",
  "transform\consumer_to_parquet.py",
  "transform\consumer_to_silver.py",
  "transform\consumer_trainer.py",
  "dashboard\training_monitor.py"
)
foreach ($p in $need) {
  if (-not (Test-Path $p)) { throw "Missing: $p" }
}

# Ensure expected folders exist
New-Item -ItemType Directory -Force -Path "data"     | Out-Null
New-Item -ItemType Directory -Force -Path "data\bronze" | Out-Null
New-Item -ItemType Directory -Force -Path "data\silver" | Out-Null

# ---- 0) Kafka ----
Write-Host "`n▶️ Starting Kafka (Redpanda)..." -ForegroundColor Yellow
# We now use the default 'docker-compose.yml' file, so the -f flag is not needed.
docker compose up -d

Start-Sleep -Seconds 3

$procs = @()

# ---- 1) Bronze consumer ----
Write-Host "`n▶️ Consumer → Bronze" -ForegroundColor Yellow
$procs += Start-Process -FilePath "python" `
  -ArgumentList "transform\consumer_to_parquet.py" `
  -WorkingDirectory $ROOT -PassThru

Start-Sleep -Seconds 2

# ---- 2) Silver consumer ----
Write-Host "`n▶️ Consumer → Silver" -ForegroundColor Yellow
$procs += Start-Process -FilePath "python" `
  -ArgumentList "transform\consumer_to_silver.py" `
  -WorkingDirectory $ROOT -PassThru

Start-Sleep -Seconds 2

# ---- 3) Trainer (reads Silver, logs metrics.csv) ----
Write-Host "`n▶️ Trainer" -ForegroundColor Yellow
$procs += Start-Process -FilePath "python" `
  -ArgumentList "transform\consumer_trainer.py" `
  -WorkingDirectory $ROOT -PassThru

Start-Sleep -Seconds 2

# ---- 4) Producer (streams from Excel) ----
Write-Host "`n▶️ Producer" -ForegroundColor Yellow
$procs += Start-Process -FilePath "python" `
  -ArgumentList "ingestion\producer.py" `
  -WorkingDirectory $ROOT -PassThru

Start-Sleep -Seconds 2

# ---- 5) Streamlit Dashboard ----
Write-Host "`n▶️ Streamlit Dashboard" -ForegroundColor Yellow
$procs += Start-Process -FilePath "streamlit" `
  -ArgumentList "run","dashboard\training_monitor.py" `
  -WorkingDirectory $ROOT -PassThru

Write-Host "`n✅ All services started. Open http://localhost:8501" -ForegroundColor Green
