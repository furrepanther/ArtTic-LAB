# ArtTic-LAB Launcher (PowerShell)
$EnvName = "ArtTic-LAB"

# Intel Optimizations
$Env:ONEAPI_DEVICE_SELECTOR = "level_zero:0"
$Env:TORCH_LLM_ALLREDUCE = "1"

if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda not found." -ForegroundColor Red
    exit 1
}

# Activate Environment
if (& conda shell.powershell hook | Out-String) {
    Invoke-Expression (& conda shell.powershell hook | Out-String)
    conda activate $EnvName
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate environment." -ForegroundColor Red
    exit 1
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "             Launching ArtTic-LAB" -ForegroundColor Cyan
Write-Host "======================================================="

# Loop for restarts
do {
    python app.py $args
    $ExitCode = $LASTEXITCODE
    
    if ($ExitCode -eq 21) {
        Write-Host "[INFO] Restarting Backend..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
} while ($ExitCode -eq 21)

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "ArtTic-LAB has closed." -ForegroundColor Cyan
Write-Host "======================================================="