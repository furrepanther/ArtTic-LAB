$ErrorActionPreference = "Stop"

$ENV_NAME = "ArtTic-LAB"
$PYTHON_VERSION = "3.11"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "           ArtTic-LAB Installer (PowerShell)           " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check for Conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda not found in PATH." -ForegroundColor Red
    Write-Host "Please install Miniconda or Anaconda and add it to your PATH." -ForegroundColor Gray
    exit 1
}

# 2. Create Environment
$envExists = conda env list | Select-String -Pattern "^$ENV_NAME\s"
if ($envExists) {
    Write-Host "[INFO] Updating existing environment '$ENV_NAME'..." -ForegroundColor Green
} else {
    Write-Host "[INFO] Creating Conda environment '$ENV_NAME'..." -ForegroundColor Green
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
}

# 3. Hardware Selection
Write-Host "-------------------------------------------------------" -ForegroundColor Yellow
Write-Host "Select your Hardware Accelerator:" -ForegroundColor Yellow
Write-Host "-------------------------------------------------------" -ForegroundColor Yellow
Write-Host "1) NVIDIA (CUDA)"
Write-Host "2) Intel ARC (XPU)"
Write-Host "3) CPU Only"
$choice = Read-Host "Enter selection (1-3)"

# 4. Activate & Install
# Note: Activating conda in a script child process is tricky. 
# We will use 'conda run' for commands to ensure they run in the env context.

Write-Host "[INFO] Upgrading pip..." -ForegroundColor Green
conda run -n $ENV_NAME python -m pip install --upgrade pip --quiet

Write-Host "[INFO] Cleaning previous PyTorch installations..." -ForegroundColor Green
conda run -n $ENV_NAME pip uninstall -y torch torchvision torchaudio 2>$null

Write-Host "[INFO] Installing PyTorch..." -ForegroundColor Green
switch ($choice) {
    "1" {
        Write-Host "Installing for NVIDIA CUDA..." -ForegroundColor Cyan
        conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
    "2" {
        Write-Host "Installing for Intel ARC (XPU)..." -ForegroundColor Cyan
        conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        
        Write-Host "[INFO] Setting Environment Variables for Intel Arc..."
        conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0 -n $ENV_NAME
    }
    "3" {
        Write-Host "Installing for CPU..." -ForegroundColor Cyan
        conda run -n $ENV_NAME pip install torch torchvision torchaudio
    }
    Default {
        Write-Host "Invalid selection. Defaulting to CPU." -ForegroundColor Red
        conda run -n $ENV_NAME pip install torch torchvision torchaudio
    }
}

Write-Host "[INFO] Installing Application Dependencies..." -ForegroundColor Green
conda run -n $ENV_NAME pip install -r requirements.txt

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Installation complete!" -ForegroundColor Green
Write-Host "Run start.bat or start.ps1 to launch." -ForegroundColor Gray
Write-Host "=======================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit..."
