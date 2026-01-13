@echo off
setlocal EnableDelayedExpansion
title ArtTic-LAB Installer

echo =======================================================
echo            ArtTic-LAB Installer (Windows)
echo =======================================================

where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please run from Anaconda Prompt.
    pause
    exit /b 1
)

set "ENV_NAME=ArtTic-LAB"
call conda create --name %ENV_NAME% python=3.11 -y
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 exit /b 1

python -m pip install --upgrade pip --quiet

echo.
echo Select your Hardware Accelerator:
echo 1) Intel ARC (XPU) - PyTorch Nightly
echo 2) NVIDIA (CUDA)   - PyTorch Stable
echo 3) CPU Only
set /p HW_CHOICE="Enter selection (1-3): "

echo.
echo [INFO] Installing PyTorch...
call pip uninstall -y torch torchvision torchaudio

if "%HW_CHOICE%"=="1" (
    echo [INFO] Installing PyTorch Nightly for Windows XPU...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
    
    echo [INFO] Configuring Conda variables...
    call conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0
    call conda env config vars set TORCH_LLM_ALLREDUCE=1
    call conda deactivate
    call conda activate %ENV_NAME%
) else if "%HW_CHOICE%"=="2" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch torchvision torchaudio
)

echo [INFO] Installing Dependencies...
:: Added 'uvicorn[standard]' to fix WebSocket issue
pip install diffusers transformers accelerate safetensors fastapi "uvicorn[standard]" jinja2 toml pyngrok pillow numpy sdnq

echo.
echo =======================================================
echo [SUCCESS] Installation complete!
echo Run 'start.bat' to launch.
echo =======================================================
pause