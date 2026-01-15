@echo off
set ENV_NAME=ArtTic-LAB
set PYTHON_VERSION=3.11

echo =======================================================
echo            ArtTic-LAB Installer (Universal)
echo =======================================================

call conda activate base >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda not found in PATH.
    pause
    exit /b 1
)

call conda create --name %ENV_NAME% python=%PYTHON_VERSION% -y
call conda activate %ENV_NAME%
python -m pip install --upgrade pip --quiet

echo.
echo -------------------------------------------------------
echo Select your Hardware Accelerator:
echo -------------------------------------------------------
echo 1) NVIDIA (CUDA)
echo 2) Intel ARC (XPU)
echo 3) CPU Only
echo -------------------------------------------------------
set /p hw_choice="Enter selection (1-3): "

echo [INFO] Cleaning previous PyTorch installations...
pip uninstall -y torch torchvision torchaudio 2>nul

if "%hw_choice%"=="1" (
    echo [INFO] Installing CUDA PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%hw_choice%"=="2" (
    echo [INFO] Installing XPU PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    call conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0
    call conda deactivate
    call conda activate %ENV_NAME%
) else (
    echo [INFO] Installing CPU PyTorch...
    pip install torch torchvision torchaudio
)

echo [INFO] Installing Application Dependencies...
pip install diffusers transformers accelerate safetensors fastapi "uvicorn[standard]" jinja2 toml pyngrok pillow numpy sdnq psutil

echo =======================================================
echo [SUCCESS] Installation complete!
echo Run start.bat to launch.
echo =======================================================
pause