@echo off
set ENV_NAME=ArtTic-LAB
set SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate environment. Run install.bat first.
    pause
    exit /b 1
)

:loop
python app.py %*
if %ERRORLEVEL% EQU 21 (
    echo [INFO] Restarting Backend...
    timeout /t 1 >nul
    goto loop
)
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] App crashed.
    pause
    exit /b %ERRORLEVEL%
)