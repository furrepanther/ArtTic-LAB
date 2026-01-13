@echo off
setlocal
title ArtTic-LAB

set "ENV_NAME=ArtTic-LAB"

:: Check Conda
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found.
    pause
    exit /b 1
)

:: Activate
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment not found. Run install.bat.
    pause
    exit /b 1
)

echo =======================================================
echo             Launching ArtTic-LAB
echo =======================================================
echo.

:loop
python app.py %*
if %ERRORLEVEL% EQU 21 (
    echo [INFO] Restarting...
    timeout /t 1 /nobreak >nul
    goto loop
)
if %ERRORLEVEL% NEQ 0 pause