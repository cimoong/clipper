@echo off
setlocal
cd /d "%~dp0"

REM --- Check that uv is installed -------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ClipForge] "uv" was not found on your PATH.
    echo             Install it first, then run this script again:
    echo.
    echo               https://docs.astral.sh/uv/getting-started/installation/
    echo.
    echo             ^(PowerShell^) powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    pause
    exit /b 1
)

REM --- Create the virtual environment on first run --------------------------
if not exist ".venv" (
    echo [ClipForge] No .venv found. Installing dependencies with "uv sync"...
    uv sync
    if errorlevel 1 (
        echo [ClipForge] "uv sync" failed. See the messages above.
        pause
        exit /b 1
    )
)

REM --- Start the server, then open the browser ------------------------------
set "URL=http://localhost:8420"
echo [ClipForge] Starting server on %URL%
echo [ClipForge] The browser will open in a moment. Press Ctrl+C to stop.
echo.

REM Wait 2s in the background, then open the default browser while uvicorn runs
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start "" %URL%"

REM Run uvicorn in the foreground so this window keeps showing the logs
uv run uvicorn clipforge.web.app:app --host 127.0.0.1 --port 8420

echo.
echo [ClipForge] Server stopped.
pause
endlocal
