@echo off
echo ========================================================
echo        STOPPING ALL AUTONOMOUS AI ENGINE PROCESSES
echo ========================================================
echo.

:: 1. Unload model from LM Studio GPU VRAM
echo [1/4] Unloading LM Studio model from GPU VRAM...
lms unload --all >nul 2>&1

:: 2. Terminate all Python processes and CMD windows for AI loop
echo [2/4] Terminating Python processes and AI windows...
taskkill /F /FI "WINDOWTITLE eq AI Brain Router*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq AI Progress Monitor*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Select AI Brain Router*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Select AI Progress Monitor*" >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
taskkill /F /IM llama-quantize.exe >nul 2>&1

:: 3. Targeted PowerShell termination for any residual processes or spawned CMD windows
echo [3/4] Terminating residual workspace processes and windows...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object CommandLine -match 'main\.py|show_progress|autonomous_loop|AI Project' | ForEach-Object { Stop-Process -Id $PSItem.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

:: 4. Free Port 8000 (AI Router Port)
echo [4/4] Freeing Port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /PID %%a /F >nul 2>&1
    echo Terminated process on port 8000 (PID: %%a)
)

echo.
echo ========================================================
echo        SYSTEM EMERGENCY STOP COMPLETE (VRAM FREED)
echo ========================================================
echo.
timeout /t 3 >nul
