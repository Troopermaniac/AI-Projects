@echo off
cd /d "%~dp0"
echo ========================================================
echo        BOOTING THE AUTONOMOUS AI EVOLUTION ENGINE
echo ========================================================
echo.

:: Kill any existing process on port 8000 (AI Router)
echo [Pre-flight] Killing any existing AI Router processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr LISTENING ^| findstr :8000') do (
    taskkill /PID %%a /F >nul 2>&1
    echo Killed process on port 8000 (PID: %%a)
)
ping 127.0.0.1 -n 2 > NUL

python -c "import json; c=json.load(open('config.json')); print('[1/2] Make sure LM Studio is running and serving on port ' + str(c.get('api_base','http://localhost:21467/v1').split(':')[2].replace('/v1','')))"
ping 127.0.0.1 -n 4 > NUL

echo [2/3] Starting the AI Brain Fast API Router...
start "AI Brain Router" cmd /k "cd /d "%~dp0" && call .\venv\Scripts\activate.bat && python main.py"

echo.
echo Waiting 6 seconds for Router to boot...
ping 127.0.0.1 -n 7 > NUL

echo [3/3] Launching Live AI Evolution Monitor...
start "AI Progress Monitor" cmd /k "cd /d "%~dp0" && call .\venv\Scripts\activate.bat && python show_progress.py --watch"

echo Starting the Darwinian Master Loop...
call .\venv\Scripts\activate.bat
python autonomous_loop.py

pause
