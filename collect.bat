@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /i "%%A"=="GITHUB_TOKEN" set "GITHUB_TOKEN=%%B"
    if /i "%%A"=="NVD_API_KEY" set "NVD_API_KEY=%%B"
)
if "%GITHUB_TOKEN%"=="" echo [WARNING] GITHUB_TOKEN is empty in .env - collection will be very slow.
if not exist logs mkdir logs
echo Collecting PyPI policies (target 300, resumable). Log: logs\collection_run.log
".venv\Scripts\python.exe" scripts\collect_dataset.py --target 300 --delay 0.5
echo.
echo Collected records:
".venv\Scripts\python.exe" -c "print(sum(1 for l in open('data/processed/pypi_dataset.jsonl', encoding='utf-8') if l.strip()))"
pause
