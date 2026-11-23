@echo off
setlocal
cd /d "%~dp0"
echo === SupportLens setup ===
where python >nul 2>nul || (echo [ERROR] Python not found. Install Python 3.10+ and tick "Add Python to PATH". & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv || (echo [ERROR] could not create .venv & pause & exit /b 1)
)
echo Installing requirements ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [ERROR] installation failed & pause & exit /b 1)
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env - open it and set GITHUB_TOKEN before collecting data.
)
echo Running the offline test suite ...
".venv\Scripts\python.exe" -m pytest
echo.
echo Setup finished. Next: set GITHUB_TOKEN in .env, then run collect.bat
pause
