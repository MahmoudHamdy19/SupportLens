@echo off
setlocal
title SupportLens Security Policy Auditor

echo ======================================================================
echo   SupportLens: Software Security Policy Extraction & Assessment
echo   System Pipeline: Layers 0 to 4
echo ======================================================================
echo.

cd /d "%~dp0"

set "PY_CMD="

if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
    goto DETECTED
)
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    goto DETECTED
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PY_CMD=%USERPROFILE%\anaconda3\python.exe"
    goto DETECTED
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PY_CMD=%USERPROFILE%\miniconda3\python.exe"
    goto DETECTED
)
if exist "%ProgramData%\anaconda3\python.exe" (
    set "PY_CMD=%ProgramData%\anaconda3\python.exe"
    goto DETECTED
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto DETECTED
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto DETECTED
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto DETECTED
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto DETECTED
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto DETECTED
)

echo [ERROR] Python not found in standard paths.
pause
exit /b 1

:DETECTED
echo [*] Python detected: %PY_CMD%
echo.
echo Select Mode to Run:
echo   [1] Launch Web Dashboard (Streamlit GUI - Recommended)
echo   [2] Run the offline test suite (pytest)
echo   [3] Audit a PyPI Package via CLI (e.g., urllib3, requests)
echo   [4] Exit
echo.
set /p choice="Enter choice [1-4] (default 1): "

if "%choice%"=="2" goto RUN_TESTS
if "%choice%"=="3" goto RUN_CLI
if "%choice%"=="4" goto END

:RUN_DASHBOARD
echo.
echo Launching SupportLens Web Dashboard...
echo Open your browser at: http://localhost:8501
"%PY_CMD%" -m streamlit run app.py --server.fileWatcherType none
goto END

:RUN_TESTS
echo.
echo Running the offline test suite...
"%PY_CMD%" -m pytest
pause
goto END

:RUN_CLI
echo.
set /p pkg_name="Enter PyPI package name to audit (default: urllib3): "
if "%pkg_name%"=="" set "pkg_name=urllib3"
echo.
echo Auditing package: %pkg_name% ...
"%PY_CMD%" supportlens_cli.py --pypi "%pkg_name%"
pause
goto END

:END
