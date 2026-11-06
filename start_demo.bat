@echo off
echo ==================================================
echo Khởi động ISIC 2024 Skin Cancer Classification Demo
echo ==================================================

:: Bật Backend trong một cửa sổ Command Prompt mới
echo 1. Dang khoi dong FastAPI Backend (Port 8000)...
set "ROOT=%~dp0"
set "PYTHON_CMD=python"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_CMD=%ROOT%.venv\Scripts\python.exe"
start "FastAPI Backend" cmd /k "cd /d ""%ROOT%backend"" && ""%PYTHON_CMD%"" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

:: Đợi 3 giây để Backend khởi động trước
timeout /t 3 /nobreak > nul

:: Bật Frontend trực tiếp tại cửa sổ hiện tại
echo 2. Dang khoi dong React Frontend (Vite)...
cd /d "%ROOT%frontend"
npm run dev
