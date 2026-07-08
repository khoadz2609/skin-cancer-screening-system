@echo off
echo ==================================================
echo Khởi động ISIC 2024 Skin Cancer Classification Demo
echo ==================================================

:: Bật Backend trong một cửa sổ Command Prompt mới
echo 1. Dang khoi dong FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "cd backend && call C:\Users\khoan\miniconda3\Scripts\activate.bat && conda activate base && uvicorn main:app --reload --host 127.0.0.1 --port 8000"

:: Đợi 3 giây để Backend khởi động trước
timeout /t 3 /nobreak > nul

:: Bật Frontend trực tiếp tại cửa sổ hiện tại
echo 2. Dang khoi dong React Frontend (Vite)...
cd frontend
npm run dev
