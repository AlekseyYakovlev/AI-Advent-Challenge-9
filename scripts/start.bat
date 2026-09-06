@echo off
cd /d %~dp0..
call .venv\Scripts\activate.bat
if not exist .env copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
