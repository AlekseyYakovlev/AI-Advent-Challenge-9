@echo off
cd /d %~dp0..
call .venv\Scripts\activate.bat
if not exist .env copy .env.example .env
rem Логи: configure_logging вызывается при импорте app/chainlit/app.py
chainlit run app/chainlit/app.py --host 0.0.0.0 --port 8000
