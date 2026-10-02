@echo off
setlocal
set "PY=D:\Anaconda\envs\yolo-asr\python.exe"
set "BASE=E:\YOLO\qwen3-asr"
title Qwen3-ASR Voice Input

rem 1) start ASR server if it is not running
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8910/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo [server] Not detected. Starting ASR server...
    start "Qwen3-ASR Server" /min "%PY%" "%BASE%\server.py"
)

rem 2) run the voice input tool (it waits for the model inside)
"%PY%" "%BASE%\voice_input.py" %*

echo.
pause