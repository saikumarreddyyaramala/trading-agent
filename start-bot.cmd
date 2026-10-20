@echo off
rem Starts the trading bot in its own minimized window. Double-click to start; use
rem stop-bot.cmd to stop it (closing the window works too, but the watchdog restarts it
rem within 5 minutes). Only one copy can run at a time: a second one refuses to start.
if exist "%~dp0data\bot-stopped" del "%~dp0data\bot-stopped"
if not "%~1"=="minimized" (
    start "Trading bot - keep this window open" /min "%~f0" minimized
    exit /b
)
title Trading bot - keep this window open
cd /d "%~dp0"
uv run --no-sync trading-agent run
echo.
echo The trading bot has stopped. Double-click start-bot.cmd to start it again.
pause
