@echo off
rem Starts the trading bot in its own window. Double-click to start; close the window to stop.
rem Only one copy can run at a time: a second one refuses to start.
title Trading bot - keep this window open
cd /d "%~dp0"
uv run --no-sync trading-agent run
echo.
echo The trading bot has stopped. Double-click start-bot.cmd to start it again.
pause
