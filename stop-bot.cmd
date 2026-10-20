@echo off
rem Stops the trading bot on purpose, so the watchdog does not start it again.
rem Double-click start-bot.cmd to run it again. Open positions keep their stop-loss.
cd /d "%~dp0"
if not exist data mkdir data
echo stopped by stop-bot.cmd on %date% %time% > data\bot-stopped
powershell -NoProfile -ExecutionPolicy Bypass -Command "$id = (Get-Content 'data\cycle.pid' -ErrorAction SilentlyContinue); if ($id) { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $id); while ($p -and $p.Name -ne 'explorer.exe' -and $p.Name -ne 'svchost.exe') { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p.ParentProcessId) } }"
echo The trading bot is stopped. Double-click start-bot.cmd to start it again.
pause
