# Trading bot watchdog: started by Windows Task Scheduler every 5 minutes.
# If the bot is not running, it starts it again (start-bot.cmd) and logs the time.
# Safe to run any time: the bot's lock means a second copy can never start.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root 'data\cycle.pid'
$stopFile = Join-Path $root 'data\bot-stopped'
$log = Join-Path $root 'data\watchdog.log'

function Write-Log([string]$text) {
    $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -Path $log -Value "$stamp  $text" -Encoding utf8
}

# Stopped on purpose with stop-bot.cmd: leave it stopped.
if (Test-Path $stopFile) { exit 0 }

$running = $false
if (Test-Path $pidFile) {
    $botPid = (Get-Content $pidFile -Raw).Trim()
    if ($botPid -match '^\d+$') {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$botPid" -ErrorAction SilentlyContinue
        # The id alone is not enough: Windows reuses ids, so check it is really the bot.
        if ($process -and $process.CommandLine -match 'trading-agent' -and $process.CommandLine -match ' run') {
            $running = $true
        }
    }
}
if ($running) { exit 0 }

Write-Log 'bot not running: starting it'
# Through Explorer, like a double-click, so the bot does not belong to this short-lived task.
Start-Process -FilePath explorer.exe -ArgumentList ('"' + (Join-Path $root 'start-bot.cmd') + '"')
