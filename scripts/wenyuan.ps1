<#
一键启动文渊并打开工作台：http://127.0.0.1:8000/ui
  scripts\wenyuan.cmd          启动（已在运行的部分会跳过）
  scripts\wenyuan.cmd -Stop    停止 API、模型服务和专用 Ollama，并停掉 Docker 容器
依次确保：Docker Desktop -> docker compose（Milvus、MySQL、Redis）-> GPU 模型服务（8001）
-> 专用 Ollama（11435，并行 2、8k 上下文，和 GPU 闸门的 2 个槽位一致）-> API（8000）。
日志在 .cache\ 和 model_server\ 下。
#>
param([switch]$Stop, [switch]$NoBrowser, [int]$ApiPort = 8000, [int]$ModelPort = 8001, [int]$OllamaPort = 11435)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path $PSScriptRoot
$logs = Join-Path $root ".cache"
New-Item -ItemType Directory -Force $logs | Out-Null

function Test-Port([int]$port) {
    $c = New-Object Net.Sockets.TcpClient
    try { return $c.ConnectAsync("127.0.0.1", $port).Wait(300) } catch { return $false } finally { $c.Dispose() }
}
function Wait-Until([scriptblock]$ready, [int]$seconds, [string]$what) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not (& $ready)) {
        if ((Get-Date) -gt $deadline) { throw "$what 在 $seconds 秒内没有就绪，请看日志：$logs" }
        Start-Sleep -Milliseconds 500
    }
}
function Find-Exe([string]$name, [string[]]$fallback) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in $fallback) { if (Test-Path $p) { return $p } }
    throw "找不到 $name，请先安装它"
}
function Test-Docker { cmd /c "docker info >nul 2>&1"; return $LASTEXITCODE -eq 0 }
function Invoke-Compose([string]$verb) {
    Push-Location $root
    try { cmd /c "docker compose $verb 2>&1" | Write-Host } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "docker compose $verb 失败" }
}
function Stop-Port([int]$port, [string]$what) {
    $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
    if ($pids) { Write-Host "已停止 $what" } else { Write-Host "$what 没有在运行" }
}

if ($Stop) {
    Stop-Port $ApiPort "API"
    Stop-Port $ModelPort "模型服务"
    Stop-Port $OllamaPort "专用 Ollama"
    Invoke-Compose "stop"
    return
}

$uv = Find-Exe "uv" @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")

# 1. Docker
if (-not (Test-Docker)) {
    Write-Host "启动 Docker Desktop…"
    $desktop = @("$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
                 "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe") | Where-Object { Test-Path $_ } |
               Select-Object -First 1
    if (-not $desktop) { throw "找不到 Docker Desktop，请先手动启动 Docker" }
    Start-Process $desktop
    Wait-Until { Test-Docker } 180 "Docker"
}
Write-Host "启动 Milvus、MySQL、Redis…"
Invoke-Compose "up -d"

# 2. GPU 模型服务
if (-not (Test-Port $ModelPort)) {
    Write-Host "启动模型服务（$ModelPort）…"
    Start-Process -ErrorAction Stop $uv -ArgumentList "run", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", $ModelPort `
        -WorkingDirectory (Join-Path $root "model_server") -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "model-server.log") -RedirectStandardError (Join-Path $logs "model-server.err.log")
}

# 3. 专用 Ollama
if (-not (Test-Port $OllamaPort)) {
    Write-Host "启动 Ollama（$OllamaPort）…"
    $ollama = Find-Exe "ollama" @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")
    $env:OLLAMA_HOST = "127.0.0.1:$OllamaPort"; $env:OLLAMA_NUM_PARALLEL = "2"; $env:OLLAMA_CONTEXT_LENGTH = "8192"
    $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"; $env:OLLAMA_KEEP_ALIVE = "30m"
    Start-Process -ErrorAction Stop $ollama -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "ollama-serve.out.log") -RedirectStandardError (Join-Path $logs "ollama-serve.log")
}

# 4. API（启动时要连 Milvus，所以等模型服务和数据库都起来后再起）
Wait-Until { Test-Port $ModelPort } 180 "模型服务"
Wait-Until { Test-Port $OllamaPort } 60 "Ollama"
if (-not (Test-Port $ApiPort)) {
    Write-Host "启动文渊 API（$ApiPort）…"
    $env:PYTHONIOENCODING = "utf-8"
    Start-Process -ErrorAction Stop $uv -ArgumentList "run", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", $ApiPort `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "api.out.log") -RedirectStandardError (Join-Path $logs "api.log")
}
Wait-Until {
    try { (Invoke-WebRequest "http://127.0.0.1:$ApiPort/healthz" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 }
    catch { $false }
} 180 "API"

Write-Host "文渊已就绪：http://127.0.0.1:$ApiPort/ui"
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$ApiPort/ui" }
