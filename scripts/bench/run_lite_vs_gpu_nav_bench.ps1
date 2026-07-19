# Lite (CPU Fast RAW) vs Full (GPU demosaic) navigation cold-suite.
# Uses Canon_Sample (~60 CR3). Clears cache between runs.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\bench\run_lite_vs_gpu_nav_bench.ps1
param(
    [string]$Entry = "D:\Development\Canon_Sample\020A0019.CR3",
    [int]$Count = 0,          # 0 = full folder
    [int]$Revisit = 5,
    [double]$Timeout = 45
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PixiPy = Join-Path $Root ".pixi\envs\default\python.exe"
$Main = Join-Path $Root "src\main.py"
$LogDir = Join-Path $Root "bench\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $PixiPy)) { throw "Missing pixi python: $PixiPy" }
if (-not (Test-Path $Entry)) { throw "Missing entry RAW: $Entry" }

function Clear-ColdStart {
    param([string]$Label)
    Write-Host ""
    Write-Host "=== clear cache: $Label ===" -ForegroundColor Yellow
    Get-Process RAWviewer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'main\.py|nav_autotest|RAWviewer') } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    $cache = Join-Path $env:USERPROFILE ".rawviewer_cache"
    if (Test-Path $cache) {
        Remove-Item -LiteralPath $cache -Recurse -Force -ErrorAction SilentlyContinue
    }
    reg delete "HKCU\Software\RAWviewer" /f 2>$null | Out-Null
    if (Test-Path $cache) { Write-Host "WARNING: cache still present" -ForegroundColor Red }
    else { Write-Host "cache cleared" }
}

function Invoke-NavBench {
    param(
        [string]$Label,
        [string]$PreferGpu,   # "0" or "1"
        [string]$OutLog
    )
    Write-Host ""
    Write-Host "=== run: $Label (PREFER_GPU=$PreferGpu) ===" -ForegroundColor Cyan

    $env:RAWVIEWER_AUTOTEST = "1"
    $env:RAWVIEWER_AUTOTEST_RAW_MODE = "1"
    $env:RAWVIEWER_AUTOTEST_REVISIT = "$Revisit"
    $env:RAWVIEWER_AUTOTEST_TIMEOUT = "$Timeout"
    $env:RAWVIEWER_AUTOTEST_SETTLE = "3.0"
    $env:RAWVIEWER_PREFER_GPU_DECODE = $PreferGpu
    $env:RAWVIEWER_FILE_LOG = "1"
    $env:RAWVIEWER_SHOW_SPLASH = "0"
    $env:RAWVIEWER_ENABLE_SEMANTIC_SEARCH = "0"
    $env:RAWVIEWER_SKIP_UPDATE_CHECK = "1"
    if ($Count -gt 0) {
        $env:RAWVIEWER_AUTOTEST_COUNT = "$Count"
    } else {
        Remove-Item Env:RAWVIEWER_AUTOTEST_COUNT -ErrorAction SilentlyContinue
    }

    $argList = @("-u", $Main, $Entry)
    $proc = Start-Process -FilePath $PixiPy -ArgumentList $argList `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError "$OutLog.err" `
        -PassThru -NoNewWindow

    $deadline = (Get-Date).AddMinutes(45)
    while (-not $proc.HasExited) {
        if ((Get-Date) -gt $deadline) {
            Write-Host "TIMEOUT killing $Label after 45 min" -ForegroundColor Red
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Seconds 5
        if ((Get-Content $OutLog -ErrorAction SilentlyContinue | Select-String -Pattern '\[AUTOTEST\] ===== SUMMARY =====' -Quiet)) {
            # give a moment for remaining lines + quit
            Start-Sleep -Seconds 8
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            break
        }
    }
    Write-Host "exit=$($proc.ExitCode) log=$OutLog"
}

function Get-SummaryLines {
    param([string]$Log)
    if (-not (Test-Path $Log)) { return @() }
    Get-Content $Log | Where-Object {
        $_ -match '\[AUTOTEST\].*(SUMMARY|cold \(first|warm \(post|folder ready|TIMEOUT|workflow:)' -or
        $_ -match '\[GPU\]|gpu demosaic|PreferGpu|PREFER_GPU|CUDA'
    }
}

# CUDA probe
Write-Host "=== CUDA probe ===" -ForegroundColor Cyan
& $PixiPy -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$liteLog = Join-Path $LogDir "lite_cpu_nav_$stamp.log"
$gpuLog = Join-Path $LogDir "full_gpu_nav_$stamp.log"

Clear-ColdStart "before-lite"
Invoke-NavBench -Label "Lite-CPU" -PreferGpu "0" -OutLog $liteLog

Clear-ColdStart "before-full-gpu"
Invoke-NavBench -Label "Full-GPU" -PreferGpu "1" -OutLog $gpuLog

Write-Host ""
Write-Host "======== Lite (CPU) summary ========" -ForegroundColor Green
Get-SummaryLines $liteLog | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "======== Full (GPU) summary ========" -ForegroundColor Green
Get-SummaryLines $gpuLog | ForEach-Object { Write-Host $_ }

# Parse medians for a one-line verdict
function Get-ColdMedian([string]$Log) {
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '\[AUTOTEST\] cold \(first pass\):.*median=([0-9.]+)s' } |
        Select-Object -Last 1
    if ($line -match 'median=([0-9.]+)s') { return [double]$Matches[1] }
    return $null
}
function Get-TimeoutCount([string]$Log) {
    @(Get-Content $Log -ErrorAction SilentlyContinue | Where-Object { $_ -match '\[AUTOTEST\] TIMEOUT' }).Count
}

$liteMed = Get-ColdMedian $liteLog
$gpuMed = Get-ColdMedian $gpuLog
$liteTo = Get-TimeoutCount $liteLog
$gpuTo = Get-TimeoutCount $gpuLog

Write-Host ""
Write-Host "======== VERDICT ========" -ForegroundColor Magenta
Write-Host ("Lite CPU  cold median: {0}  TIMEOUTs: {1}" -f $(if ($null -eq $liteMed) { 'n/a' } else { "$liteMed s" }), $liteTo)
Write-Host ("Full GPU  cold median: {0}  TIMEOUTs: {1}" -f $(if ($null -eq $gpuMed) { 'n/a' } else { "$gpuMed s" }), $gpuTo)
if ($null -ne $liteMed -and $null -ne $gpuMed -and $liteMed -gt 0) {
    $ratio = $liteMed / $gpuMed
    Write-Host ("GPU vs Lite speed ratio (lite/gpu): {0:N2}x  ( >1 means GPU faster )" -f $ratio)
    if ($gpuTo -gt 0) {
        Write-Host "NOTE: Full GPU had TIMEOUTs — acceleration claim not clean." -ForegroundColor Yellow
    } elseif ($gpuMed -le $liteMed * 1.15) {
        Write-Host "PASS: Full GPU is within ~15% of Lite or faster on cold RAW nav median." -ForegroundColor Green
    } else {
        Write-Host "WARN: Full GPU cold median is >15% slower than Lite on this set." -ForegroundColor Yellow
    }
} else {
    Write-Host "INCOMPLETE: could not parse both cold medians — check logs." -ForegroundColor Red
}

Write-Host "Logs: $liteLog"
Write-Host "      $gpuLog"
