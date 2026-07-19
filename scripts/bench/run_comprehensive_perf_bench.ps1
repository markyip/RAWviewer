# Comprehensive RAWviewer performance suite:
#   A) RAW_Sample Lite vs Full GPU nav (RAW HQ)
#   B) Japan Trip gallery cold suite
#   C) JPEG Mach Loop folder nav (embedded workflow)
param(
    [string]$RawEntry = "I:\RAW_Sample\020A0019.CR3",
    [string]$JapanEntry = "I:\Photos\Japan Trip\DSC03534.ARW",
    [string]$JpegEntry = "I:\Photos\23092025 Mach Loop JPEG\sRGB Glossy Paper Selected\DSC03348.jpg",
    [int]$RawCount = 0,
    [int]$JpegCount = 0,
    [int]$Revisit = 5,
    [double]$Timeout = 45,
    [switch]$SkipA,
    [switch]$SkipB,
    [switch]$SkipC
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PixiPy = Join-Path $Root ".pixi\envs\default\python.exe"
$Main = Join-Path $Root "src\main.py"
$LogDir = Join-Path $Root "bench\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $PixiPy)) { throw "Missing pixi python: $PixiPy" }
if (-not (Test-Path I:\)) { throw "I: drive not mounted" }
foreach ($e in @($RawEntry, $JapanEntry, $JpegEntry)) {
    if (-not (Test-Path -LiteralPath $e)) { throw "Missing entry: $e" }
}

function Clear-ColdStart {
    param([string]$Label)
    Write-Host ""
    Write-Host "=== clear cache: $Label ===" -ForegroundColor Yellow
    Get-Process RAWviewer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'main\.py|nav_autotest|gallery_autotest|RAWviewer') } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    $cache = Join-Path $env:USERPROFILE ".rawviewer_cache"
    if (Test-Path $cache) {
        Remove-Item -LiteralPath $cache -Recurse -Force -ErrorAction SilentlyContinue
    }
    cmd /c "reg delete HKCU\Software\RAWviewer /f >nul 2>&1" | Out-Null
    if (Test-Path $cache) { Write-Host "WARNING: cache still present" -ForegroundColor Red }
    else { Write-Host "cache cleared" }
}

function Clear-AutotestEnv {
    @(
        "RAWVIEWER_AUTOTEST",
        "RAWVIEWER_AUTOTEST_RAW_MODE",
        "RAWVIEWER_AUTOTEST_COUNT",
        "RAWVIEWER_AUTOTEST_REVISIT",
        "RAWVIEWER_AUTOTEST_TIMEOUT",
        "RAWVIEWER_AUTOTEST_SETTLE",
        "RAWVIEWER_GALLERY_AUTOTEST",
        "RAWVIEWER_GALLERY_AUTOTEST_SECONDS",
        "RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ",
        "RAWVIEWER_GALLERY_AUTOTEST_SETTLE",
        "RAWVIEWER_GALLERY_AUTOTEST_GALLERY_WAIT",
        "RAWVIEWER_GALLERY_AUTOTEST_FIRST_RENDER_WAIT",
        "RAWVIEWER_GALLERY_AUTOTEST_DIRECTION",
        "RAWVIEWER_PREFER_GPU_DECODE",
        "RAWVIEWER_FILE_LOG",
        "RAWVIEWER_SHOW_SPLASH",
        "RAWVIEWER_ENABLE_SEMANTIC_SEARCH",
        "RAWVIEWER_SKIP_UPDATE_CHECK"
    ) | ForEach-Object {
        Remove-Item "Env:$_" -ErrorAction SilentlyContinue
    }
}

function Wait-ProcessOrMarker {
    param(
        [System.Diagnostics.Process]$Proc,
        [string]$OutLog,
        [string]$MarkerRegex,
        [int]$MaxMinutes = 45,
        [int]$PostMarkerSeconds = 8
    )
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while (-not $Proc.HasExited) {
        if ((Get-Date) -gt $deadline) {
            Write-Host "TIMEOUT killing after $MaxMinutes min" -ForegroundColor Red
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Seconds 5
        if ((Get-Content $OutLog -ErrorAction SilentlyContinue | Select-String -Pattern $MarkerRegex -Quiet)) {
            Start-Sleep -Seconds $PostMarkerSeconds
            if (-not $Proc.HasExited) {
                Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            }
            break
        }
    }
}

function Invoke-NavBench {
    param(
        [string]$Label,
        [string]$Entry,
        [string]$PreferGpu,
        [string]$RawMode,
        [int]$Count,
        [string]$OutLog,
        [int]$MaxMinutes = 45
    )
    Write-Host ""
    Write-Host "=== run: $Label (PREFER_GPU=$PreferGpu RAW_MODE=$RawMode) ===" -ForegroundColor Cyan
    Clear-AutotestEnv

    $env:RAWVIEWER_AUTOTEST = "1"
    $env:RAWVIEWER_AUTOTEST_RAW_MODE = $RawMode
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
    }

    $argList = "-u `"$Main`" `"$Entry`""
    $proc = Start-Process -FilePath $PixiPy -ArgumentList $argList `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError "$OutLog.err" `
        -PassThru -NoNewWindow

    Wait-ProcessOrMarker -Proc $proc -OutLog $OutLog `
        -MarkerRegex '\[AUTOTEST\] ===== SUMMARY =====' `
        -MaxMinutes $MaxMinutes -PostMarkerSeconds 8
    Write-Host "exit=$($proc.ExitCode) log=$OutLog"
}

function Invoke-GalleryBench {
    param(
        [string]$Label,
        [string]$Entry,
        [string]$PreferGpu,
        [string]$OutLog,
        [int]$MaxMinutes = 30
    )
    Write-Host ""
    Write-Host "=== run: $Label (PREFER_GPU=$PreferGpu gallery) ===" -ForegroundColor Cyan
    Clear-AutotestEnv

    $env:RAWVIEWER_GALLERY_AUTOTEST = "1"
    $env:RAWVIEWER_GALLERY_AUTOTEST_SECONDS = "20"
    $env:RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ = "30"
    $env:RAWVIEWER_GALLERY_AUTOTEST_SETTLE = "3.0"
    $env:RAWVIEWER_GALLERY_AUTOTEST_GALLERY_WAIT = "900"
    $env:RAWVIEWER_GALLERY_AUTOTEST_FIRST_RENDER_WAIT = "600"
    $env:RAWVIEWER_GALLERY_AUTOTEST_DIRECTION = "up"
    $env:RAWVIEWER_PREFER_GPU_DECODE = $PreferGpu
    $env:RAWVIEWER_FILE_LOG = "1"
    $env:RAWVIEWER_SHOW_SPLASH = "0"
    $env:RAWVIEWER_ENABLE_SEMANTIC_SEARCH = "0"
    $env:RAWVIEWER_SKIP_UPDATE_CHECK = "1"

    $argList = "-u `"$Main`" `"$Entry`""
    $proc = Start-Process -FilePath $PixiPy -ArgumentList $argList `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError "$OutLog.err" `
        -PassThru -NoNewWindow

    Wait-ProcessOrMarker -Proc $proc -OutLog $OutLog `
        -MarkerRegex '\[GALLERYTEST\] DONE|\[GALLERYTEST\] quit:' `
        -MaxMinutes $MaxMinutes -PostMarkerSeconds 10
    Write-Host "exit=$($proc.ExitCode) log=$OutLog"
}

function Get-ColdMedian([string]$Log) {
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '\[AUTOTEST\] cold \(first pass\):.*median=([0-9.]+)s' } |
        Select-Object -Last 1
    if ($line -match 'median=([0-9.]+)s') { return [double]$Matches[1] }
    return $null
}
function Get-ColdMean([string]$Log) {
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '\[AUTOTEST\] cold \(first pass\):.*mean=([0-9.]+)s' } |
        Select-Object -Last 1
    if ($line -match 'mean=([0-9.]+)s') { return [double]$Matches[1] }
    return $null
}
function Get-ColdP95([string]$Log) {
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '\[AUTOTEST\] cold \(first pass\):.*p95=([0-9.]+)s' } |
        Select-Object -Last 1
    if ($line -match 'p95=([0-9.]+)s') { return [double]$Matches[1] }
    return $null
}
function Get-TimeoutCount([string]$Log) {
    @(Get-Content $Log -ErrorAction SilentlyContinue | Where-Object { $_ -match '\[AUTOTEST\] TIMEOUT' }).Count
}
function Get-GpuCudaDecodeCount([string]$Log) {
    $patterns = @('\[GPU\].*cuda', 'gpu demosaic', 'CUDA demosaic', 'GPU decode', 'cuda_decode', 'demosaic.*cuda')
    $n = 0
    Get-Content $Log -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($p in $patterns) {
            if ($_ -match $p) { $n++; break }
        }
    }
    return $n
}
function Get-GalleryField([string]$Log, [string]$Field) {
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '\[GALLERYTEST\] DONE' } |
        Select-Object -Last 1
    if ($null -eq $line) {
        $line = Get-Content $Log -ErrorAction SilentlyContinue |
            Where-Object { $_ -match "\[GALLERYTEST\].*$Field=" } |
            Select-Object -Last 1
    }
    if ($line -match "$Field=([0-9.]+|None)") { return $Matches[1] }
    return $null
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$results = [ordered]@{}

Write-Host "=== CUDA probe ===" -ForegroundColor Cyan
& $PixiPy -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

$liteLog = Join-Path $LogDir "A_raw_sample_lite_$stamp.log"
$gpuLog = Join-Path $LogDir "A_raw_sample_gpu_$stamp.log"
$galleryLog = Join-Path $LogDir "B_japan_gallery_$stamp.log"
$jpegLog = Join-Path $LogDir "C_jpeg_machloop_$stamp.log"

if (-not $SkipA) {
    Clear-ColdStart "A-before-lite"
    Invoke-NavBench -Label "A-Lite-CPU" -Entry $RawEntry -PreferGpu "0" -RawMode "1" `
        -Count $RawCount -OutLog $liteLog -MaxMinutes 50

    Clear-ColdStart "A-before-full-gpu"
    Invoke-NavBench -Label "A-Full-GPU" -Entry $RawEntry -PreferGpu "1" -RawMode "1" `
        -Count $RawCount -OutLog $gpuLog -MaxMinutes 50

    $results.A_lite_median = Get-ColdMedian $liteLog
    $results.A_lite_mean = Get-ColdMean $liteLog
    $results.A_lite_p95 = Get-ColdP95 $liteLog
    $results.A_lite_timeouts = Get-TimeoutCount $liteLog
    $results.A_gpu_median = Get-ColdMedian $gpuLog
    $results.A_gpu_mean = Get-ColdMean $gpuLog
    $results.A_gpu_p95 = Get-ColdP95 $gpuLog
    $results.A_gpu_timeouts = Get-TimeoutCount $gpuLog
    $results.A_gpu_cuda_decode = Get-GpuCudaDecodeCount $gpuLog
}

if (-not $SkipB) {
    Clear-ColdStart "B-before-japan-gallery"
    Invoke-GalleryBench -Label "B-Japan-Gallery" -Entry $JapanEntry -PreferGpu "1" `
        -OutLog $galleryLog -MaxMinutes 45

    $results.B_gallery_ready = Get-GalleryField $galleryLog "gallery_ready_s"
    $results.B_first_render = Get-GalleryField $galleryLog "first_render_s"
    $results.B_tiles = Get-GalleryField $galleryLog "rendered_tiles"
}

if (-not $SkipC) {
    Clear-ColdStart "C-before-jpeg-nav"
    Invoke-NavBench -Label "C-JPEG-Nav" -Entry $JpegEntry -PreferGpu "0" -RawMode "0" `
        -Count $JpegCount -OutLog $jpegLog -MaxMinutes 30

    $results.C_median = Get-ColdMedian $jpegLog
    $results.C_mean = Get-ColdMean $jpegLog
    $results.C_p95 = Get-ColdP95 $jpegLog
    $results.C_timeouts = Get-TimeoutCount $jpegLog
}

Clear-ColdStart "after-suite"

Write-Host ""
Write-Host "======== COMPARISON TABLE ========" -ForegroundColor Magenta
Write-Host ("{0,-28} {1,-14} {2,-14} {3,-10} {4}" -f "Suite", "median(s)", "mean(s)", "TIMEOUT", "notes")
Write-Host ("-" * 90)
if (-not $SkipA) {
    Write-Host ("{0,-28} {1,-14} {2,-14} {3,-10} {4}" -f "A RAW_Sample Lite",
        $(if ($null -eq $results.A_lite_median) { "n/a" } else { "{0:N3}" -f $results.A_lite_median }),
        $(if ($null -eq $results.A_lite_mean) { "n/a" } else { "{0:N3}" -f $results.A_lite_mean }),
        $results.A_lite_timeouts, "p95=$($results.A_lite_p95)")
    Write-Host ("{0,-28} {1,-14} {2,-14} {3,-10} {4}" -f "A RAW_Sample Full GPU",
        $(if ($null -eq $results.A_gpu_median) { "n/a" } else { "{0:N3}" -f $results.A_gpu_median }),
        $(if ($null -eq $results.A_gpu_mean) { "n/a" } else { "{0:N3}" -f $results.A_gpu_mean }),
        $results.A_gpu_timeouts, "cuda_decode~$($results.A_gpu_cuda_decode) p95=$($results.A_gpu_p95)")
}
if (-not $SkipB) {
    Write-Host ("{0,-28} {1,-14} {2,-14} {3,-10} {4}" -f "B Japan gallery",
        "ready=$($results.B_gallery_ready)",
        "1st=$($results.B_first_render)",
        "-",
        "tiles=$($results.B_tiles)")
}
if (-not $SkipC) {
    Write-Host ("{0,-28} {1,-14} {2,-14} {3,-10} {4}" -f "C JPEG Mach Loop",
        $(if ($null -eq $results.C_median) { "n/a" } else { "{0:N3}" -f $results.C_median }),
        $(if ($null -eq $results.C_mean) { "n/a" } else { "{0:N3}" -f $results.C_mean }),
        $results.C_timeouts, "p95=$($results.C_p95)")
}

Write-Host ""
Write-Host "======== VERDICT ========" -ForegroundColor Magenta
if (-not $SkipA -and $null -ne $results.A_lite_median -and $null -ne $results.A_gpu_median -and $results.A_lite_median -gt 0) {
    $ratio = $results.A_lite_median / $results.A_gpu_median
    Write-Host ("GPU vs Lite speed ratio (lite/gpu): {0:N2}x  ( >1 means GPU faster )" -f $ratio)
    if ($results.A_gpu_timeouts -gt 0) {
        Write-Host "NOTE: Full GPU had TIMEOUTs - acceleration claim not clean." -ForegroundColor Yellow
        Write-Host "VERDICT: INCONCLUSIVE (GPU timeouts)" -ForegroundColor Yellow
    } elseif ($results.A_gpu_median -lt $results.A_lite_median) {
        Write-Host "VERDICT: Full GPU ACCELERATES vs Lite on RAW_Sample cold median." -ForegroundColor Green
    } elseif ($results.A_gpu_median -le $results.A_lite_median * 1.15) {
        Write-Host "VERDICT: Full GPU ~parity with Lite (within 15%) on RAW_Sample." -ForegroundColor Green
    } else {
        Write-Host "VERDICT: Full GPU does NOT accelerate vs Lite on RAW_Sample (slower cold median)." -ForegroundColor Yellow
    }
} elseif (-not $SkipA) {
    Write-Host "VERDICT: INCOMPLETE - could not parse both A cold medians." -ForegroundColor Red
}

Write-Host ""
Write-Host "Logs:"
Write-Host "  $liteLog"
Write-Host "  $gpuLog"
Write-Host "  $galleryLog"
Write-Host "  $jpegLog"

$summaryPath = Join-Path $LogDir "comprehensive_summary_$stamp.txt"
$results.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host "Summary file: $summaryPath"
