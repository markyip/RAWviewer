# GPU demosaic vs Lite CPU Fast RAW — size-vs-speed justification suite.
#
# A) Variation: I:\RAW_Sample (multi-manufacturer RAWs), full folder
# B) Stress:    first 1000 files in a large work folder (default Japan Trip)
#
# Clears ~/.rawviewer_cache + QSettings between every run. RAW HQ workflow.
#
# Usage (repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\bench\run_gpu_vs_cpu_demosaic_justification_bench.ps1
param(
    [string]$VariationEntry = "I:\RAW_Sample\020A0019.CR3",
    [string]$StressEntry = "I:\Photos\Japan Trip\DSC00001.ARW",
    [int]$StressCount = 1000,
    [int]$Revisit = 5,
    [double]$Timeout = 60,
    [int]$VariationMaxMinutes = 90,
    [int]$StressMaxMinutes = 240,
    [switch]$SkipVariation,
    [switch]$SkipStress
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PixiPy = Join-Path $Root ".pixi\envs\default\python.exe"
$Main = Join-Path $Root "src\main.py"
$LogDir = Join-Path $Root "bench\logs"
$Report = Join-Path $LogDir ("gpu_vs_cpu_demosaic_report_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $PixiPy)) { throw "Missing pixi python: $PixiPy" }
if (-not $SkipVariation -and -not (Test-Path -LiteralPath $VariationEntry)) {
    throw "Missing variation entry: $VariationEntry"
}
if (-not $SkipStress -and -not (Test-Path -LiteralPath $StressEntry)) {
    throw "Missing stress entry: $StressEntry"
}

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
        "RAWVIEWER_PREFER_GPU_DECODE",
        "RAWVIEWER_FILE_LOG",
        "RAWVIEWER_SHOW_SPLASH",
        "RAWVIEWER_ENABLE_SEMANTIC_SEARCH",
        "RAWVIEWER_SKIP_UPDATE_CHECK"
    ) | ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }
}

function Wait-ProcessOrMarker {
    param(
        [System.Diagnostics.Process]$Proc,
        [string]$OutLog,
        [string]$MarkerRegex,
        [int]$MaxMinutes,
        [int]$PostMarkerSeconds = 10
    )
    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while (-not $Proc.HasExited) {
        if ((Get-Date) -gt $deadline) {
            Write-Host "TIMEOUT killing after $MaxMinutes min" -ForegroundColor Red
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Seconds 10
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
        [int]$Count,
        [string]$OutLog,
        [int]$MaxMinutes
    )
    Write-Host ""
    Write-Host "=== run: $Label (PREFER_GPU=$PreferGpu COUNT=$Count) ===" -ForegroundColor Cyan
    Clear-AutotestEnv

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
    }

    $argList = "-u `"$Main`" `"$Entry`""
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $PixiPy -ArgumentList $argList `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError "$OutLog.err" `
        -PassThru -NoNewWindow

    Wait-ProcessOrMarker -Proc $proc -OutLog $OutLog `
        -MarkerRegex '\[AUTOTEST\] ===== SUMMARY =====' `
        -MaxMinutes $MaxMinutes
    $sw.Stop()
    Write-Host ("exit={0} wall={1:N1}min log={2}" -f $proc.ExitCode, $sw.Elapsed.TotalMinutes, $OutLog)
}

function Get-Metric([string]$Log, [string]$Label, [string]$Key) {
    # Label e.g. "cold (first pass)"; Key e.g. median|mean|p95|n
    $line = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "\[AUTOTEST\] $([regex]::Escape($Label)):" } |
        Select-Object -Last 1
    if (-not $line) { return $null }
    if ($Key -eq "n" -and $line -match 'n=(\d+)') { return [int]$Matches[1] }
    if ($line -match "$Key=([0-9.]+)s") { return [double]$Matches[1] }
    return $null
}

function Get-TimeoutCount([string]$Log) {
    @(Get-Content $Log -ErrorAction SilentlyContinue | Where-Object { $_ -match '\[AUTOTEST\] TIMEOUT' }).Count
}

function Get-GpuHitCount([string]$Log) {
    @(Get-Content $Log -ErrorAction SilentlyContinue | Where-Object {
        $_ -match 'pytorch_cuda|gpu demosaic|GPU demosaic|backend=pytorch'
    }).Count
}

function Write-PairSummary {
    param(
        [string]$Suite,
        [string]$CpuLog,
        [string]$GpuLog,
        [System.Collections.Generic.List[string]]$Lines
    )
    $cpuMed = Get-Metric $CpuLog "cold (first pass)" "median"
    $gpuMed = Get-Metric $GpuLog "cold (first pass)" "median"
    $cpuMean = Get-Metric $CpuLog "cold (first pass)" "mean"
    $gpuMean = Get-Metric $GpuLog "cold (first pass)" "mean"
    $cpuP95 = Get-Metric $CpuLog "cold (first pass)" "p95"
    $gpuP95 = Get-Metric $GpuLog "cold (first pass)" "p95"
    $cpuN = Get-Metric $CpuLog "cold (first pass)" "n"
    $gpuN = Get-Metric $GpuLog "cold (first pass)" "n"
    $cpuTo = Get-TimeoutCount $CpuLog
    $gpuTo = Get-TimeoutCount $GpuLog
    $gpuHits = Get-GpuHitCount $GpuLog

    $ratio = $null
    if ($null -ne $cpuMed -and $null -ne $gpuMed -and $gpuMed -gt 0) {
        $ratio = $cpuMed / $gpuMed
    }

    $block = @(
        "",
        "======== $Suite ========",
        ("CPU Fast RAW  n={0}  median={1}  mean={2}  p95={3}  TIMEOUT={4}" -f $cpuN, $cpuMed, $cpuMean, $cpuP95, $cpuTo),
        ("GPU demosaic  n={0}  median={1}  mean={2}  p95={3}  TIMEOUT={4}  gpu_log_hits={5}" -f $gpuN, $gpuMed, $gpuMean, $gpuP95, $gpuTo, $gpuHits),
        ("speedup (cpu_median/gpu_median): {0}" -f $(if ($null -eq $ratio) { "n/a" } else { "{0:N2}x" -f $ratio })),
        "CPU log: $CpuLog",
        "GPU log: $GpuLog"
    )
    foreach ($l in $block) {
        Write-Host $l -ForegroundColor Green
        $Lines.Add($l) | Out-Null
    }

    return @{
        Suite = $Suite
        CpuMedian = $cpuMed
        GpuMedian = $gpuMed
        Ratio = $ratio
        CpuTimeout = $cpuTo
        GpuTimeout = $gpuTo
    }
}

$reportLines = [System.Collections.Generic.List[string]]::new()
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$results = @()

Write-Host "=== CUDA probe ===" -ForegroundColor Cyan
& $PixiPy -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$reportLines.Add("GPU demosaic vs Lite CPU Fast RAW justification bench") | Out-Null
$reportLines.Add("stamp=$stamp") | Out-Null
$reportLines.Add("variation_entry=$VariationEntry") | Out-Null
$reportLines.Add("stress_entry=$StressEntry stress_count=$StressCount") | Out-Null

# ----- A) Variation (multi-manufacturer) -----
if (-not $SkipVariation) {
    $varCpu = Join-Path $LogDir "var_cpu_$stamp.log"
    $varGpu = Join-Path $LogDir "var_gpu_$stamp.log"

    Clear-ColdStart "variation-before-cpu"
    Invoke-NavBench -Label "Variation-CPU" -Entry $VariationEntry -PreferGpu "0" `
        -Count 0 -OutLog $varCpu -MaxMinutes $VariationMaxMinutes

    Clear-ColdStart "variation-before-gpu"
    Invoke-NavBench -Label "Variation-GPU" -Entry $VariationEntry -PreferGpu "1" `
        -Count 0 -OutLog $varGpu -MaxMinutes $VariationMaxMinutes

    $results += ,(Write-PairSummary -Suite "A) Variation RAW_Sample (multi-manufacturer)" `
        -CpuLog $varCpu -GpuLog $varGpu -Lines $reportLines)
}

# ----- B) Stress (1000 images) -----
if (-not $SkipStress) {
    $stressCpu = Join-Path $LogDir "stress_cpu_$stamp.log"
    $stressGpu = Join-Path $LogDir "stress_gpu_$stamp.log"

    Clear-ColdStart "stress-before-cpu"
    Invoke-NavBench -Label "Stress-CPU" -Entry $StressEntry -PreferGpu "0" `
        -Count $StressCount -OutLog $stressCpu -MaxMinutes $StressMaxMinutes

    Clear-ColdStart "stress-before-gpu"
    Invoke-NavBench -Label "Stress-GPU" -Entry $StressEntry -PreferGpu "1" `
        -Count $StressCount -OutLog $stressGpu -MaxMinutes $StressMaxMinutes

    $results += ,(Write-PairSummary -Suite "B) Stress $StressCount images" `
        -CpuLog $stressCpu -GpuLog $stressGpu -Lines $reportLines)
}

# ----- Verdict -----
$reportLines.Add("") | Out-Null
$reportLines.Add("======== SIZE vs SPEED VERDICT ========") | Out-Null
$reportLines.Add("CUDA Full install ~5.9 GB vs Lite ~0.9 GB (~6.5x disk).") | Out-Null
$reportLines.Add("GPU demosaic requires the ~4.5 GB torch+cu124 stack.") | Out-Null

$ratios = @($results | Where-Object { $null -ne $_.Ratio } | ForEach-Object { $_.Ratio })
if ($ratios.Count -gt 0) {
    $avg = ($ratios | Measure-Object -Average).Average
    $minR = ($ratios | Measure-Object -Minimum).Minimum
    $maxR = ($ratios | Measure-Object -Maximum).Maximum
    $reportLines.Add(("Observed speedup range: {0:N2}x – {1:N2}x (avg {2:N2}x)" -f $minR, $maxR, $avg)) | Out-Null
    if ($avg -ge 1.5) {
        $reportLines.Add("JUSTIFIED for users who navigate RAW HQ heavily: >=1.5x median cold speedup.") | Out-Null
    } elseif ($avg -ge 1.2) {
        $reportLines.Add("MARGINAL: ~20%+ faster — optional for power users, not worth 5x size for everyone.") | Out-Null
    } else {
        $reportLines.Add("NOT JUSTIFIED as default: speedup <1.2x does not warrant ~5x install size.") | Out-Null
    }
} else {
    $reportLines.Add("INCOMPLETE: could not compute speedup ratios — inspect logs.") | Out-Null
}

$reportLines | Set-Content -LiteralPath $Report -Encoding UTF8
Write-Host ""
Write-Host "Report: $Report" -ForegroundColor Magenta
Get-Content -LiteralPath $Report | ForEach-Object { Write-Host $_ }
