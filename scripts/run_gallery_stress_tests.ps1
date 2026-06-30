# Gallery crash isolation — runs stress_gallery_thumbs.py in isolated subprocesses.
# Does not commit; local testing only.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$folder = "I:\Photos\London"
if (-not (Test-Path $folder)) {
    Write-Error "Test folder missing: $folder"
}

$common = @{
    RAWVIEWER_USE_PROCESS_POOL = "1"
    RAWVIEWER_FILE_LOG           = "0"
    RAWVIEWER_FATAL_DUMP         = "1"
    QT_QPA_PLATFORM              = "offscreen"
}

$tests = @(
    @{
        Name = "baseline"
        Env  = @{}
    },
    @{
        Name = "A_no_semantic"
        Env  = @{ RAWVIEWER_ENABLE_SEMANTIC_SEARCH = "0" }
    },
    @{
        Name = "B_no_gpu_view"
        Env  = @{ RAWVIEWER_GPU_VIEW = "0" }
    },
    @{
        Name = "C_low_gallery_caps"
        Env  = @{
            RAWVIEWER_GALLERY_ACTIVE_CAP    = "12"
            RAWVIEWER_GALLERY_MAX_TASKS     = "8"
            RAWVIEWER_GALLERY_MAX_WIDGETS   = "24"
            RAWVIEWER_EXTERNAL_VOLUME_MAX_WORKERS = "8"
            RAWVIEWER_EXTERNAL_VOLUME_RAW_LIMIT   = "4"
        }
    },
    @{
        Name = "D_no_process_pool"
        Env  = @{ RAWVIEWER_USE_PROCESS_POOL = "0" }
    }
)

function Invoke-StressTest {
    param(
        [string]$Label,
        [hashtable]$ExtraEnv
    )
    foreach ($k in $common.Keys) {
        Set-Item -Path "Env:$k" -Value $common[$k]
    }
    foreach ($k in $ExtraEnv.Keys) {
        Set-Item -Path "Env:$k" -Value $ExtraEnv[$k]
    }

    Write-Host ""
    Write-Host "========== $Label ==========" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & pixi run python scripts/stress_gallery_thumbs.py `
        --label $Label `
        --folder $folder `
        --file-limit 320 `
        --waves 6 `
        --batch-size 64 `
        --wave-pause 2.0
    $code = $LASTEXITCODE
    $sw.Stop()
    $hex = if ($code -lt 0) { "0x{0:X8}" -f [uint32]$code } else { $code.ToString() }
    $crashed = ($code -ne 0)
    [pscustomobject]@{
        Test     = $Label
        ExitCode = $code
        Hex      = $hex
        Seconds  = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        Crashed  = $crashed
    }
}

$results = @()
foreach ($t in $tests) {
    $results += Invoke-StressTest -Label $t.Name -ExtraEnv $t.Env
}

Write-Host ""
Write-Host "========== SUMMARY ==========" -ForegroundColor Yellow
$results | Format-Table -AutoSize

$crashers = $results | Where-Object { $_.Crashed }
if ($crashers) {
    Write-Host "Crashed/failed:" ($crashers.Test -join ", ")
} else {
    Write-Host "All headless stress tests completed without process crash."
    Write-Host "Native crash may require full GUI gallery scroll — see stress_gallery_gui.py"
}
