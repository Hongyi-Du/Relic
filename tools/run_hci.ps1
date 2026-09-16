<#
.SYNOPSIS
    Run the human member workspace on Windows. No WSL, no dataset, no API key.

.DESCRIPTION
    Finds a Python 3.12+, creates a small private virtualenv the first time
    (.venv-hci, four packages), then starts the workspace. If the repository
    already has a usable .venv it uses that instead of making a second one.

.EXAMPLE
    .\tools\run_hci.ps1
    .\tools\run_hci.ps1 -BindAll        # let other machines take seats
    .\tools\run_hci.ps1 -Llm            # LLM-backed working agent
#>
[CmdletBinding()]
param(
    [int]$Port = 8100,
    [switch]$BindAll,
    [switch]$Llm,
    [switch]$Paused,
    [int]$Warmup = 36,
    [double]$TickSeconds = 15,
    [switch]$OpenBrowser,
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$hciVenv = Join-Path $repo '.venv-hci'

# Single-quoted so PowerShell leaves the Python alone.
$VersionProbe = 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'
$ImportProbe  = 'import fastapi, uvicorn'

# Probes are expected to fail; a native command writing to stderr would abort
# the whole script under ErrorActionPreference=Stop, so run them isolated.
function Invoke-Probe {
    param([string]$Exe, [string[]]$Argv)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Argv 2>&1 | Out-Null
        return $LASTEXITCODE
    } catch {
        return 1
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Python([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    return (Invoke-Probe $exe @('-c', $VersionProbe)) -eq 0
}

function Test-Deps([string]$exe) {
    return (Invoke-Probe $exe @('-c', $ImportProbe)) -eq 0
}

function Find-BasePython {
    # `py` is the Windows launcher and the most reliable way to reach a specific
    # version; fall back to whatever `python` is on PATH.
    if (Get-Command 'py' -ErrorAction SilentlyContinue) {
        foreach ($tag in @('-3.13', '-3.12', '-3')) {
            if ((Invoke-Probe 'py' @($tag, '-c', $VersionProbe)) -eq 0) {
                return @('py', $tag)
            }
        }
    }
    if (Get-Command 'python' -ErrorAction SilentlyContinue) {
        if ((Invoke-Probe 'python' @('-c', $VersionProbe)) -eq 0) { return @('python') }
    }
    return $null
}

Push-Location $repo
try {
    if ($Rebuild -and (Test-Path $hciVenv)) {
        Write-Host 'Removing .venv-hci...'
        Remove-Item -Recurse -Force $hciVenv
    }

    $python = $null
    foreach ($candidate in @((Join-Path $hciVenv 'Scripts\python.exe'),
                             (Join-Path $repo '.venv\Scripts\python.exe'))) {
        if (Test-Python $candidate) { $python = $candidate; break }
    }

    if (-not $python) {
        $base = Find-BasePython
        if (-not $base) {
            Write-Host ''
            Write-Host 'No Python 3.12 or newer was found.' -ForegroundColor Red
            Write-Host ''
            Write-Host 'Install it from https://www.python.org/downloads/windows/'
            Write-Host '(tick "Add python.exe to PATH"), or run the workspace in'
            Write-Host 'Docker instead, which needs no local Python:'
            Write-Host ''
            Write-Host '    docker compose -f docker-compose.hci.yml up'
            Write-Host ''
            exit 2
        }
        Write-Host "Creating .venv-hci with $($base -join ' ')..."
        & $base[0] @($base[1..($base.Count - 1)]) -m venv $hciVenv
        if ($LASTEXITCODE -ne 0) { Write-Host 'venv creation failed' -ForegroundColor Red; exit 1 }
        $python = Join-Path $hciVenv 'Scripts\python.exe'
    }

    if (-not (Test-Deps $python)) {
        Write-Host 'Installing the four packages the workspace needs...'
        # pip narrates to stderr even when it succeeds.
        $ErrorActionPreference = 'Continue'
        & $python -m pip install --disable-pip-version-check -q --upgrade pip 2>&1 | Out-Null
        & $python -m pip install --disable-pip-version-check -q -r (Join-Path $repo 'requirements-hci.txt') 2>&1 |
            ForEach-Object { Write-Host "  $_" }
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) { Write-Host 'pip install failed' -ForegroundColor Red; exit 1 }
        if (-not (Test-Deps $python)) {
            Write-Host 'fastapi/uvicorn still missing after install' -ForegroundColor Red
            exit 1
        }
    }

    $argv = @((Join-Path $repo 'tools\run_hci.py'),
              '--port', $Port, '--warmup', $Warmup, '--tick-seconds', $TickSeconds)
    if ($BindAll)     { $argv += @('--host', '0.0.0.0') }
    if ($Llm)         { $argv += '--llm' }
    if ($Paused)      { $argv += '--paused' }
    if ($OpenBrowser) { $argv += '--open' }

    $env:PYTHONPATH = $repo
    & $python @argv
}
finally {
    Pop-Location
}
