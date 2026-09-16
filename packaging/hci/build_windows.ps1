param(
    [string]$Version = "0.1.13"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$python = (Resolve-Path (Join-Path $repoRoot ".venv-hci\Scripts\python.exe")).Path
$config = (Resolve-Path (Join-Path $repoRoot "config\llm.local.yaml")).Path
$isccCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)
$iscc = $isccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup first."
}

Push-Location $repoRoot
try {
    & $python -m pip install --disable-pip-version-check `
        -r (Join-Path $scriptRoot "requirements-build.txt")
    if ($LASTEXITCODE -ne 0) { throw "Build dependency install failed with exit code $LASTEXITCODE" }

    npm --prefix "environments\org_env\frontend\app" run build
    $env:SECRETARY_LLM_CONFIG = $config
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath (Join-Path $scriptRoot "build\dist") `
        --workpath (Join-Path $scriptRoot "build\work") `
        (Join-Path $scriptRoot "secretary.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    foreach ($mode in @("--p2", "--p3")) {
        & (Join-Path $scriptRoot "build\dist\Secretary\Secretary.exe") --self-test $mode
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged Secretary $mode self-test failed with exit code $LASTEXITCODE"
        }
    }

    $runnerRepo = Join-Path $repoRoot "benchmarks\relic-main-v1\packs\traffic_watch_v1\starter_repo"
    Push-Location $runnerRepo
    try {
        $env:PYTHONDONTWRITEBYTECODE = "1"
        $embeddedImport = Start-Process `
            -FilePath (Join-Path $scriptRoot "build\dist\Secretary\Secretary.exe") `
            -ArgumentList "-c", "import", "traffic_violation_system" `
            -WorkingDirectory $runnerRepo `
            -WindowStyle Hidden `
            -PassThru `
            -Wait
        if ($embeddedImport.ExitCode -ne 0) {
            throw "Packaged fragmented -c import failed with exit code $($embeddedImport.ExitCode)"
        }
        & (Join-Path $scriptRoot "build\dist\Secretary\Secretary.exe") `
            -m pytest tests/public/test_smoke.py -q -k package_imports `
            --confcutdir=tests/public -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Packaged worker test runner failed with exit code $LASTEXITCODE" }
    } finally {
        Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
        Pop-Location
    }

    # Keep exactly one selectable installer.  Leaving 0.1.0 next to the fixed
    # build made it too easy to launch the known-broken no-console package.
    Get-ChildItem -LiteralPath (Join-Path $scriptRoot "release") `
        -Filter "Relic-Secretary-Setup-*.exe" -ErrorAction SilentlyContinue |
        Remove-Item -Force

    & $iscc "/DMyAppVersion=$Version" (Join-Path $scriptRoot "secretary-installer.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Get-ChildItem -LiteralPath (Join-Path $scriptRoot "release") -Filter "*.exe"
