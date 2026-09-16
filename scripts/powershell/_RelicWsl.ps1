Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RelicWslConfiguration {
    [CmdletBinding()]
    param(
        [string]$Distribution,
        [string]$RepositoryPath
    )

    if ($null -eq (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'wsl.exe was not found. Install WSL2 and run this launcher from Windows PowerShell.'
    }

    if ([string]::IsNullOrWhiteSpace($Distribution)) {
        $Distribution = $env:RELIC_WSL_DISTRIBUTION
    }
    if ([string]::IsNullOrWhiteSpace($Distribution)) {
        throw 'Specify -Distribution or set RELIC_WSL_DISTRIBUTION to the target WSL distribution.'
    }
    if ($Distribution.IndexOf([char]0) -ge 0) {
        throw 'The WSL distribution name cannot contain a null character.'
    }

    if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
        $RepositoryPath = $env:RELIC_WSL_REPO
    }
    if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
        throw 'Specify -RepositoryPath or set RELIC_WSL_REPO to the Linux absolute path of the Relic repository.'
    }
    if ($RepositoryPath.IndexOf([char]0) -ge 0) {
        throw 'The WSL repository path cannot contain a null character.'
    }
    if (-not $RepositoryPath.StartsWith('/')) {
        throw 'The Relic repository path must be a Linux absolute path inside the selected WSL distribution.'
    }
    if ($RepositoryPath -match '^/mnt(?:/|$)') {
        throw 'The Relic repository must live on the WSL filesystem, not under /mnt.'
    }

    $listArguments = @('--list', '--quiet')
    $availableOutput = & wsl.exe @listArguments
    if ($LASTEXITCODE -ne 0) {
        throw "wsl.exe could not list installed distributions (exit code $LASTEXITCODE)."
    }
    $availableDistributions = @(
        $availableOutput |
            ForEach-Object { $_.ToString().Trim().Trim([char]0) } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($availableDistributions -notcontains $Distribution) {
        throw "WSL distribution '$Distribution' was not found. Use 'wsl.exe --list --verbose' to see installed distributions."
    }

    $kernelArguments = @(
        '--distribution'
        $Distribution
        '--exec'
        'cat'
        '/proc/sys/kernel/osrelease'
    )
    $kernelRelease = (& wsl.exe @kernelArguments | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $kernelRelease -notmatch '(?i)(microsoft-standard|wsl2)') {
        throw "WSL distribution '$Distribution' is not running on the supported WSL2 kernel."
    }

    $repositoryProbeArguments = @(
        '--distribution'
        $Distribution
        '--exec'
        'test'
        '-d'
        $RepositoryPath
    )
    & wsl.exe @repositoryProbeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The Relic repository path does not exist in WSL distribution '$Distribution': $RepositoryPath"
    }

    return [pscustomobject]@{
        Distribution = $Distribution
        RepositoryPath = $RepositoryPath
    }
}

function Invoke-RelicWslBashScript {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Distribution,

        [Parameter(Mandatory)]
        [string]$RepositoryPath,

        [Parameter(Mandatory)]
        [ValidateSet('check_env', 'smoke', 'run_cell', 'run_main_120', 'evaluate', 'start_inspector')]
        [string]$ScriptName,

        [string[]]$ScriptArguments = @()
    )

    # Do not use Join-Path: on Windows it would turn this Linux path into a backslash path.
    $wslScript = "{0}/scripts/bash/{1}.sh" -f $RepositoryPath.TrimEnd('/'), $ScriptName
    $scriptProbeArguments = @(
        '--distribution'
        $Distribution
        '--exec'
        'test'
        '-f'
        $wslScript
    )
    & wsl.exe @scriptProbeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Expected WSL launcher script was not found: $wslScript"
    }

    # Each item remains an argument; do not build a shell command string here.
    $wslArguments = @(
        '--distribution'
        $Distribution
        '--exec'
        'bash'
        $wslScript
    )
    if ($null -ne $ScriptArguments) {
        $wslArguments += @($ScriptArguments)
    }
    # Capture only the native success stream so this function returns one
    # object instead of mixing CLI output lines with the exit code.
    $commandOutput = @(& wsl.exe @wslArguments)
    $commandExitCode = $LASTEXITCODE
    return [pscustomobject]@{
        ExitCode = [int]$commandExitCode
        Output = @($commandOutput)
    }
}
