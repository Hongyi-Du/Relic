[CmdletBinding()]
param(
    [string]$Distribution,
    [string]$RepositoryPath
)

. (Join-Path $PSScriptRoot '_RelicWsl.ps1')

$configuration = Get-RelicWslConfiguration `
    -Distribution $Distribution `
    -RepositoryPath $RepositoryPath
Write-Output ("WSL distribution: {0}" -f $configuration.Distribution)
Write-Output ("WSL repository: {0}" -f $configuration.RepositoryPath)
