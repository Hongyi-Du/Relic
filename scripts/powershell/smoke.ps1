[CmdletBinding()]
param(
    [string]$Distribution,
    [string]$RepositoryPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)

. (Join-Path $PSScriptRoot '_RelicWsl.ps1')

$configuration = Get-RelicWslConfiguration `
    -Distribution $Distribution `
    -RepositoryPath $RepositoryPath
$result = Invoke-RelicWslBashScript `
    -Distribution $configuration.Distribution `
    -RepositoryPath $configuration.RepositoryPath `
    -ScriptName 'smoke' `
    -ScriptArguments $ScriptArguments
$result.Output | Write-Output
exit $result.ExitCode
