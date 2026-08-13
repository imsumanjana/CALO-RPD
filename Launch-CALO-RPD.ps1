[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ApplicationArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    Write-Error @"
CALO-RPD Studio's repository virtual environment was not found:
  $PythonExecutable

First setup is separate from routine launch. From this directory run:
  py -3.11 -m venv .venv
  .\.venv\Scripts\python.exe bootstrap.py --setup

No packages were installed or upgraded by this launcher.
"@
    exit 2
}

Push-Location -LiteralPath $ProjectRoot
try {
    & $PythonExecutable -m calo_rpd_studio.app.application @ApplicationArguments
    $ApplicationExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $ApplicationExitCode
