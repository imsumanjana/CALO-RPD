param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& python (Join-Path $root "tools/ai_repo_intelligence.py") @Arguments
exit $LASTEXITCODE
