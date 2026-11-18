param([ValidateRange(1024,65535)][int]$Port = 8780)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Run deploy\setup.ps1 first' }
Push-Location -LiteralPath $projectRoot
try {
    Write-Output "Open http://127.0.0.1:$Port"
    & $runtime -m facade_mcp.web_app --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Web service exited with an error' }
} finally {
    Pop-Location
}
