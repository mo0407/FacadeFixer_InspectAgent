param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $runtime)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment' }
    }
    & $runtime -m pip install -e '.[dev,web,pdf]'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
    }
    Write-Output 'Installed. Edit .env, then run .\deploy\start-web.ps1'
} finally {
    Pop-Location
}
