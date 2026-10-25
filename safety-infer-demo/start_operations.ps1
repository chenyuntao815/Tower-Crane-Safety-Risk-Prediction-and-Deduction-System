param(
    [string]$Python = "python",
    [ValidateRange(1, 65535)][int]$Port = 8000
)
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Crane operations: http://localhost:$Port/twin/"
Write-Host "Sensors, Neo4j and an LLM key are optional."
& $Python -m uvicorn app:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
