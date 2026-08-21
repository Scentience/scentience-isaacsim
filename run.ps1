#requires -version 5
<#
  Convenience launcher for scentience-olfaction.
  Usage:  .\run.ps1 [test|examples|physics|provenance|isaac|shell]
  See SETUP.md. Isaac status: docs/ISAAC_COMPATIBILITY.md
#>
param([string]$Target = "test")

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$py   = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "No venv at $py -- see SETUP.md to create it."
}

switch ($Target.ToLower()) {
    "test"       { & $py -m pytest -m "not isaac" -q }
    "physics"    { & $py (Join-Path $repo "scripts\validate_physics.py") }
    "provenance" { & $py (Join-Path $repo "scripts\provenance_demo.py") }
    "examples"   {
        Get-ChildItem (Join-Path $repo "examples") -Filter "*.py" | Sort-Object Name | ForEach-Object {
            Write-Host "`n===== $($_.Name) =====" -ForegroundColor Cyan
            & $py $_.FullName
        }
    }
    "isaac"      {
        # Runs inside Isaac Sim's own interpreter, not the venv.
        $isaac = "C:\isaacsim\python.bat"
        if (-not (Test-Path $isaac)) { Write-Error "Isaac Sim not found at C:\isaacsim" }
        Write-Host "Expected to FAIL on this machine -- see docs/ISAAC_COMPATIBILITY.md" -ForegroundColor Yellow
        $env:PYTHONPATH = $repo
        $env:OMNI_KIT_ACCEPT_EULA = "YES"
        & $isaac (Join-Path $repo "scripts\validate_install.py")
    }
    "shell"      { & $py }
    default      { Write-Error "Unknown target '$Target'. Use: test|examples|physics|provenance|isaac|shell" }
}
