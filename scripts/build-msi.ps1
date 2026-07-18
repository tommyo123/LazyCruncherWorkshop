#!/usr/bin/env pwsh
# Build the Windows MSI installer locally, end to end.
#
#   ./scripts/build-msi.ps1
#
# Needs cargo-wix (`cargo install cargo-wix`) and the WiX Toolset 3.x.
# The MSI lands in target\wix\.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    cargo msi @args
    if ($LASTEXITCODE -ne 0) { throw "cargo msi failed" }
    Get-ChildItem target\wix\*.msi | Select-Object Name, Length
}
finally {
    Pop-Location
}
