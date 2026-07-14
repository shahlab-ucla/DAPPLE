[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Recreate,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$previousLocation = Get-Location

try {
    Set-Location -LiteralPath $repoRoot

    $venvRoot = Join-Path $repoRoot ".venv"
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if ($Recreate -and (Test-Path -LiteralPath $venvRoot)) {
        # This is intentionally strict: recursive deletion is allowed only for the
        # repository-local directory named exactly '.venv'.
        $venvItem = Get-Item -LiteralPath $venvRoot -Force
        if (($venvItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to recursively remove a .venv symlink or junction: $venvRoot"
        }
        $expectedVenv = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".venv"))
        $resolvedRepo = [System.IO.Path]::GetFullPath($repoRoot)
        $venvParent = [System.IO.Directory]::GetParent($expectedVenv).FullName
        if (($venvParent -ne $resolvedRepo) -or
            ([System.IO.Path]::GetFileName($expectedVenv) -ne ".venv")) {
            throw "Refusing to remove unexpected virtual-environment path: $expectedVenv"
        }
        Write-Host "Removing repository-local .venv because -Recreate was requested..."
        Remove-Item -LiteralPath $expectedVenv -Recurse -Force
    }

    if ((Test-Path -LiteralPath $venvRoot) -and
        -not (Test-Path -LiteralPath $venvPython)) {
        throw "The existing .venv is incomplete or stale. Rerun with -Recreate to replace only this repository's .venv."
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($Python) {
            $bootstrapExecutable = $Python
            $bootstrapArguments = @()
        }
        elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $bootstrapExecutable = "py"
            $bootstrapArguments = @("-3")
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            $bootstrapExecutable = "python"
            $bootstrapArguments = @()
        }
        else {
            throw "Python 3.11 or 3.12 was not found. Install a supported Python, then rerun this script."
        }

        try {
            Invoke-Checked -Executable $bootstrapExecutable -Arguments (
                $bootstrapArguments + @(
                    "-c",
                    "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 'DAPPLE requires Python 3.11 or 3.12')"
                )
            )
        }
        catch {
            throw "The selected bootstrap interpreter is not usable Python 3.11 or 3.12. Pass -Python with a supported interpreter path. Original error: $($_.Exception.Message)"
        }

        Write-Host "Creating .venv with $bootstrapExecutable..."
        Invoke-Checked -Executable $bootstrapExecutable -Arguments (
            $bootstrapArguments + @("-m", "venv", ".venv")
        )
    }
    else {
        Write-Host "Reusing existing .venv."
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Virtual environment creation did not produce $venvPython"
    }

    # Detect stale environments (for example, one whose base Python was removed)
    # before attempting a less-informative pip invocation.
    try {
        Invoke-Checked -Executable $venvPython -Arguments @(
            "-c",
            "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 'DAPPLE requires Python 3.11 or 3.12')"
        )
    }
    catch {
        if (-not $Recreate) {
            throw "The existing .venv is stale, broken, or uses an unsupported Python. Rerun with -Recreate to replace only this repository's .venv. Original error: $($_.Exception.Message)"
        }
        throw
    }

    Write-Host "Updating packaging tools..."
    Invoke-Checked -Executable $venvPython -Arguments @(
        "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
    )

    if ($Dev) {
        Write-Host "Installing DAPPLE with GUI and contributor dependencies..."
        Invoke-Checked -Executable $venvPython -Arguments @(
            "-m", "pip", "install", "--editable", ".[gui,dev]"
        )
    }
    else {
        Write-Host "Installing DAPPLE with the tested GUI dependency set..."
        Invoke-Checked -Executable $venvPython -Arguments @(
            "-m", "pip", "install", ".[gui]"
        )
    }

    Write-Host "Checking installed dependency compatibility..."
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "check")

    Write-Host "Running installation checks..."
    Invoke-Checked -Executable $venvPython -Arguments @("-m", "dapple.cli.doctor")

    Write-Host ""
    Write-Host "Installation complete. Launch with:"
    Write-Host "  .\.venv\Scripts\python.exe -m napari"
}
finally {
    Set-Location -LiteralPath $previousLocation.Path
}
