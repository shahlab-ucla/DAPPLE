#!/usr/bin/env sh
set -eu

dev=0
recreate=0
python_cmd="${PYTHON:-python3}"

usage() {
    echo "Usage: sh scripts/bootstrap.sh [--dev] [--recreate] [--python PATH]"
    echo "  --dev          Install editable GUI + contributor dependencies."
    echo "  --recreate     Replace only this repository's .venv if it is stale."
    echo "  --python PATH  Python 3.11 or 3.12 executable used to create .venv."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dev)
            dev=1
            shift
            ;;
        --recreate)
            recreate=1
            shift
            ;;
        --python)
            if [ "$#" -lt 2 ]; then
                echo "--python requires an executable path" >&2
                exit 2
            fi
            python_cmd="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

venv_python="$repo_root/.venv/bin/python"
venv_root="$repo_root/.venv"
if [ "$recreate" -eq 1 ] && { [ -e "$venv_root" ] || [ -L "$venv_root" ]; }; then
    # Keep recursive deletion constrained to the literal repository-local .venv.
    if [ "$venv_root" != "$repo_root/.venv" ] || [ "${venv_root##*/}" != ".venv" ]; then
        echo "Refusing to remove unexpected virtual-environment path: $venv_root" >&2
        exit 1
    fi
    echo "Removing repository-local .venv because --recreate was requested..."
    rm -rf -- "$venv_root"
fi

if { [ -e "$venv_root" ] || [ -L "$venv_root" ]; } && [ ! -x "$venv_python" ]; then
    echo "The existing .venv is incomplete or stale." >&2
    echo "Rerun with --recreate to replace only this repository's .venv." >&2
    exit 1
fi

if [ ! -x "$venv_python" ]; then
    if ! "$python_cmd" -c \
        "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 'DAPPLE requires Python 3.11 or 3.12')"
    then
        echo "The selected bootstrap interpreter is not usable Python 3.11 or 3.12." >&2
        echo "Pass --python with a supported interpreter path." >&2
        exit 1
    fi
    echo "Creating .venv with $python_cmd..."
    "$python_cmd" -m venv .venv
else
    echo "Reusing existing .venv."
fi

# Detect stale environments before attempting a less-informative pip command.
if ! "$venv_python" -c \
    "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 'DAPPLE requires Python 3.11 or 3.12')"
then
    if [ "$recreate" -eq 0 ]; then
        echo "The existing .venv is stale, broken, or uses an unsupported Python." >&2
        echo "Rerun with --recreate to replace only this repository's .venv." >&2
    fi
    exit 1
fi

echo "Updating packaging tools..."
"$venv_python" -m pip install --upgrade pip setuptools wheel

if [ "$dev" -eq 1 ]; then
    echo "Installing DAPPLE with GUI and contributor dependencies..."
    "$venv_python" -m pip install --editable ".[gui,dev]"
else
    echo "Installing DAPPLE with the tested GUI dependency set..."
    "$venv_python" -m pip install ".[gui]"
fi

echo "Checking installed dependency compatibility..."
"$venv_python" -m pip check

echo "Running installation checks..."
"$venv_python" -m dapple.cli.doctor

echo ""
echo "Installation complete. Launch with:"
echo "  .venv/bin/python -m napari"
