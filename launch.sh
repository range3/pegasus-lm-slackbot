#!/usr/bin/bash
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE:-$0}" )" &> /dev/null && pwd )"

cd "${SCRIPT_DIR}"
source .envrc
source .venv/bin/activate
python run.py
