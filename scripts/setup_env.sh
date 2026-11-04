#!/bin/bash
# CPU environment for reproducing the final submission. usage: bash scripts/setup_env.sh [python executable]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${1:-python3}
"$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt" -r "$ROOT/requirements-notebook.txt"
"$ROOT/.venv/bin/python" -c "import numpy, scipy, sklearn; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'scikit-learn', sklearn.__version__)"
echo "done. next: PYTHON=$ROOT/.venv/bin/python bash scripts/reproduce_best.sh"
