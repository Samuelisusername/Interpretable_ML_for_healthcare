#!/bin/bash
#SBATCH --job-name=cnn-classifier
#SBATCH --account=ml4h
#SBATCH --partition=jobs
#SBATCH --output=cnn_classifier-%j.out
#SBATCH --error=cnn_classifier-%j.err
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$WORK_DIR" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
DATA_DIR="${SCRIPT_DIR}/data"

cd "$SCRIPT_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch torchvision pandas pillow

if ![ -d "$DATA_DIR" ]; then
    echo "Data not generated yet."
    python "${SCRIPT_DIR}/preprocessing.py"
fi



python "${SCRIPT_DIR}/cnn_classifier.py"