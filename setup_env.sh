#!/bin/bash
# ------------------------------------------------------------
# NCAR environment setup for Bayesian swarm sweep (CUDA 12.x)
# ------------------------------------------------------------

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/swarm_venv"

echo "Root directory: ${ROOT_DIR}"
echo "Virtual environment: ${VENV_DIR}"

echo "Loading NCAR modules..."
module purge
module load ncarenv/25.10
module load conda/latest
module load cuda
module load cudnn

# Initialize conda (required on NCAR)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Recreate venv cleanly (recommended when changing CUDA)
if [ -d "${VENV_DIR}" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "${VENV_DIR}"
fi

echo "Creating virtual environment..."
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch (CUDA 12.1 build)..."
pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining dependencies..."
pip install \
    numpy scipy scikit-learn \
    matplotlib seaborn pandas \
    tqdm pyyaml einops \
    tensorboard \
    wandb

echo "✅ Environment setup complete."
echo "Activate later with:"
echo "  source ${VENV_DIR}/bin/activate"