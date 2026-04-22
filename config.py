"""
config.py

Environment-aware path and device resolution.

Import this module in notebooks and scripts to get DRIVE_ROOT, CHECKPOINT_DIR,
DATA_ROOT, DEVICE, and IN_COLAB without hardcoding paths.

Resolution order for DRIVE_ROOT:
    1. Colab  : /content/drive/MyDrive/Final_Project
    2. Local  : ~/Library/CloudStorage/GoogleDrive-*/My Drive/Final_Project
    3. Fallback: <repo_root>/outputs   (gitignored, always works)
"""

import os
import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent

_COLAB_DRIVE = Path('/content/drive/MyDrive/Final_Project')
_gdrive_candidates = sorted(
    Path.home().glob('Library/CloudStorage/GoogleDrive-*/My Drive/Final_Project')
)
_LOCAL_DRIVE = _gdrive_candidates[0] if _gdrive_candidates else None

if _COLAB_DRIVE.exists():
    DRIVE_ROOT = _COLAB_DRIVE
elif _LOCAL_DRIVE is not None and _LOCAL_DRIVE.exists():
    DRIVE_ROOT = _LOCAL_DRIVE
else:
    DRIVE_ROOT = _REPO_ROOT / 'outputs'

CHECKPOINT_DIR = DRIVE_ROOT / 'experiments' / 'checkpoints'
DATA_ROOT = _REPO_ROOT / 'data'

# ---------------------------------------------------------------------------
# Device: CUDA > MPS > CPU
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
