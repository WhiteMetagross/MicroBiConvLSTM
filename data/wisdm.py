"""
WISDM Dataset Loader (wrapper for MicroBiConvLSTM self-contained project)
"""
import sys
from pathlib import Path

# Add parent repo to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from nanoharmamba.data.wisdm import WisdmDataset, getWisdmLoaders, ACTIVITIES

__all__ = ['WisdmDataset', 'getWisdmLoaders', 'ACTIVITIES']
