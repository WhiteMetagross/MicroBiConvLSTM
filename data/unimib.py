"""
UniMiB-SHAR dataset loader wrapper for the standalone MicroBiConvLSTM repository.
"""
import sys
from pathlib import Path

# Add parent repo to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from nanoharmamba.data.unimib import UniMiBSHARDataset, getUniMiBLoaders

# Backward-compatible alias used by the MicroBiConvLSTM training scripts.
UniMiBDataset = UniMiBSHARDataset
getUnimibLoaders = getUniMiBLoaders

__all__ = ['UniMiBDataset', 'getUnimibLoaders']
