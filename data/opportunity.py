"""
Opportunity Dataset Loader (wrapper for LightDeepConvLSTM self-contained project)
"""
import sys
from pathlib import Path

# Add parent repo to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from nanoharmamba.data.opportunity import OpportunityDataset, getOpportunityLoaders

__all__ = ['OpportunityDataset', 'getOpportunityLoaders']
