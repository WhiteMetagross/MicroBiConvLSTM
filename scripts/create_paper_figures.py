"""
Publication-Quality Visualizations for μBiConvLSTM (LightDeepConvLSTM) Research Paper
Creates TinyHAR-style figures with datasets as columns and metrics as rows.

Author: Senior AI Researcher
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION STYLE SETTINGS
# ============================================================================

# Default style (for large grid figures)
DEFAULT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
}

# Enhanced style with 2x larger text for single/small figures
ENHANCED_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': False,
}

# Extra enhanced style with 2.5x larger text (for figures 3, 5, 6, 7)
EXTRA_ENHANCED_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 22,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'axes.grid': False,
}

plt.rcParams.update(DEFAULT_STYLE)

# ============================================================================
# MODEL AND DATA DEFINITIONS
# ============================================================================

# Model display names (μ = micro symbol U+03BC)
MODEL_NAMES = {
    'LightDeepConvLSTM': 'μBiConvLSTM',
    'DeepConvLSTM': 'DeepConvLSTM', 
    'TinyHAR': 'TinyHAR',
    'TinierHAR': 'TinierHAR',
}

# Model order for plotting (our model first, highlighted)
MODELS = ['LightDeepConvLSTM', 'DeepConvLSTM', 'TinyHAR', 'TinierHAR']

# VIBRANT color scheme - highly distinguishable
MODEL_COLORS = {
    'μBiConvLSTM': '#FF2D55',      # Vibrant Pink-Red (our model - prominent)
    'DeepConvLSTM': '#5856D6',     # Vibrant Purple
    'TinyHAR': '#34C759',          # Vibrant Green
    'TinierHAR': '#FF9500',        # Vibrant Orange
}

# Lighter versions for error bands
MODEL_COLORS_LIGHT = {
    'μBiConvLSTM': '#FF8FA3',
    'DeepConvLSTM': '#A5A4E3',
    'TinyHAR': '#8EE5A1',
    'TinierHAR': '#FFCC80',
}

# Datasets in order
DATASETS = ['UCI-HAR', 'MotionSense', 'WISDM', 'PAMAP2', 'Opportunity', 'UniMiB', 'Skoda', 'Daphnet']

# ============================================================================
# BENCHMARK DATA (from results files)
# ============================================================================

BENCHMARK_DATA = {
    'UCI-HAR': {
        'LightDeepConvLSTM': {'params': 10454, 'macs': 420128, 'flops': 840256, 'f1': 93.41, 'f1_std': 0.35, 'acc': 93.33},
        'DeepConvLSTM': {'params': 132038, 'macs': 16621952, 'flops': 33243904, 'f1': 93.53, 'f1_std': 0.26, 'acc': 93.44},
        'TinyHAR': {'params': 42704, 'macs': 4890000, 'flops': 9780000, 'f1': 96.53, 'f1_std': 0.41, 'acc': 96.46},
        'TinierHAR': {'params': 16931, 'macs': 879968, 'flops': 1759936, 'f1': 96.37, 'f1_std': 0.57, 'acc': 96.30},
    },
    'MotionSense': {
        'LightDeepConvLSTM': {'params': 10214, 'macs': 389408, 'flops': 778816, 'f1': 91.65, 'f1_std': 0.43, 'acc': 92.71},
        'DeepConvLSTM': {'params': 131078, 'macs': 16499072, 'flops': 32998144, 'f1': 92.90, 'f1_std': 0.96, 'acc': 94.12},
        'TinyHAR': {'params': 39248, 'macs': 3458064, 'flops': 6916128, 'f1': 92.67, 'f1_std': 0.67, 'acc': 94.00},
        'TinierHAR': {'params': 12323, 'macs': 603104, 'flops': 1206208, 'f1': 91.99, 'f1_std': 0.60, 'acc': 93.28},
    },
    'WISDM': {
        'LightDeepConvLSTM': {'params': 9974, 'macs': 358688, 'flops': 717376, 'f1': 73.17, 'f1_std': 12.42, 'acc': 81.73},
        'DeepConvLSTM': {'params': 130118, 'macs': 16376192, 'flops': 32752384, 'f1': 81.84, 'f1_std': 1.46, 'acc': 83.17},
        'TinyHAR': {'params': 35792, 'macs': 2026128, 'flops': 4052256, 'f1': 77.09, 'f1_std': 4.95, 'acc': 83.83},
        'TinierHAR': {'params': 7715, 'macs': 326240, 'flops': 652480, 'f1': 83.06, 'f1_std': 3.24, 'acc': 86.35},
    },
    'PAMAP2': {
        'LightDeepConvLSTM': {'params': 11548, 'macs': 522816, 'flops': 1045632, 'f1': 60.75, 'f1_std': 1.76, 'acc': 62.17},
        'DeepConvLSTM': {'params': 135628, 'macs': 17031936, 'flops': 34063872, 'f1': 67.79, 'f1_std': 1.50, 'acc': 67.50},
        'TinyHAR': {'params': 54518, 'macs': 9663408, 'flops': 19326816, 'f1': 73.22, 'f1_std': 3.58, 'acc': 74.98},
        'TinierHAR': {'params': 32489, 'macs': 1803040, 'flops': 3606080, 'f1': 74.07, 'f1_std': 1.16, 'acc': 73.64},
    },
    'Opportunity': {
        'LightDeepConvLSTM': {'params': 16005, 'macs': 1136880, 'flops': 2273760, 'f1': 87.58, 'f1_std': 0.73, 'acc': 86.62},
        'DeepConvLSTM': {'params': 154373, 'macs': 19489088, 'flops': 38978176, 'f1': 88.30, 'f1_std': 0.72, 'acc': 86.74},
        'TinyHAR': {'params': 123295, 'macs': 38301792, 'flops': 76603584, 'f1': 88.69, 'f1_std': 0.38, 'acc': 87.45},
        'TinierHAR': {'params': 124418, 'macs': 7340096, 'flops': 14680192, 'f1': 87.09, 'f1_std': 0.90, 'acc': 86.19},
    },
    'UniMiB': {
        'LightDeepConvLSTM': {'params': 10121, 'macs': 358832, 'flops': 717664, 'f1': 79.43, 'f1_std': 1.66, 'acc': 91.13},
        'DeepConvLSTM': {'params': 130313, 'macs': 16376384, 'flops': 32752768, 'f1': 85.83, 'f1_std': 1.22, 'acc': 92.71},
        'TinyHAR': {'params': 35939, 'macs': 2026272, 'flops': 4052544, 'f1': 77.61, 'f1_std': 2.23, 'acc': 90.49},
        'TinierHAR': {'params': 7814, 'macs': 326336, 'flops': 652672, 'f1': 79.67, 'f1_std': 4.45, 'acc': 90.30},
    },
    'Skoda': {
        'LightDeepConvLSTM': {'params': 12379, 'macs': 482768, 'flops': 965536, 'f1': 94.46, 'f1_std': 1.31, 'acc': 94.39},
        'DeepConvLSTM': {'params': 139083, 'macs': 13385152, 'flops': 26770304, 'f1': 94.63, 'f1_std': 2.47, 'acc': 94.88},
        'TinyHAR': {'params': 67141, 'macs': 11479968, 'flops': 22959936, 'f1': 97.01, 'f1_std': 0.53, 'acc': 97.14},
        'TinierHAR': {'params': 49352, 'macs': 2123868, 'flops': 4247736, 'f1': 96.99, 'f1_std': 0.76, 'acc': 96.88},
    },
    'Daphnet': {
        'LightDeepConvLSTM': {'params': 10258, 'macs': 210016, 'flops': 420032, 'f1': 88.98, 'f1_std': 1.64, 'acc': 97.37},
        'DeepConvLSTM': {'params': 131778, 'macs': 8310912, 'flops': 16621824, 'f1': 88.95, 'f1_std': 2.26, 'acc': 97.37},
        'TinyHAR': {'params': 42508, 'macs': 2452176, 'flops': 4904352, 'f1': 86.42, 'f1_std': 3.64, 'acc': 96.44},
        'TinierHAR': {'params': 16799, 'macs': 439968, 'flops': 879936, 'f1': 89.84, 'f1_std': 1.90, 'acc': 97.41},
    },
}

# ============================================================================
# ABLATION DATA (from AblationsResults.md)
# ============================================================================

# Ablation variants
ABLATION_VARIANTS = ['A0 (Base)', 'A1 (No Pool)', 'A2 (UniDir)', 'A3 (Single Conv)', 'A4 (Mean Pool)']
ABLATION_IDS = ['A0', 'A1', 'A2', 'A3', 'A4']

ABLATION_DATA = {
    'UCI-HAR': {
        'A0': {'f1': 92.84, 'f1_std': 0.27, 'params': 10454, 'macs': 420000},
        'A1': {'f1': 92.61, 'f1_std': 0.64, 'params': 10454, 'macs': 1239000},
        'A2': {'f1': 93.67, 'f1_std': 0.45, 'params': 6278, 'macs': 297000},
        'A3': {'f1': 92.85, 'f1_std': 0.62, 'params': 9126, 'macs': 584000},
        'A4': {'f1': 91.94, 'f1_std': 0.0, 'params': 10454, 'macs': 420000},
    },
    'MotionSense': {
        'A0': {'f1': 90.76, 'f1_std': 0.70, 'params': 10214, 'macs': 389000},
        'A1': {'f1': 91.22, 'f1_std': 0.69, 'params': 10214, 'macs': 1209000},
        'A2': {'f1': 90.01, 'f1_std': 1.36, 'params': 6038, 'macs': 266000},
        'A3': {'f1': 90.23, 'f1_std': 0.57, 'params': 8886, 'macs': 553000},
        'A4': {'f1': 87.46, 'f1_std': 0.0, 'params': 10214, 'macs': 389000},
    },
    'WISDM': {
        'A0': {'f1': 79.60, 'f1_std': 1.72, 'params': 9974, 'macs': 359000},
        'A1': {'f1': 77.59, 'f1_std': 1.30, 'params': 9974, 'macs': 1178000},
        'A2': {'f1': 79.56, 'f1_std': 0.66, 'params': 5798, 'macs': 236000},
        'A3': {'f1': 79.48, 'f1_std': 1.69, 'params': 8646, 'macs': 523000},
        'A4': {'f1': 71.35, 'f1_std': 0.0, 'params': 9974, 'macs': 359000},
    },
    'PAMAP2': {
        'A0': {'f1': 66.24, 'f1_std': 2.19, 'params': 11548, 'macs': 523000},
        'A1': {'f1': 63.28, 'f1_std': 2.42, 'params': 11548, 'macs': 1342000},
        'A2': {'f1': 64.84, 'f1_std': 4.43, 'params': 7228, 'macs': 400000},
        'A3': {'f1': 63.93, 'f1_std': 3.36, 'params': 10220, 'macs': 687000},
        'A4': {'f1': 58.88, 'f1_std': 0.0, 'params': 11548, 'macs': 523000},
    },
    'Opportunity': {
        'A0': {'f1': 86.53, 'f1_std': 0.56, 'params': 16005, 'macs': 1137000},
        'A1': {'f1': 86.21, 'f1_std': 0.42, 'params': 16005, 'macs': 1956000},
        'A2': {'f1': 87.06, 'f1_std': 0.64, 'params': 11853, 'macs': 1014000},
        'A3': {'f1': 86.31, 'f1_std': 0.83, 'params': 14677, 'macs': 1301000},
        'A4': {'f1': 83.39, 'f1_std': 0.0, 'params': 16005, 'macs': 1137000},
    },
    'UniMiB': {
        'A0': {'f1': 74.03, 'f1_std': 3.54, 'params': 10121, 'macs': 359000},
        'A1': {'f1': 74.68, 'f1_std': 2.02, 'params': 10121, 'macs': 1178000},
        'A2': {'f1': 76.59, 'f1_std': 1.40, 'params': 5873, 'macs': 236000},
        'A3': {'f1': 70.92, 'f1_std': 4.36, 'params': 8793, 'macs': 523000},
        'A4': {'f1': 21.60, 'f1_std': 0.0, 'params': 10121, 'macs': 359000},
    },
    'Skoda': {
        'A0': {'f1': 95.34, 'f1_std': 1.14, 'params': 12379, 'macs': 483000},
        'A1': {'f1': 92.93, 'f1_std': 1.97, 'params': 12379, 'macs': 1114000},
        'A2': {'f1': 94.13, 'f1_std': 0.83, 'params': 8083, 'macs': 390000},
        'A3': {'f1': 93.42, 'f1_std': 2.07, 'params': 11051, 'macs': 612000},
        'A4': {'f1': 91.15, 'f1_std': 0.0, 'params': 12379, 'macs': 483000},
    },
    'Daphnet': {
        'A0': {'f1': 87.74, 'f1_std': 2.60, 'params': 10258, 'macs': 210000},
        'A1': {'f1': 85.08, 'f1_std': 2.33, 'params': 10258, 'macs': 620000},
        'A2': {'f1': 83.26, 'f1_std': 9.18, 'params': 6178, 'macs': 149000},
        'A3': {'f1': 87.41, 'f1_std': 3.89, 'params': 8930, 'macs': 292000},
        'A4': {'f1': 81.37, 'f1_std': 0.0, 'params': 10258, 'macs': 210000},
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_count(x, pos):
    """Format large numbers with K/M suffixes."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def get_display_name(model):
    """Get display name for model."""
    return MODEL_NAMES.get(model, model)


# ============================================================================
# MAIN BENCHMARK COMPARISON FIGURE
# ============================================================================

def create_benchmark_comparison_figure(output_dir):
    """
    Create TinyHAR-style benchmark comparison figure.
    Layout: 7 rows (metrics) × 8 columns (datasets)
    Each cell contains grouped bar chart with 4 models.
    """
    # Use enhanced style for this figure (2x text size)
    plt.rcParams.update(ENHANCED_STYLE)
    
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    
    # Metrics to display
    metrics = [
        ('params', 'Parameters', lambda d: d['params'], lambda d: 0, False, 'linear'),
        ('macs', 'MACs', lambda d: d['macs'], lambda d: 0, False, 'log'),
        ('flops', 'FLOPs', lambda d: d['flops'], lambda d: 0, False, 'log'),
        ('f1', 'F1 Score (%)', lambda d: d['f1'], lambda d: d.get('f1_std', 0), True, 'linear'),
        ('size_kb', 'Model Size (KB)', lambda d: d['params'] * 4 / 1024, lambda d: 0, False, 'linear'),
        ('eff_mac', 'F1/MACs (×10⁶)', lambda d: d['f1'] / d['macs'] * 1e6, lambda d: d.get('f1_std', 0) / d['macs'] * 1e6, True, 'linear'),
        ('eff_param', 'F1/Params (×10³)', lambda d: d['f1'] / d['params'] * 1e3, lambda d: d.get('f1_std', 0) / d['params'] * 1e3, True, 'linear'),
    ]
    
    fig, axes = plt.subplots(len(metrics), n_datasets, figsize=(18, 16))
    
    bar_width = 0.2
    x = np.arange(n_models)
    
    for row_idx, (metric_key, metric_label, metric_func, std_func, higher_better, scale) in enumerate(metrics):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            
            values = []
            errors = []
            colors = []
            
            for model in MODELS:
                display_name = get_display_name(model)
                data = BENCHMARK_DATA[dataset][model]
                values.append(metric_func(data))
                errors.append(std_func(data))
                colors.append(MODEL_COLORS[display_name])
            
            # Create bars with error bars
            bars = ax.bar(x, values, bar_width * 3.5, color=colors, 
                         edgecolor='white', linewidth=1.2, 
                         yerr=errors if any(e > 0 for e in errors) else None,
                         capsize=2, error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': '#333333'})
            
            # Highlight our model with thicker border
            bars[0].set_edgecolor('#8B0000')
            bars[0].set_linewidth(2.5)
            
            # Scale
            if scale == 'log' and min(values) > 0:
                ax.set_yscale('log')
            
            # Format y-axis for counts
            if metric_key in ['params', 'macs', 'flops']:
                ax.yaxis.set_major_formatter(FuncFormatter(format_count))
            
            # Only show x-tick labels on bottom row
            if row_idx == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([get_display_name(m) for m in MODELS], 
                                  rotation=45, ha='right', fontsize=7)
            else:
                ax.set_xticks([])
            
            # Only show y-label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=17, fontweight='bold')  # 2.2x larger minus 15%
            
            # Only show title on top row
            if row_idx == 0:
                ax.set_title(dataset, fontsize=10, fontweight='bold')
            
            # Clean up
            ax.tick_params(axis='y', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add subtle grid for readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
    
    # Add legend at bottom
    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS['μBiConvLSTM'], edgecolor='#8B0000', 
                      linewidth=2, label='μBiConvLSTM (Ours)'),
        mpatches.Patch(facecolor=MODEL_COLORS['DeepConvLSTM'], edgecolor='white',
                      linewidth=1, label='DeepConvLSTM'),
        mpatches.Patch(facecolor=MODEL_COLORS['TinyHAR'], edgecolor='white',
                      linewidth=1, label='TinyHAR'),
        mpatches.Patch(facecolor=MODEL_COLORS['TinierHAR'], edgecolor='white',
                      linewidth=1, label='TinierHAR'),
    ]
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, 
              fontsize=11, frameon=True, fancybox=False, 
              edgecolor='black', bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Benchmark Comparison: μBiConvLSTM vs Baselines Across HAR Datasets', 
                fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, hspace=0.15, wspace=0.25)
    
    plt.savefig(output_dir / 'benchmark_comparison_grid.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(DEFAULT_STYLE)  # Reset to default
    print("✓ Saved: benchmark_comparison_grid.png")


# ============================================================================
# ABLATION STUDY FIGURE (Δ from baseline)
# ============================================================================

def create_ablation_study_figure(output_dir):
    """
    Create ablation study figure showing Δ (change) from baseline A0.
    Layout: 4 rows (metrics) × 8 columns (datasets)
    Shows net gain/loss for each ablation variant, with A0 as reference (0).
    """
    # Use extra enhanced style for this figure (4x text size)
    plt.rcParams.update(EXTRA_ENHANCED_STYLE)
    
    n_datasets = len(DATASETS)
    ablation_labels = ['A0\n(Base)', 'A1\n(No Pool)', 'A2\n(UniDir)', 'A3\n(Single Conv)', 'A4\n(Mean Pool)']
    ablation_keys = ['A0', 'A1', 'A2', 'A3', 'A4']
    n_ablations = len(ablation_keys)
    
    # Vibrant colors for ablations
    ABLATION_COLORS = {
        'A0': '#FF2D55',  # Vibrant Pink-Red (baseline)
        'A1': '#00CED1',  # Dark Turquoise
        'A2': '#9370DB',  # Medium Purple
        'A3': '#20B2AA',  # Light Sea Green
        'A4': '#FF6B6B',  # Coral Red
    }
    
    # Metrics for ablation
    metrics = [
        ('f1_delta', 'ΔF1 Score (%)', lambda d, base: d['f1'] - base['f1'], lambda d: d.get('f1_std', 0)),
        ('params_delta', 'ΔParameters (%)', lambda d, base: (d['params'] - base['params']) / base['params'] * 100, lambda d: 0),
        ('macs_delta', 'ΔMACs (%)', lambda d, base: (d['macs'] - base['macs']) / base['macs'] * 100, lambda d: 0),
        ('eff_delta', 'ΔEfficiency (%)', lambda d, base: (d['f1']/d['macs'] - base['f1']/base['macs']) / (base['f1']/base['macs']) * 100, lambda d: 0),
    ]
    
    fig, axes = plt.subplots(len(metrics), n_datasets, figsize=(18, 10))
    
    bar_width = 0.18
    x = np.arange(n_ablations)
    
    for row_idx, (metric_key, metric_label, metric_func, std_func) in enumerate(metrics):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            
            base_data = ABLATION_DATA[dataset]['A0']
            values = []
            errors = []
            colors = []
            
            for abl_key in ablation_keys:
                abl_data = ABLATION_DATA[dataset][abl_key]
                delta = metric_func(abl_data, base_data)
                values.append(delta)
                errors.append(std_func(abl_data))
                colors.append(ABLATION_COLORS[abl_key])
            
            # Create bars with error bars
            bars = ax.bar(x, values, bar_width * 4, color=colors,
                         edgecolor='white', linewidth=1.2,
                         yerr=errors if any(e > 0 for e in errors) else None,
                         capsize=2, error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': '#333333'})
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
            
            # Color bars based on improvement/degradation with alpha
            for i, (bar, val) in enumerate(zip(bars, values)):
                # Highlight A0 baseline with thick border
                if i == 0:  # A0 baseline
                    bar.set_edgecolor('#8B0000')
                    bar.set_linewidth(2.5)
                elif metric_key == 'f1_delta' or metric_key == 'eff_delta':
                    if val < 0:
                        bar.set_alpha(0.6)
                        bar.set_edgecolor('#8B0000')
                        bar.set_linewidth(1.5)
                else:
                    if val > 0:
                        bar.set_alpha(0.6)
                        bar.set_edgecolor('#8B0000')
                        bar.set_linewidth(1.5)
            
            # Only show x-tick labels on bottom row
            if row_idx == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(ablation_labels, fontsize=7, rotation=45, ha='right')
            else:
                ax.set_xticks([])
            
            # Only show y-label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=9, fontweight='bold')
            
            # Only show title on top row
            if row_idx == 0:
                ax.set_title(dataset, fontsize=10, fontweight='bold')
            
            # Clean up
            ax.tick_params(axis='y', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add subtle grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
    
    # Add legend
    legend_handles = [
        mpatches.Patch(facecolor=ABLATION_COLORS['A1'], edgecolor='white', label='A1: No Pooling'),
        mpatches.Patch(facecolor=ABLATION_COLORS['A2'], edgecolor='white', label='A2: Unidirectional'),
        mpatches.Patch(facecolor=ABLATION_COLORS['A3'], edgecolor='white', label='A3: Single Conv'),
        mpatches.Patch(facecolor=ABLATION_COLORS['A4'], edgecolor='white', label='A4: Mean Pool'),
    ]
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
              fontsize=10, frameon=True, fancybox=False,
              edgecolor='black', bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Ablation Study: Change (Δ) from Baseline μBiConvLSTM (A0)', 
                fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.15, wspace=0.25)
    
    plt.savefig(output_dir / 'ablation_study_grid.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(DEFAULT_STYLE)  # Reset to default
    print("✓ Saved: ablation_study_grid.png")


# ============================================================================
# ADDITIONAL SUMMARY FIGURES
# ============================================================================

def create_summary_bar_chart(output_dir):
    """Create summary bar chart showing average metrics across all datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Calculate averages
    avg_metrics = {model: {'params': [], 'macs': [], 'f1': [], 'f1_std': []} for model in MODELS}
    
    for dataset in DATASETS:
        for model in MODELS:
            data = BENCHMARK_DATA[dataset][model]
            avg_metrics[model]['params'].append(data['params'])
            avg_metrics[model]['macs'].append(data['macs'])
            avg_metrics[model]['f1'].append(data['f1'])
            avg_metrics[model]['f1_std'].append(data.get('f1_std', 0))
    
    for model in MODELS:
        avg_metrics[model]['params_mean'] = np.mean(avg_metrics[model]['params'])
        avg_metrics[model]['macs_mean'] = np.mean(avg_metrics[model]['macs'])
        avg_metrics[model]['f1_mean'] = np.mean(avg_metrics[model]['f1'])
        avg_metrics[model]['f1_std_mean'] = np.mean(avg_metrics[model]['f1_std'])
        avg_metrics[model]['size_kb'] = avg_metrics[model]['params_mean'] * 4 / 1024
        avg_metrics[model]['eff_mac'] = avg_metrics[model]['f1_mean'] / avg_metrics[model]['macs_mean'] * 1e6
        avg_metrics[model]['eff_param'] = avg_metrics[model]['f1_mean'] / avg_metrics[model]['params_mean'] * 1e3
    
    metrics_to_plot = [
        ('params_mean', 'Avg Parameters', 0, False),
        ('macs_mean', 'Avg MACs', 0, False),
        ('f1_mean', 'Avg F1 Score (%)', 'f1_std_mean', True),
        ('size_kb', 'Avg Model Size (KB)', 0, False),
        ('eff_mac', 'Avg Efficiency (F1/MACs×10⁶)', 0, True),
        ('eff_param', 'Avg Efficiency (F1/Params×10³)', 0, True),
    ]
    
    for idx, (metric_key, metric_label, std_key, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        display_names = [get_display_name(m) for m in MODELS]
        values = [avg_metrics[m][metric_key] for m in MODELS]
        errors = [avg_metrics[m][std_key] if std_key else 0 for m in MODELS]
        colors = [MODEL_COLORS[n] for n in display_names]
        
        bars = ax.bar(display_names, values, color=colors, edgecolor='white', linewidth=1.5,
                     yerr=errors if any(e > 0 for e in errors) else None,
                     capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
        
        # Highlight our model
        bars[0].set_edgecolor('#8B0000')
        bars[0].set_linewidth(3)
        
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        
        if metric_key in ['params_mean', 'macs_mean']:
            ax.yaxis.set_major_formatter(FuncFormatter(format_count))
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric_key in ['params_mean', 'macs_mean']:
                label = f'{val/1e3:.0f}K' if val < 1e6 else f'{val/1e6:.1f}M'
            elif metric_key == 'f1_mean':
                label = f'{val:.1f}%'
            elif metric_key == 'size_kb':
                label = f'{val:.1f}'
            else:
                label = f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02, label,
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle('Average Performance Summary Across All Datasets', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'summary_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: summary_comparison.png")


def create_pareto_figure(output_dir):
    """Create Pareto efficiency plot: F1 vs Model Complexity with dataset markers."""
    # Use extra enhanced style for this figure (4x text size)
    plt.rcParams.update(EXTRA_ENHANCED_STYLE)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Dataset markers - unique for each dataset
    DATASET_MARKERS = {
        'UCI-HAR': 'o',
        'MotionSense': 's',
        'WISDM': '^',
        'PAMAP2': 'D',
        'Opportunity': 'p',
        'UniMiB': 'h',
        'Skoda': '*',
        'Daphnet': 'X',
    }
    
    # Plot 1: F1 vs Parameters
    ax1 = axes[0]
    for model in MODELS:
        display_name = get_display_name(model)
        
        for dataset in DATASETS:
            data = BENCHMARK_DATA[dataset][model]
            params = data['params'] / 1000  # K
            f1 = data['f1']
            f1_std = data.get('f1_std', 0)
            
            size = 180 if display_name == 'μBiConvLSTM' else 100
            edgecolor = '#8B0000' if display_name == 'μBiConvLSTM' else 'black'
            linewidth = 2.5 if display_name == 'μBiConvLSTM' else 1
            zorder = 10 if display_name == 'μBiConvLSTM' else 5
            
            ax1.errorbar(params, f1, yerr=f1_std, fmt='none',
                        ecolor=MODEL_COLORS_LIGHT[display_name], elinewidth=1.5, capsize=2,
                        capthick=1, alpha=0.6, zorder=zorder-1)
            ax1.scatter(params, f1, c=MODEL_COLORS[display_name],
                       s=size, marker=DATASET_MARKERS[dataset],
                       edgecolor=edgecolor, linewidth=linewidth, alpha=0.9, zorder=zorder)
    
    ax1.set_xlabel('Parameters (K)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=11)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('(a) F1 vs Parameters', fontweight='bold', fontsize=12)
    
    # Plot 2: F1 vs MACs
    ax2 = axes[1]
    for model in MODELS:
        display_name = get_display_name(model)
        
        for dataset in DATASETS:
            data = BENCHMARK_DATA[dataset][model]
            macs = data['macs'] / 1e6  # M
            f1 = data['f1']
            f1_std = data.get('f1_std', 0)
            
            size = 180 if display_name == 'μBiConvLSTM' else 100
            edgecolor = '#8B0000' if display_name == 'μBiConvLSTM' else 'black'
            linewidth = 2.5 if display_name == 'μBiConvLSTM' else 1
            zorder = 10 if display_name == 'μBiConvLSTM' else 5
            
            ax2.errorbar(macs, f1, yerr=f1_std, fmt='none',
                        ecolor=MODEL_COLORS_LIGHT[display_name], elinewidth=1.5, capsize=2,
                        capthick=1, alpha=0.6, zorder=zorder-1)
            ax2.scatter(macs, f1, c=MODEL_COLORS[display_name],
                       s=size, marker=DATASET_MARKERS[dataset],
                       edgecolor=edgecolor, linewidth=linewidth, alpha=0.9, zorder=zorder)
    
    ax2.set_xlabel('MACs (Millions)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=11)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('(b) F1 vs MACs', fontweight='bold', fontsize=12)
    
    # Create combined legend for models (colors) and datasets (markers)
    # Model legend
    model_handles = [mpatches.Patch(facecolor=MODEL_COLORS[get_display_name(m)], 
                                    edgecolor='black', label=get_display_name(m)) for m in MODELS]
    
    # Dataset legend (using scatter for markers)
    dataset_handles = [plt.Line2D([0], [0], marker=DATASET_MARKERS[d], color='gray', 
                                   linestyle='None', markersize=8, label=d) for d in DATASETS]
    
    # Add legends to figure
    leg1 = fig.legend(handles=model_handles, loc='lower left', ncol=2, fontsize=9, 
                     title='Models', title_fontsize=10, bbox_to_anchor=(0.08, -0.02),
                     framealpha=0.95, edgecolor='black')
    leg2 = fig.legend(handles=dataset_handles, loc='lower right', ncol=4, fontsize=9,
                     title='Datasets', title_fontsize=10, bbox_to_anchor=(0.95, -0.02),
                     framealpha=0.95, edgecolor='black')
    
    plt.suptitle('Pareto Efficiency Analysis: μBiConvLSTM vs Baselines', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    plt.savefig(output_dir / 'pareto_efficiency.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(DEFAULT_STYLE)  # Reset to default
    print("✓ Saved: pareto_efficiency.png")


def create_memory_comparison(output_dir):
    """Create memory footprint comparison figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Memory data from MemoryFootprintResults.md
    memory_data = {
        'μBiConvLSTM': {'fp32': 41.9, 'int8': 21.2},
        'TinierHAR': {'fp32': 68.3, 'int8': 34.1},
        'TinyHAR': {'fp32': 169.4, 'int8': 84.7},
        'DeepConvLSTM': {'fp32': 528.5, 'int8': 264.3},
    }
    
    models = list(memory_data.keys())
    x = np.arange(len(models))
    width = 0.35
    
    fp32_vals = [memory_data[m]['fp32'] for m in models]
    int8_vals = [memory_data[m]['int8'] for m in models]
    
    bars1 = ax.bar(x - width/2, fp32_vals, width, label='FP32', 
                   color='#5856D6', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, int8_vals, width, label='INT8',
                   color='#FF9500', edgecolor='white', linewidth=1.5)
    
    # Highlight our model
    bars1[0].set_edgecolor('#8B0000')
    bars1[0].set_linewidth(3)
    bars2[0].set_edgecolor('#8B0000')
    bars2[0].set_linewidth(3)
    
    ax.set_ylabel('Model Size (KB)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 8,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.title('Model Size Comparison (UCI-HAR)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'memory_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: memory_comparison.png")


# ============================================================================
# RADAR CHART
# ============================================================================

def create_radar_chart(output_dir):
    """Create radar chart comparing models across multiple metrics."""
    # Use enhanced style for this figure (2x text size)
    plt.rcParams.update(ENHANCED_STYLE)
    
    # Metrics for radar (normalized 0-1, higher is better for all)
    # Reorder metrics to prevent label overlap - MAC Efficiency moved to opposite side
    metrics = ['F1 Score', 'Memory Efficiency', 'MAC Efficiency', 'Param Efficiency']
    n_metrics = len(metrics)
    
    # Calculate normalized scores (higher is better)
    scores = {}
    for model in MODELS:
        display_name = get_display_name(model)
        
        # Average F1
        f1_scores = [BENCHMARK_DATA[d][model]['f1'] for d in DATASETS]
        avg_f1 = np.mean(f1_scores)
        
        # Parameter efficiency (inverse of params, normalized)
        avg_params = np.mean([BENCHMARK_DATA[d][model]['params'] for d in DATASETS])
        
        # MAC efficiency (inverse of MACs, normalized)
        avg_macs = np.mean([BENCHMARK_DATA[d][model]['macs'] for d in DATASETS])
        
        scores[display_name] = {
            'F1 Score': avg_f1,
            'Param Efficiency': 1 / avg_params * 1e4,  # Scale for visibility
            'MAC Efficiency': 1 / avg_macs * 1e6,
            'Memory Efficiency': 1 / (avg_params * 4 / 1024),  # Inverse of KB
        }
    
    # Normalize to 0-1 for each metric
    for metric in metrics:
        max_val = max(scores[m][metric] for m in scores)
        for m in scores:
            scores[m][metric] = scores[m][metric] / max_val
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    for model in MODELS:
        display_name = get_display_name(model)
        values = [scores[display_name][m] for m in metrics]
        values += values[:1]
        
        linewidth = 3.5 if display_name == 'μBiConvLSTM' else 2.5
        alpha = 0.3 if display_name == 'μBiConvLSTM' else 0.15
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
               label=display_name, color=MODEL_COLORS[display_name], markersize=10)
        ax.fill(angles, values, alpha=alpha, color=MODEL_COLORS[display_name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Clear default labels, we'll position manually
    ax.set_ylim(0, 1.1)
    
    # Manually position labels with custom offsets to prevent overlap
    # metrics order: ['F1 Score', 'Memory Efficiency', 'MAC Efficiency', 'Param Efficiency']
    # angles: 0 (top), π/2 (right), π (bottom), 3π/2 (left)
    label_configs = [
        ('F1 Score', 0.0, 0.0),              # Top - centered
        ('Memory Efficiency', 0.0, 0.0),     # Right - centered  
        ('MAC Efficiency', 0.0, 0.12),       # Bottom - moved left to prevent overlap
        ('Param Efficiency', 0.0, 0.0),      # Left - centered
    ]
    
    for idx, (label, angle_off, r_off) in enumerate(label_configs):
        angle = angles[idx] + angle_off
        r = 1.28 + r_off
        ax.text(angle, r, label, fontsize=16, fontweight='bold',
               ha='center', va='center')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=14)
    ax.set_title('Multi-Metric Comparison Radar', fontsize=18, fontweight='bold', pad=25)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: radar_comparison.png")
    
    # Reset to default style
    plt.rcParams.update(DEFAULT_STYLE)


# ============================================================================
# F1 SCORE LINE PLOT WITH ERROR BANDS
# ============================================================================

def create_f1_line_plot(output_dir):
    """Create F1 score line plot across datasets with error bands."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(DATASETS))
    
    for model in MODELS:
        display_name = get_display_name(model)
        f1_scores = [BENCHMARK_DATA[d][model]['f1'] for d in DATASETS]
        f1_stds = [BENCHMARK_DATA[d][model].get('f1_std', 0) for d in DATASETS]
        
        linewidth = 3 if display_name == 'μBiConvLSTM' else 2
        marker = 's' if display_name == 'μBiConvLSTM' else 'o'
        markersize = 10 if display_name == 'μBiConvLSTM' else 7
        zorder = 10 if display_name == 'μBiConvLSTM' else 5
        
        # Plot line
        ax.plot(x, f1_scores, marker=marker, linewidth=linewidth, 
               label=display_name, color=MODEL_COLORS[display_name],
               markersize=markersize, zorder=zorder)
        
        # Add error band
        ax.fill_between(x, 
                       np.array(f1_scores) - np.array(f1_stds),
                       np.array(f1_scores) + np.array(f1_stds),
                       alpha=0.2, color=MODEL_COLORS[display_name], zorder=zorder-1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(50, 100)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title('F1 Score Comparison Across Datasets (with ±1σ bands)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'f1_line_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: f1_line_comparison.png")


# ============================================================================
# EFFICIENCY HEATMAP
# ============================================================================

def create_efficiency_heatmap(output_dir):
    """Create heatmap showing efficiency (F1/MACs) across models and datasets."""
    # Use enhanced style for this figure (2x text size)
    plt.rcParams.update(ENHANCED_STYLE)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1/MACs efficiency
    eff_mac_data = np.zeros((len(MODELS), len(DATASETS)))
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            data = BENCHMARK_DATA[dataset][model]
            eff_mac_data[i, j] = data['f1'] / data['macs'] * 1e6
    
    ax1 = axes[0]
    im1 = ax1.imshow(eff_mac_data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(np.arange(len(DATASETS)))
    ax1.set_yticks(np.arange(len(MODELS)))
    ax1.set_xticklabels(DATASETS, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels([get_display_name(m) for m in MODELS], fontsize=10)
    
    # Add text annotations
    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            text = ax1.text(j, i, f'{eff_mac_data[i, j]:.1f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if eff_mac_data[i, j] > eff_mac_data.max() * 0.6 else 'black')
    
    ax1.set_title('(a) F1/MACs Efficiency (×10⁶)', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # F1/Params efficiency
    eff_param_data = np.zeros((len(MODELS), len(DATASETS)))
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            data = BENCHMARK_DATA[dataset][model]
            eff_param_data[i, j] = data['f1'] / data['params'] * 1e3
    
    ax2 = axes[1]
    im2 = ax2.imshow(eff_param_data, cmap='YlGnBu', aspect='auto')
    ax2.set_xticks(np.arange(len(DATASETS)))
    ax2.set_yticks(np.arange(len(MODELS)))
    ax2.set_xticklabels(DATASETS, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels([get_display_name(m) for m in MODELS], fontsize=10)
    
    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            text = ax2.text(j, i, f'{eff_param_data[i, j]:.1f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if eff_param_data[i, j] > eff_param_data.max() * 0.6 else 'black')
    
    ax2.set_title('(b) F1/Params Efficiency (×10³)', fontweight='bold', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.suptitle('Efficiency Heatmaps: μBiConvLSTM vs Baselines', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'efficiency_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: efficiency_heatmap.png")
    
    # Reset to default style
    plt.rcParams.update(DEFAULT_STYLE)


# ============================================================================
# ABLATION F1 BAR CHART (ABSOLUTE VALUES)
# ============================================================================

def create_ablation_absolute_figure(output_dir):
    """Create ablation study figure showing absolute F1 values with error bars."""
    # Use extra enhanced style for this figure (4x text size)
    plt.rcParams.update(EXTRA_ENHANCED_STYLE)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    ablation_keys = ['A0', 'A1', 'A2', 'A3', 'A4']
    ablation_labels = ['A0\n(Base)', 'A1\n(No Pool)', 'A2\n(UniDir)', 'A3\n(1 Conv)', 'A4\n(Mean)']
    
    ABLATION_COLORS_ABS = {
        'A0': '#FF2D55',  # Our model - vibrant pink
        'A1': '#00CED1',  # Turquoise
        'A2': '#9370DB',  # Purple
        'A3': '#20B2AA',  # Sea green
        'A4': '#FF6B6B',  # Coral
    }
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        
        values = [ABLATION_DATA[dataset][k]['f1'] for k in ablation_keys]
        errors = [ABLATION_DATA[dataset][k].get('f1_std', 0) for k in ablation_keys]
        colors = [ABLATION_COLORS_ABS[k] for k in ablation_keys]
        
        x = np.arange(len(ablation_keys))
        bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1.2,
                     yerr=errors, capsize=3, 
                     error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
        
        # Highlight baseline (A0)
        bars[0].set_edgecolor('#8B0000')
        bars[0].set_linewidth(2.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_labels, fontsize=8)
        ax.set_ylabel('F1 Score (%)', fontweight='bold') if idx % 4 == 0 else None
        ax.set_title(dataset, fontweight='bold', fontsize=11)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set y-axis to start from reasonable value
        min_val = min(values) - max(errors) - 5
        ax.set_ylim(max(0, min_val), 100)
    
    plt.suptitle('Ablation Study: F1 Scores by Dataset (with ±1σ error bars)', 
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'ablation_absolute.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: ablation_absolute.png")
    
    # Reset to default style
    plt.rcParams.update(DEFAULT_STYLE)


# ============================================================================
# COMPLEXITY COMPARISON (LOG SCALE)
# ============================================================================

def create_complexity_comparison(output_dir):
    """Create complexity comparison showing params vs MACs vs FLOPs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average across datasets
    avg_data = {}
    for model in MODELS:
        display_name = get_display_name(model)
        avg_data[display_name] = {
            'params': np.mean([BENCHMARK_DATA[d][model]['params'] for d in DATASETS]),
            'macs': np.mean([BENCHMARK_DATA[d][model]['macs'] for d in DATASETS]),
            'flops': np.mean([BENCHMARK_DATA[d][model]['flops'] for d in DATASETS]),
        }
    
    display_names = [get_display_name(m) for m in MODELS]
    
    # Parameters
    ax1 = axes[0]
    params = [avg_data[n]['params'] for n in display_names]
    colors = [MODEL_COLORS[n] for n in display_names]
    bars1 = ax1.bar(display_names, params, color=colors, edgecolor='white', linewidth=1.5)
    bars1[0].set_edgecolor('#8B0000')
    bars1[0].set_linewidth(3)
    ax1.set_ylabel('Parameters', fontweight='bold', fontsize=11)
    ax1.set_yscale('log')
    ax1.set_title('(a) Parameters (log scale)', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='x', rotation=30)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_count))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # MACs
    ax2 = axes[1]
    macs = [avg_data[n]['macs'] for n in display_names]
    bars2 = ax2.bar(display_names, macs, color=colors, edgecolor='white', linewidth=1.5)
    bars2[0].set_edgecolor('#8B0000')
    bars2[0].set_linewidth(3)
    ax2.set_ylabel('MACs', fontweight='bold', fontsize=11)
    ax2.set_yscale('log')
    ax2.set_title('(b) MACs (log scale)', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', rotation=30)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_count))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # FLOPs
    ax3 = axes[2]
    flops = [avg_data[n]['flops'] for n in display_names]
    bars3 = ax3.bar(display_names, flops, color=colors, edgecolor='white', linewidth=1.5)
    bars3[0].set_edgecolor('#8B0000')
    bars3[0].set_linewidth(3)
    ax3.set_ylabel('FLOPs', fontweight='bold', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_title('(c) FLOPs (log scale)', fontweight='bold', fontsize=12)
    ax3.tick_params(axis='x', rotation=30)
    ax3.yaxis.set_major_formatter(FuncFormatter(format_count))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle('Computational Complexity Comparison (Average Across Datasets)', 
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'complexity_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: complexity_comparison.png")


# ============================================================================
# ABLATION RELATIVE F1 FIGURE (Figure 7)
# ============================================================================

def create_ablation_relative_figure(output_dir, original_output_dir=None):
    """
    Create Figure 7: Ablation study showing relative F1 changes from baseline (A0).
    Negative values indicate degradation. Green for increase, red for decrease.
    Percentage values positioned above error bars.
    """
    # Use extra enhanced style for this figure
    plt.rcParams.update(EXTRA_ENHANCED_STYLE)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    ablation_keys = ['A0', 'A1', 'A2', 'A3', 'A4']
    ablation_labels = ['A0\n(Base)', 'A1\n(No Pool)', 'A2\n(UniDir)', 'A3\n(1 Conv)', 'A4\n(Mean)']
    
    # Vibrant, distinct colors for each ablation variant
    ABLATION_COLORS_REL = {
        'A0': '#FF2D55',  # Vibrant Pink-Red (baseline)
        'A1': '#00CED1',  # Dark Turquoise
        'A2': '#9370DB',  # Medium Purple
        'A3': '#20B2AA',  # Light Sea Green  
        'A4': '#FF6B6B',  # Coral Red
    }
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        
        base_f1 = ABLATION_DATA[dataset]['A0']['f1']
        
        # Calculate relative changes from baseline
        values = []
        errors = []
        colors = []
        for k in ablation_keys:
            f1_val = ABLATION_DATA[dataset][k]['f1']
            f1_std = ABLATION_DATA[dataset][k].get('f1_std', 0)
            relative_change = f1_val - base_f1  # Absolute difference in F1 points
            values.append(relative_change)
            errors.append(f1_std)
            colors.append(ABLATION_COLORS_REL[k])
        
        x = np.arange(len(ablation_keys))
        bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1.2,
                     yerr=errors, capsize=3,
                     error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
        
        # Highlight baseline (A0)
        bars[0].set_edgecolor('#8B0000')
        bars[0].set_linewidth(2.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
        
        # Add percentage labels above bars (above error bars)
        for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
            # Position label above error bar
            if val >= 0:
                y_pos = val + err + 0.5
                color = '#008000'  # Green for positive
                label = f'+{val:.2f}%'
            else:
                y_pos = val + err + 0.5
                color = '#CC0000'  # Red for negative
                label = f'{val:.2f}%'
            
            # Special case for A0 (always 0)
            if i == 0:
                label = '0.00%'
                color = '#333333'  # Neutral color for baseline
            
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   color=color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_labels, fontsize=8)
        ax.set_ylabel('ΔF1 Score (%)', fontweight='bold') if idx % 4 == 0 else None
        ax.set_title(dataset, fontweight='bold', fontsize=11)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set y-axis limits with padding for labels
        min_val = min(values) - max(errors) - 3
        max_val = max(values) + max(errors) + 4
        ax.set_ylim(min_val, max_val)
    
    plt.suptitle('Ablation Study: Relative F1 Change from Baseline (A0)', 
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save to both original and compressed directories
    if original_output_dir:
        original_output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(original_output_dir / 'ablation_relative.png', format='png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved original: {original_output_dir / 'ablation_relative.png'}")
    
    plt.savefig(output_dir / 'ablation_relative.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: ablation_relative.png")
    
    # Reset to default style
    plt.rcParams.update(DEFAULT_STYLE)


# ============================================================================
# QUANTIZATION COMPARISON
# ============================================================================

def create_quantization_figure(output_dir):
    """Create quantization impact comparison figure."""
    # Use extra enhanced style for this figure (4x text size)
    plt.rcParams.update(EXTRA_ENHANCED_STYLE)
    
    # Data from AblationsResults.md - Quantization section
    quant_data = {
        'UCI-HAR': {'fp32': 92.69, 'int8': 92.69, 'error': 0.00},
        'MotionSense': {'fp32': 90.88, 'int8': 90.46, 'error': 0.42},
        'WISDM': {'fp32': 76.84, 'int8': 76.82, 'error': 0.02},
        'PAMAP2': {'fp32': 63.73, 'int8': 63.79, 'error': -0.06},
        'Opportunity': {'fp32': 87.03, 'int8': 87.09, 'error': -0.06},
        'Skoda': {'fp32': 95.71, 'int8': 95.64, 'error': 0.07},
        'Daphnet': {'fp32': 88.13, 'int8': 87.05, 'error': 1.09},
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: FP32 vs INT8 F1 scores
    ax1 = axes[0]
    datasets = list(quant_data.keys())
    x = np.arange(len(datasets))
    width = 0.35
    
    fp32_vals = [quant_data[d]['fp32'] for d in datasets]
    int8_vals = [quant_data[d]['int8'] for d in datasets]
    
    bars1 = ax1.bar(x - width/2, fp32_vals, width, label='FP32', 
                   color='#5856D6', edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, int8_vals, width, label='INT8',
                   color='#FF9500', edgecolor='white', linewidth=1.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('F1 Score (%)', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title('(a) FP32 vs INT8 F1 Scores', fontweight='bold', fontsize=12)
    ax1.set_ylim(60, 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Quantization error
    ax2 = axes[1]
    errors = [quant_data[d]['error'] for d in datasets]
    colors = ['#34C759' if e <= 0 else '#FF2D55' for e in errors]
    
    bars = ax2.bar(x, errors, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Quantization Error (%)', fontweight='bold', fontsize=11)
    ax2.set_title('(b) INT8 Quantization Error (FP32 - INT8)', fontweight='bold', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle('μBiConvLSTM INT8 Quantization Analysis', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'quantization_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: quantization_comparison.png")
    
    # Reset to default style
    plt.rcParams.update(DEFAULT_STYLE)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all publication figures."""
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / 'docs' / 'img'
    original_output_dir = script_dir / 'docs' / 'OriginalImg'
    output_dir.mkdir(parents=True, exist_ok=True)
    original_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Publication Figures for μBiConvLSTM (LightDeepConvLSTM)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Original output directory: {original_output_dir}")
    print()
    
    print("1. Creating main benchmark comparison grid...")
    create_benchmark_comparison_figure(output_dir)
    
    print("\n2. Creating ablation study grid (delta from baseline)...")
    create_ablation_study_figure(output_dir)
    
    print("\n3. Creating ablation absolute F1 scores...")
    create_ablation_absolute_figure(output_dir)
    
    print("\n4. Creating summary bar chart...")
    create_summary_bar_chart(output_dir)
    
    print("\n5. Creating Pareto efficiency plots...")
    create_pareto_figure(output_dir)
    
    print("\n6. Creating memory comparison...")
    create_memory_comparison(output_dir)
    
    print("\n7. Creating radar comparison chart...")
    create_radar_chart(output_dir)
    
    print("\n8. Creating F1 line plot with error bands...")
    create_f1_line_plot(output_dir)
    
    print("\n9. Creating efficiency heatmaps...")
    create_efficiency_heatmap(output_dir)
    
    print("\n10. Creating complexity comparison...")
    create_complexity_comparison(output_dir)
    
    print("\n11. Creating quantization comparison...")
    create_quantization_figure(output_dir)
    
    print("\n12. Creating Figure 7: Ablation relative F1 changes...")
    create_ablation_relative_figure(output_dir, original_output_dir)
    
    print()
    print("=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)
    print("Generated files:")
    print("  1. benchmark_comparison_grid.png (7×8 grid with error bars)")
    print("  2. ablation_study_grid.png (4×8 delta from baseline)")
    print("  3. ablation_absolute.png (F1 scores with error bars)")
    print("  4. summary_comparison.png (average metrics)")
    print("  5. pareto_efficiency.png (F1 vs complexity with error bars)")
    print("  6. memory_comparison.png (FP32 vs INT8)")
    print("  7. radar_comparison.png (multi-metric radar)")
    print("  8. f1_line_comparison.png (F1 across datasets with bands)")
    print("  9. efficiency_heatmap.png (F1/MACs and F1/Params)")
    print(" 10. complexity_comparison.png (params/MACs/FLOPs log scale)")
    print(" 11. quantization_comparison.png (INT8 quantization impact)")
    print(" 12. ablation_relative.png (Figure 7: Relative F1 changes)")
    print("=" * 70)


if __name__ == '__main__':
    main()
