#!/usr/bin/env python3
"""
Sensor Time Series Visualization - Vertical Wave Display
Creates publication-quality visualization of accelerometer and gyroscope data
from the UCI-HAR dataset showing different human activities.

Author: Research Team
Date: January 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path


def load_ucihar_sample(dataset_path, sample_indices=None):
    """Load sample data from UCI-HAR dataset."""
    
    inertial_path = dataset_path / 'train' / 'Inertial Signals'
    
    # Load all 9 channels
    channels = {
        'body_acc_x': np.loadtxt(inertial_path / 'body_acc_x_train.txt'),
        'body_acc_y': np.loadtxt(inertial_path / 'body_acc_y_train.txt'),
        'body_acc_z': np.loadtxt(inertial_path / 'body_acc_z_train.txt'),
        'body_gyro_x': np.loadtxt(inertial_path / 'body_gyro_x_train.txt'),
        'body_gyro_y': np.loadtxt(inertial_path / 'body_gyro_y_train.txt'),
        'body_gyro_z': np.loadtxt(inertial_path / 'body_gyro_z_train.txt'),
        'total_acc_x': np.loadtxt(inertial_path / 'total_acc_x_train.txt'),
        'total_acc_y': np.loadtxt(inertial_path / 'total_acc_y_train.txt'),
        'total_acc_z': np.loadtxt(inertial_path / 'total_acc_z_train.txt'),
    }
    
    # Load labels
    labels = np.loadtxt(dataset_path / 'train' / 'y_train.txt', dtype=int)
    
    activity_names = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS', 
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    return channels, labels, activity_names


def create_vertical_wave_visualization():
    """Create a vertical visualization of sensor time series as waves."""
    
    # Dataset path
    dataset_path = Path(__file__).parent.parent.parent / 'datasets' / 'UCI HAR Dataset'
    
    print("Loading UCI-HAR dataset...")
    channels, labels, activity_names = load_ucihar_sample(dataset_path)
    
    # Find good sample indices for each activity
    activity_samples = {}
    for activity_id in range(1, 7):
        indices = np.where(labels == activity_id)[0]
        # Pick a representative sample (middle of the set)
        activity_samples[activity_id] = indices[len(indices) // 2]
    
    # Select activities to display (pick 4 diverse ones)
    selected_activities = [1, 3, 4, 6]  # Walking, Walking Downstairs, Sitting, Laying
    
    # Color palette for channels
    colors = {
        'acc_x': '#E53935',   # Red
        'acc_y': '#43A047',   # Green  
        'acc_z': '#1E88E5',   # Blue
        'gyro_x': '#FB8C00',  # Orange
        'gyro_y': '#8E24AA',  # Purple
        'gyro_z': '#00ACC1',  # Cyan
    }
    
    # Activity colors (backgrounds)
    activity_colors = {
        1: '#E3F2FD',  # Walking - Light Blue
        2: '#F3E5F5',  # Walking Upstairs - Light Purple
        3: '#FFF3E0',  # Walking Downstairs - Light Orange
        4: '#E8F5E9',  # Sitting - Light Green
        5: '#FFF8E1',  # Standing - Light Amber
        6: '#FCE4EC',  # Laying - Light Pink
    }
    
    # Create figure
    fig, axes = plt.subplots(len(selected_activities), 1, figsize=(14, 16), 
                             dpi=300, gridspec_kw={'hspace': 0.35})
    
    # Title
    fig.suptitle('UCI-HAR Sensor Time Series Visualization', 
                 fontsize=22, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, '9-Channel Accelerometer & Gyroscope Data (128 timesteps @ 50Hz = 2.56 seconds)',
             ha='center', fontsize=12, color='#546E7A', style='italic')
    
    # Time axis (128 samples at 50Hz)
    time = np.arange(128) / 50.0  # Convert to seconds
    
    for ax_idx, activity_id in enumerate(selected_activities):
        ax = axes[ax_idx]
        sample_idx = activity_samples[activity_id]
        
        # Set background color
        ax.set_facecolor(activity_colors[activity_id])
        
        # Channel groups for this sample
        channel_groups = [
            ('Body Acc', ['body_acc_x', 'body_acc_y', 'body_acc_z'], 
             [colors['acc_x'], colors['acc_y'], colors['acc_z']], ['X', 'Y', 'Z']),
            ('Body Gyro', ['body_gyro_x', 'body_gyro_y', 'body_gyro_z'],
             [colors['gyro_x'], colors['gyro_y'], colors['gyro_z']], ['X', 'Y', 'Z']),
            ('Total Acc', ['total_acc_x', 'total_acc_y', 'total_acc_z'],
             [colors['acc_x'], colors['acc_y'], colors['acc_z']], ['X', 'Y', 'Z']),
        ]
        
        # Plot each channel with offset for visibility
        offset = 0
        offset_step = 2.5
        plotted_lines = []
        plotted_labels = []
        
        for group_name, channel_names, group_colors, axis_labels in channel_groups:
            for ch_name, color, axis_label in zip(channel_names, group_colors, axis_labels):
                data = channels[ch_name][sample_idx]
                # Normalize to [-1, 1] range for better visualization
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                
                # Determine line style based on sensor type
                if 'gyro' in ch_name:
                    linestyle = '--'
                    linewidth = 1.8
                else:
                    linestyle = '-'
                    linewidth = 2.0
                
                line, = ax.plot(time, data_norm + offset, color=color, 
                               linestyle=linestyle, linewidth=linewidth, alpha=0.85)
                
                # Add channel label on the left
                if axis_label == 'X':
                    ax.text(-0.08, offset, group_name, fontsize=9, fontweight='bold',
                           ha='right', va='center', color='#37474F', transform=ax.get_yaxis_transform())
                
                plotted_lines.append(line)
                plotted_labels.append(f'{group_name} {axis_label}')
                
            offset += offset_step
        
        # Activity label
        activity_name = activity_names[activity_id]
        ax.text(0.02, 0.95, f'Activity: {activity_name}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left', color='#1A237E',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='#90A4AE', alpha=0.95, linewidth=1.5))
        
        # Sample info
        ax.text(0.98, 0.95, f'Sample #{sample_idx}', 
                transform=ax.transAxes, fontsize=10,
                va='top', ha='right', color='#546E7A', style='italic')
        
        # Styling
        ax.set_xlim(0, time[-1])
        ax.set_ylim(-1.5, offset + 1)
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Amplitude', fontsize=11, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#B0BEC5')
        ax.axhline(y=0, color='#78909C', linestyle='-', linewidth=1, alpha=0.5)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_color('#90A4AE')
            spine.set_linewidth(1.5)
        
        # Y-axis ticks - show sensor groups
        ax.set_yticks([0, offset_step, 2*offset_step])
        ax.set_yticklabels(['Body\nAcc', 'Body\nGyro', 'Total\nAcc'], fontsize=9)
    
    # Legend (shared)
    legend_elements = [
        mpatches.Patch(facecolor=colors['acc_x'], edgecolor='#B71C1C', linewidth=1.5, label='X-axis'),
        mpatches.Patch(facecolor=colors['acc_y'], edgecolor='#2E7D32', linewidth=1.5, label='Y-axis'),
        mpatches.Patch(facecolor=colors['acc_z'], edgecolor='#1565C0', linewidth=1.5, label='Z-axis'),
        plt.Line2D([0], [0], color='#37474F', linestyle='-', linewidth=2, label='Accelerometer'),
        plt.Line2D([0], [0], color='#37474F', linestyle='--', linewidth=2, label='Gyroscope'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
               fontsize=11, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.01))
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.08, top=0.93, left=0.12, right=0.95)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'UCIHAR_Sensor_Waves_Vertical.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print(f"✓ Visualization saved to: {output_path}")
    
    # Also save to results
    alt_output = Path(__file__).parent.parent / 'results' / 'UCIHAR_Sensor_Waves_Vertical.png'
    plt.savefig(alt_output, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"✓ Also saved to: {alt_output}")
    
    plt.close()
    return output_path


def create_single_activity_detailed_view():
    """Create a detailed single-activity view with all 9 channels stacked vertically."""
    
    dataset_path = Path(__file__).parent.parent.parent / 'datasets' / 'UCI HAR Dataset'
    
    print("\nLoading UCI-HAR dataset for detailed view...")
    channels, labels, activity_names = load_ucihar_sample(dataset_path)
    
    # Pick WALKING activity for detailed visualization
    activity_id = 1
    indices = np.where(labels == activity_id)[0]
    sample_idx = indices[50]  # Pick a good sample
    
    # Color palette - vibrant and distinct
    channel_colors = [
        '#E53935',  # Body Acc X - Red
        '#43A047',  # Body Acc Y - Green
        '#1E88E5',  # Body Acc Z - Blue
        '#FB8C00',  # Body Gyro X - Orange
        '#8E24AA',  # Body Gyro Y - Purple
        '#00ACC1',  # Body Gyro Z - Cyan
        '#D81B60',  # Total Acc X - Pink
        '#7CB342',  # Total Acc Y - Light Green
        '#3949AB',  # Total Acc Z - Indigo
    ]
    
    channel_names = [
        ('body_acc_x', 'Body Acc X'),
        ('body_acc_y', 'Body Acc Y'),
        ('body_acc_z', 'Body Acc Z'),
        ('body_gyro_x', 'Body Gyro X'),
        ('body_gyro_y', 'Body Gyro Y'),
        ('body_gyro_z', 'Body Gyro Z'),
        ('total_acc_x', 'Total Acc X'),
        ('total_acc_y', 'Total Acc Y'),
        ('total_acc_z', 'Total Acc Z'),
    ]
    
    # Create figure with 9 subplots stacked vertically
    fig, axes = plt.subplots(9, 1, figsize=(14, 18), dpi=300,
                             gridspec_kw={'hspace': 0.12})
    
    # Title
    fig.suptitle('UCI-HAR: 9-Channel Sensor Time Series', 
                 fontsize=24, fontweight='bold', y=0.995)
    fig.text(0.5, 0.975, f'Activity: {activity_names[activity_id]} | Sample #{sample_idx} | 128 timesteps @ 50Hz',
             ha='center', fontsize=13, color='#37474F', style='italic')
    
    # Time axis
    time = np.arange(128) / 50.0
    
    for idx, ((ch_key, ch_label), color) in enumerate(zip(channel_names, channel_colors)):
        ax = axes[idx]
        data = channels[ch_key][sample_idx]
        
        # Plot as filled wave
        ax.fill_between(time, 0, data, alpha=0.3, color=color)
        ax.plot(time, data, color=color, linewidth=2.0, alpha=0.9)
        
        # Zero line
        ax.axhline(y=0, color='#90A4AE', linestyle='-', linewidth=0.8, alpha=0.6)
        
        # Channel label
        ax.text(-0.01, 0.5, ch_label, transform=ax.transAxes, fontsize=11,
                fontweight='bold', ha='right', va='center', color=color)
        
        # Styling
        ax.set_xlim(0, time[-1])
        ax.set_ylabel('')
        
        # Only show x-axis label on bottom plot
        if idx == 8:
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
        
        # Light grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B0BEC5')
        ax.spines['bottom'].set_color('#B0BEC5')
        
        # Background gradient effect
        if 'acc' in ch_key and 'gyro' not in ch_key:
            ax.set_facecolor('#FAFAFA')
        else:
            ax.set_facecolor('#F5F5F5')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.04, top=0.95, left=0.14, right=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'UCIHAR_9Channel_Waves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print(f"✓ Detailed view saved to: {output_path}")
    
    alt_output = Path(__file__).parent.parent / 'results' / 'UCIHAR_9Channel_Waves.png'
    plt.savefig(alt_output, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"✓ Also saved to: {alt_output}")
    
    plt.close()
    return output_path


def create_activity_comparison_grid():
    """Create a grid showing all 6 activities with their sensor signatures."""
    
    dataset_path = Path(__file__).parent.parent.parent / 'datasets' / 'UCI HAR Dataset'
    
    print("\nLoading UCI-HAR dataset for activity comparison...")
    channels, labels, activity_names = load_ucihar_sample(dataset_path)
    
    # Create 2x3 grid for 6 activities
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=300,
                             gridspec_kw={'hspace': 0.25, 'wspace': 0.15})
    axes = axes.flatten()
    
    # Title
    fig.suptitle('UCI-HAR: Activity Comparison - Accelerometer Signatures', 
                 fontsize=22, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, 'Body Acceleration (X: Red, Y: Green, Z: Blue) | 128 timesteps @ 50Hz',
             ha='center', fontsize=12, color='#546E7A', style='italic')
    
    # Colors for XYZ
    xyz_colors = ['#E53935', '#43A047', '#1E88E5']
    
    # Activity background colors
    bg_colors = {
        1: '#E3F2FD',  # Walking - Light Blue
        2: '#F3E5F5',  # Walking Upstairs - Light Purple
        3: '#FFF3E0',  # Walking Downstairs - Light Orange
        4: '#E8F5E9',  # Sitting - Light Green
        5: '#FFF8E1',  # Standing - Light Amber
        6: '#FCE4EC',  # Laying - Light Pink
    }
    
    time = np.arange(128) / 50.0
    
    for ax_idx, activity_id in enumerate(range(1, 7)):
        ax = axes[ax_idx]
        
        # Find sample for this activity
        indices = np.where(labels == activity_id)[0]
        sample_idx = indices[len(indices) // 3]  # Pick representative sample
        
        # Set background
        ax.set_facecolor(bg_colors[activity_id])
        
        # Plot body acceleration XYZ
        for ch_idx, (ch_suffix, color) in enumerate(zip(['x', 'y', 'z'], xyz_colors)):
            data = channels[f'body_acc_{ch_suffix}'][sample_idx]
            ax.plot(time, data, color=color, linewidth=1.8, alpha=0.85,
                   label=f'{ch_suffix.upper()}-axis')
        
        # Activity label
        activity_name = activity_names[activity_id]
        ax.set_title(activity_name, fontsize=14, fontweight='bold', 
                    color='#1A237E', pad=10)
        
        # Styling
        ax.set_xlim(0, time[-1])
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Acceleration (g)', fontsize=10)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color('#90A4AE')
            spine.set_linewidth(1.2)
        
        # Add legend to first plot only
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.06, top=0.92, left=0.06, right=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'UCIHAR_Activity_Comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print(f"✓ Activity comparison saved to: {output_path}")
    
    alt_output = Path(__file__).parent.parent / 'results' / 'UCIHAR_Activity_Comparison.png'
    plt.savefig(alt_output, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"✓ Also saved to: {alt_output}")
    
    plt.close()
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("UCI-HAR Sensor Time Series Visualization")
    print("=" * 60)
    
    # Generate all visualizations
    create_vertical_wave_visualization()
    create_single_activity_detailed_view()
    create_activity_comparison_grid()
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
