"""
Statistical visualization utilities for model performance analysis.

This module provides functions for creating box plots, violin plots, and combined
visualizations to analyze statistical variation of different models across signal samples.
These plots are designed to work with ROC output data from the roc_per_signal mode.

Functions:
    parse_roc_output: Parse ROC output file to extract performance metrics.
    create_box_plots: Generate box plots showing statistical variation.
    create_violin_plots: Generate violin plots showing performance distributions.
    create_combined_box_violin_plots: Create overlaid box and violin plots.
    generate_statistical_plots_from_roc_output: Main function to generate all plots.

Note:
    This module requires pandas for data manipulation. It's kept separate from the
    main plotting module to avoid adding pandas as a core dependency.
"""

import os
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_roc_output(output_file_path, verbose=False):
    """
    Parse ROC output file to extract AUC and TPR values for each model and signal.
    
    Parameters
    ----------
    output_file_path : str
        Path to the ROC output text file
    verbose : bool, optional
        Whether to print parsing progress, default is False
        
    Returns
    -------
    dict
        Dictionary with workspace names as keys, each containing model data
        Structure: {workspace_name: {model_name: [{signal_name: str, auc: float, 
                   tpr_1e-4: float, tpr_1e-3: float, tpr_1e-2: float, tpr_1e-1: float}]}}
    """
    if verbose:
        print(f"Parsing ROC output from: {output_file_path}")
    
    data = defaultdict(lambda: defaultdict(list))
    
    with open(output_file_path, 'r') as file:
        content = file.read()
    
    # Split by sections that start with "Generating per-signal ROC plots..."
    sections = re.split(r'Generating per-signal ROC plots\.\.\.', content)
    
    for section in sections[1:]:  # Skip the first empty section
        current_workspace = None
        current_model = None
        
        lines = section.strip().split('\n')
        
        for i, line in enumerate(lines):
            # Look for signal processing lines
            signal_match = re.match(r'Processing signal file: (\w+) \(indices', line)
            if signal_match:
                signal_name = signal_match.group(1)
                
                # Skip if this is the problematic sneaky5000R075 signal
                if signal_name == "sneaky5000R075":
                    continue
                
                # Look for the next lines containing AUC and TPR values
                if i + 1 < len(lines):
                    auc_line = lines[i + 1].strip()
                    auc_match = re.search(r'(\w+) - LOSS AUC: ([0-9.]+)', auc_line)
                    if auc_match:
                        auc_value = float(auc_match.group(2))
                        
                        # Extract TPR values from the next 4 lines
                        tpr_values = {}
                        fpr_levels = ['1.0e-04', '1.0e-03', '1.0e-02', '1.0e-01']
                        
                        for j, fpr_level in enumerate(fpr_levels):
                            if i + 2 + j < len(lines):
                                tpr_line = lines[i + 2 + j].strip()
                                tpr_match = re.search(f'TPR at FPR {re.escape(fpr_level)}: ([0-9.]+)', tpr_line)
                                if tpr_match:
                                    tpr_values[f'tpr_{fpr_level}'] = float(tpr_match.group(1))
                        
                        # Look for the saved plot line to extract workspace and model
                        for k in range(i + 6, min(i + 10, len(lines))):
                            if k < len(lines) and 'Saved per-signal ROC plot:' in lines[k]:
                                plot_path = lines[k].split(': ')[1]
                                # Extract workspace and model from path like:
                                # bead/workspaces/csf_results/convvae/output/plots/loss/roc_sneaky1000R025.pdf
                                path_parts = plot_path.split('/')
                                if len(path_parts) >= 5 and path_parts[1] == 'workspaces':
                                    current_workspace = path_parts[2]
                                    current_model = path_parts[3]
                                    
                                    # Create the data entry
                                    entry = {
                                        'signal_name': signal_name,
                                        'auc': auc_value,
                                        **tpr_values
                                    }
                                    
                                    data[current_workspace][current_model].append(entry)
                                    
                                    if verbose:
                                        print(f"Added data for {current_workspace}/{current_model}/{signal_name}: AUC={auc_value:.4f}")
                                break
    
    return dict(data)


def _convert_to_dataframe(parsed_data, workspace_name):
    """
    Convert parsed data to a structured format for plotting.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    workspace_name : str
        Name of the workspace to process
        
    Returns
    -------
    list
        List of dictionaries with structured data for plotting
    """
    workspace_data = parsed_data.get(workspace_name, {})
    plot_data = []
    
    for model_name, model_data in workspace_data.items():
        for entry in model_data:
            plot_data.append({
                'model': model_name,
                'signal': entry['signal_name'],
                'AUC': entry['auc'],
                'TPR_1e-4': entry.get('tpr_1.0e-04', np.nan),
                'TPR_1e-3': entry.get('tpr_1.0e-03', np.nan),
                'TPR_1e-2': entry.get('tpr_1.0e-02', np.nan),
                'TPR_1e-1': entry.get('tpr_1.0e-01', np.nan)
            })
    
    return plot_data


def create_box_plots(parsed_data, save_dir, verbose=False):
    """
    Create box plots showing statistical variation across models and signals.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating box plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2', 'TPR_1e-1']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            box_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    box_data.append(model_values)
                    box_labels.append(model)
                    box_colors.append(colors[j])
            
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 4})
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Comparison - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_box_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved box plot: {save_path}")


def create_violin_plots(parsed_data, save_dir, verbose=False):
    """
    Create violin plots showing statistical variation across models and signals.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2', 'TPR_1e-1']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot
            violin_data = []
            violin_labels = []
            violin_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    violin_data.append(model_values)
                    violin_labels.append(model)
                    violin_colors.append(colors[j])
            
            if violin_data:
                vp = ax.violinplot(violin_data, positions=range(1, len(violin_data) + 1), 
                                  showmeans=True, showmedians=True)
                
                # Color the violins
                for pc, color in zip(vp['bodies'], violin_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(1, len(violin_labels) + 1))
                ax.set_xticklabels(violin_labels, rotation=45)
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Distribution - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_violin_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved violin plot: {save_path}")


def create_combined_box_violin_plots(parsed_data, save_dir, verbose=False):
    """
    Create combined plots with both box and violin plots overlaid.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating combined box and violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2', 'TPR_1e-1']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for both plots
            plot_data_list = []
            plot_labels = []
            plot_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    plot_data_list.append(model_values)
                    plot_labels.append(model)
                    plot_colors.append(colors[j])
            
            if plot_data_list:
                positions = range(1, len(plot_data_list) + 1)
                
                # Create violin plot first (background)
                vp = ax.violinplot(plot_data_list, positions=positions, 
                                  showmeans=False, showmedians=False, showextrema=False)
                
                # Color the violins with transparency
                for pc, color in zip(vp['bodies'], plot_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                
                # Create box plot on top
                bp = ax.boxplot(plot_data_list, positions=positions, patch_artist=True, 
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 3},
                               widths=0.3)  # Make boxes narrower
                
                # Color the boxes with higher opacity
                for patch, color in zip(bp['boxes'], plot_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(plot_labels, rotation=45)
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance (Box + Violin) - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_combined_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved combined plot: {save_path}")


def generate_statistical_plots_from_roc_output(output_file_path, save_dir=None, verbose=False):
    """
    Generate all statistical plots (box, violin, and combined) from ROC output file.
    
    Parameters
    ----------
    output_file_path : str
        Path to the ROC output text file
    save_dir : str, optional
        Directory to save plots, if None uses current directory
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if save_dir is None:
        save_dir = os.path.dirname(output_file_path)
    
    if verbose:
        print("Generating statistical plots from ROC output...")
    
    # Parse the ROC output
    parsed_data = parse_roc_output(output_file_path, verbose=verbose)
    
    if not parsed_data:
        print("No data found in ROC output file")
        return
    
    # Create all types of plots
    create_box_plots(parsed_data, save_dir, verbose=verbose)
    create_violin_plots(parsed_data, save_dir, verbose=verbose)
    create_combined_box_violin_plots(parsed_data, save_dir, verbose=verbose)
    
    if verbose:
        print(f"All statistical plots saved to: {save_dir}")