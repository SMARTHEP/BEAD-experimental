# Statistical Plotting for ROC Analysis

This document describes the new statistical plotting functionality added to BEAD for analyzing model performance variation across different signal samples.

## Overview

The statistical plotting functions parse terminal output from the `roc_per_signal` mode and generate visualizations showing the statistical variation of different models across each available signal sample individually. This provides insights into model performance consistency and highlights differences between models.

## Features

### Plot Types

1. **Box Plots** (`*_box_plots.pdf`)
   - Show statistical summaries (median, quartiles, outliers) for each model
   - Each metric (AUC, TPR at different FPR levels) has its own subplot
   - Models are displayed along the y-axis with different colors

2. **Violin Plots** (`*_violin_plots.pdf`)
   - Show the full distribution shape for each model's performance
   - Includes density estimation showing probability distribution
   - Reveals multi-modal distributions and distribution skewness

3. **Combined Plots** (`*_combined_plots.pdf`)
   - Overlay box plots on top of violin plots
   - Violin plots provide background distribution information
   - Box plots provide statistical summary overlays
   - Best of both visualization types

### Metrics Analyzed

- **AUC**: Area Under the ROC Curve
- **TPR at FPR 1e-4**: True Positive Rate at False Positive Rate 10^-4
- **TPR at FPR 1e-3**: True Positive Rate at False Positive Rate 10^-3  
- **TPR at FPR 1e-2**: True Positive Rate at False Positive Rate 10^-2
- **TPR at FPR 1e-1**: True Positive Rate at False Positive Rate 10^-1

### Color Schemes

#### CSF Results Workspace
- `convvae`: Blue
- `convvae_planar`: Orange
- `convvae_house`: Green
- `ntx_convvae`: Red
- `dvae`: Violet
- `convvae_sc`: Brown
- `convvae_house_sc_anneal`: Pink

#### Other Workspaces
- Uses matplotlib's default tab10 colormap for consistent color assignment

## Usage

### Method 1: Standalone Script

```bash
# Basic usage
uv run python generate_statistical_plots.py roc_output.txt

# With custom output directory
uv run python generate_statistical_plots.py roc_output.txt ./my_plots/

# With verbose output
uv run python generate_statistical_plots.py roc_output.txt -v
```

### Method 2: Python Import

```python
from bead.src.utils.statistical_plotting import generate_statistical_plots_from_roc_output

# Generate all plots
generate_statistical_plots_from_roc_output(
    output_file_path="roc_output.txt",
    save_dir="./statistical_plots/",
    verbose=True
)
```

### Method 3: Individual Plot Types

```python
from bead.src.utils.statistical_plotting import (
    parse_roc_output,
    create_box_plots,
    create_violin_plots,
    create_combined_box_violin_plots
)

# Parse the data
parsed_data = parse_roc_output("roc_output.txt", verbose=True)

# Generate specific plot types
create_box_plots(parsed_data, "./plots/", verbose=True)
create_violin_plots(parsed_data, "./plots/", verbose=True)
create_combined_box_violin_plots(parsed_data, "./plots/", verbose=True)
```

## Input Format

The functions expect terminal output from BEAD's `roc_per_signal` mode. The expected format includes:

```
Processing signal file: sneaky1000R025 (indices 1407905:1419763)
  sneaky1000R025 - LOSS AUC: 0.6364
    TPR at FPR 1.0e-04: 0.0027
    TPR at FPR 1.0e-03: 0.0192
    TPR at FPR 1.0e-02: 0.0712
    TPR at FPR 1.0e-01: 0.2576
Saved per-signal ROC plot: bead/workspaces/csf_results/convvae/output/plots/loss/roc_sneaky1000R025.pdf
```

## Output Files

For each workspace found in the ROC output, three PDF files are generated:

- `{workspace_name}_box_plots.pdf`
- `{workspace_name}_violin_plots.pdf` 
- `{workspace_name}_combined_plots.pdf`

## Signal Processing

- The `sneaky5000R075` signal is automatically skipped due to known data length mismatches
- Signal names are parsed to extract mediator mass and R_invisible values
- Each signal's performance metrics are collected across all models in the workspace

## Dependencies

The statistical plotting functionality requires:
- `matplotlib` (for plotting)
- `numpy` (for numerical operations)
- Standard library modules: `os`, `re`, `warnings`, `collections`

Note: This functionality is kept in a separate module (`statistical_plotting.py`) to avoid adding dependencies to the main plotting module.

## File Structure

```
bead/src/utils/
├── plotting.py              # Main plotting functions (no new dependencies)
└── statistical_plotting.py  # Statistical plotting functions (requires additional deps)
```

## Error Handling

- Validates input file existence
- Handles missing data gracefully
- Provides informative error messages
- Includes optional verbose mode for debugging
- Skips problematic signals automatically

## Future Enhancements

Potential improvements could include:
- Support for additional metrics beyond AUC and TPR
- Interactive plots with hover information
- Statistical significance testing between models
- Custom color scheme configuration
- Export to other formats (PNG, SVG, etc.)