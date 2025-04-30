# Copyright 2025 BEAD Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BEAD (Behavior and Anomaly Detection) main module.

This module serves as the entry point for the BEAD framework, providing command-line
interface functionality for anomaly detection workflows. It orchestrates the complete
pipeline from data preparation to model training, inference, and result visualization.
"""

import os

from src.utils import ggl

# __all__ = (
#     "create_new_project",  # noqa: F822
#     "convert_csv",  # noqa: F822
#     "prepare_inputs",  # noqa: F822
#     "run_training",  # noqa: F822
#     "run_inference",  # noqa: F822
#     "run_plots",  # noqa: F822
#     "run_full_chain",  # noqa: F822
#     "run_diagnostics",  # noqa: F822
# )


def main():
    """Process command-line arguments to execute BEAD functionality.

    Parses command-line arguments and executes the appropriate functionality based on the
    specified mode. The available modes are:

    Args:
        Arguments are parsed from command line, not passed directly to this function.

    Modes:
        new_project: Create a new project with default configuration.
        convert_csv: Convert CSV data to HDF5 or NPY format.
        prepare_inputs: Process data files into PyTorch tensors for training.
        train: Train a model using the prepared data.
        detect: Run inference on data to detect anomalies.
        plot: Generate visualization plots for results.
        diagnostics: Run performance profiling and diagnostics.
        chain: Execute the full pipeline from data conversion to visualization.

    Raises:
        NameError: If the specified mode is not recognized.
    """
    (
        config,
        mode,
        options,
        workspace_name,
        project_name,
        verbose,
    ) = ggl.get_arguments()

    # Define paths dict for the different paths used frequently in the pipeline
    paths = {
        "workspace_path": os.path.join("workspaces", workspace_name),
        "project_path": os.path.join("workspaces", workspace_name, project_name),
        "data_path": os.path.join("workspaces", workspace_name, "data"),
        "output_path": os.path.join(
            "workspaces", workspace_name, project_name, "output"
        ),
    }

    # Check what the options flag is set to and override the default if necessary
    if options == "h5" or options == "npy":
        config.file_type = options

    # Call the appropriate ggl function based on the mode
    if mode == "new_project":
        ggl.create_new_project(workspace_name, project_name, verbose)
    elif mode == "convert_csv":
        ggl.convert_csv(paths, config, verbose)
    elif mode == "prepare_inputs":
        ggl.prepare_inputs(paths, config, verbose)
    elif mode == "train":
        ggl.run_training(paths, config, verbose)
    elif mode == "detect":
        ggl.run_inference(paths, config, verbose)
    elif mode == "plot":
        ggl.run_plots(paths, config, verbose)
    elif mode == "diagnostics":
        ggl.run_diagnostics(paths, config, verbose)
    elif mode == "chain":
        ggl.run_full_chain(
            workspace_name, project_name, paths, config, options, verbose
        )
    else:
        raise NameError(
            "BEAD mode "
            + mode
            + " not recognised. Use < bead --help > to see available modes."
        )
