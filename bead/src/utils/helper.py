# This file contains functions that help manipulate different artifacts as required
# in the pipeline. The functions in this file are used to manipulate data, models, and # tensors.
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import Dataset

from ..models import models
from . import loss


def get_device(config=None):
    """
    Returns the appropriate processing device.
    If DDP is active, uses the local_rank.
    Otherwise, uses cuda:0 if available, else cpu.

    Args:
        config (dataClass): Base class selecting user inputs.

    Returns:
        torch.device: The device to be used for processing.

    """
    if (
        config
        and hasattr(config, "is_ddp_active")
        and config.is_ddp_active
        and torch.cuda.is_available()
    ):
        return torch.device(f"cuda:{config.local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def detach_device(tensor):
    """
    Detaches a given tensor to ndarray

    Args:
        tensor (torch.Tensor): The PyTorch tensor one wants to convert to a ndarray

    Returns:
        ndarray: Converted torch.Tensor to ndarray
    """
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data):
    """
    Converts ndarray to torch.Tensors.

    Args:
        data (ndarray): The data you wish to convert from ndarray to torch.Tensor.

    Returns:
        torch.Tensor: Your data as a tensor
    """
    return torch.tensor(data, dtype=torch.float32)


def numpy_to_tensor(data):
    """
    Converts ndarray to torch.Tensors.

    Args:
        data (ndarray): The data you wish to convert from ndarray to torch.Tensor.

    Returns:
        torch.Tensor: Your data as a tensor
    """
    return torch.from_numpy(data)


def save_model(model, model_path: str, config=None) -> None:
    """
    Saves the models state dictionary as a `.pt` file to the given path.
    Handles DDP model saving.

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.
        config (dataClass): Base class selecting user inputs. Used to check if DDP is active.

    Returns:
        None: Saved model state dictionary as `.pt` file.
    """
    if config and hasattr(config, "is_ddp_active") and config.is_ddp_active:
        if config.local_rank == 0:  # Only save from rank 0
            torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


class Log1pScaler(BaseEstimator, TransformerMixin):
    """
    Log(1+x) transformer for positive-skewed HEP features
    """

    def __init__(self):
        self.epsilon = 1e-8  # Small value to prevent log(0)

    def fit(self, X, y=None):
        if np.any(X + self.epsilon <= 0):
            raise ValueError("Data contains values <= 0 after epsilon addition")
        return self

    def transform(self, X):
        return np.log1p(X + self.epsilon)

    def inverse_transform(self, X):
        return np.expm1(X) - self.epsilon


class L2Normalizer(BaseEstimator, TransformerMixin):
    """
    L2 normalization per feature of data
    """

    def __init__(self):
        self.norms = None

    def fit(self, X, y=None):
        self.norms = np.linalg.norm(X, axis=0)
        self.norms[self.norms == 0] = 1.0  # Prevent division by zero
        return self

    def transform(self, X):
        return X / self.norms

    def inverse_transform(self, X):
        return X * self.norms


class SinCosTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms an angle (in radians) into two features:
    [sin(angle), cos(angle)]. Inverse transformation uses arctan2.
    """

    def fit(self, X, y=None):
        # Nothing to learn
        return self

    def transform(self, X):
        # Ensure X is of shape (N,1)
        X = np.asarray(X).reshape(-1, 1)
        sin_part = np.sin(X)
        cos_part = np.cos(X)
        return np.hstack([sin_part, cos_part])

    def inverse_transform(self, X):
        if X.shape[1] != 2:
            raise ValueError(
                "Expected input with 2 columns for inverse transformation."
            )
        sin_part = X[:, 0]
        cos_part = X[:, 1]
        angles = np.arctan2(sin_part, cos_part).reshape(-1, 1)
        return angles


class ChainedScaler(BaseEstimator, TransformerMixin):
    """
    Chains a list of scaler transformations.
    The transformation is applied sequentially (in the order provided)
    and the inverse transformation is applied in reverse order.
    """

    def __init__(self, scalers):
        self.scalers = scalers

    def fit(self, X, y=None):
        data = X
        for scaler in self.scalers:
            scaler.fit(data)
            data = scaler.transform(data)
        return self

    def transform(self, X):
        data = X
        for scaler in self.scalers:
            data = scaler.transform(data)
        return data

    def inverse_transform(self, X):
        data = X
        for scaler in reversed(self.scalers):
            data = scaler.inverse_transform(data)
        return data


def normalize_data(data, normalization_type):
    """
    Normalizes jet data for VAE-based anomaly detection.

    Args:
        data: 2D numpy array (n_jets, n_features)
        normalization_type: A string indicating the normalization method(s).
            It can be a single method or a chain of methods separated by '+'.
            Valid options include:
            'minmax'  - MinMaxScaler (scales features to [0,1])
            'standard'- StandardScaler (zero mean, unit variance)
            'robust'  - RobustScaler (less sensitive to outliers)
            'log'     - Log1pScaler (applies log1p transformation)
            'l2'      - L2Normalizer (scales each feature by its L2 norm)
            'power'   - PowerTransformer (using Yeo-Johnson)
            'quantile'- QuantileTransformer (transforms features to follow a normal or uniform distribution)
            'maxabs'  - MaxAbsScaler (scales each feature by its maximum absolute value)
            'sincos'  - SinCosTransformer (converts angles to sin/cos features)
            Example: 'log+standard' applies a log transformation followed by standard scaling.

    Returns:
        normalized_data: Transformed data array.
        scaler: Fitted scaler object (or chained scaler) for inverse transformations.
    """
    # Handle potential NaN/inf in HEP data
    if np.any(~np.isfinite(data)):
        raise ValueError("Input data contains NaN/infinite values")

    # Mapping from method names to corresponding scaler constructors.
    scaler_map = {
        "minmax": lambda: MinMaxScaler(feature_range=(0, 1)),
        "standard": lambda: StandardScaler(),
        "robust": lambda: RobustScaler(
            quantile_range=(5, 95)
        ),  # Reduced outlier sensitivity
        "log": lambda: Log1pScaler(),
        "l2": lambda: L2Normalizer(),
        "power": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
        "quantile": lambda: QuantileTransformer(output_distribution="normal"),
        "maxabs": lambda: MaxAbsScaler(),
        "sincos": lambda: SinCosTransformer(),
    }

    # Parse the chain of normalization methods.
    methods = (
        normalization_type.split("+")
        if "+" in normalization_type
        else [normalization_type]
    )

    scalers = []
    transformed_data = data.copy()

    for method in methods:
        method = method.strip().lower()
        if method not in scaler_map:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                "Valid options: " + ", ".join(scaler_map.keys())
            )
        scaler = scaler_map[method]()
        scaler.fit(transformed_data)
        transformed_data = scaler.transform(transformed_data)
        scalers.append(scaler)

    # If multiple scalers are used, return a chained scaler; otherwise the single scaler.
    if len(scalers) > 1:
        composite_scaler = ChainedScaler(scalers)
    else:
        composite_scaler = scalers[0]

    return transformed_data, composite_scaler


def invert_normalize_data(normalized_data, scaler):
    """
    Inverts a chained normalization transformation.

    This function accepts normalized data (for example, the output of a VAE's preprocessed input)
    and the scaler (or ChainedScaler) that was used to perform the forward transformation.
    It then returns the original data by calling the scaler's inverse_transform method.

    Args:
        normalized_data (np.ndarray): The transformed data array.
        scaler: The scaler object (or a ChainedScaler instance) used for the forward transformation,
            which must implement an `inverse_transform` method.

    Returns:
        np.ndarray: The data mapped back to its original scale.
    """
    if not hasattr(scaler, "inverse_transform"):
        raise ValueError(
            "The provided scaler object does not have an inverse_transform method."
        )
    return scaler.inverse_transform(normalized_data)


def load_tensors(folder_path, keyword="sig_test", include_efp=False):
    """
    Searches through the specified folder for all '.pt' files containing the given keyword in their names.
    Categorizes these files based on the presence of 'jets', 'events', 'constituents', or 'efp' in their filenames,
    loads them into PyTorch tensors, concatenates them along axis=0, and returns the resulting tensors.

    Args:
        folder_path (str): The path to the folder to search.
        keyword (str): The keyword to filter files ('bkg_train', 'bkg_test', or 'sig_test').
        include_efp (bool): Whether to include EFP features in loading. Default False for backward compatibility.

    Returns:
        tuple: A tuple containing PyTorch tensors. If include_efp=False: (events, jets, constituents).
               If include_efp=True: (events, jets, constituents, efp).

    Raises:
        ValueError: If any required category ('jets', 'events', 'constituents') has no matching files.
                   EFP files are optional and won't raise errors if missing.
    """
    if keyword not in ["bkg_train", "bkg_test", "sig_test"]:
        raise ValueError(
            "Invalid keyword. Please choose from 'bkg_train', 'bkg_test', or 'sig_test'."
        )

    # Initialize dictionaries to hold file lists for each category
    file_categories = {"jets": [], "events": [], "constituents": []}
    if include_efp:
        file_categories["efp"] = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".pt") and keyword in filename:
            # Categorize files based on their names
            for category in file_categories:
                if category in filename:
                    file_categories[category].append(
                        os.path.join(folder_path, filename)
                    )

    # Function to load and concatenate a list of .pt files along axis 0
    def load_and_concat(file_list):
        tensors = [torch.load(file) for file in file_list]
        return torch.cat(tensors, dim=0)

    # Load and concatenate tensors for each category
    result_tensors = {}
    for category, files in file_categories.items():
        if not files:
            if category == "efp" and include_efp:
                # EFP files are optional - return None if missing
                result_tensors[category] = None
                continue
            else:
                raise ValueError(
                    f"Required files with keyword, '{keyword}' not found. Please run the --mode convert_csv and prepare_inputs before retrying."
                )
        result_tensors[category] = load_and_concat(files)

    if include_efp:
        return (
            result_tensors["events"],
            result_tensors["jets"],
            result_tensors["constituents"],
            result_tensors["efp"],
        )
    else:
        return (
            result_tensors["events"],
            result_tensors["jets"],
            result_tensors["constituents"],
        )


def get_signal_file_info_from_csv(csv_folder_path, keyword="sig_test"):
    """
    Get information about signal files for per-signal ROC plotting by counting CSV lines.
    
    Args:
        csv_folder_path (str): The path to the folder containing CSV files.
        keyword (str): The keyword to filter files (default: 'sig_test').
    
    Returns:
        list: List of dictionaries containing file info with keys:
              'filename', 'sig_filename', 'events_count', 'start_idx', 'end_idx'
    """
    import csv
    
    # Get all signal CSV files and sort them to ensure consistent ordering
    signal_files = []
    for filename in sorted(os.listdir(csv_folder_path)):
        if filename.endswith(".csv") and keyword in filename:
            signal_files.append(filename)
    
    # Extract signal file info by counting CSV lines
    file_info = []
    current_idx = 0
    
    for filename in signal_files:
        # Count lines in CSV file (excluding header)
        csv_path = os.path.join(csv_folder_path, filename)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header and count data rows
            next(reader, None)  # Skip header
            events_count = sum(1 for row in reader)
        
        # Extract signal filename from CSV filename
        # Convert from 'sig_test_sneaky1000R025.csv' to 'sneaky1000R025'
        base_name = filename.replace('.csv', '')
        sig_filename = base_name.replace(f'{keyword}_', '')
        
        file_info.append({
            'filename': filename,
            'sig_filename': sig_filename,
            'events_count': events_count,
            'start_idx': current_idx,
            'end_idx': current_idx + events_count
        })
        
        current_idx += events_count
    
    return file_info


def get_bkg_test_count_from_csv(csv_folder_path):
    """
    Get the total count of background test events by counting CSV lines.
    
    Args:
        csv_folder_path (str): The path to the folder containing CSV files.
    
    Returns:
        int: Total number of background test events
    """
    import csv
    
    total_bkg_count = 0
    
    # Get all background test CSV files
    for filename in os.listdir(csv_folder_path):
        if filename.endswith(".csv") and filename.startswith("bkg_test"):
            csv_path = os.path.join(csv_folder_path, filename)
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header and count data rows
                next(reader, None)  # Skip header
                events_count = sum(1 for row in reader)
                total_bkg_count += events_count
    
    return total_bkg_count


def load_augment_tensors(folder_path, keyword):
    """
    Searches through the specified folder for all '.pt' files whose names contain the specified
    keyword (e.g., 'bkg_train', 'bkg_test', or 'sig_test'). Files are then categorized by whether
    their filename contains one of the three substrings: 'jets', 'events', or 'constituents'.

    For 'bkg_train', each file must contain one of the generator names: 'herwig', 'pythia', or 'sherpa'.
    For each file, the tensor is loaded and a new feature is appended along the last dimension:
    - 0 for files containing 'herwig'
    - 1 for files containing 'pythia'
    - 2 for files containing 'sherpa'

    For 'bkg_test' and 'sig_test', the appended new feature is filled with -1, since generator info
    is not available at test time.

    Finally, for each category the resulting tensors are concatenated along axis=0.

    Args:
        folder_path (str): The path to the folder to search.
        keyword (str): The keyword to filter files (e.g., 'bkg_train', 'bkg_test', or 'sig_test').

    Returns:
        tuple: A tuple of three PyTorch tensors: (jets_tensor, events_tensor, constituents_tensor)
            corresponding to the concatenated tensors for each category.

    Raises:
        ValueError: If any category does not have at least one file for each generator type.
            The error message is:
            "required files not found. please run the --mode convert_csv and prepare inputs before retrying"
    """
    # Check if the keyword is valid
    if keyword not in ["bkg_train", "bkg_test"]:
        raise ValueError(
            "Invalid keyword. Please choose from 'bkg_train' or 'bkg_test'."
        )

    # Define the categories and generator subcategories.
    categories = ["jets", "events", "constituents"]
    generators = {"herwig": 0, "pythia": 1, "sherpa": 2}

    # Initialize dictionary to store files per category and generator.
    file_categories = {cat: {gen: [] for gen in generators} for cat in categories}

    # Iterate over files in the folder.
    keyword_found = False
    for filename in os.listdir(folder_path):
        # Only consider files ending with '.pt' that contain the specified keyword.
        if not filename.endswith(".pt"):
            continue
        if keyword not in filename:
            continue

        keyword_found = True
        lower_filename = filename.lower()
        # Determine category based on substring in the filename.
        for cat in categories:
            if cat in lower_filename:
                # Determine generator type.
                for gen, gen_val in generators.items():
                    if gen in lower_filename:
                        full_path = os.path.join(folder_path, filename)
                        file_categories[cat][gen].append((full_path, gen_val))
                # Note: if a file contains multiple generator substrings (unlikely), it will be added
                # to all matching generator groups.
    if not keyword_found:
        raise ValueError("No files found with the specified keyword, " + keyword)

    # For each category in 'bkg_train', ensure that each generator type has at least one file.
    # if keyword == "bkg_train":
    #     for cat in categories:
    #         for gen in generators:
    #             if len(file_categories[cat][gen]) == 0:
    #                 raise ValueError(
    #                     "Required files not found. Please run the --mode convert_csv and prepare inputs before retrying."
    #                 )

    # For each file, load its tensor and append the generator feature.
    def load_and_augment(file_info):
        """
        Given a tuple (file_path, generator_value), load the tensor and append a new feature column
        with the constant generator_value along the last dimension.
        Works for both 2D and 3D tensors.
        """
        file_path, gen_val = file_info
        tensor = torch.load(file_path)
        # Create a constant tensor with the same device and dtype as tensor.
        if tensor.dim() == 2:
            # For a 2D tensor of shape (m, n), create a (m, 1) tensor.
            constant_feature = torch.full(
                (tensor.size(0), 1), gen_val, dtype=tensor.dtype, device=tensor.device
            )
            augmented = torch.cat([tensor, constant_feature], dim=1)
        elif tensor.dim() == 3:
            # For a 3D tensor of shape (m, p, n), create a (m, p, 1) tensor.
            constant_feature = torch.full(
                (tensor.size(0), tensor.size(1), 1),
                gen_val,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            augmented = torch.cat([tensor, constant_feature], dim=2)
        else:
            raise ValueError(
                "Tensor from {} has unsupported dimensions: {}".format(
                    file_path, tensor.dim()
                )
            )
        return augmented

    # For each category, load the tensors for each generator, augment them, and then concatenate.
    concatenated = {}
    for cat in categories:
        cat_tensors = []
        for gen in generators:
            # Get the list of file infos (tuples) for this generator.
            file_list = file_categories[cat][gen]
            # For each file, load and augment.
            for file_info in file_list:
                cat_tensors.append(load_and_augment(file_info))
        # Before concatenation, we want to split the data into a multiple of the sample count
        # (here we simply concatenate along axis=0).
        concatenated[cat] = torch.cat(cat_tensors, dim=0)

    return concatenated["events"], concatenated["jets"], concatenated["constituents"]


def select_features(jets_tensor, constituents_tensor, input_features):
    """
    Process the jets_tensor and constituents_tensor based on the input_features flag.

    Parameters:
        jets_tensor (torch.Tensor): Tensor with features
            [evt_id, jet_id, num_constituents, b_tagged, jet_pt, jet_eta, jet_phi_sin, jet_phi_cos, generator_id]
        constituents_tensor (torch.Tensor): Tensor with features
            [evt_id, jet_id, constit_id, b_tagged, constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id]
        input_features (str): The flag to determine which features to select.
            Options:
            - 'all': return tensors as is.
            - '4momentum': select [pt, eta, phi_sin, phi_cos, generator_id] for both.
            - '4momentum_btag': select [b_tagged, pt, eta, phi_sin, phi_cos, generator_id] for both.
            - 'pj_custom': select everything except [evt_id, jet_id] for jets and except [evt_id, jet_id, constit_id] for constituents.

    Returns:
        tuple: Processed jets_tensor and constituents_tensor.
    """

    if input_features == "all":
        # Return tensors unchanged.
        return jets_tensor, constituents_tensor

    elif input_features == "4momentum":
        # For jets: [jet_pt, jet_eta, jet_phi_sin, jet_phi_cos, generator_id] -> indices [4, 5, 6, 7, 8]
        jets_out = jets_tensor[:, :, 4:]
        # For constituents: [constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id] -> indices [4, 5, 6, 7, 8]
        constituents_out = constituents_tensor[:, :, 4:]
        return jets_out, constituents_out

    elif input_features == "4momentum_btag":
        # For jets: [b_tagged, jet_pt, jet_eta, jet_phi_sin, jet_phi_cos, generator_id] -> indices [3, 4, 5, 6, 7, 8]
        jets_out = jets_tensor[:, :, 3:]
        # For constituents: [b_tagged, constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id] -> indices [3, 4, 5, 6, 7, 8]
        constituents_out = constituents_tensor[:, :, 3:]
        return jets_out, constituents_out

    elif input_features == "pj_custom":
        # For jets: exclude [evt_id, jet_id] -> remove indices [0, 1]
        jets_out = jets_tensor[:, :, 2:]  # returns indices 2 to end
        # For constituents: exclude [evt_id, jet_id, constit_id] -> remove indices [0, 1, 2]
        constituents_out = constituents_tensor[:, :, 3:]  # returns indices 3 to end
        return jets_out, constituents_out

    else:
        raise ValueError("Invalid input_features flag provided.")


def train_val_split(tensor, train_ratio):
    """
    Splits a tensor into training and validation sets based on the specified train_ratio.
    The split is done by sampling indices randomly ensuring that the data is shuffled.

    Args:
        tensor (torch.Tensor): The input tensor to be split.
        train_ratio (float): Proportion of data to be used for training (e.g., 0.8 for 80% training data).

    Returns:
        tuple: A tuple containing two tensors:
            - train_tensor: Tensor containing the training data.
            - val_tensor: Tensor containing the validation data.

    Raises:
        ValueError: If train_ratio is not between 0 and 1.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be a float between 0 and 1.")

    # Set the random seed for reproducibility.
    torch.manual_seed(42)

    # Determine the split sizes
    total_size = tensor.size(0)
    train_size = int(train_ratio * total_size)

    # Generate a random permutation of indices.
    indices = torch.randperm(total_size)

    # Split the indices into train and validation indices.
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Use these indices to index into your tensor.
    train_tensor = tensor[train_indices]
    val_tensor = tensor[val_indices]

    return train_tensor, val_tensor


def add_sig_bkg_label(tensors: tuple, label: str) -> tuple:
    """
    Adds a new feature to the last dimension of each tensor in the tuple.
    The new feature is filled with 0 for "bkg" and 1 for "sig".

    Args:
        tensors: A tuple of three tensors (events, jets, constituents).
        label: A string, either "bkg" or "sig", to determine the value of the new feature.

    Returns:
        A tuple of the three tensors with the new feature added to the last dimension.
    """
    if label not in ["bkg", "sig"]:
        raise ValueError("label must be either 'bkg' or 'sig'")

    # Determine the value for the new feature
    feature_value = 0 if label == "bkg" else 1

    def add_feature(tensor: torch.Tensor) -> torch.Tensor:
        """Helper function to add the feature to a single tensor."""
        # Get shape for the new feature tensor (same as input tensor but last dim=1)
        feature_shape = tensor.shape[:-1] + (1,)

        # Create a tensor filled with the feature value, matching device and dtype
        feature = torch.full(
            feature_shape, feature_value, dtype=tensor.dtype, device=tensor.device
        )

        # Concatenate along the last dimension
        return torch.cat([tensor, feature], dim=-1)

    # Apply the feature addition to each tensor in the tuple
    events, jets, constituents = tensors
    events = add_feature(events)
    jets = add_feature(jets)
    constituents = add_feature(constituents)

    return events, jets, constituents


def data_label_split(data):
    """
    Splits the data into features and labels.

    Args:
        data (ndarray): The data you wish to split into features and labels.

    Returns:
        tuple: A tuple containing two ndarrays:
            - data: The features of the data.
            - labels: The labels of the data.
    """
    (
        events,
        jets,
        constituents,
    ) = data

    data = (
        events[:, :-1],
        jets[:, :, :-1],
        constituents[:, :, :-1],
    )

    labels = (
        events[:, -1],
        jets[:, 0, -1].squeeze(),
        constituents[:, 0, -1].squeeze(),
    )
    return data, labels


# Define the custom dataset class
class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling paired data and label tensors, with optional EFP features.

    This dataset provides a simple interface for accessing data points and their
    corresponding labels, which is compatible with PyTorch's DataLoader.

    Attributes:
        data (torch.Tensor): The data tensor containing features.
        labels (torch.Tensor): The labels tensor associated with the data.
        efp_data (torch.Tensor, optional): The EFP features tensor. None if not provided.
    """

    def __init__(self, data_tensor, label_tensor, efp_tensor=None):
        self.data = data_tensor
        self.labels = label_tensor
        self.efp_data = efp_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.efp_data is not None:
            return self.data[idx], self.labels[idx], self.efp_data[idx]
        else:
            return self.data[idx], self.labels[idx]


# Function to create datasets
def create_datasets(
    events_train,
    jets_train,
    constituents_train,
    events_val,
    jets_val,
    constituents_val,
    events_train_label,
    jets_train_label,
    constituents_train_label,
    events_val_label,
    jets_val_label,
    constituents_val_label,
    efp_train=None,
    efp_val=None,
    efp_train_label=None,
    efp_val_label=None,
):
    """
    Creates CustomDataset objects for training and validation data.

    This function pairs data tensors with their corresponding label tensors
    to create dataset objects for events, jets, constituents, and optionally EFP data.

    Args:
        events_train (torch.Tensor): Training events data.
        jets_train (torch.Tensor): Training jets data.
        constituents_train (torch.Tensor): Training constituents data.
        events_val (torch.Tensor): Validation events data.
        jets_val (torch.Tensor): Validation jets data.
        constituents_val (torch.Tensor): Validation constituents data.
        events_train_label (torch.Tensor): Labels for training events.
        jets_train_label (torch.Tensor): Labels for training jets.
        constituents_train_label (torch.Tensor): Labels for training constituents.
        events_val_label (torch.Tensor): Labels for validation events.
        jets_val_label (torch.Tensor): Labels for validation jets.
        constituents_val_label (torch.Tensor): Labels for validation constituents.
        efp_train (torch.Tensor, optional): Training EFP features data.
        efp_val (torch.Tensor, optional): Validation EFP features data.
        efp_train_label (torch.Tensor, optional): Labels for training EFP features.
        efp_val_label (torch.Tensor, optional): Labels for validation EFP features.

    Returns:
        dict: A dictionary containing CustomDataset objects for all data types.
    """
    # Create datasets for training data
    events_train_dataset = CustomDataset(events_train, events_train_label)
    jets_train_dataset = CustomDataset(jets_train, jets_train_label)
    constituents_train_dataset = CustomDataset(
        constituents_train, constituents_train_label
    )

    # Create datasets for validation data
    events_val_dataset = CustomDataset(events_val, events_val_label)
    jets_val_dataset = CustomDataset(jets_val, jets_val_label)
    constituents_val_dataset = CustomDataset(constituents_val, constituents_val_label)

    # Return all datasets as a dictionary for easy access
    datasets = {
        "events_train": events_train_dataset,
        "jets_train": jets_train_dataset,
        "constituents_train": constituents_train_dataset,
        "events_val": events_val_dataset,
        "jets_val": jets_val_dataset,
        "constituents_val": constituents_val_dataset,
    }
    
    # Add EFP datasets if provided
    if efp_train is not None and efp_train_label is not None:
        datasets["efp_train"] = CustomDataset(efp_train, efp_train_label)
    if efp_val is not None and efp_val_label is not None:
        datasets["efp_val"] = CustomDataset(efp_val, efp_val_label)
    return datasets


def calculate_in_shape(data, config, test_mode=False):
    """
    Calculates the input shapes for the models based on the data.

    Args:
        data (ndarray): The data you wish to calculate the input shapes for.
        config (dataClass): Base class selecting user inputs.
        test_mode (bool): A flag to indicate if the function is being called in test mode.

    Returns:
        tuple: A tuple containing the input shapes for the models.
    """
    if test_mode:
        bs = 1
    else:
        bs = config.batch_size
    
    # Handle data unpacking with optional EFP features
    if len(data) == 8:  # EFP features included
        (
            events_train,
            jets_train,
            constituents_train,
            efp_train,
            events_val,
            jets_val,
            constituents_val,
            efp_val,
        ) = data
    elif len(data) == 6:  # No EFP features
        (
            events_train,
            jets_train,
            constituents_train,
            events_val,
            jets_val,
            constituents_val,
        ) = data
    else:
        raise ValueError(f"Expected 6 or 8 data tensors, got {len(data)}")

    # Get the shapes of the data
    # Calculate the input shapes to initialize the model

    in_shape_e = [bs] + list(events_train.shape[1:])
    in_shape_j = [bs] + list(jets_train.shape[1:])
    in_shape_c = [bs] + list(constituents_train.shape[1:])

    if config.model_name == "pj_ensemble":
        # Make in_shape tuple
        in_shape = (in_shape_e, in_shape_j, in_shape_c)

    else:
        if config.input_level == "event":
            in_shape = in_shape_e
        elif config.input_level == "jet":
            in_shape = in_shape_j
        elif config.input_level == "constituent":
            in_shape = in_shape_c

    return in_shape


def model_init(in_shape, config):
    """
    Initializing the models attributes to a model_object variable.

    Args:
        model_name (str): The name of the model you wish to initialize. This should correspond to what your Model name.
        init (str): The initialization method you wish to use (Xavier support currently). Default is None.
        config (dataClass): Base class selecting user inputs.

    Returns:
        class: Object with the models class attributes
    """

    def xavier_init_weights(m):
        """
        Applies Xavier initialization to the weights of the given module.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model_object = getattr(models, config.model_name)

    if config.model_name == "pj_custom":
        model = model_object(*in_shape, z_dim=config.latent_space_size)

    else:
        model = model_object(in_shape, z_dim=config.latent_space_size)

    if config.model_init == "xavier":
        model.apply(xavier_init_weights)

    return model


def get_loss(loss_function: str):
    """
    Returns the loss_object based on the string provided.

    Args:
        loss_function (str): The loss function you wish to use. Options include:
            - 'mse': Mean Squared Error
            - 'bce': Binary Cross Entropy
            - 'mae': Mean Absolute Error
            - 'huber': Huber Loss
            - 'l1': L1 Loss
            - 'l2': L2 Loss
            - 'smoothl1': Smooth L1 Loss

    Returns:
        class: The loss function object
    """
    loss_object = getattr(loss, loss_function)

    return loss_object


def get_optimizer(optimizer_name, parameters, lr):
    """
    Returns a PyTorch optimizer configured with optimal arguments for training a large VAE.

    Args:
        optimizer_name (str): One of "adam", "adamw", "rmsprop", "sgd", "radam", "adagrad".
        parameters (iterable): The parameters (or parameter groups) of your model.
        lr (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: An instantiated optimizer with specified hyperparameters.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    opt = optimizer_name.lower()

    if opt == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),  # Default values
            eps=1e-8,
            weight_decay=0,  # Set to a small value like 1e-5 if regularization is needed
        )
    elif opt == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,  # L2 regularization
        )
    elif opt == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            alpha=0.99,  # Smoothing constant
            eps=1e-8,
            weight_decay=1e-2,  # L2 regularization
            momentum=0.9,  # Set to a value like 0.9 if momentum is desired
        )
    elif opt == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=0.9,  # Momentum term
            weight_decay=0,  # Set to a small value like 1e-5 if regularization is needed
            nesterov=True,  # Set to True if Nesterov momentum is desired
        )
    elif opt == "radam":
        return torch.optim.RAdam(
            parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
    elif opt == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=lr,
            lr_decay=0,  # Learning rate decay over each update
            weight_decay=0,
            initial_accumulator_value=0,  # Starting value for the accumulators
            eps=1e-10,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def call_forward(model, inputs):
    """
    Calls the `forward` method of the given object.
    If the return value is not a tuple, packs it into a tuple.

    Args:
        model: An object that has a `forward` method.
        inputs: The input data to pass to the model.

    Returns:
        A tuple containing the result(s) of the `forward` method.
    """
    # Call the forward method
    result = model(inputs)

    # Ensure the result is a tuple
    if isinstance(result, tuple):
        return result
    else:
        return (result,)


def unpack_model_outputs(outputs):
    """
    Standardizes model outputs to a consistent 6-tuple format regardless of model type.
    
    This function takes the raw outputs from different model types and ensures they all
    conform to the standard 6-tuple format: (recon, mu, logvar, ldj, z0, zk).
    
    For models that don't naturally produce all these values:
    - AE models (len(outputs) == 2): Returns (recon, zeros, zeros, zeros, z, z)
    - VAE models (len(outputs) == 4): Returns (recon, mu, logvar, zeros, z, z)
    - Flow models (len(outputs) == 6): Returns (recon, mu, logvar, ldj, z0, zk)
    - Dirichlet VAE (len(outputs) == 6): Returns (recon, mu, logvar, zeros, G_z, D_z)
    
    Args:
        outputs (tuple): The raw outputs tuple from a model's forward method.
            Could be one of:
            - (recon, z) for basic autoencoders (AE, AE_Dropout_BN, ConvAE)
            - (recon, mu, logvar, z) for VAEs (ConvVAE)
            - (recon, mu, logvar, ldj, z0, zk) for flow-based models (Planar_ConvVAE, etc.)
            - (recon, mu, logvar, G_z, G_z, D_z) for Dirichlet_ConvVAE
    
    Returns:
        tuple: A standardized 6-tuple (recon, mu, logvar, ldj, z0, zk) where:
            - recon: Reconstructed input
            - mu: Mean of the latent distribution (or zeros for AEs)
            - logvar: Log variance of the latent distribution (or zeros for AEs)
            - ldj: Log determinant of the Jacobian (or zeros for non-flow models)
            - z0: Initial sample from the latent distribution (G_z for Dirichlet VAE)
            - zk: Final transformed latent variable (D_z for Dirichlet VAE)
    
    Raises:
        ValueError: If the length of outputs is not 2, 4, or 6.
    """
    if len(outputs) == 2:  # Basic AE model: (recon, z)     
        recon, zk = outputs
        # Create zero tensors with proper shape and device for mu, logvar, ldj
        shape = zk.shape
        device = zk.device
        mu = torch.zeros(shape, device=device)
        logvar = torch.zeros(shape, device=device)
        ldj = torch.zeros(1, device=device)
        z0 = zk  # For AE, initial and final latent are the same
        return recon, mu, logvar, ldj, z0, zk
        
    elif len(outputs) == 4:  # VAE model: (recon, mu, logvar, z)
        recon, mu, logvar, zk = outputs
        # Create zero tensor for ldj
        ldj = torch.zeros(1, device=recon.device)
        z0 = zk  # For VAEs without flows, initial and final latent are the same
        return recon, mu, logvar, ldj, z0, zk
        
    elif len(outputs) == 6:  # Could be Flow model or Dirichlet VAE
        recon, mu, logvar = outputs[0], outputs[1], outputs[2]
        
        # Check if this is likely a Dirichlet VAE output (recon, mu, logvar, G_z, G_z, D_z)
        # We can identify this by checking if the 4th and 5th elements are identical (both G_z)
        # This is a heuristic that should work reliably for the known models
        if torch.equal(outputs[3], outputs[4]):
            # This is likely a Dirichlet VAE output
            G_z, _, D_z = outputs[3], outputs[4], outputs[5]
            ldj = torch.zeros(1, device=recon.device)  # DVAE has no ldj
            z0, zk = G_z, D_z  # Use G_z as z0 and D_z as zk
            return recon, mu, logvar, ldj, z0, zk
        else:
            # This is likely a Flow model output (already in correct format)
            return outputs
        
    else:
        raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}. Expected 2, 4, or 6.")


class EarlyStopping:
    """
    Class to perform early stopping during model training.

    Args:
        patience (int): The number of epochs to wait before stopping the training process if the validation loss doesn't improve.
        min_delta (float): The minimum difference between the new loss and the previous best loss for the new loss to be considered an improvement.

    Attributes:
        counter (int): Counts the number of times the validation loss hasn't improved.
        best_loss (float): The best validation loss observed so far.
        early_stop (bool): Flag that indicates whether early stopping criteria have been met.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience  # Nr of times we allow val. loss to not improve before early stopping
        self.min_delta = min_delta  # min(new loss - best loss) for new loss to be considered improvement
        self.counter = 0  # counts nr of times val_loss dosent improve
        self.best_loss = None
        self.early_stop = False

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss

        elif self.best_loss - train_loss > self.min_delta:
            self.best_loss = train_loss
            self.counter = 0  # Resets if val_loss improves

        elif self.best_loss - train_loss < self.min_delta:
            self.counter += 1

            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping")
                self.early_stop = True


class LRScheduler:
    """
    A learning rate scheduler that adjusts the learning rate of an optimizer based on the training loss.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        patience (int): The number of epochs with no improvement in training loss after which the learning rate will be reduced.
        min_lr (float, optional): The minimum learning rate that can be reached (default: 1e-6).
        factor (float, optional): The factor by which the learning rate will be reduced (default: 0.1).

    Attributes:
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The PyTorch learning rate scheduler that actually performs the adjustments.

    Example usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = LRScheduler(optimizer, patience=3, min_lr=1e-6, factor=0.5)
        for epoch in range(num_epochs):
        train_loss = train(model, train_data_loader)
        lr_scheduler(train_loss)
        # ...
    """

    def __init__(self, optimizer, patience, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        # Maybe add if statements for selectment of lr schedulers
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
        )

    def __call__(self, loss):
        self.lr_scheduler.step(loss)


def load_model(model_path: str, in_shape, config):
    """

    Loads the state dictionary of the trained model into a model variable. This variable is then used for passing
    data through the encoding and decoding functions.

    Args:
        model_path (str): Path to model
        in_shape (tuple): Input shape
        config (Config): Configuration object

    Returns: nn.Module: Returns a model object with the attributes of the model class, with the selected state dictionary loaded into it.
    """
    model_object = getattr(models, config.model_name)

    if config.model_name == "pj_custom":
        model = model_object(*in_shape, z_dim=config.latent_space_size)

    else:
        model = model_object(in_shape, z_dim=config.latent_space_size)

    # Load state_dict to CPU first to avoid device mismatches,
    # especially if saved from a specific GPU or DDP setup.
    state_dict = torch.load(str(model_path), map_location="cpu")

    new_state_dict = {}
    is_ddp_checkpoint = any(key.startswith("module.") for key in state_dict.keys())

    if is_ddp_checkpoint:
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(
            new_state_dict, strict=True
        )  # Be strict if we've cleaned keys
    else:
        model.load_state_dict(state_dict, strict=True)

    return model


def get_ntxent_outputs(model, inputs, config):
    """
    Performs a dual forward pass through the model with augmented input views for NT-Xent contrastive learning.
    
    This function:
    1. Generates two augmented views of the input data using naive gaussian smearing strategy
    2. Passes each view through the model
    3. Unpacks the model outputs from each view
    4. Returns all necessary outputs for NT-Xent loss calculation
    
    Args:
        model (nn.Module): The model to perform forward passes with
        inputs (torch.Tensor): Input data batch
        config (dataClass): Configuration object containing NT-Xent parameters
        
    Returns:
        tuple: A tuple containing:
            - recon_i: Reconstruction from the first view
            - mu_i: Mean latent vector from the first view
            - logvar_i: Log variance from the first view
            - ldj_i: Log-determinant of Jacobian from the first view
            - z0_i: Initial latent vector from the first view
            - zk_i: Final latent vector from the first view
            - zk_j: Final latent vector from the second view
    """
    from ..utils.ntxent_utils import generate_augmented_views
    
    # Generate two augmented views using naive gaussian smearing strategy
    # The sigma (noise level) is controlled by the config
    x_i, x_j = generate_augmented_views(inputs, sigma=config.ntxent_sigma)
    
    # Perform forward pass with the first view
    out_i = call_forward(model, x_i)
    recon_i, mu_i, logvar_i, ldj_i, z0_i, zk_i = unpack_model_outputs(out_i)
    
    # Perform forward pass with the second view (we only need zk_j for NT-Xent)
    out_j = call_forward(model, x_j)
    _, _, _, _, _, zk_j = unpack_model_outputs(out_j)
    
    # Return all outputs needed for loss calculation
    return recon_i, mu_i, logvar_i, ldj_i, z0_i, zk_i, zk_j


def save_loss_components(loss_data, component_names, suffix, save_dir="loss_outputs"):
    """
    This function unpacks loss_data into separate components, converts each into a NumPy array,
    and saves each array as a .npy file with a filename of the form:
    <component_name>_<suffix>.npy

    Args:
      - loss_data (list): a list of tuples, where each tuple contains loss components
      - component_names (list): a list of strings naming each component in the tuple
      - suffix (str): a string keyword to be appended (separated by '_') to each filename
      - save_dir (path): directory to save .npy files (default "loss_outputs")

    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    if not loss_data:
        raise ValueError("loss_data is empty.")

    # Check that the number of components in each tuple matches the number of names provided.
    n_components = len(loss_data[0])
    if n_components != len(component_names):
        raise ValueError(
            "The length of each loss tuple must match the number of component names provided."
        )

    # Unpack the list of tuples into a list of components using zip.
    # Each element in 'components' is a tuple containing that component from every iteration.
    components = list(zip(*loss_data, strict=False))

    # def reshape_tensor_lists(original_list):
    #     transformed_list = []
    #     for inner_list in original_list:
    #         # Concatenate all tensors in the inner list along the 0th dimension
    #         concatenated_tensor = torch.cat(inner_list, dim=0)
    #         transformed_list.append([concatenated_tensor])
    #     return transformed_list
    # components = reshape_tensor_lists(components)

    # Process and save each component.
    for name, comp in zip(component_names, components, strict=False):
        # Convert each element to a NumPy array if it's a PyTorch tensor.
        converted = []
        for val in comp:
            if hasattr(val, "detach"):  # likely a PyTorch tensor
                converted.append(val.detach().cpu().numpy())
            else:
                converted.append(val)
        arr = np.array(converted)

        # Create filename with component name and appended suffix
        filename = os.path.join(save_dir, f"{name}_{suffix}.npy")
        np.save(filename, arr)
