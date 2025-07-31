"""
EFP integration utilities for BEAD model input preparation.

This module provides helper functions to integrate Energy-Flow Polynomial (EFP) features
with existing model inputs through direct concatenation, maintaining backward compatibility
with models that don't use EFP features.
"""

import torch
from typing import Optional


def prepare_model_input(base_features: torch.Tensor, 
                       efp_features: Optional[torch.Tensor] = None, 
                       config=None) -> torch.Tensor:
    """
    Prepare model input by optionally concatenating EFP features with base features.
    
    This function provides a single integration point for EFP features, allowing models
    to receive enhanced input without requiring architecture changes. The LazyLinear
    layers in BEAD models automatically adapt to the increased input dimension.
    
    Args:
        base_features (torch.Tensor): Base input features, typically flattened constituents
            Shape: (batch_size, base_feature_dim)
        efp_features (torch.Tensor, optional): EFP features per jet
            Shape: (batch_size, n_jets, n_efp_features)
        config: Configuration object with enable_efp flag
        
    Returns:
        torch.Tensor: Combined input features for model
            Shape: (batch_size, base_feature_dim + n_jets * n_efp_features) if EFPs enabled
            Shape: (batch_size, base_feature_dim) if EFPs disabled or not provided
            
    Example:
        >>> base_input = torch.randn(32, 1500)  # 32 samples, 1500 constituent features
        >>> efp_input = torch.randn(32, 10, 140)  # 32 samples, 10 jets, 140 EFP features
        >>> combined = prepare_model_input(base_input, efp_input, config)
        >>> print(combined.shape)  # torch.Size([32, 2900])  # 1500 + 10*140
    """
    # Check if EFP features should be used
    if not _should_use_efp(config) or efp_features is None:
        return base_features
    
    # Validate input shapes
    _validate_input_shapes(base_features, efp_features)
    
    # Flatten EFP features: (batch_size, n_jets, n_efp) -> (batch_size, n_jets * n_efp)
    batch_size = efp_features.size(0)
    efp_flat = efp_features.view(batch_size, -1)
    
    # Concatenate with base features along feature dimension
    combined_features = torch.cat([base_features, efp_flat], dim=1)
    
    return combined_features


def _should_use_efp(config) -> bool:
    """
    Check if EFP features should be used based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if EFP features should be used
    """
    if config is None:
        return False
    
    return getattr(config, 'enable_efp', False)


def _validate_input_shapes(base_features: torch.Tensor, efp_features: torch.Tensor) -> None:
    """
    Validate that input tensors have compatible shapes for concatenation.
    
    Args:
        base_features: Base input tensor
        efp_features: EFP input tensor
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if base_features.dim() != 2:
        raise ValueError(f"base_features must be 2D (batch_size, features), got shape {base_features.shape}")
    
    if efp_features.dim() != 3:
        raise ValueError(f"efp_features must be 3D (batch_size, n_jets, n_efp), got shape {efp_features.shape}")
    
    if base_features.size(0) != efp_features.size(0):
        raise ValueError(f"Batch size mismatch: base_features {base_features.size(0)} vs efp_features {efp_features.size(0)}")


def get_combined_input_dim(base_dim: int, n_jets: int, n_efp_features: int, config=None) -> int:
    """
    Calculate the combined input dimension when EFP features are concatenated.
    
    Args:
        base_dim: Dimension of base features
        n_jets: Number of jets per event
        n_efp_features: Number of EFP features per jet
        config: Configuration object
        
    Returns:
        int: Combined input dimension
    """
    if not _should_use_efp(config):
        return base_dim
    
    return base_dim + (n_jets * n_efp_features)
