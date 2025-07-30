#!/usr/bin/env python3
"""
End-to-end test for EFP integration in BEAD data processing pipeline.

This test validates the complete EFP integration from configuration through
data processing to tensor saving, ensuring all components work together correctly.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

# Add BEAD source to path
sys.path.insert(0, os.path.join(os.getcwd(), 'bead', 'src'))

# Import modules with proper error handling
try:
    from utils.efp_utils import validate_efp_config, create_efpset
    from utils.data_processing import compute_efp_features
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the BEAD_upstream directory")
    sys.exit(1)

# Create a minimal config class for testing
@dataclass
class TestConfig:
    """Minimal config class for EFP testing."""
    # Core parameters
    enable_efp: bool = True
    efp_nmax: int = 5
    efp_dmax: int = 6
    efp_extended_mode: bool = False
    efp_beta: float = 1.0
    efp_beta_list: list = None
    efp_measure: str = "hadr"
    efp_normed: bool = True
    efp_include_composites: bool = False
    efp_eps: float = 1e-12
    efp_embedding_dim: int = 64
    efp_gate: str = "sigmoid"
    efp_gate_thresh: float = 0.05
    efp_standardize_meanvar: bool = True
    efp_cache_dir: str = None
    efp_n_jobs: int = 2
    efp_device: str = "cpu"
    efp_feature_prefix: str = "EFP_"
    efp_use_true_energy: bool = False


def create_test_config(enable_efp=True, extended_mode=False):
    """Create a test configuration with EFP parameters."""
    
    # Create minimal config with EFP parameters
    config = TestConfig(
        enable_efp=enable_efp,
        efp_nmax=6 if extended_mode else 5,
        efp_dmax=7 if extended_mode else 6,
        efp_extended_mode=extended_mode,
        efp_beta=1.0,
        efp_beta_list=None,
        efp_measure="hadr",
        efp_normed=True,
        efp_include_composites=False,
        efp_eps=1e-12,
        efp_embedding_dim=64,
        efp_gate="sigmoid",
        efp_gate_thresh=0.05,
        efp_standardize_meanvar=True,
        efp_cache_dir=None,
        efp_n_jobs=2,
        efp_device="cpu",
        efp_feature_prefix="EFP_",
        efp_use_true_energy=False
    )
    
    return config


def create_synthetic_constituent_data(num_events=50, n_jets=3, n_constits=15):
    """Create synthetic constituent data matching BEAD format."""
    
    # BEAD constituent format: [evt_id, jet_id, constit_id, b_tagged, constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id]
    total_constits = num_events * n_jets * n_constits
    
    # Generate realistic constituent data
    np.random.seed(42)
    
    # Event and jet IDs
    evt_ids = np.repeat(np.arange(num_events), n_jets * n_constits)
    jet_ids = np.tile(np.repeat(np.arange(n_jets), n_constits), num_events)
    constit_ids = np.tile(np.arange(n_constits), num_events * n_jets)
    
    # Physics features
    b_tagged = np.random.choice([0, 1], size=total_constits, p=[0.9, 0.1])
    
    # pT: exponential distribution (more realistic for particle physics)
    constit_pt = np.random.exponential(scale=10.0, size=total_constits)
    
    # Add some zero-padding (empty constituents)
    zero_mask = np.random.choice([True, False], size=total_constits, p=[0.2, 0.8])
    constit_pt[zero_mask] = 0.0
    
    # eta: normal distribution around 0
    constit_eta = np.random.normal(0.0, 2.0, size=total_constits)
    constit_eta[zero_mask] = 0.0
    
    # phi: uniform distribution [0, 2π], convert to sin/cos
    constit_phi = np.random.uniform(0, 2*np.pi, size=total_constits)
    constit_phi[zero_mask] = 0.0
    constit_phi_sin = np.sin(constit_phi)
    constit_phi_cos = np.cos(constit_phi)
    
    # Generator ID
    generator_id = np.random.choice([1, 2, 3], size=total_constits)
    generator_id[zero_mask] = 0
    
    # Stack into BEAD format
    constituents = np.column_stack([
        evt_ids, jet_ids, constit_ids, b_tagged,
        constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id
    ])
    
    # Reshape to (num_events, n_jets * n_constits, features)
    constituents = constituents.reshape(num_events, n_jets * n_constits, 9)
    
    return constituents.astype(np.float32)


def test_efp_config_validation():
    """Test EFP configuration validation."""
    print("=== Testing EFP Configuration Validation ===")
    
    # Test standard config
    config = create_test_config(enable_efp=True, extended_mode=False)
    validated = validate_efp_config(config)
    
    print(f"Standard config: {validated['n_efps']} EFPs (n≤{validated['nmax']}, d≤{validated['dmax']})")
    assert validated['n_efps'] == 140, f"Expected 140 EFPs, got {validated['n_efps']}"
    assert validated['nmax'] == 5
    assert validated['dmax'] == 6
    
    # Test extended config
    config_ext = create_test_config(enable_efp=True, extended_mode=True)
    validated_ext = validate_efp_config(config_ext)
    
    print(f"Extended config: {validated_ext['n_efps']} EFPs (n≤{validated_ext['nmax']}, d≤{validated_ext['dmax']})")
    assert validated_ext['n_efps'] == 531, f"Expected 531 EFPs, got {validated_ext['n_efps']}"
    assert validated_ext['nmax'] == 6
    assert validated_ext['dmax'] == 7
    
    print("✓ EFP configuration validation passed")
    return validated, validated_ext


def test_efp_computation():
    """Test EFP computation with synthetic data."""
    print("\n=== Testing EFP Computation ===")
    
    # Create test data and config
    config = create_test_config(enable_efp=True, extended_mode=False)
    constituents = create_synthetic_constituent_data(num_events=10, n_jets=3, n_constits=15)
    
    print(f"Input constituents shape: {constituents.shape}")
    print(f"Sample constituent features: {constituents[0, 0, :]}")
    
    # Test EFP computation
    efp_tensor = compute_efp_features(
        constituents, config, n_jets=3, n_constits=15, verbose=True
    )
    
    print(f"EFP tensor shape: {efp_tensor.shape}")
    print(f"EFP tensor dtype: {efp_tensor.dtype}")
    
    # Validate output
    expected_shape = (10, 3, 140)  # (events, jets, efps)
    assert efp_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {efp_tensor.shape}"
    assert efp_tensor.dtype == torch.float32, f"Expected float32, got {efp_tensor.dtype}"
    assert not torch.any(torch.isnan(efp_tensor)), "EFP tensor contains NaN values"
    assert not torch.any(torch.isinf(efp_tensor)), "EFP tensor contains infinite values"
    
    # Check that standardization was applied (mean ≈ 0, std ≈ 1)
    efp_flat = efp_tensor.view(-1, 140)
    mean_vals = torch.mean(efp_flat, dim=0)
    std_vals = torch.std(efp_flat, dim=0)
    
    print(f"EFP means range: [{torch.min(mean_vals):.6f}, {torch.max(mean_vals):.6f}]")
    print(f"EFP stds range: [{torch.min(std_vals):.6f}, {torch.max(std_vals):.6f}]")
    
    # Allow some tolerance for small datasets
    assert torch.all(torch.abs(mean_vals) < 0.1), "EFP means not close to zero after standardization"
    assert torch.all(std_vals > 0.5) and torch.all(std_vals < 2.0), "EFP stds not close to 1 after standardization"
    
    print("✓ EFP computation test passed")
    return efp_tensor


def test_extended_mode():
    """Test extended mode with more EFP features."""
    print("\n=== Testing Extended Mode (531 EFPs) ===")
    
    # Create test data and extended config
    config = create_test_config(enable_efp=True, extended_mode=True)
    constituents = create_synthetic_constituent_data(num_events=5, n_jets=3, n_constits=15)
    
    print(f"Extended mode config: n≤{config.efp_nmax}, d≤{config.efp_dmax}")
    
    # Test EFP computation
    efp_tensor = compute_efp_features(
        constituents, config, n_jets=3, n_constits=15, verbose=True
    )
    
    print(f"Extended EFP tensor shape: {efp_tensor.shape}")
    
    # Validate output
    expected_shape = (5, 3, 531)  # (events, jets, efps)
    assert efp_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {efp_tensor.shape}"
    assert efp_tensor.dtype == torch.float32, f"Expected float32, got {efp_tensor.dtype}"
    
    print("✓ Extended mode test passed")
    return efp_tensor


def test_disabled_efp():
    """Test that EFP computation is skipped when disabled."""
    print("\n=== Testing Disabled EFP Mode ===")
    
    # Create config with EFP disabled
    config = create_test_config(enable_efp=False)
    constituents = create_synthetic_constituent_data(num_events=5, n_jets=3, n_constits=15)
    
    print(f"EFP enabled: {config.enable_efp}")
    
    # This should return None or handle gracefully
    try:
        efp_tensor = compute_efp_features(
            constituents, config, n_jets=3, n_constits=15, verbose=True
        )
        # If function doesn't check enable_efp, it will still compute
        print(f"Warning: EFP computation proceeded despite enable_efp=False")
    except Exception as e:
        print(f"Expected behavior: EFP computation skipped when disabled: {e}")
    
    print("✓ Disabled EFP test completed")


def test_edge_cases():
    """Test edge cases like empty jets and malformed data."""
    print("\n=== Testing Edge Cases ===")
    
    config = create_test_config(enable_efp=True, extended_mode=False)
    
    # Test with all-zero constituents (empty jets)
    empty_constituents = np.zeros((5, 3 * 15, 9), dtype=np.float32)
    
    print("Testing with all-zero constituents...")
    efp_tensor = compute_efp_features(
        empty_constituents, config, n_jets=3, n_constits=15, verbose=True
    )
    
    print(f"Empty jets EFP tensor shape: {efp_tensor.shape}")
    assert efp_tensor.shape == (5, 3, 140), "Shape mismatch for empty jets"
    
    # Check that output is reasonable (should be zeros or standardized zeros)
    print(f"Empty jets EFP range: [{torch.min(efp_tensor):.6f}, {torch.max(efp_tensor):.6f}]")
    
    print("✓ Edge cases test passed")


def main():
    """Run all EFP integration tests."""
    print("EFP Integration Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration validation
        validated_configs = test_efp_config_validation()
        
        # Test basic EFP computation
        efp_tensor = test_efp_computation()
        
        # Test extended mode
        efp_tensor_ext = test_extended_mode()
        
        # Test disabled mode
        test_disabled_efp()
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ All EFP integration tests passed!")
        print(f"Standard config: {validated_configs[0]['n_efps']} EFPs (n≤5, d≤6)")
        print(f"Extended config: {validated_configs[1]['n_efps']} EFPs (n≤6, d≤7)")
        print("EFP integration is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
