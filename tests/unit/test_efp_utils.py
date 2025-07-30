#!/usr/bin/env python3
"""
Test script for EFP utilities module.

Validates EFP configuration, preprocessing, and computation functions.
"""

import numpy as np
import sys
from pathlib import Path

# Add BEAD src to path
sys.path.insert(0, str(Path(__file__).parent / "bead" / "src"))

from utils.efp_utils import (
    validate_efp_config, 
    create_efpset,
    preprocess_jet_constituents,
    compute_efps_for_jet,
    compute_efps_batch,
    standardize_efps,
    get_efp_cache_path
)

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, **kwargs):
        # Default values
        self.efp_nmax = 5
        self.efp_dmax = 6
        self.efp_extended_mode = False
        self.efp_beta = 1.0
        self.efp_measure = 'hadr'
        self.efp_normed = True
        self.efp_n_jobs = 4
        self.efp_cache_dir = None
        self.efp_standardize_meanvar = True
        self.efp_feature_prefix = 'EFP_'
        self.efp_eps = 1e-12
        
        # Override with provided values
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_config_validation():
    """Test EFP configuration validation."""
    print("=== Testing EFP Configuration Validation ===")
    
    # Test standard configuration
    config = MockConfig()
    validated = validate_efp_config(config)
    
    print(f"Standard config (n≤{validated['nmax']}, d≤{validated['dmax']}): {validated['n_efps']} EFPs")
    assert validated['n_efps'] == 140, f"Expected 140 EFPs, got {validated['n_efps']}"
    assert validated['nmax'] == 5
    assert validated['dmax'] == 6
    assert not validated['extended_mode']
    
    # Test extended configuration
    config_ext = MockConfig(efp_extended_mode=True)
    validated_ext = validate_efp_config(config_ext)
    
    print(f"Extended config (n≤{validated_ext['nmax']}, d≤{validated_ext['dmax']}): {validated_ext['n_efps']} EFPs")
    assert validated_ext['n_efps'] == 531, f"Expected 531 EFPs, got {validated_ext['n_efps']}"
    assert validated_ext['nmax'] == 6
    assert validated_ext['dmax'] == 7
    assert validated_ext['extended_mode']
    
    print("✓ Configuration validation tests passed")
    return validated, validated_ext


def test_efpset_creation():
    """Test EFPSet creation."""
    print("\n=== Testing EFPSet Creation ===")
    
    config = MockConfig()
    validated = validate_efp_config(config)
    efpset = create_efpset(validated)
    
    n_efps = len(efpset.graphs())  # Use graph count
    print(f"Created EFPSet with {n_efps} EFPs")
    assert n_efps == 140, f"Expected 140 EFPs, got {n_efps}"
    
    print("✓ EFPSet creation test passed")
    return efpset


def test_preprocessing():
    """Test constituent preprocessing."""
    print("\n=== Testing Constituent Preprocessing ===")
    
    # Test normal jet
    constituents = np.array([
        [50.0, 0.1, 0.5],   # High pT
        [30.0, -0.2, 1.0],  # Medium pT
        [0.0, 0.0, 0.0],    # Zero-padded (should be filtered)
        [5.0, 0.3, -1.0],   # Low pT
        [0.0, 0.0, 0.0],    # Zero-padded (should be filtered)
    ])
    
    filtered, mask = preprocess_jet_constituents(constituents)
    print(f"Original: {len(constituents)} particles, Filtered: {len(filtered)} particles")
    assert len(filtered) == 3, f"Expected 3 valid particles, got {len(filtered)}"
    assert np.sum(mask) == 3, f"Expected 3 True values in mask, got {np.sum(mask)}"
    
    # Test empty jet
    empty_constituents = np.zeros((5, 3))
    filtered_empty, mask_empty = preprocess_jet_constituents(empty_constituents)
    print(f"Empty jet: {len(filtered_empty)} particles after filtering")
    assert len(filtered_empty) == 1, "Empty jet should have 1 dummy particle"
    assert not np.any(mask_empty), "Empty jet mask should be all False"
    
    print("✓ Preprocessing tests passed")


def test_efp_computation():
    """Test EFP computation for individual jets and batches."""
    print("\n=== Testing EFP Computation ===")
    
    # Create EFPSet
    config = MockConfig()
    validated = validate_efp_config(config)
    efpset = create_efpset(validated)
    
    # Test single jet computation
    jet_constituents = np.array([
        [50.0, 0.1, 0.5],
        [30.0, -0.2, 1.0],
        [20.0, 0.5, -0.5],
        [15.0, -0.1, 2.0],
        [10.0, 0.3, -1.0],
    ])
    
    efp_values = compute_efps_for_jet(jet_constituents, efpset)
    print(f"Single jet EFPs: shape {efp_values.shape}, dtype {efp_values.dtype}")
    expected_n_efps = len(efpset.graphs())  # Use graph count
    assert efp_values.shape == (expected_n_efps,), f"Expected shape ({expected_n_efps},), got {efp_values.shape}"
    assert efp_values.dtype == np.float32, f"Expected float32, got {efp_values.dtype}"
    assert not np.any(np.isnan(efp_values)), "EFP values should not contain NaN"
    
    # Test batch computation
    batch_constituents = np.array([
        # Jet 1: normal jet
        [[50.0, 0.1, 0.5], [30.0, -0.2, 1.0], [20.0, 0.5, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # Jet 2: smaller jet
        [[40.0, 0.2, -0.3], [15.0, -0.1, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # Jet 3: empty jet
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    
    batch_efps = compute_efps_batch(batch_constituents, efpset, n_jobs=2)
    print(f"Batch EFPs: shape {batch_efps.shape}, dtype {batch_efps.dtype}")
    expected_n_efps = len(efpset.graphs())  # Use graph count
    assert batch_efps.shape == (3, expected_n_efps), f"Expected shape (3, {expected_n_efps}), got {batch_efps.shape}"
    assert batch_efps.dtype == np.float32, f"Expected float32, got {batch_efps.dtype}"
    
    # Check that empty jet (index 2) has all zeros
    assert np.allclose(batch_efps[2], 0.0), "Empty jet should have zero EFPs"
    
    # Check that non-empty jets have non-zero EFPs
    assert not np.allclose(batch_efps[0], 0.0), "Non-empty jet should have non-zero EFPs"
    assert not np.allclose(batch_efps[1], 0.0), "Non-empty jet should have non-zero EFPs"
    
    print("✓ EFP computation tests passed")
    return batch_efps


def test_standardization():
    """Test EFP standardization."""
    print("\n=== Testing EFP Standardization ===")
    
    # Create sample EFP data (use dynamic size)
    np.random.seed(42)
    n_efps = 140  # Standard configuration (corrected)
    efp_data = np.random.exponential(2.0, (100, n_efps)).astype(np.float32)
    
    # Add some variation
    efp_data[:, :50] *= 10  # Make first 50 features larger
    efp_data[:, 50:] *= 0.1  # Make last features smaller
    
    # Standardize
    standardized, stats = standardize_efps(efp_data)
    
    print(f"Original data: mean={np.mean(efp_data):.3f}, std={np.std(efp_data):.3f}")
    print(f"Standardized data: mean={np.mean(standardized):.6f}, std={np.std(standardized):.6f}")
    
    # Check standardization
    assert standardized.shape == efp_data.shape, "Shape should be preserved"
    assert standardized.dtype == np.float32, "Dtype should be float32"
    assert abs(np.mean(standardized)) < 1e-6, "Mean should be close to zero"
    assert abs(np.std(standardized) - 1.0) < 1e-6, "Std should be close to 1.0"
    
    # Check stats
    assert 'mean' in stats and 'std' in stats, "Stats should contain mean and std"
    assert stats['mean'].shape == (n_efps,), f"Mean should have shape ({n_efps},)"
    assert stats['std'].shape == (n_efps,), f"Std should have shape ({n_efps},)"
    
    print("✓ Standardization tests passed")


def test_cache_path():
    """Test cache path generation."""
    print("\n=== Testing Cache Path Generation ===")
    
    config = MockConfig()
    validated = validate_efp_config(config)
    
    cache_path = get_efp_cache_path("/tmp/test_cache", validated, prefix="test")
    print(f"Cache path: {cache_path}")
    
    expected_parts = ["test", "n5", "d6", "b1.0", "hadr", "efp.pt"]
    for part in expected_parts:
        assert part in str(cache_path), f"Expected '{part}' in cache path"
    
    # Test extended mode
    validated_ext = validate_efp_config(MockConfig(efp_extended_mode=True))
    cache_path_ext = get_efp_cache_path("/tmp/test_cache", validated_ext)
    print(f"Extended cache path: {cache_path_ext}")
    
    assert "n6" in str(cache_path_ext), "Extended mode should have n6"
    assert "d7" in str(cache_path_ext), "Extended mode should have d7"
    assert "ext" in str(cache_path_ext), "Extended mode should have 'ext' marker"
    
    print("✓ Cache path tests passed")


def main():
    """Run all tests."""
    print("EFP Utils Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        validated_configs = test_config_validation()
        efpset = test_efpset_creation()
        test_preprocessing()
        batch_efps = test_efp_computation()
        test_standardization()
        test_cache_path()
        
        print("\n" + "=" * 50)
        print("✅ All EFP utils tests passed!")
        print(f"Standard config: {validated_configs[0]['n_efps']} EFPs (n≤5, d≤6)")
        print(f"Extended config: {validated_configs[1]['n_efps']} EFPs (n≤6, d≤7)")
        print("Ready for integration into BEAD pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
