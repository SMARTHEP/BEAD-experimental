#!/usr/bin/env python3
"""
Standalone test for EFP utilities integration.

This test validates the EFP utilities without complex imports,
focusing on the core EFP computation functionality.
"""

import os
import sys
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

# Add BEAD source to path
bead_src_path = os.path.join(os.getcwd(), 'bead', 'src')
sys.path.insert(0, bead_src_path)

# Test energyflow availability first
try:
    import energyflow as ef
    print("✓ EnergyFlow library available")
except ImportError:
    print("❌ EnergyFlow library not available")
    sys.exit(1)

# Import EFP utilities
try:
    # Import directly from the utils directory
    utils_path = os.path.join(bead_src_path, 'utils')
    sys.path.insert(0, utils_path)
    
    import efp_utils
    print("✓ EFP utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import EFP utilities: {e}")
    sys.exit(1)


@dataclass
class TestConfig:
    """Minimal config class for EFP testing."""
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
    efp_standardize_meanvar: bool = True
    efp_n_jobs: int = 2


def create_synthetic_jet_data(n_jets=10, max_particles=15):
    """Create synthetic jet data for testing."""
    np.random.seed(42)
    
    jets = []
    for _ in range(n_jets):
        # Random number of particles (1 to max_particles)
        n_particles = np.random.randint(1, max_particles + 1)
        
        # Generate realistic particle data
        pt = np.random.exponential(scale=10.0, size=n_particles)
        eta = np.random.normal(0.0, 2.0, size=n_particles)
        phi = np.random.uniform(0, 2*np.pi, size=n_particles)
        
        # Stack into (n_particles, 3) format
        jet = np.column_stack([pt, eta, phi])
        jets.append(jet)
    
    return jets


def test_efp_config_validation():
    """Test EFP configuration validation."""
    print("\n=== Testing EFP Configuration Validation ===")
    
    # Test standard config
    config = TestConfig(efp_nmax=5, efp_dmax=6)
    validated = efp_utils.validate_efp_config(config)
    
    print(f"Standard config: {validated['n_efps']} EFPs (n≤{validated['nmax']}, d≤{validated['dmax']})")
    assert validated['n_efps'] == 140, f"Expected 140 EFPs, got {validated['n_efps']}"
    
    # Test extended config
    config_ext = TestConfig(efp_nmax=6, efp_dmax=7, efp_extended_mode=True)
    validated_ext = efp_utils.validate_efp_config(config_ext)
    
    print(f"Extended config: {validated_ext['n_efps']} EFPs (n≤{validated_ext['nmax']}, d≤{validated_ext['dmax']})")
    assert validated_ext['n_efps'] == 531, f"Expected 531 EFPs, got {validated_ext['n_efps']}"
    
    print("✓ EFP configuration validation passed")
    return validated, validated_ext


def test_efpset_creation():
    """Test EFPSet creation."""
    print("\n=== Testing EFPSet Creation ===")
    
    config = TestConfig()
    validated = efp_utils.validate_efp_config(config)
    efpset = efp_utils.create_efpset(validated)
    
    print(f"EFPSet created with {len(efpset.efps)} EFP objects")
    print(f"Expected {validated['n_efps']} features from compute()")
    
    # Test with sample data
    sample_jet = np.array([[10.0, 0.0, 0.0], [5.0, 1.0, 1.5], [3.0, -0.5, 2.0]])
    efp_values = efpset.compute(sample_jet)
    
    print(f"Sample computation: {len(efp_values)} EFP values")
    assert len(efp_values) == validated['n_efps'], f"Expected {validated['n_efps']} values, got {len(efp_values)}"
    
    print("✓ EFPSet creation and computation passed")
    return efpset


def test_jet_preprocessing():
    """Test jet constituent preprocessing."""
    print("\n=== Testing Jet Preprocessing ===")
    
    # Test with normal jet
    jet_normal = np.array([[10.0, 0.0, 0.0], [5.0, 1.0, 1.5], [3.0, -0.5, 2.0]])
    filtered, mask = efp_utils.preprocess_jet_constituents(jet_normal)
    
    print(f"Normal jet: {jet_normal.shape} -> {filtered.shape}, mask: {mask.sum()}/{len(mask)}")
    assert np.array_equal(filtered, jet_normal), "Normal jet should not be filtered"
    assert np.all(mask), "All particles should be valid"
    
    # Test with zero-padded jet
    jet_padded = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, -0.5, 2.0], [0.0, 0.0, 0.0]])
    filtered, mask = efp_utils.preprocess_jet_constituents(jet_padded)
    
    print(f"Padded jet: {jet_padded.shape} -> {filtered.shape}, mask: {mask.sum()}/{len(mask)}")
    assert filtered.shape[0] == 2, "Should filter out 2 zero-padded particles"
    assert mask.sum() == 2, "Should identify 2 valid particles"
    
    # Test with empty jet
    jet_empty = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    filtered, mask = efp_utils.preprocess_jet_constituents(jet_empty)
    
    print(f"Empty jet: {jet_empty.shape} -> {filtered.shape}, mask: {mask.sum()}/{len(mask)}")
    # Empty jets get a dummy particle to avoid EFP computation errors
    assert filtered.shape[0] == 1, "Should return dummy particle for empty jet"
    assert mask.sum() == 0, "Should identify no valid particles"
    
    print("✓ Jet preprocessing passed")


def test_single_jet_efp():
    """Test EFP computation for single jets."""
    print("\n=== Testing Single Jet EFP Computation ===")
    
    config = TestConfig()
    validated = efp_utils.validate_efp_config(config)
    efpset = efp_utils.create_efpset(validated)
    
    # Test normal jet
    jet_normal = np.array([[10.0, 0.0, 0.0], [5.0, 1.0, 1.5], [3.0, -0.5, 2.0]])
    efps = efp_utils.compute_efps_for_jet(jet_normal, efpset)
    
    print(f"Normal jet EFPs: shape {efps.shape}, range [{np.min(efps):.6f}, {np.max(efps):.6f}]")
    assert efps.shape == (validated['n_efps'],), f"Expected shape ({validated['n_efps']},), got {efps.shape}"
    assert not np.any(np.isnan(efps)), "EFPs should not contain NaN"
    assert not np.any(np.isinf(efps)), "EFPs should not contain inf"
    
    # Test empty jet
    jet_empty = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    efps_empty = efp_utils.compute_efps_for_jet(jet_empty, efpset)
    
    print(f"Empty jet EFPs: shape {efps_empty.shape}, all zeros: {np.allclose(efps_empty, 0)}")
    assert efps_empty.shape == (validated['n_efps'],), "Empty jet should have same shape"
    assert np.allclose(efps_empty, 0), "Empty jet EFPs should be zero"
    
    print("✓ Single jet EFP computation passed")


def test_batch_efp_computation():
    """Test batch EFP computation."""
    print("\n=== Testing Batch EFP Computation ===")
    
    config = TestConfig()
    validated = efp_utils.validate_efp_config(config)
    efpset = efp_utils.create_efpset(validated)
    
    # Create synthetic jet data
    jets = create_synthetic_jet_data(n_jets=20, max_particles=10)
    
    # Pad jets to same length for batch processing
    max_particles = max(jet.shape[0] for jet in jets)
    jets_padded = np.zeros((len(jets), max_particles, 3))
    
    for i, jet in enumerate(jets):
        jets_padded[i, :jet.shape[0], :] = jet
    
    print(f"Batch input: {jets_padded.shape}")
    
    # Compute EFPs
    import time
    start_time = time.time()
    efps_batch = efp_utils.compute_efps_batch(jets_padded, efpset, n_jobs=2)
    computation_time = time.time() - start_time
    
    print(f"Batch EFPs: shape {efps_batch.shape}, time {computation_time:.3f}s")
    print(f"EFP range: [{np.min(efps_batch):.6f}, {np.max(efps_batch):.6f}]")
    
    expected_shape = (len(jets), validated['n_efps'])
    assert efps_batch.shape == expected_shape, f"Expected shape {expected_shape}, got {efps_batch.shape}"
    assert not np.any(np.isnan(efps_batch)), "Batch EFPs should not contain NaN"
    assert not np.any(np.isinf(efps_batch)), "Batch EFPs should not contain inf"
    
    print("✓ Batch EFP computation passed")
    return efps_batch


def test_efp_standardization():
    """Test EFP standardization."""
    print("\n=== Testing EFP Standardization ===")
    
    # Create sample EFP data
    np.random.seed(42)
    n_samples, n_efps = 100, 140
    efp_data = np.random.normal(loc=5.0, scale=2.0, size=(n_samples, n_efps))
    
    print(f"Original data: mean {np.mean(efp_data):.3f}, std {np.std(efp_data):.3f}")
    
    # Standardize
    standardized, stats = efp_utils.standardize_efps(efp_data)
    
    print(f"Standardized data: mean {np.mean(standardized):.6f}, std {np.std(standardized):.6f}")
    print(f"Stats: mean shape {stats['mean'].shape}, std shape {stats['std'].shape}")
    
    # Validate standardization
    assert np.allclose(np.mean(standardized), 0, atol=1e-10), "Mean should be close to zero"
    assert np.allclose(np.std(standardized), 1, atol=1e-10), "Std should be close to one"
    assert stats['mean'].shape == (n_efps,), "Mean stats should match feature count"
    assert stats['std'].shape == (n_efps,), "Std stats should match feature count"
    
    print("✓ EFP standardization passed")


def main():
    """Run all EFP standalone tests."""
    print("EFP Standalone Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration validation
        validated_configs = test_efp_config_validation()
        
        # Test EFPSet creation
        efpset = test_efpset_creation()
        
        # Test preprocessing
        test_jet_preprocessing()
        
        # Test single jet computation
        test_single_jet_efp()
        
        # Test batch computation
        efps_batch = test_batch_efp_computation()
        
        # Test standardization
        test_efp_standardization()
        
        print("\n" + "=" * 50)
        print("✅ All EFP standalone tests passed!")
        print(f"Standard config: {validated_configs[0]['n_efps']} EFPs (n≤5, d≤6)")
        print(f"Extended config: {validated_configs[1]['n_efps']} EFPs (n≤6, d≤7)")
        print("EFP utilities are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
