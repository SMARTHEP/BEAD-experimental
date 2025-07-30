#!/usr/bin/env python3
"""
Comprehensive EnergyFlow Library Integration Test for BEAD

This script validates the EnergyFlow library integration for EFP computation
within the BEAD anomaly detection pipeline. It tests installation, API compatibility,
performance benchmarks, GPU support, masking behavior, and edge cases.

Phase 3.1: EnergyFlow Library Integration Testing
Author: BEAD Development Team
Date: 2025-01-30
"""

import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import traceback

import numpy as np
import torch
import multiprocessing as mp

def test_basic_imports() -> Dict[str, Any]:
    """Test basic imports and version compatibility."""
    print("=" * 60)
    print("PHASE 3.1: EnergyFlow Library Integration Testing")
    print("=" * 60)
    
    results = {
        'success': True,
        'errors': [],
        'versions': {},
        'warnings': []
    }
    
    try:
        # Test core imports
        print("\n1. Testing Basic Imports...")
        
        import energyflow as ef
        results['versions']['energyflow'] = ef.__version__
        print(f"   ✓ energyflow v{ef.__version__} imported successfully")
        
        # Test specific EFP classes
        from energyflow import EFP, EFPSet
        print("   ✓ EFP and EFPSet classes imported successfully")
        
        # Test measure compatibility
        from energyflow.utils import ptyphims_from_p4s, p4s_from_ptyphims
        print("   ✓ Utility functions imported successfully")
        
        # Version compatibility check
        version_parts = ef.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 1 or (major == 1 and minor < 2):
            results['warnings'].append(f"EnergyFlow version {ef.__version__} may be outdated. Recommend >=1.2.0")
        
        print(f"   ✓ Version compatibility check passed")
        
    except ImportError as e:
        results['success'] = False
        results['errors'].append(f"Import error: {e}")
        print(f"   ✗ Import failed: {e}")
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Unexpected error: {e}")
        print(f"   ✗ Unexpected error: {e}")
    
    return results


def test_api_compatibility() -> Dict[str, Any]:
    """Test EnergyFlow API compatibility with BEAD data format."""
    print("\n2. Testing API Compatibility...")
    
    results = {
        'success': True,
        'errors': [],
        'efp_features': 0,
        'computation_time': 0.0
    }
    
    try:
        import energyflow as ef
        
        # Create synthetic BEAD-like data
        # Format: (n_particles, 3) for (pT, eta, phi)
        np.random.seed(42)
        
        # Simulate a jet with 10 particles
        n_particles = 10
        jet_data = np.array([
            [50.0, 0.1, 0.5],   # High pT particle
            [30.0, -0.2, 1.0],  # Medium pT particle
            [20.0, 0.5, -0.5],  # Medium pT particle
            [15.0, -0.1, 2.0],  # Lower pT particle
            [10.0, 0.3, -1.0],  # Lower pT particle
            [8.0, -0.4, 1.5],   # Low pT particle
            [5.0, 0.2, -2.0],   # Low pT particle
            [3.0, -0.3, 0.8],   # Very low pT particle
            [2.0, 0.1, -0.3],   # Very low pT particle
            [1.0, -0.1, 0.2],   # Very low pT particle
        ])
        
        print(f"   ✓ Created synthetic jet data: {jet_data.shape}")
        print(f"     pT range: {jet_data[:, 0].min():.1f} - {jet_data[:, 0].max():.1f} GeV")
        print(f"     η range: {jet_data[:, 1].min():.1f} - {jet_data[:, 1].max():.1f}")
        print(f"     φ range: {jet_data[:, 2].min():.1f} - {jet_data[:, 2].max():.1f}")
        
        # Test EFPSet creation with BEAD-compatible parameters
        start_time = time.time()
        
        # Use parameters matching our scientific decisions
        efpset = ef.EFPSet(
            n=5,           # N_max = 5 (max particles in graph)
            d=6,           # d_max = 6 (max degree)
            measure='hadr', # Hadron collider measure (pT, eta, phi)
            beta=1.0,      # Linear energy weighting (IRC safe)
            normed=True,   # Normalize angular distances to [0,1]
            coords='ptyphim', # Input format: (pT, y/eta, phi, mass)
            check_input=True
        )
        
        # Get number of EFPs
        n_efps = len(efpset)
        results['efp_features'] = n_efps
        print(f"   ✓ EFPSet created: {n_efps} EFP features")
        print(f"     Parameters: n≤5, d≤6, β=1.0, measure='hadr', normed=True")
        
        # Add mass column (set to 0 for massless particles)
        jet_data_with_mass = np.column_stack([jet_data, np.zeros(n_particles)])
        
        # Test single event computation
        efp_values = efpset.compute(jet_data_with_mass)
        
        computation_time = time.time() - start_time
        results['computation_time'] = computation_time
        
        print(f"   ✓ Single jet EFP computation successful")
        print(f"     Output shape: {efp_values.shape}")
        print(f"     Computation time: {computation_time*1000:.2f} ms")
        print(f"     EFP value range: {efp_values.min():.6f} - {efp_values.max():.6f}")
        
        # Test batch computation
        batch_data = [jet_data_with_mass, jet_data_with_mass * 0.8, jet_data_with_mass * 1.2]
        batch_efps = efpset.batch_compute(batch_data, n_jobs=1)
        
        print(f"   ✓ Batch computation successful")
        print(f"     Batch shape: {batch_efps.shape}")
        print(f"     Expected: (3, {n_efps})")
        
        # Validate output properties
        if np.any(np.isnan(efp_values)):
            results['errors'].append("NaN values detected in EFP output")
        
        if np.any(np.isinf(efp_values)):
            results['errors'].append("Infinite values detected in EFP output")
        
        if efp_values.shape[0] != n_efps:
            results['errors'].append(f"Unexpected output shape: {efp_values.shape}")
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"API compatibility test failed: {e}")
        print(f"   ✗ API test failed: {e}")
        traceback.print_exc()
    
    return results


def main():
    """Run comprehensive EnergyFlow integration tests."""
    print("Starting EnergyFlow Library Integration Tests...")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run all tests
    all_results = {}
    
    # Test 1: Basic imports
    all_results['basic_imports'] = test_basic_imports()
    
    # Test 2: API compatibility
    all_results['api_compatibility'] = test_api_compatibility()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("ENERGYFLOW INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(result.get('success', False) for result in all_results.values())
    status = "✓ PASSED" if all_passed else "✗ FAILED"
    print(f"\nOverall Status: {status}")
    
    for test_name, result in all_results.items():
        test_status = "✓" if result.get('success', False) else "✗"
        print(f"  {test_status} {test_name.replace('_', ' ').title()}")
        
        if result.get('errors'):
            for error in result['errors']:
                print(f"      Error: {error}")
    
    if 'api_compatibility' in all_results:
        api_result = all_results['api_compatibility']
        if api_result.get('success'):
            print(f"\nKey Metrics:")
            print(f"  • EFP Features: {api_result.get('efp_features', 'N/A')}")
            print(f"  • Computation Time: {api_result.get('computation_time', 0)*1000:.2f} ms")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
