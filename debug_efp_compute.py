#!/usr/bin/env python3
"""
Debug EFP computation to understand the shape discrepancy.
"""

import numpy as np
import energyflow as ef

def debug_efp_computation():
    """Debug EFP computation shapes."""
    print("EFP Computation Debug")
    print("=" * 40)
    
    # Create EFPSet
    efpset = ef.EFPSet(
        'n<=5',
        'd<=6',
        measure='hadr',
        beta=1.0,
        normed=True,
        coords='ptyphim',
        check_input=False
    )
    
    print(f"EFPSet info:")
    print(f"  len(efpset.efps): {len(efpset.efps)}")
    print(f"  len(efpset.graphs()): {len(efpset.graphs())}")
    
    # Create test jet
    jet_constituents = np.array([
        [50.0, 0.1, 0.5],
        [30.0, -0.2, 1.0],
        [20.0, 0.5, -0.5],
        [15.0, -0.1, 2.0],
        [10.0, 0.3, -1.0],
    ])
    
    print(f"\nJet constituents shape: {jet_constituents.shape}")
    
    # Test different computation methods
    try:
        result1 = efpset.compute(jet_constituents)
        print(f"efpset.compute() result shape: {result1.shape}")
        print(f"efpset.compute() result type: {type(result1)}")
    except Exception as e:
        print(f"efpset.compute() failed: {e}")
    
    try:
        result2 = efpset.batch_compute([jet_constituents])
        print(f"efpset.batch_compute([jet]) result shape: {result2.shape}")
        print(f"efpset.batch_compute([jet]) result type: {type(result2)}")
        if len(result2.shape) == 2:
            print(f"First jet result shape: {result2[0].shape}")
    except Exception as e:
        print(f"efpset.batch_compute() failed: {e}")
    
    # Check if there are any special attributes or methods
    print(f"\nEFPSet methods and attributes:")
    methods = [attr for attr in dir(efpset) if not attr.startswith('_')]
    for method in methods[:10]:  # Show first 10
        print(f"  {method}")
    
    # Test with different settings
    print(f"\nTesting different EFPSet configurations:")
    
    # Test without normed
    efpset_no_norm = ef.EFPSet(
        'n<=5', 'd<=6',
        measure='hadr', beta=1.0, normed=False,
        coords='ptyphim', check_input=False
    )
    result_no_norm = efpset_no_norm.compute(jet_constituents)
    print(f"Without normed: {result_no_norm.shape}")
    
    # Test smaller configuration
    efpset_small = ef.EFPSet(
        'n<=3', 'd<=4',
        measure='hadr', beta=1.0, normed=True,
        coords='ptyphim', check_input=False
    )
    result_small = efpset_small.compute(jet_constituents)
    print(f"Small config (n<=3, d<=4): {result_small.shape}, expected {len(efpset_small.efps)}")

if __name__ == "__main__":
    debug_efp_computation()
