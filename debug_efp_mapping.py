#!/usr/bin/env python3
"""
Debug EFP mapping between graphs and EFP features.
"""

import numpy as np
import energyflow as ef

def debug_efp_mapping():
    """Debug the mapping between graphs and EFP features."""
    print("EFP Mapping Debug")
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
    
    # Check if there's a mapping or indexing issue
    print(f"\nEFPSet attributes:")
    for attr in ['efps', 'graphs', 'specs', 'cols']:
        if hasattr(efpset, attr):
            try:
                value = getattr(efpset, attr)
                if callable(value):
                    value = value()
                print(f"  {attr}: {type(value)} of length {len(value) if hasattr(value, '__len__') else 'N/A'}")
            except Exception as e:
                print(f"  {attr}: Error accessing - {e}")
    
    # Test computation and see if we can understand the mapping
    jet_constituents = np.array([
        [50.0, 0.1, 0.5],
        [30.0, -0.2, 1.0],
        [20.0, 0.5, -0.5],
    ])
    
    result = efpset.compute(jet_constituents)
    print(f"\nComputation result:")
    print(f"  Shape: {result.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(result)}")
    print(f"  Min value: {np.min(result):.6f}")
    print(f"  Max value: {np.max(result):.6f}")
    
    # Check if there's a way to get the "correct" number of features
    # Maybe we need to use a different method or access pattern
    
    # Try to understand the specs
    if hasattr(efpset, 'specs'):
        specs = efpset.specs
        print(f"\nSpecs info:")
        print(f"  Type: {type(specs)}")
        print(f"  Length: {len(specs) if hasattr(specs, '__len__') else 'N/A'}")
        print(f"  Shape: {specs.shape if hasattr(specs, 'shape') else 'N/A'}")
        if hasattr(specs, '__len__') and len(specs) > 0:
            print(f"  First few specs: {specs[:5] if len(specs) > 5 else specs}")
    
    # Check cols attribute which might be related to the output columns
    if hasattr(efpset, 'cols'):
        cols = efpset.cols
        print(f"\nCols info:")
        print(f"  Type: {type(cols)}")
        print(f"  Length: {len(cols) if hasattr(cols, '__len__') else 'N/A'}")
        print(f"  Cols shape: {cols.shape if hasattr(cols, 'shape') else 'N/A'}")
        print(f"  Cols content: {cols}")
    
    # Check if there's a way to map from 140 to 111
    print(f"\nLooking for mapping from graphs to EFPs:")
    print(f"  140 graphs -> 111 EFPs suggests some graphs are equivalent or filtered")
    
    # Try to access the actual efps list
    efps_list = efpset.efps
    print(f"\nEFPs list info:")
    print(f"  Type: {type(efps_list)}")
    print(f"  Length: {len(efps_list)}")
    if len(efps_list) > 0:
        print(f"  First EFP type: {type(efps_list[0])}")
        print(f"  First EFP: {efps_list[0]}")

if __name__ == "__main__":
    debug_efp_mapping()
