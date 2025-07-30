#!/usr/bin/env python3
"""
Final test to understand EFP mapping and determine correct feature count.
"""

import numpy as np
import energyflow as ef

def test_efp_final_mapping():
    """Test to determine the correct EFP feature count."""
    print("Final EFP Mapping Test")
    print("=" * 40)
    
    # Test different configurations to see the pattern
    configs = [
        ('n<=3', 'd<=4'),
        ('n<=4', 'd<=5'),
        ('n<=5', 'd<=6'),
        ('n<=6', 'd<=7'),
    ]
    
    print("Configuration analysis:")
    print("Config\t\tGraphs\tEFPs\tCompute")
    print("-" * 40)
    
    for n_spec, d_spec in configs:
        efpset = ef.EFPSet(
            n_spec, d_spec,
            measure='hadr',
            beta=1.0,
            normed=True,
            coords='ptyphim',
            check_input=False
        )
        
        n_graphs = len(efpset.graphs())
        n_efps = len(efpset.efps)
        
        # Test computation
        jet_constituents = np.array([
            [50.0, 0.1, 0.5],
            [30.0, -0.2, 1.0],
            [20.0, 0.5, -0.5],
        ])
        
        result = efpset.compute(jet_constituents)
        n_compute = len(result)
        
        print(f"{n_spec}, {d_spec}\t\t{n_graphs}\t{n_efps}\t{n_compute}")
    
    print("\nConclusion:")
    print("The compute() method returns one value per graph.")
    print("The efps list contains EFP objects (may be different organization).")
    print("For feature extraction, we should use the compute() output length.")
    
    # Test our target configuration
    efpset = ef.EFPSet(
        'n<=5', 'd<=6',
        measure='hadr', beta=1.0, normed=True,
        coords='ptyphim', check_input=False
    )
    
    print(f"\nOur configuration (n<=5, d<=6):")
    print(f"  Graphs: {len(efpset.graphs())}")
    print(f"  EFPs: {len(efpset.efps)}")
    print(f"  Compute output: {len(efpset.compute(jet_constituents))}")
    print(f"  -> Use {len(efpset.compute(jet_constituents))} as feature count")

if __name__ == "__main__":
    test_efp_final_mapping()
