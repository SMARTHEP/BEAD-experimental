#!/usr/bin/env python3
"""
Debug script to investigate EFP count discrepancy.
"""

import energyflow as ef

def debug_efp_counts():
    """Debug EFP count differences."""
    print("EFP Count Debug")
    print("=" * 40)
    
    # Test our exact configuration
    efpset = ef.EFPSet(
        'n<=5',
        'd<=6',
        measure='hadr',
        beta=1.0,
        normed=True,
        coords='ptyphim',
        check_input=False
    )
    
    n_efps = len(efpset.efps)
    print(f"EFPSet configuration:")
    print(f"  n<=5, d<=6")
    print(f"  measure='hadr'")
    print(f"  beta=1.0")
    print(f"  normed=True")
    print(f"  coords='ptyphim'")
    print(f"  Actual EFP count: {n_efps}")
    
    # Check if there are any additional parameters affecting count
    print(f"\nEFPSet attributes:")
    try:
        if hasattr(efpset, 'graphs'):
            graphs = efpset.graphs()
            print(f"  efpset.graphs(): {len(graphs)}")
        else:
            print(f"  efpset.graphs: N/A")
    except:
        print(f"  efpset.graphs: N/A (method call failed)")
    print(f"  efpset.efps: {len(efpset.efps)}")
    
    # Test different configurations to understand the pattern
    configs = [
        ('n<=3', 'd<=4'),
        ('n<=4', 'd<=5'),
        ('n<=5', 'd<=6'),
        ('n<=6', 'd<=7'),
    ]
    
    print(f"\nConfiguration comparison:")
    for n_spec, d_spec in configs:
        try:
            test_efpset = ef.EFPSet(
                n_spec, d_spec,
                measure='hadr',
                beta=1.0,
                normed=True,
                coords='ptyphim',
                check_input=False
            )
            count = len(test_efpset.efps)
            print(f"  {n_spec}, {d_spec}: {count} EFPs")
        except Exception as e:
            print(f"  {n_spec}, {d_spec}: ERROR - {e}")

if __name__ == "__main__":
    debug_efp_counts()
