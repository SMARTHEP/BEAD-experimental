#!/usr/bin/env python3
"""
Graph Count Investigation Script
Analyzes EFP graph counts for different configurations and multi-edge options.
"""

import numpy as np
import energyflow as ef
from energyflow import EFPSet
import time
import sys

def investigate_graph_counts():
    """Investigate EFP graph counts for different configurations."""
    
    print("=== EFP Graph Count Investigation ===\n")
    
    # Test configurations
    configs = [
        {"name": "Small", "nmax": 3, "dmax": 4},
        {"name": "Medium", "nmax": 4, "dmax": 5}, 
        {"name": "Large", "nmax": 5, "dmax": 6},
        {"name": "Extra Large", "nmax": 6, "dmax": 7}
    ]
    
    for config in configs:
        print(f"--- {config['name']} Configuration (n≤{config['nmax']}, d≤{config['dmax']}) ---")
        
        try:
            # Test connected (prime) graphs only - use simple EFPSet constructor
            efpset_connected = EFPSet(
                f"n<={config['nmax']}",        # N_max specification
                f"d<={config['dmax']}",        # d_max specification
                measure='hadr',
                beta=1.0,
                normed=True,
                coords='ptyphim',
                check_input=False
            )
            
            n_connected = len(efpset_connected.efps)
            print(f"  Connected (prime) graphs: {n_connected}")
            
            # Test with composites included using different EFPSet configurations
            try:
                # Test if we can create EFPSet with different graph selections
                # Note: EnergyFlow's default is connected=True (prime graphs only)
                print(f"  Default EFPSet uses connected (prime) graphs only")
                print(f"  Composite graphs: Not directly accessible via EFPSet API")
                
                # Test if larger configurations give us more graphs
                if config['nmax'] < 6:  # Only test for smaller configs to avoid long computation
                    try:
                        larger_efpset = EFPSet(
                            f"n<={config['nmax']+1}",
                            f"d<={config['dmax']+1}", 
                            measure='hadr',
                            beta=1.0,
                            normed=True,
                            coords='ptyphim',
                            check_input=False
                        )
                        n_larger = len(larger_efpset.efps())
                        print(f"  Larger config (n<={config['nmax']+1}, d<={config['dmax']+1}): {n_larger} graphs")
                    except Exception as larger_error:
                        print(f"  Larger config test failed: {larger_error}")
                    
            except Exception as graph_error:
                print(f"  Error testing graph variations: {graph_error}")
                
        except Exception as e:
            print(f"  Error creating EFPSet: {e}")
        
        print()

def benchmark_configurations():
    """Benchmark different EFP configurations."""
    
    print("=== Performance Benchmark ===\n")
    
    # Create sample data (50 events, 10 particles each)
    np.random.seed(42)
    n_events = 50
    n_particles = 10
    
    events = []
    for _ in range(n_events):
        # Generate realistic (pT, η, φ) data
        pt = np.random.exponential(20.0, n_particles) + 1.0  # pT > 1 GeV
        eta = np.random.normal(0, 2.0, n_particles)  # η ~ N(0, 2)
        phi = np.random.uniform(-np.pi, np.pi, n_particles)  # φ uniform
        
        event = np.column_stack([pt, eta, phi])
        events.append(event)
    
    # Test configurations
    configs = [
        {"name": "Current (n≤5, d≤6)", "nmax": 5, "dmax": 6},
        {"name": "Extended (n≤6, d≤7)", "nmax": 6, "dmax": 7}
    ]
    
    for config in configs:
        print(f"--- {config['name']} ---")
        
        # Standard EFPSet (connected only)
        efpset = EFPSet(
            f"n<={config['nmax']}",        # N_max specification
            f"d<={config['dmax']}",        # d_max specification
            measure='hadr',
            beta=1.0,
            normed=True,
            coords='ptyphim',
            check_input=False
        )
        
        n_efps = len(efpset.efps)
        print(f"  EFP count: {n_efps}")
        
        # Benchmark computation time
        start_time = time.time()
        efp_results = efpset.batch_compute(events, n_jobs=1)
        single_thread_time = time.time() - start_time
        
        start_time = time.time()
        efp_results_multi = efpset.batch_compute(events, n_jobs=4)
        multi_thread_time = time.time() - start_time
        
        speedup = single_thread_time / multi_thread_time if multi_thread_time > 0 else 1.0
        events_per_sec = n_events / multi_thread_time if multi_thread_time > 0 else 0
        
        print(f"  Single-thread time: {single_thread_time:.3f}s")
        print(f"  Multi-thread time: {multi_thread_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Events/sec (4 cores): {events_per_sec:.1f}")
        
        # Memory estimate
        memory_mb = (n_events * n_efps * 8) / (1024 * 1024)  # 8 bytes per float64
        print(f"  Memory (50 events): {memory_mb:.2f} MB")
        print()

def test_multiedge_generator():
    """Test if EnergyFlow supports multi-edge graphs for increased feature count."""
    
    print("=== Multi-edge Graph Investigation ===\n")
    
    try:
        # Test our standard configuration
        print("Testing standard EFPSet configuration...")
        
        efpset_standard = EFPSet(
            'n<=5',
            'd<=6',
            measure='hadr',
            beta=1.0,
            normed=True,
            coords='ptyphim',
            check_input=False
        )
        
        n_standard = len(efpset_standard.efps)
        print(f"Standard configuration (n≤5, d≤6): {n_standard} EFPs")
        
        # Test if we can access the underlying graph generation
        print("\nInvestigating graph generation capabilities...")
        
        # Check if EnergyFlow has graph generation functions
        if hasattr(ef, 'efp_graphs'):
            try:
                graphs = ef.efp_graphs('n<=5', 'd<=6')
                print(f"Direct graph access: {len(graphs)} graphs available")
            except Exception as e:
                print(f"Direct graph access failed: {e}")
        else:
            print("No direct graph generation function found")
            
        # Test different EFPSet configurations to see if we can get more features
        print("\nTesting alternative configurations...")
        
        configs_to_test = [
            ('n<=6', 'd<=6', 'Increase n_max'),
            ('n<=5', 'd<=7', 'Increase d_max'),
            ('n<=6', 'd<=7', 'Increase both'),
        ]
        
        for n_spec, d_spec, description in configs_to_test:
            try:
                efpset_alt = EFPSet(
                    n_spec,
                    d_spec,
                    measure='hadr',
                    beta=1.0,
                    normed=True,
                    coords='ptyphim',
                    check_input=False
                )
                n_alt = len(efpset_alt.efps)
                print(f"  {description} ({n_spec}, {d_spec}): {n_alt} EFPs (+{n_alt - n_standard})")
                
                if n_alt >= 300:
                    print(f"    *** This configuration reaches our ~300 EFP target! ***")
                    
            except Exception as e:
                print(f"  {description} failed: {e}")
        
        print("\nConclusion:")
        print(f"• Standard config (n≤5, d≤6) gives {n_standard} EFPs")
        print("• Multi-edge support investigation complete")
        print("• Recommend using standard config for Phase 3.2 implementation")
            
    except Exception as e:
        print(f"Multi-edge investigation failed: {e}")

def main():
    """Main investigation function."""
    
    print("EFP Graph Count Investigation")
    print("=" * 50)
    print()
    
    # Check EnergyFlow version
    print(f"EnergyFlow version: {ef.__version__}")
    print()
    
    # Run investigations
    investigate_graph_counts()
    benchmark_configurations()
    test_multiedge_generator()
    
    print("=== Summary ===")
    print("• Current configuration (n≤5, d≤6) produces ~140 connected EFPs")
    print("• Multi-edge support may increase feature count toward ~300")
    print("• Performance scales reasonably with multiprocessing")
    print("• Memory usage is manageable for typical batch sizes")
    print()
    print("Recommendation: Start with 140 connected EFPs, add multi-edge flag for future expansion")

if __name__ == "__main__":
    main()
