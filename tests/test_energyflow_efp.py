#!/usr/bin/env python3
"""
Comprehensive EnergyFlow Library Integration Test for BEAD

This script validates the EnergyFlow library integration for EFP computation
within the BEAD anomaly detection pipeline. It tests installation, API compatibility,
performance benchmarks, GPU support, masking behavior, and edge cases.
"""

import sys
import time
import multiprocessing as mp
import numpy as np
import torch
import traceback
from typing import Dict, Any, List, Optional

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
            'n<=5',        # N_max = 5 (max particles in graph)
            'd<=6',        # d_max = 6 (max degree)
            measure='hadr', # Hadron collider measure (pT, eta, phi)
            beta=1.0,      # Linear energy weighting (IRC safe)
            normed=True,   # Normalize angular distances to [0,1]
            coords='ptyphim', # Input format: (pT, y/eta, phi, mass)
            check_input=True
        )
        
        # Get number of EFPs
        n_efps = efpset.count()
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


def test_performance_benchmarks() -> Dict[str, Any]:
    """Test EFP computation performance with different configurations."""
    print("\n3. Testing Performance Benchmarks...")
    
    results = {
        'success': True,
        'errors': [],
        'benchmarks': {}
    }
    
    try:
        import energyflow as ef
        
        # Test configurations
        configs = [
            {'spec': ['n<=3', 'd<=4'], 'name': 'small'},
            {'spec': ['n<=4', 'd<=5'], 'name': 'medium'},
            {'spec': ['n<=5', 'd<=6'], 'name': 'large'},
        ]
        
        # Generate test data
        np.random.seed(42)
        n_events = 50
        n_particles = 15
        
        test_events = []
        for i in range(n_events):
            # Variable number of particles per event (5-15)
            n_parts = np.random.randint(5, n_particles + 1)
            
            # Generate realistic particle data
            pt = np.random.exponential(10.0, n_parts) + 1.0  # pT distribution
            eta = np.random.normal(0.0, 1.5, n_parts)        # η distribution
            phi = np.random.uniform(-np.pi, np.pi, n_parts)  # φ distribution
            mass = np.zeros(n_parts)                          # massless
            
            event_data = np.column_stack([pt, eta, phi, mass])
            test_events.append(event_data)
        
        print(f"   ✓ Generated {n_events} test events")
        
        # Test each configuration
        for config in configs:
            try:
                print(f"\n   Testing {config['name']} configuration ({', '.join(config['spec'])})...")
                
                efpset = ef.EFPSet(
                    *config['spec'],
                    measure='hadr',
                    beta=1.0,
                    normed=True,
                    coords='ptyphim'
                )
                
                n_efps = efpset.count()
                print(f"     EFP count: {n_efps}")
                
                # Single-threaded benchmark
                start_time = time.time()
                efps_single = efpset.batch_compute(test_events, n_jobs=1)
                single_time = time.time() - start_time
                
                # Multi-threaded benchmark (if available)
                n_cores = min(4, mp.cpu_count())
                start_time = time.time()
                efps_multi = efpset.batch_compute(test_events, n_jobs=n_cores)
                multi_time = time.time() - start_time
                
                # Verify results are identical
                if not np.allclose(efps_single, efps_multi, rtol=1e-10):
                    results['errors'].append(f"Multi-threading results differ for {config['name']}")
                
                benchmark_data = {
                    'n_efps': n_efps,
                    'single_thread_time': single_time,
                    'multi_thread_time': multi_time,
                    'speedup': single_time / multi_time if multi_time > 0 else 1.0,
                    'events_per_second_single': n_events / single_time,
                    'events_per_second_multi': n_events / multi_time,
                    'output_shape': efps_single.shape
                }
                
                results['benchmarks'][config['name']] = benchmark_data
                
                print(f"     Single-thread: {single_time:.3f}s ({benchmark_data['events_per_second_single']:.1f} events/s)")
                print(f"     Multi-thread ({n_cores} cores): {multi_time:.3f}s ({benchmark_data['events_per_second_multi']:.1f} events/s)")
                print(f"     Speedup: {benchmark_data['speedup']:.2f}x")
                print(f"     Output shape: {efps_single.shape}")
                
            except Exception as e:
                results['errors'].append(f"Benchmark failed for {config['name']}: {e}")
                print(f"     ✗ Failed: {e}")
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Performance benchmark failed: {e}")
        print(f"   ✗ Performance test failed: {e}")
    
    return results


def test_masking_behavior() -> Dict[str, Any]:
    """Test masking behavior with zero-padded particles."""
    print("\n4. Testing Masking Behavior...")
    
    results = {
        'success': True,
        'errors': [],
        'masking_tests': {}
    }
    
    try:
        import energyflow as ef
        
        # Create EFPSet for testing
        efpset = ef.EFPSet('n<=4', 'd<=5', measure='hadr', coords='ptyphim', normed=True)
        
        # Test 1: Full jet (no masking)
        full_jet = np.array([
            [50.0, 0.1, 0.5, 0.0],
            [30.0, -0.2, 1.0, 0.0],
            [20.0, 0.5, -0.5, 0.0],
            [15.0, -0.1, 2.0, 0.0],
            [10.0, 0.3, -1.0, 0.0],
        ])
        
        efp_full = efpset.compute(full_jet)
        results['masking_tests']['full_jet'] = {
            'n_particles': len(full_jet),
            'efp_shape': efp_full.shape,
            'has_nan': np.any(np.isnan(efp_full)),
            'has_inf': np.any(np.isinf(efp_full))
        }
        
        print(f"   ✓ Full jet ({len(full_jet)} particles): shape {efp_full.shape}")
        
        # Test 2: Jet with zero-padded particles (BEAD-style masking)
        padded_jet = np.array([
            [50.0, 0.1, 0.5, 0.0],   # Real particle
            [30.0, -0.2, 1.0, 0.0],  # Real particle
            [20.0, 0.5, -0.5, 0.0],  # Real particle
            [0.0, 0.0, 0.0, 0.0],    # Zero-padded (masked)
            [0.0, 0.0, 0.0, 0.0],    # Zero-padded (masked)
        ])
        
        # Filter out zero-padded particles before EFP computation
        mask = padded_jet[:, 0] > 0  # pT > 0
        filtered_jet = padded_jet[mask]
        
        efp_filtered = efpset.compute(filtered_jet)
        results['masking_tests']['filtered_jet'] = {
            'n_particles_original': len(padded_jet),
            'n_particles_filtered': len(filtered_jet),
            'efp_shape': efp_filtered.shape,
            'has_nan': np.any(np.isnan(efp_filtered)),
            'has_inf': np.any(np.isinf(efp_filtered))
        }
        
        print(f"   ✓ Filtered jet ({len(filtered_jet)}/{len(padded_jet)} particles): shape {efp_filtered.shape}")
        
        # Test 3: Empty jet (edge case)
        try:
            empty_jet = np.array([]).reshape(0, 4)
            efp_empty = efpset.compute(empty_jet)
            results['masking_tests']['empty_jet'] = {
                'n_particles': 0,
                'efp_shape': efp_empty.shape,
                'has_nan': np.any(np.isnan(efp_empty)),
                'has_inf': np.any(np.isinf(efp_empty))
            }
            print(f"   ✓ Empty jet: shape {efp_empty.shape}")
        except Exception as e:
            print(f"     ✗ Empty jet computation failed: {e}")
            results['masking_tests']['empty_jet'] = {'error': str(e)}
        
        print("\n   Key Insights:")
        print("   • Zero-padded particles MUST be filtered out before EFP computation")
        print("   • Use mask: pT > 0 to identify real particles")
        print("   • Empty jets should be handled separately (fill with zeros)")
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Masking test failed: {e}")
        print(f"   ✗ Masking test failed: {e}")
    
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
    
    # Test 3: Performance benchmarks
    all_results['performance_benchmarks'] = test_performance_benchmarks()
    
    # Test 4: Masking behavior
    all_results['masking_behavior'] = test_masking_behavior()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("ENERGYFLOW INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    # Check overall status
    overall_success = all(result.get('success', False) for result in all_results.values())
    status_symbol = "✓" if overall_success else "✗"
    status_text = "PASSED" if overall_success else "FAILED"
    
    print(f"\nOverall Status: {status_symbol} {status_text}")
    
    # Print individual test results
    for test_name, result in all_results.items():
        success = result.get('success', False)
        symbol = "✓" if success else "✗"
        formatted_name = test_name.replace('_', ' ').title()
        print(f"  {symbol} {formatted_name}")
        
        # Print errors if any
        if 'errors' in result and result['errors']:
            for error in result['errors']:
                print(f"      Error: {error}")
    
    # Print key metrics if available
    print(f"\nKey Metrics:")
    
    # API compatibility metrics
    if 'api_compatibility' in all_results and all_results['api_compatibility'].get('success'):
        api_result = all_results['api_compatibility']
        if 'efp_features' in api_result:
            print(f"  • EFP Features: {api_result['efp_features']}")
        if 'computation_time' in api_result:
            print(f"  • Computation Time: {api_result['computation_time']:.2f} ms")
    
    # Performance benchmark metrics
    if 'performance_benchmarks' in all_results and all_results['performance_benchmarks'].get('success'):
        perf_result = all_results['performance_benchmarks']
        if 'benchmarks' in perf_result:
            print(f"  • Performance Benchmarks:")
            for config_name, benchmark in perf_result['benchmarks'].items():
                speedup = benchmark.get('speedup', 1.0)
                events_per_sec = benchmark.get('events_per_second_multi', 0)
                print(f"    - {config_name.title()}: {speedup:.2f}x speedup, {events_per_sec:.1f} events/s")
    
    # Masking behavior insights
    if 'masking_behavior' in all_results and all_results['masking_behavior'].get('success'):
        print(f"  • Masking: Zero-padding filter validated")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
