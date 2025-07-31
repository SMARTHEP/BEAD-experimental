#!/usr/bin/env python3
"""
Core EFP integration test focusing on the essential functionality.
This test validates EFP integration without complex Config instantiation.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_efp_utils_functions():
    """Test core EFP utility functions."""
    print("Testing EFP utility functions...")
    
    try:
        from src.utils.efp_utils import validate_efp_config, create_efpset, compute_efps_batch
        print("✓ Successfully imported EFP utility functions")
    except Exception as e:
        print(f"✗ Failed to import EFP utilities: {e}")
        return False
    
    # Create a mock config object
    class MockConfig:
        def __init__(self):
            self.efp_nmax = 5
            self.efp_dmax = 6
            self.efp_extended_mode = False
            self.efp_beta = 1.0
            self.efp_measure = "hadr"
            self.efp_normed = True
            self.efp_include_composites = False
    
    try:
        config = MockConfig()
        
        # Test validate_efp_config
        efp_config = validate_efp_config(config)
        print(f"✓ validate_efp_config successful: {efp_config['n_efps']} EFPs")
        
        # Test create_efpset
        efpset = create_efpset(efp_config)
        print(f"✓ create_efpset successful: {len(efpset.efps)} graphs")
        
        # Test compute_efps_batch with mock data
        # Create mock jet data: (n_jets, max_constituents, 3)
        n_jets, max_constituents = 10, 15
        jets_data = np.random.uniform(0.1, 10.0, (n_jets, max_constituents, 3))
        
        # Set some constituents to zero (padding)
        jets_data[:, 10:, :] = 0  # Zero out last 5 constituents
        
        efp_results = compute_efps_batch(jets_data, efpset, n_jobs=1)
        print(f"✓ compute_efps_batch successful: shape {efp_results.shape}")
        
        # Verify output shape
        expected_shape = (n_jets, efp_config['n_efps'])
        if efp_results.shape == expected_shape:
            print(f"✓ EFP output shape matches expected: {expected_shape}")
        else:
            print(f"✗ EFP output shape mismatch: got {efp_results.shape}, expected {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ EFP utility functions test failed: {e}")
        return False


def test_load_tensors_efp_integration():
    """Test load_tensors function with EFP integration."""
    print("Testing load_tensors EFP integration...")
    
    try:
        from src.utils import helper
        print("✓ Successfully imported helper module")
    except Exception as e:
        print(f"✗ Failed to import helper module: {e}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock tensor files
        batch_size = 50
        n_jets = 3
        n_constituents = 45
        n_efp_features = 140
        
        # Create and save mock tensors
        events_tensor = torch.randn(batch_size, 4)
        jets_tensor = torch.randn(batch_size, n_jets, 8)
        constituents_tensor = torch.randn(batch_size, n_constituents, 5)
        efp_tensor = torch.randn(batch_size, n_jets, n_efp_features)
        
        torch.save(events_tensor, os.path.join(temp_dir, "bkg_test_events.pt"))
        torch.save(jets_tensor, os.path.join(temp_dir, "bkg_test_jets.pt"))
        torch.save(constituents_tensor, os.path.join(temp_dir, "bkg_test_constituents.pt"))
        torch.save(efp_tensor, os.path.join(temp_dir, "bkg_test_efp.pt"))
        
        try:
            # Test loading with EFP
            result = helper.load_tensors(temp_dir, keyword="bkg_test", include_efp=True)
            
            if len(result) == 4:
                events, jets, constituents, efp = result
                print(f"✓ load_tensors with EFP successful")
                print(f"  Events shape: {events.shape}")
                print(f"  Jets shape: {jets.shape}")
                print(f"  Constituents shape: {constituents.shape}")
                print(f"  EFP shape: {efp.shape}")
                
                # Verify shapes
                assert events.shape == events_tensor.shape
                assert jets.shape == jets_tensor.shape
                assert constituents.shape == constituents_tensor.shape
                assert efp.shape == efp_tensor.shape
                
                print("✓ All tensor shapes match expected values")
            else:
                print(f"✗ Expected 4 tensors, got {len(result)}")
                return False
            
        except Exception as e:
            print(f"✗ load_tensors with EFP failed: {e}")
            return False
        
        try:
            # Test loading without EFP
            result = helper.load_tensors(temp_dir, keyword="bkg_test", include_efp=False)
            
            if len(result) == 3:
                events, jets, constituents = result
                print(f"✓ load_tensors without EFP successful")
            else:
                print(f"✗ Expected 3 tensors, got {len(result)}")
                return False
            
        except Exception as e:
            print(f"✗ load_tensors without EFP failed: {e}")
            return False
    
    return True


def test_create_datasets_efp():
    """Test create_datasets function with EFP data."""
    print("Testing create_datasets EFP integration...")
    
    try:
        from src.utils import helper
        print("✓ Successfully imported helper module")
    except Exception as e:
        print(f"✗ Failed to import helper module: {e}")
        return False
    
    try:
        # Create mock data and labels
        batch_size = 100
        val_size = 20
        
        # Data tensors
        events_train = torch.randn(batch_size, 4)
        jets_train = torch.randn(batch_size, 3, 8)
        constituents_train = torch.randn(batch_size, 45, 5)
        efp_train = torch.randn(batch_size, 3, 140)
        
        events_val = torch.randn(val_size, 4)
        jets_val = torch.randn(val_size, 3, 8)
        constituents_val = torch.randn(val_size, 45, 5)
        efp_val = torch.randn(val_size, 3, 140)
        
        # Label tensors
        events_train_label = torch.randint(0, 2, (batch_size,))
        jets_train_label = torch.randint(0, 2, (batch_size,))
        constituents_train_label = torch.randint(0, 2, (batch_size,))
        efp_train_label = torch.randint(0, 2, (batch_size,))
        
        events_val_label = torch.randint(0, 2, (val_size,))
        jets_val_label = torch.randint(0, 2, (val_size,))
        constituents_val_label = torch.randint(0, 2, (val_size,))
        efp_val_label = torch.randint(0, 2, (val_size,))
        
        # Test create_datasets with EFP data
        datasets = helper.create_datasets(
            events_train, jets_train, constituents_train,
            events_val, jets_val, constituents_val,
            events_train_label, jets_train_label, constituents_train_label,
            events_val_label, jets_val_label, constituents_val_label,
            efp_train, efp_val, efp_train_label, efp_val_label
        )
        
        print(f"✓ create_datasets with EFP successful")
        print(f"  Created datasets: {list(datasets.keys())}")
        
        # Check that EFP datasets were created
        if "efp_train" in datasets and "efp_val" in datasets:
            print("✓ EFP datasets successfully created")
        else:
            print("✗ EFP datasets missing from created datasets")
            return False
        
        # Test create_datasets without EFP data
        datasets_no_efp = helper.create_datasets(
            events_train, jets_train, constituents_train,
            events_val, jets_val, constituents_val,
            events_train_label, jets_train_label, constituents_train_label,
            events_val_label, jets_val_label, constituents_val_label
        )
        
        print(f"✓ create_datasets without EFP successful")
        print(f"  Created datasets: {list(datasets_no_efp.keys())}")
        
        # Check that EFP datasets were NOT created
        if "efp_train" not in datasets_no_efp and "efp_val" not in datasets_no_efp:
            print("✓ EFP datasets correctly omitted")
        else:
            print("✗ Unexpected EFP datasets found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ create_datasets test failed: {e}")
        return False


def test_data_processing_efp_import():
    """Test that data_processing module can import EFP functions correctly."""
    print("Testing data_processing EFP imports...")
    
    try:
        from src.utils import data_processing
        print("✓ Successfully imported data_processing module")
        
        # Check if compute_efp_features function exists
        if hasattr(data_processing, 'compute_efp_features'):
            print("✓ compute_efp_features function found")
        else:
            print("✗ compute_efp_features function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ data_processing import test failed: {e}")
        return False


def main():
    """Run all core EFP integration tests."""
    print("=" * 60)
    print("BEAD Core EFP Integration Tests")
    print("=" * 60)
    
    tests = [
        test_efp_utils_functions,
        test_load_tensors_efp_integration,
        test_create_datasets_efp,
        test_data_processing_efp_import,
    ]
    
    results = []
    for test_func in tests:
        print(f"\n{'-' * 40}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n{'=' * 60}")
    print("Test Summary:")
    print(f"{'=' * 60}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All core EFP integration tests PASSED!")
        return True
    else:
        print("❌ Some core EFP integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
