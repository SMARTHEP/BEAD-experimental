#!/usr/bin/env python3
"""
Test EFP integration with actual BEAD helper functions.
This script runs from the repository root and properly imports BEAD modules.
"""

import os
import sys
import tempfile
import torch
from pathlib import Path

# Add the BEAD src directory to Python path
bead_src_path = Path(__file__).parent / "bead" / "src"
sys.path.insert(0, str(bead_src_path))

def test_load_tensors_function():
    """Test the actual load_tensors function from helper.py."""
    print("Testing actual load_tensors function...")
    
    # Import from the correct location
    try:
        # Change to bead directory to avoid import issues
        original_cwd = os.getcwd()
        os.chdir(str(bead_src_path))
        
        # Now import the helper module
        from utils import helper
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        print("✓ Successfully imported helper module")
        
    except Exception as e:
        print(f"✗ Failed to import helper module: {e}")
        return False
    
    # Test the load_tensors function
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
            events, jets, constituents, efp = helper.load_tensors(
                temp_dir, keyword="bkg_test", include_efp=True
            )
            
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
            
        except Exception as e:
            print(f"✗ load_tensors with EFP failed: {e}")
            return False
        
        try:
            # Test loading without EFP
            events, jets, constituents = helper.load_tensors(
                temp_dir, keyword="bkg_test", include_efp=False
            )
            
            print(f"✓ load_tensors without EFP successful")
            print(f"  Events shape: {events.shape}")
            print(f"  Jets shape: {jets.shape}")
            print(f"  Constituents shape: {constituents.shape}")
            
        except Exception as e:
            print(f"✗ load_tensors without EFP failed: {e}")
            return False
    
    return True


def test_calculate_in_shape_function():
    """Test the calculate_in_shape function with EFP data."""
    print("Testing calculate_in_shape function...")
    
    try:
        # Import from the correct location
        original_cwd = os.getcwd()
        os.chdir(str(bead_src_path))
        
        from utils import helper
        from utils.ggl import Config
        
        os.chdir(original_cwd)
        
        print("✓ Successfully imported helper and Config modules")
        
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        return False
    
    try:
        # Create mock data with EFP (8 tensors)
        batch_size = 100
        data_with_efp = (
            torch.randn(batch_size, 4),      # events_train
            torch.randn(batch_size, 3, 8),   # jets_train  
            torch.randn(batch_size, 45, 5),  # constituents_train
            torch.randn(batch_size, 3, 140), # efp_train
            torch.randn(20, 4),              # events_val
            torch.randn(20, 3, 8),           # jets_val
            torch.randn(20, 45, 5),          # constituents_val
            torch.randn(20, 3, 140),         # efp_val
        )
        
        # Create config
        config = Config()
        config.input_level = "constituent"
        config.enable_efp = True
        
        # Test calculate_in_shape with EFP data
        input_shape = helper.calculate_in_shape(data_with_efp, config)
        print(f"✓ calculate_in_shape with EFP successful: {input_shape}")
        
        # Test without EFP (6 tensors)
        data_without_efp = data_with_efp[:3] + data_with_efp[4:7]  # Remove EFP tensors
        config.enable_efp = False
        
        input_shape = helper.calculate_in_shape(data_without_efp, config)
        print(f"✓ calculate_in_shape without EFP successful: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ calculate_in_shape test failed: {e}")
        return False


def test_create_datasets_function():
    """Test the create_datasets function with EFP data."""
    print("Testing create_datasets function...")
    
    try:
        # Import from the correct location
        original_cwd = os.getcwd()
        os.chdir(str(bead_src_path))
        
        from utils import helper
        
        os.chdir(original_cwd)
        
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


def main():
    """Run all EFP helper function tests."""
    print("=" * 60)
    print("EFP Helper Function Integration Tests")
    print("=" * 60)
    
    tests = [
        test_load_tensors_function,
        test_calculate_in_shape_function,
        test_create_datasets_function,
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
        print("🎉 All EFP helper function tests PASSED!")
        return True
    else:
        print("❌ Some EFP helper function tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
