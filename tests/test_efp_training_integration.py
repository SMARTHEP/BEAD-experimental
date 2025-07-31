#!/usr/bin/env python3
"""
Test script for EFP integration in BEAD training pipeline.

This script validates that EFP features are properly loaded, processed, and 
integrated into the training pipeline without breaking existing functionality.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bead" / "src"))

# Import with absolute paths to avoid relative import issues
import utils.helper as helper
import utils.data_processing as data_processing
from utils.ggl import Config


def create_mock_efp_data(events_shape, jets_shape, constituents_shape, n_efp_features=140):
    """Create mock EFP data matching the expected shapes."""
    batch_size = events_shape[0]
    n_jets = 3  # Standard BEAD configuration
    
    # Create EFP tensor: (batch_size, n_jets, n_efp_features)
    efp_shape = (batch_size, n_jets, n_efp_features)
    efp_data = torch.randn(efp_shape, dtype=torch.float32)
    
    return efp_data


def create_mock_data_with_efp():
    """Create mock training data including EFP features."""
    # Standard BEAD data shapes
    batch_size = 100
    n_jets = 3
    n_constituents = 45  # 3 jets * 15 constituents
    n_features = 5
    n_efp_features = 140
    
    # Create mock tensors
    events_train = torch.randn(batch_size, 4, dtype=torch.float32)
    jets_train = torch.randn(batch_size, n_jets, 8, dtype=torch.float32)
    constituents_train = torch.randn(batch_size, n_constituents, n_features, dtype=torch.float32)
    efp_train = torch.randn(batch_size, n_jets, n_efp_features, dtype=torch.float32)
    
    # Validation data (smaller)
    val_size = 20
    events_val = torch.randn(val_size, 4, dtype=torch.float32)
    jets_val = torch.randn(val_size, n_jets, 8, dtype=torch.float32)
    constituents_val = torch.randn(val_size, n_constituents, n_features, dtype=torch.float32)
    efp_val = torch.randn(val_size, n_jets, n_efp_features, dtype=torch.float32)
    
    # Create labels (binary classification)
    events_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    jets_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    constituents_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    efp_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    
    events_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    jets_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    constituents_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    efp_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    
    data = (
        events_train, jets_train, constituents_train, efp_train,
        events_val, jets_val, constituents_val, efp_val
    )
    
    labels = (
        events_train_label, jets_train_label, constituents_train_label, efp_train_label,
        events_val_label, jets_val_label, constituents_val_label, efp_val_label
    )
    
    return data, labels


def create_mock_data_without_efp():
    """Create mock training data without EFP features."""
    # Standard BEAD data shapes
    batch_size = 100
    n_jets = 3
    n_constituents = 45  # 3 jets * 15 constituents
    n_features = 5
    
    # Create mock tensors
    events_train = torch.randn(batch_size, 4, dtype=torch.float32)
    jets_train = torch.randn(batch_size, n_jets, 8, dtype=torch.float32)
    constituents_train = torch.randn(batch_size, n_constituents, n_features, dtype=torch.float32)
    
    # Validation data (smaller)
    val_size = 20
    events_val = torch.randn(val_size, 4, dtype=torch.float32)
    jets_val = torch.randn(val_size, n_jets, 8, dtype=torch.float32)
    constituents_val = torch.randn(val_size, n_constituents, n_features, dtype=torch.float32)
    
    # Create labels (binary classification)
    events_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    jets_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    constituents_train_label = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    
    events_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    jets_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    constituents_val_label = torch.randint(0, 2, (val_size,), dtype=torch.long)
    
    data = (
        events_train, jets_train, constituents_train,
        events_val, jets_val, constituents_val
    )
    
    labels = (
        events_train_label, jets_train_label, constituents_train_label,
        events_val_label, jets_val_label, constituents_val_label
    )
    
    return data, labels


def test_data_unpacking_with_efp():
    """Test that training function correctly unpacks data with EFP features."""
    print("Testing data unpacking with EFP features...")
    
    data, labels = create_mock_data_with_efp()
    
    # Create minimal config
    config = Config()
    config.batch_size = 16
    config.input_level = "constituent"
    config.model_name = "ConvAE"
    config.input_features = "4momentum"
    config.enable_efp = True
    config.epochs = 1
    config.is_ddp_active = False
    config.local_rank = 0
    config.world_size = 1
    
    # Test calculate_in_shape with EFP data
    try:
        input_shape = helper.calculate_in_shape(data, config)
        print(f"✓ Input shape calculation successful: {input_shape}")
    except Exception as e:
        print(f"✗ Input shape calculation failed: {e}")
        return False
    
    # Test create_datasets with EFP data
    try:
        if len(data) == 8:
            datasets = helper.create_datasets(
                data[0], data[1], data[2], data[4], data[5], data[6],  # events, jets, constituents
                labels[0], labels[1], labels[2], labels[4], labels[5], labels[6],  # labels
                data[3], data[7], labels[3], labels[7]  # EFP data and labels
            )
        print(f"✓ Dataset creation successful. Created {len(datasets)} datasets")
        
        # Check that EFP datasets were created
        if "efp_train" in datasets and "efp_val" in datasets:
            print("✓ EFP datasets successfully created")
        else:
            print("✗ EFP datasets missing from created datasets")
            return False
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False
    
    return True


def test_data_unpacking_without_efp():
    """Test that training function correctly unpacks data without EFP features."""
    print("Testing data unpacking without EFP features...")
    
    data, labels = create_mock_data_without_efp()
    
    # Create minimal config
    config = Config()
    config.batch_size = 16
    config.input_level = "constituent"
    config.model_name = "ConvAE"
    config.input_features = "4momentum"
    config.enable_efp = False
    config.epochs = 1
    config.is_ddp_active = False
    config.local_rank = 0
    config.world_size = 1
    
    # Test calculate_in_shape without EFP data
    try:
        input_shape = helper.calculate_in_shape(data, config)
        print(f"✓ Input shape calculation successful: {input_shape}")
    except Exception as e:
        print(f"✗ Input shape calculation failed: {e}")
        return False
    
    # Test create_datasets without EFP data
    try:
        datasets = helper.create_datasets(*data, *labels)
        print(f"✓ Dataset creation successful. Created {len(datasets)} datasets")
        
        # Check that EFP datasets were NOT created
        if "efp_train" not in datasets and "efp_val" not in datasets:
            print("✓ EFP datasets correctly omitted")
        else:
            print("✗ Unexpected EFP datasets found")
            return False
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False
    
    return True


def test_load_tensors_with_efp():
    """Test load_tensors function with EFP features."""
    print("Testing load_tensors with EFP features...")
    
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
            
            print(f"✓ Loaded tensors successfully")
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
    
    return True


def test_load_tensors_without_efp():
    """Test load_tensors function without EFP features."""
    print("Testing load_tensors without EFP features...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock tensor files (no EFP)
        batch_size = 50
        n_jets = 3
        n_constituents = 45
        
        # Create and save mock tensors
        events_tensor = torch.randn(batch_size, 4)
        jets_tensor = torch.randn(batch_size, n_jets, 8)
        constituents_tensor = torch.randn(batch_size, n_constituents, 5)
        
        torch.save(events_tensor, os.path.join(temp_dir, "bkg_test_events.pt"))
        torch.save(jets_tensor, os.path.join(temp_dir, "bkg_test_jets.pt"))
        torch.save(constituents_tensor, os.path.join(temp_dir, "bkg_test_constituents.pt"))
        
        try:
            # Test loading without EFP
            events, jets, constituents = helper.load_tensors(
                temp_dir, keyword="bkg_test", include_efp=False
            )
            
            print(f"✓ Loaded tensors successfully")
            print(f"  Events shape: {events.shape}")
            print(f"  Jets shape: {jets.shape}")
            print(f"  Constituents shape: {constituents.shape}")
            
            # Verify shapes
            assert events.shape == events_tensor.shape
            assert jets.shape == jets_tensor.shape
            assert constituents.shape == constituents_tensor.shape
            
            print("✓ All tensor shapes match expected values")
            
        except Exception as e:
            print(f"✗ load_tensors without EFP failed: {e}")
            return False
    
    return True


def main():
    """Run all EFP integration tests."""
    print("=" * 60)
    print("BEAD EFP Training Integration Tests")
    print("=" * 60)
    
    tests = [
        test_data_unpacking_with_efp,
        test_data_unpacking_without_efp,
        test_load_tensors_with_efp,
        test_load_tensors_without_efp,
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
        print("🎉 All EFP integration tests PASSED!")
        return True
    else:
        print("❌ Some EFP integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
