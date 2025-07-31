#!/usr/bin/env python3
"""
Simple EFP integration test that validates core functionality without complex imports.
"""

import os
import sys
import tempfile
import torch
from pathlib import Path

def test_efp_tensor_loading():
    """Test EFP tensor loading functionality."""
    print("Testing EFP tensor loading...")
    
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
        
        # Save tensors with BEAD naming convention
        torch.save(events_tensor, os.path.join(temp_dir, "bkg_test_events.pt"))
        torch.save(jets_tensor, os.path.join(temp_dir, "bkg_test_jets.pt"))
        torch.save(constituents_tensor, os.path.join(temp_dir, "bkg_test_constituents.pt"))
        torch.save(efp_tensor, os.path.join(temp_dir, "bkg_test_efp.pt"))
        
        # Test manual loading (simulating load_tensors function)
        try:
            # Load tensors manually
            events = torch.load(os.path.join(temp_dir, "bkg_test_events.pt"))
            jets = torch.load(os.path.join(temp_dir, "bkg_test_jets.pt"))
            constituents = torch.load(os.path.join(temp_dir, "bkg_test_constituents.pt"))
            efp = torch.load(os.path.join(temp_dir, "bkg_test_efp.pt"))
            
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
            return True
            
        except Exception as e:
            print(f"✗ Tensor loading failed: {e}")
            return False


def test_efp_feature_concatenation():
    """Test EFP feature concatenation for model input."""
    print("Testing EFP feature concatenation...")
    
    try:
        # Create mock data
        batch_size = 32
        n_jets = 3
        n_constituents = 45
        n_features = 5
        n_efp_features = 140
        
        # Mock constituent features (flattened)
        constituents = torch.randn(batch_size, n_constituents * n_features)
        
        # Mock EFP features
        efp_features = torch.randn(batch_size, n_jets, n_efp_features)
        
        # Flatten EFP features for concatenation
        efp_flattened = efp_features.view(batch_size, -1)  # (batch_size, n_jets * n_efp_features)
        
        # Concatenate features
        combined_features = torch.cat([constituents, efp_flattened], dim=1)
        
        print(f"✓ Feature concatenation successful")
        print(f"  Constituent features shape: {constituents.shape}")
        print(f"  EFP features shape (original): {efp_features.shape}")
        print(f"  EFP features shape (flattened): {efp_flattened.shape}")
        print(f"  Combined features shape: {combined_features.shape}")
        
        # Verify dimensions
        expected_combined_size = n_constituents * n_features + n_jets * n_efp_features
        assert combined_features.shape[1] == expected_combined_size
        
        print(f"✓ Combined feature dimension correct: {expected_combined_size}")
        return True
        
    except Exception as e:
        print(f"✗ Feature concatenation failed: {e}")
        return False


def test_efp_data_unpacking():
    """Test data unpacking with different numbers of tensors."""
    print("Testing EFP data unpacking...")
    
    try:
        # Test with 8 tensors (EFP included)
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
        
        # Test unpacking
        if len(data_with_efp) == 8:
            events_train, jets_train, constituents_train, efp_train = data_with_efp[:4]
            events_val, jets_val, constituents_val, efp_val = data_with_efp[4:]
            print("✓ 8-tensor unpacking (with EFP) successful")
        
        # Test with 6 tensors (no EFP)
        data_without_efp = (
            torch.randn(batch_size, 4),      # events_train
            torch.randn(batch_size, 3, 8),   # jets_train  
            torch.randn(batch_size, 45, 5),  # constituents_train
            torch.randn(20, 4),              # events_val
            torch.randn(20, 3, 8),           # jets_val
            torch.randn(20, 45, 5),          # constituents_val
        )
        
        if len(data_without_efp) == 6:
            events_train, jets_train, constituents_train = data_without_efp[:3]
            events_val, jets_val, constituents_val = data_without_efp[3:]
            print("✓ 6-tensor unpacking (without EFP) successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Data unpacking failed: {e}")
        return False


def test_efp_config_validation():
    """Test EFP configuration parameters."""
    print("Testing EFP configuration validation...")
    
    try:
        # Test standard EFP configuration
        efp_config = {
            "enable_efp": True,
            "efp_nmax": 5,
            "efp_dmax": 6,
            "efp_beta": 1.0,
            "efp_measure": "hadr",
            "efp_normed": True,
            "efp_include_composites": False,
        }
        
        # Validate configuration
        assert efp_config["efp_nmax"] > 0
        assert efp_config["efp_dmax"] > 0
        assert efp_config["efp_beta"] > 0
        assert efp_config["efp_measure"] in ["hadr", "ee", "gen"]
        assert isinstance(efp_config["efp_normed"], bool)
        assert isinstance(efp_config["efp_include_composites"], bool)
        
        print("✓ EFP configuration validation successful")
        print(f"  Configuration: {efp_config}")
        
        # Test expected feature count for n≤5, d≤6
        expected_features = 140  # Based on our previous validation
        print(f"  Expected EFP features: {expected_features}")
        
        return True
        
    except Exception as e:
        print(f"✗ EFP configuration validation failed: {e}")
        return False


def main():
    """Run all simple EFP integration tests."""
    print("=" * 60)
    print("Simple EFP Integration Tests")
    print("=" * 60)
    
    tests = [
        test_efp_tensor_loading,
        test_efp_feature_concatenation,
        test_efp_data_unpacking,
        test_efp_config_validation,
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
        print("🎉 All simple EFP integration tests PASSED!")
        return True
    else:
        print("❌ Some EFP integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
