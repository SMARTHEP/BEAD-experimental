#!/usr/bin/env python3
"""
Unit tests for EFP model integration utilities.

This module tests the EFP integration helper functions that prepare model inputs
by concatenating EFP features with base features.
"""

import os
import sys
import unittest
import torch
from dataclasses import dataclass

# Add BEAD source to path
sys.path.insert(0, os.path.join(os.getcwd(), 'bead', 'src'))

from utils.efp_integration import (
    prepare_model_input,
    get_combined_input_dim,
    _should_use_efp,
    _validate_input_shapes
)


@dataclass
class MockConfig:
    """Mock configuration class for testing."""
    enable_efp: bool = True


class TestEFPModelIntegration(unittest.TestCase):
    """Test cases for EFP model integration utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.base_dim = 1500
        self.n_jets = 10
        self.n_efp = 140
        
        # Create test tensors
        self.base_features = torch.randn(self.batch_size, self.base_dim)
        self.efp_features = torch.randn(self.batch_size, self.n_jets, self.n_efp)
        
        # Create test configs
        self.config_enabled = MockConfig(enable_efp=True)
        self.config_disabled = MockConfig(enable_efp=False)
    
    def test_prepare_model_input_with_efp_enabled(self):
        """Test model input preparation with EFP features enabled."""
        result = prepare_model_input(
            self.base_features, 
            self.efp_features, 
            self.config_enabled
        )
        
        expected_dim = self.base_dim + (self.n_jets * self.n_efp)
        self.assertEqual(result.shape, (self.batch_size, expected_dim))
        
        # Check that base features are preserved at the beginning
        torch.testing.assert_close(
            result[:, :self.base_dim], 
            self.base_features,
            msg="Base features should be preserved"
        )
    
    def test_prepare_model_input_with_efp_disabled(self):
        """Test model input preparation with EFP features disabled."""
        result = prepare_model_input(
            self.base_features, 
            self.efp_features, 
            self.config_disabled
        )
        
        # Should return unchanged base features
        torch.testing.assert_close(
            result, 
            self.base_features,
            msg="Should return unchanged base features when EFP disabled"
        )
    
    def test_prepare_model_input_no_efp_features(self):
        """Test model input preparation when no EFP features provided."""
        result = prepare_model_input(
            self.base_features, 
            None, 
            self.config_enabled
        )
        
        # Should return unchanged base features
        torch.testing.assert_close(
            result, 
            self.base_features,
            msg="Should return unchanged base features when EFP features are None"
        )
    
    def test_prepare_model_input_no_config(self):
        """Test model input preparation when no config provided."""
        result = prepare_model_input(
            self.base_features, 
            self.efp_features, 
            None
        )
        
        # Should return unchanged base features
        torch.testing.assert_close(
            result, 
            self.base_features,
            msg="Should return unchanged base features when config is None"
        )
    
    def test_validate_input_shapes_valid(self):
        """Test input shape validation with valid shapes."""
        # Should not raise any exception
        _validate_input_shapes(self.base_features, self.efp_features)
    
    def test_validate_input_shapes_invalid_base_dim(self):
        """Test input shape validation with invalid base features dimension."""
        invalid_base = torch.randn(self.batch_size, self.base_dim, 10)  # 3D instead of 2D
        
        with self.assertRaises(ValueError) as context:
            _validate_input_shapes(invalid_base, self.efp_features)
        
        self.assertIn("base_features must be 2D", str(context.exception))
    
    def test_validate_input_shapes_invalid_efp_dim(self):
        """Test input shape validation with invalid EFP features dimension."""
        invalid_efp = torch.randn(self.batch_size, self.n_jets * self.n_efp)  # 2D instead of 3D
        
        with self.assertRaises(ValueError) as context:
            _validate_input_shapes(self.base_features, invalid_efp)
        
        self.assertIn("efp_features must be 3D", str(context.exception))
    
    def test_validate_input_shapes_batch_size_mismatch(self):
        """Test input shape validation with mismatched batch sizes."""
        mismatched_efp = torch.randn(self.batch_size + 5, self.n_jets, self.n_efp)
        
        with self.assertRaises(ValueError) as context:
            _validate_input_shapes(self.base_features, mismatched_efp)
        
        self.assertIn("Batch size mismatch", str(context.exception))
    
    def test_should_use_efp_enabled(self):
        """Test EFP usage check with enabled config."""
        self.assertTrue(_should_use_efp(self.config_enabled))
    
    def test_should_use_efp_disabled(self):
        """Test EFP usage check with disabled config."""
        self.assertFalse(_should_use_efp(self.config_disabled))
    
    def test_should_use_efp_no_config(self):
        """Test EFP usage check with no config."""
        self.assertFalse(_should_use_efp(None))
    
    def test_should_use_efp_missing_attribute(self):
        """Test EFP usage check with config missing enable_efp attribute."""
        @dataclass
        class IncompleteConfig:
            other_param: str = "test"
        
        incomplete_config = IncompleteConfig()
        self.assertFalse(_should_use_efp(incomplete_config))
    
    def test_get_combined_input_dim_enabled(self):
        """Test combined input dimension calculation with EFP enabled."""
        result = get_combined_input_dim(
            self.base_dim, 
            self.n_jets, 
            self.n_efp, 
            self.config_enabled
        )
        
        expected = self.base_dim + (self.n_jets * self.n_efp)
        self.assertEqual(result, expected)
    
    def test_get_combined_input_dim_disabled(self):
        """Test combined input dimension calculation with EFP disabled."""
        result = get_combined_input_dim(
            self.base_dim, 
            self.n_jets, 
            self.n_efp, 
            self.config_disabled
        )
        
        self.assertEqual(result, self.base_dim)
    
    def test_efp_flattening_consistency(self):
        """Test that EFP features are flattened consistently."""
        result = prepare_model_input(
            self.base_features, 
            self.efp_features, 
            self.config_enabled
        )
        
        # Extract the EFP portion
        efp_portion = result[:, self.base_dim:]
        
        # Manually flatten EFP features and compare
        expected_efp_flat = self.efp_features.view(self.batch_size, -1)
        
        torch.testing.assert_close(
            efp_portion, 
            expected_efp_flat,
            msg="EFP features should be flattened consistently"
        )
    
    def test_gradient_flow(self):
        """Test that gradients flow through the concatenation operation."""
        base_features = torch.randn(self.batch_size, self.base_dim, requires_grad=True)
        efp_features = torch.randn(self.batch_size, self.n_jets, self.n_efp, requires_grad=True)
        
        result = prepare_model_input(base_features, efp_features, self.config_enabled)
        
        # Compute a simple loss and backpropagate
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(base_features.grad)
        self.assertIsNotNone(efp_features.grad)
        
        # Check gradient shapes
        self.assertEqual(base_features.grad.shape, base_features.shape)
        self.assertEqual(efp_features.grad.shape, efp_features.shape)


if __name__ == '__main__':
    unittest.main()
