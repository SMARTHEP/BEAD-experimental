#!/usr/bin/env python3
"""
Unit tests for the hyperparameter annealing functionality.

These tests verify that the annealing manager correctly handles different
annealing strategies and properly updates parameters according to the
specified configuration.
"""

import unittest
import torch
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from bead.src.utils.annealing import AnnealingManager, AnnealingStrategy


class TestAnnealingBasic(unittest.TestCase):
    """Test basic annealing functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = SimpleNamespace()
        self.config.epochs = 10
        self.config.lr = 0.1
        self.config.batch_size = 32
        self.config.beta = 1.0
        
        # Add annealing parameters
        self.config.annealing_params = {
            "lr": {
                "strategy": "CONSTANT_PACE",
                "start_value": 0.1,
                "end_value": 0.001,
                "total_steps": 5
            },
            "beta": {
                "strategy": "SCHEDULED",
                "schedule": {
                    2: 1.5,
                    5: 2.0,
                    8: 2.5
                }
            }
        }
        
        # Create annealing manager
        self.manager = AnnealingManager(self.config)
    
    def test_constant_pace_annealing(self):
        """Test that constant pace annealing works correctly."""
        # Initial values
        self.assertEqual(self.config.lr, 0.1)
        
        # Step 1: Should change lr
        annealed = self.manager.step(epoch=1)
        self.assertIn("lr", annealed)
        expected_lr = 0.1 - 0.099 * (1 / 5)  # start - diff * step/total_steps
        self.assertAlmostEqual(self.config.lr, expected_lr, places=5)
        
        # Step 4: Should be on 4th step of annealing
        self.manager.step(epoch=2)
        self.manager.step(epoch=3)
        annealed = self.manager.step(epoch=4)
        self.assertIn("lr", annealed)
        expected_lr = 0.1 - 0.099 * (4 / 5)  # start - diff * step/total_steps
        self.assertAlmostEqual(self.config.lr, expected_lr, places=5)
        
        # Step 5: Should be at end value
        annealed = self.manager.step(epoch=5)
        self.assertIn("lr", annealed)
        self.assertAlmostEqual(self.config.lr, 0.001, places=5)
        
        # Step 6: No more changes
        annealed = self.manager.step(epoch=6)
        self.assertNotIn("lr", annealed)
        self.assertAlmostEqual(self.config.lr, 0.001, places=5)
    
    def test_scheduled_annealing(self):
        """Test that scheduled annealing works correctly."""
        # Initial values
        self.assertEqual(self.config.beta, 1.0)
        
        # No change at epoch 0 and 1
        self.manager.step(epoch=0)
        annealed = self.manager.step(epoch=1)
        self.assertNotIn("beta", annealed)
        self.assertEqual(self.config.beta, 1.0)
        
        # Change at epoch 2
        annealed = self.manager.step(epoch=2)
        self.assertIn("beta", annealed)
        self.assertEqual(self.config.beta, 1.5)
        
        # No change at epochs 3-4
        self.manager.step(epoch=3)
        annealed = self.manager.step(epoch=4)
        self.assertNotIn("beta", annealed)
        self.assertEqual(self.config.beta, 1.5)
        
        # Change at epoch 5
        annealed = self.manager.step(epoch=5)
        self.assertIn("beta", annealed)
        self.assertEqual(self.config.beta, 2.0)
        
        # Change at epoch 8
        self.manager.step(epoch=6)
        self.manager.step(epoch=7)
        annealed = self.manager.step(epoch=8)
        self.assertIn("beta", annealed)
        self.assertEqual(self.config.beta, 2.5)
        
        # No change after that
        annealed = self.manager.step(epoch=9)
        self.assertNotIn("beta", annealed)
        self.assertEqual(self.config.beta, 2.5)


class TestAnnealingTriggerBased(unittest.TestCase):
    """Test trigger-based annealing functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = SimpleNamespace()
        self.config.epochs = 10
        self.config.dropout = 0.5
        self.config.reg_param = 0.01
        
        # Add annealing parameters
        self.config.annealing_params = {
            "dropout": {
                "strategy": "TRIGGER_BASED",
                "values": [0.5, 0.3, 0.2, 0.1],
                "trigger_source": "early_stopper_triggered"
            },
            "reg_param": {
                "strategy": "TRIGGER_BASED",
                "values": [0.01, 0.05, 0.1, 0.5],
                "trigger_source": "lr_scheduler_triggered"
            }
        }
        
        # Create annealing manager
        self.manager = AnnealingManager(self.config)
    
    def test_trigger_based_annealing(self):
        """Test that trigger-based annealing works correctly."""
        # Initial values
        self.assertEqual(self.config.dropout, 0.5)
        self.assertEqual(self.config.reg_param, 0.01)
        
        # Trigger early_stopper (dropout should change)
        metrics = {"early_stopper_triggered": True, "lr_scheduler_triggered": False}
        annealed = self.manager.step(metrics=metrics)
        self.assertIn("dropout", annealed)
        self.assertEqual(self.config.dropout, 0.3)
        self.assertNotIn("reg_param", annealed)
        
        # Trigger lr_scheduler (reg_param should change)
        metrics = {"early_stopper_triggered": False, "lr_scheduler_triggered": True}
        annealed = self.manager.step(metrics=metrics)
        self.assertNotIn("dropout", annealed)
        self.assertIn("reg_param", annealed)
        self.assertEqual(self.config.reg_param, 0.05)
        
        # No triggers (no changes)
        metrics = {"early_stopper_triggered": False, "lr_scheduler_triggered": False}
        annealed = self.manager.step(metrics=metrics)
        self.assertFalse(annealed)
        
        # Both triggers (both should change)
        metrics = {"early_stopper_triggered": True, "lr_scheduler_triggered": True}
        annealed = self.manager.step(metrics=metrics)
        self.assertIn("dropout", annealed)
        self.assertIn("reg_param", annealed)
        self.assertEqual(self.config.dropout, 0.2)
        self.assertEqual(self.config.reg_param, 0.1)
    
    def test_trigger_based_value_wrap(self):
        """Test that trigger-based annealing stops at the last value."""
        # Initial values
        self.assertEqual(self.config.dropout, 0.5)
        
        # Trigger 4 times (should go through all values)
        metrics = {"early_stopper_triggered": True}
        
        # First trigger: 0.5 -> 0.3
        self.manager.step(metrics=metrics)
        self.assertEqual(self.config.dropout, 0.3)
        
        # Second trigger: 0.3 -> 0.2
        metrics = {"early_stopper_triggered": False}  # Reset trigger
        metrics = {"early_stopper_triggered": True}
        self.manager.step(metrics=metrics)
        self.assertEqual(self.config.dropout, 0.2)
        
        # Third trigger: 0.2 -> 0.1
        metrics = {"early_stopper_triggered": False}  # Reset trigger
        metrics = {"early_stopper_triggered": True}
        self.manager.step(metrics=metrics)
        self.assertEqual(self.config.dropout, 0.1)
        
        # Fourth trigger: 0.1 -> 0.1 (should stay at last value)
        metrics = {"early_stopper_triggered": False}  # Reset trigger
        metrics = {"early_stopper_triggered": True}
        annealed = self.manager.step(metrics=metrics)
        self.assertNotIn("dropout", annealed)
        self.assertEqual(self.config.dropout, 0.1)


class TestAnnealingNested(unittest.TestCase):
    """Test annealing of nested attributes."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = SimpleNamespace()
        self.config.epochs = 5
        
        # Create a nested structure
        self.config.model_params = SimpleNamespace()
        self.config.model_params.network = {
            "layers": [
                {"units": 64, "dropout": 0.3},
                {"units": 32, "dropout": 0.2}
            ]
        }
        
        # Add annealing parameters for nested attributes
        self.config.annealing_params = {
            # Anneal dropout in first layer
            "layer0_dropout": {
                "object": self.config.model_params,
                "attr_name": "network.layers.0.dropout",
                "strategy": "CONSTANT_PACE",
                "start_value": 0.3,
                "end_value": 0.1,
                "total_steps": 5
            },
            # Anneal units in second layer
            "layer1_units": {
                "object": self.config.model_params,
                "attr_name": "network.layers.1.units",
                "strategy": "SCHEDULED",
                "schedule": {
                    2: 48,
                    4: 64
                }
            }
        }
        
        # Create annealing manager
        self.manager = AnnealingManager(self.config)
    
    def test_nested_attribute_annealing(self):
        """Test that annealing of nested attributes works correctly."""
        # Initial values
        self.assertEqual(self.config.model_params.network["layers"][0]["dropout"], 0.3)
        self.assertEqual(self.config.model_params.network["layers"][1]["units"], 32)
        
        # Step 1: Should change dropout
        annealed = self.manager.step(epoch=1)
        self.assertIn("layer0_dropout", annealed)
        expected_dropout = 0.3 - 0.2 * (1 / 5)  # start - diff * step/total_steps
        self.assertAlmostEqual(
            self.config.model_params.network["layers"][0]["dropout"], 
            expected_dropout, 
            places=5
        )
        
        # Step 2: Should change dropout and units
        annealed = self.manager.step(epoch=2)
        self.assertIn("layer0_dropout", annealed)
        self.assertIn("layer1_units", annealed)
        expected_dropout = 0.3 - 0.2 * (2 / 5)
        self.assertAlmostEqual(
            self.config.model_params.network["layers"][0]["dropout"], 
            expected_dropout, 
            places=5
        )
        self.assertEqual(self.config.model_params.network["layers"][1]["units"], 48)
        
        # Step 4: Should change dropout and units
        self.manager.step(epoch=3)
        annealed = self.manager.step(epoch=4)
        self.assertIn("layer0_dropout", annealed)
        self.assertIn("layer1_units", annealed)
        expected_dropout = 0.3 - 0.2 * (4 / 5)
        self.assertAlmostEqual(
            self.config.model_params.network["layers"][0]["dropout"], 
            expected_dropout, 
            places=5
        )
        self.assertEqual(self.config.model_params.network["layers"][1]["units"], 64)


class TestAnnealingDDP(unittest.TestCase):
    """Test annealing with DDP (Distributed Data Parallel)."""
    
    @patch('torch.distributed.broadcast')
    def test_ddp_broadcast(self, mock_broadcast):
        """Test that parameters are broadcasted in DDP mode."""
        # Create a config with DDP enabled
        config = SimpleNamespace()
        config.epochs = 5
        config.is_ddp_active = True
        config.world_size = 2
        config.rank = 0
        config.lr = 0.1
        
        # Add annealing parameter
        config.annealing_params = {
            "lr": {
                "strategy": "CONSTANT_PACE",
                "start_value": 0.1,
                "end_value": 0.001,
                "total_steps": 5
            }
        }
        
        # Create annealing manager
        manager = AnnealingManager(config)
        
        # Call step method
        manager.step(epoch=1)
        
        # Check that broadcast was called
        self.assertTrue(mock_broadcast.called)
    
    @patch('torch.distributed.broadcast')
    def test_ddp_non_root_receive(self, mock_broadcast):
        """Test that non-root processes receive parameters."""
        # Create a config with DDP enabled (non-root rank)
        config = SimpleNamespace()
        config.epochs = 5
        config.is_ddp_active = True
        config.world_size = 2
        config.rank = 1
        config.lr = 0.1
        
        # Add annealing parameter
        config.annealing_params = {
            "lr": {
                "strategy": "CONSTANT_PACE",
                "start_value": 0.1,
                "end_value": 0.001,
                "total_steps": 5
            }
        }
        
        # Create annealing manager
        manager = AnnealingManager(config)
        
        # Prepare mock for _receive_annealed_params
        with patch.object(manager, '_receive_annealed_params') as mock_receive:
            mock_receive.return_value = {"lr": 0.08}
            
            # Call step method
            result = manager.step(epoch=1)
            
            # Check that _receive_annealed_params was called
            self.assertTrue(mock_receive.called)
            self.assertEqual(result, {"lr": 0.08})


if __name__ == '__main__':
    unittest.main()
