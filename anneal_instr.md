# Hyperparameter Annealing in BEAD

This document explains how to use BEAD's hyperparameter annealing capabilities to dynamically adjust parameters during training.

## Overview

Hyperparameter annealing (gradually changing hyperparameters during training) can help improve model performance by adapting parameters as training progresses. BEAD supports several annealing strategies:

1. **CONSTANT_PACE**: Gradually change a parameter from a start value to an end value over a specified number of steps
2. **TRIGGER_BASED**: Change a parameter value when triggered by signals like early stopping or learning rate scheduling
3. **SCHEDULED**: Change a parameter according to a predefined schedule based on epochs

All annealing strategies are compatible with Distributed Data Parallel (DDP) training, where decisions are made by rank 0 and broadcasted to all processes.

## Configuration

To enable annealing for hyperparameters, add an `annealing_params` dictionary to your project's config file. The structure is:

```python
def set_config(c):
    # ... other config settings ...
    
    # Annealing configuration
    c.annealing_params = {
        "parameter_name": {
            # Annealing settings for this parameter
        }
    }
```

## Annealing Strategies

### CONSTANT_PACE Strategy

Gradually changes a parameter's value from a start value to an end value over a specified number of steps.

```python
"parameter_name": {
    "strategy": "CONSTANT_PACE",
    "start_value": 0.1,           # Initial value
    "end_value": 0.01,            # Final value to reach
    "total_steps": 10             # Number of steps to reach end_value
}
```

### TRIGGER_BASED Strategy

Changes parameter values when triggered by signals like early stopping or learning rate scheduling.

```python
"parameter_name": {
    "strategy": "TRIGGER_BASED",
    "values": [0.001, 0.005, 0.01, 0.05],  # Values to cycle through
    "trigger_source": "early_stopper_counter",  # Trigger to watch for
    "current_index": 0               # Starting index in the values list
}
```

Supported trigger sources:
- `early_stopper_counter`: Triggers when early stopper counter increases
- `early_stopper_half_patience`: Triggers when early stopper counter reaches half of its patience value
- `early_stopper_triggered`: Triggers when early stopping is activated
- `lr_scheduler_triggered`: Triggers when learning rate scheduler reduces learning rate

### SCHEDULED Strategy

Changes parameter values according to a predefined schedule based on training epochs.

```python
"parameter_name": {
    "strategy": "SCHEDULED",
    "schedule": {
        5: 128,   # Change to 128 at epoch 5
        10: 192,  # Change to 192 at epoch 10
        15: 256   # Change to 256 at epoch 15
    }
}
```

## Example Configuration

Here's a complete example showing different annealing strategies:

```python
def set_config(c):
    # ... other config settings ...
    
    # Annealing configuration
    c.annealing_params = {
        # Example 1: Anneal contrastive_temperature with constant pace
        "contrastive_temperature": {
            "strategy": "CONSTANT_PACE",
            "start_value": 0.1,
            "end_value": 0.01,
            "total_steps": 10  # Gradually reduce over first 10 epochs
        },
        
        # Example 2: Anneal regularization parameter based on early stopper trigger
        "reg_param": {
            "strategy": "TRIGGER_BASED",
            "values": [0.001, 0.005, 0.01, 0.05],  # Values to cycle through
            "trigger_source": "early_stopper_counter",  # Will increase when counter increases
            "current_index": 0
        },
        
        # Example 3: Anneal batch size based on a schedule
        "batch_size": {
            "strategy": "SCHEDULED",
            "schedule": {
                5: 128,   # Change to 128 at epoch 5
                10: 192,  # Change to 192 at epoch 10
                15: 256   # Change to 256 at epoch 15
            }
        }
    }
```

## Advanced Usage

### Annealing Optimizer Parameters

To anneal optimizer parameters (like weight decay or momentum), use dot notation to specify nested attributes:

```python
"optimizer.param_groups.0.weight_decay": {
    "strategy": "CONSTANT_PACE",
    "start_value": 0.01,
    "end_value": 0.001,
    "total_steps": 5
}
```

### Custom Objects

By default, parameters are assumed to be attributes of the config object. To anneal parameters in other objects, specify the target object:

```python
"weight_decay": {
    "object": optimizer,
    "attr_name": "param_groups.0.weight_decay",
    "strategy": "CONSTANT_PACE",
    "start_value": 0.01,
    "end_value": 0.001,
    "total_steps": 5
}
```

## Implementation Details

The annealing functionality is implemented in the `AnnealingManager` class in `bead/src/utils/annealing.py`. This class is initialized with the configuration object and automatically manages parameter annealing during training.

To use the annealing manager in your training loop:

```python
# Initialize annealing manager
annealing_manager = AnnealingManager(config)

# In your training loop
for epoch in range(config.epochs):
    # Train and validate
    train_loss = train(...)
    val_loss = validate(...)
    
    # Step the annealing manager with current epoch and metrics
    metrics = {
        "early_stopper_counter": early_stopper.counter,
        "early_stopper_triggered": early_stopper.early_stop,
        "lr_scheduler_triggered": lr_scheduler.reduced_lr,
    }
    annealed_params = annealing_manager.step(epoch=epoch, metrics=metrics)
    
    # Log annealed parameters if needed
    if annealed_params and (rank == 0 or not config.is_ddp_active):
        print(f"Annealed parameters at epoch {epoch}: {annealed_params}")
```

## Default Configuration

By default, BEAD sets up annealing for `reg_param` and `contrastive_weight` parameters to trigger when the early stopper counter reaches half of its patience value.
