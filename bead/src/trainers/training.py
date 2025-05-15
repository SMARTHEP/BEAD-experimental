"""
Training functionality for anomaly detection models.

This module provides functionality for training neural network models for anomaly detection.
It includes functions for model fitting, validation, and the main training loop that handles
data loading, model initialization, optimization, and early stopping.

Functions:
    fit: Performs one epoch of training on the training set.
    validate: Evaluates the model on the validation set.
    seed_worker: Sets seeds for workers to ensure reproducibility.
    train: Main function that handles the entire training process.
"""

import os
import random
import time
import warnings

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from ..utils import diagnostics, helper

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def fit(
    config,
    model,
    dataloader,
    loss_fn,
    reg_param,
    optimizer,
    device,
    scaler,
    is_ddp_active,
    local_rank,
):
    """
    This function trains the model on the train set. It computes the losses and does the backwards propagation, and updates the optimizer as well.

    Args:
        config (dataClass): Base class selecting user inputs
        model (modelObject): The model you wish to train
        dataloader (torch.DataLoader): Defines the batched data which the model is trained on
        loss_fn (lossObject): Defines the loss function used to train the model
        reg_param (float): Determines proportionality constant to balance different components of the loss.
        optimizer (torch.optim): Chooses optimizer for gradient descent.
        device (torch.device): Chooses which device to use with torch
        scaler (torch.cuda.amp.GradScaler): Scaler for mixed precision training
        is_ddp_active (bool): Flag indicating if DDP is active
        local_rank (int): Local rank of the process in DDP

    Returns:
        list, model object: Training losses, Epoch_loss and trained model
    """
    model = model.module if isinstance(model, DDP) else model
    model.train()
    running_loss = 0.0

    # Use tqdm only on rank 0 to avoid multiple progress bars
    if not is_ddp_active or local_rank == 0:
        pbar = tqdm(dataloader, desc="Training Batch")
    else:
        pbar = dataloader

    for idx, batch in enumerate(pbar):
        inputs, labels = batch
        inputs = inputs.to(
            device, non_blocking=True
        )  # non_blocking=True only if pin_memory=True
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16 if config.use_amp else torch.float32,
            enabled=(config.use_amp and torch.cuda.is_available()),
        ):
            out = helper.call_forward(model, inputs)
            recon, mu, logvar, ldj, z0, zk = out
            losses = loss_fn.calculate(
                recon=recon,
                target=inputs,
                mu=mu,
                logvar=logvar,
                parameters=model.parameters(),
                log_det_jacobian=ldj
                if hasattr(ldj, "item")
                else torch.tensor(0.0, device=device),
            )
        loss, *_ = losses

        scaler.scale(loss).backward()
        # Optional: Gradient clipping for faster training - performance TBD
        # scaler.unscale_(optimizer) # Unscale before clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    epoch_loss_train = running_loss / (idx + 1)
    epoch_loss_tensor = torch.tensor(epoch_loss_train, device=device)

    if is_ddp_active:  # Average loss across all processes
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)

    if not is_ddp_active or local_rank == 0:
        print(f"# Training Loss: {epoch_loss_tensor.item():.6f}")

    return losses, epoch_loss_tensor, model


def validate(
    config, model, dataloader, loss_fn, reg_param, device, is_ddp_active, local_rank
):
    """
    Function used to validate the training. Not necessary for doing compression, but gives a good indication of wether the model selected is a good fit or not.

    Args:
        config (dataClass): Base class selecting user inputs
        model (modelObject): Defines the model one wants to validate. The model used here is passed directly from `fit()`.
        dataloader (torch.DataLoader): Defines the batched data which the model is validated on
        loss_fn (lossObject): Defines the loss function used to train the model
        reg_param (float): Determines proportionality constant to balance different components of the loss.
        device (torch.device): Chooses which device to use with torch
        is_ddp_active (bool): Flag indicating if DDP is active
        local_rank (int): Local rank of the process in DDP

    Returns:
        float: Validation loss
    """
    model = model.module if isinstance(model, DDP) else model

    model.eval()
    running_loss = 0.0

    if not is_ddp_active or local_rank == 0:
        pbar = tqdm(dataloader, desc="Validation Batch")
    else:
        pbar = dataloader

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16 if config.use_amp else torch.float32,
                enabled=(config.use_amp and torch.cuda.is_available()),
            ):
                out = helper.call_forward(model, inputs)
                recon, mu, logvar, ldj, z0, zk = out
                losses = loss_fn.calculate(
                    recon=recon,
                    target=inputs,
                    mu=mu,
                    logvar=logvar,
                    parameters=model.parameters(),
                    log_det_jacobian=ldj
                    if hasattr(ldj, "item")
                    else torch.tensor(0.0, device=device),
                )
            loss, *_ = losses
            running_loss += loss.item()

    epoch_loss_val = running_loss / (idx + 1)
    epoch_loss_tensor = torch.tensor(epoch_loss_val, device=device)

    if is_ddp_active:
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)

    if not is_ddp_active or local_rank == 0:
        print(f"# Validation Loss: {epoch_loss_tensor.item():.6f}")

    return losses, epoch_loss_tensor


def seed_worker(worker_id):
    """PyTorch implementation to fix the seeds

    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    data,
    labels,
    output_path,
    config,
    verbose: bool = False,
):
    """
    Processes the entire training loop by calling the `fit()` and `validate()`. Appart from this, this is the main function where the data is converted
    to the correct type for it to be trained, via `torch.Tensor()`. Furthermore, the batching is also done here, based on `config.batch_size`,
    and it is the `torch.utils.data.DataLoader` doing the splitting.
    Torch AMP and DDP are also implemented here, if the user has selected them in the config file. Applying either `EarlyStopping` or `LR Scheduler` is also done here, all based on their respective `config` arguments.
    For reproducibility, the seeds can also be fixed in this function using the deterministic_algorithm `config` flag.

    Args:
        model (modelObject): The model you wish to train
        data (Tuple): Tuple containing the training and validation data
        labels (Tuple): Tuple containing the training and validation labels
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints additional information during training

    Returns:
        modelObject: fully trained model ready to perform inference
    """

    # Get DDP parameters from config
    is_ddp_active = config.is_ddp_active
    local_rank = config.local_rank
    world_size = config.world_size

    # Get the device and move tensors to the device
    device = helper.get_device(config)

    # Reshape tensors to pass to conv layers
    if "ConvVAE" in config.model_name or "ConvAE" in config.model_name:
        (
            events_train,
            jets_train,
            constituents_train,
            events_val,
            jets_val,
            constituents_val,
        ) = [x.unsqueeze(1).float() for x in data]

        data = (
            events_train,
            jets_train,
            constituents_train,
            events_val,
            jets_val,
            constituents_val,
        )

    (
        events_train,
        jets_train,
        constituents_train,
        events_val,
        jets_val,
        constituents_val,
    ) = data

    (
        events_train_label,
        jets_train_label,
        constituents_train_label,
        events_val_label,
        jets_val_label,
        constituents_val_label,
    ) = labels

    # Create datasets
    ds = helper.create_datasets(*data, *labels)

    if verbose and (not is_ddp_active or local_rank == 0):
        # Print input shapes
        print("Events - Training set shape:         ", events_train.shape)
        print("Events - Validation set shape:       ", events_val.shape)
        print("Jets - Training set shape:           ", jets_train.shape)
        print("Jets - Validation set shape:         ", jets_val.shape)
        print("Constituents - Training set shape:   ", constituents_train.shape)
        print("Constituents - Validation set shape: ", constituents_val.shape)

        # Print label shapes
        print("Events - Training set labels shape:         ", events_train_label.shape)
        print("Events - Validation set labels shape:       ", events_val_label.shape)
        print("Jets - Training set labels shape:           ", jets_train_label.shape)
        print("Jets - Validation set labels shape:         ", jets_val_label.shape)
        print(
            "Constituents - Training set labels shape:   ",
            constituents_train_label.shape,
        )
        print(
            "Constituents - Validation set labels shape: ", constituents_val_label.shape
        )

    # Calculate the input shapes to initialize the model
    in_shape = helper.calculate_in_shape(data, config)

    # Instantiate and Initialize the model
    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Calculated input shape: {in_shape}")
        print(f"Intitalizing Model with Latent Size - {config.latent_space_size}")
    model = helper.model_init(in_shape, config)
    if verbose and (not is_ddp_active or local_rank == 0):
        if config.model_init == "xavier":
            print("Model initialized using Xavier initialization")
        else:
            print("Model initialized using default PyTorch initialization")
        print(f"Model architecture:\n{model}")

    model = model.to(device)

    if is_ddp_active:
        # find_unused_parameters can be True if parts of the model aren't used in forward pass which can happen with complex models or conditional execution paths.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                f"Distributed Data Parallel (DDP) initialized. Running on {world_size} GPUs."
            )

    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Device used for training: {device}")
        print("Model moved to device")

    if verbose and (not is_ddp_active or local_rank == 0):
        print(
            "Loading data into DataLoader and using batch size of ", config.batch_size
        )

    # DataLoaders with DistributedSampler
    train_dataset_map = {
        "event": ds["events_train"],
        "jet": ds["jets_train"],
        "constituent": ds["constituents_train"],
    }
    val_dataset_map = {
        "event": ds["events_val"],
        "jet": ds["jets_val"],
        "constituent": ds["constituents_val"],
    }

    train_ds_selected = train_dataset_map[config.input_level]
    val_ds_selected = val_dataset_map[config.input_level]

    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Input data is of {config.input_level} level")

    train_sampler = None
    val_sampler = None
    shuffle_train = not is_ddp_active  # sampler handles it for DDP

    if is_ddp_active:
        train_sampler = DistributedSampler(
            train_ds_selected,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds_selected,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=True,
        )

    common_loader_args = {
        "batch_size": config.batch_size,
        "drop_last": True,  # Important for DDP to have same number of batches per GPU
        "num_workers": config.parallel_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker if config.deterministic_algorithm else None,
        "generator": torch.Generator().manual_seed(0)
        if config.deterministic_algorithm
        else None,
    }

    train_dl = DataLoader(
        train_ds_selected,
        sampler=train_sampler,
        shuffle=shuffle_train,
        **common_loader_args,
    )
    valid_dl = DataLoader(
        val_ds_selected,
        sampler=val_sampler,
        shuffle=False,
        **common_loader_args,
    )

    # Select Loss Function
    try:
        loss_object = helper.get_loss(config.loss_function)
        loss_fn = loss_object(config=config)
        if verbose and (not is_ddp_active or local_rank == 0):
            print(f"Loss Function: {config.loss_function}")
    except ValueError as e:
        print(e)

    # Select Optimizer
    try:
        optimizer = helper.get_optimizer(
            config.optimizer, model.parameters(), lr=config.lr
        )
        if verbose and (not is_ddp_active or local_rank == 0):
            print(f"Optimizer: {config.optimizer}")
    except ValueError as e:
        print(e)

    # AMP GradScaler
    amp_scaler = torch.amp.GradScaler(
        enabled=(config.use_amp and torch.cuda.is_available())
    )

    # Activate early stopping
    if config.early_stopping:
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                "Early stopping is activated with patience of ",
                config.early_stopping_patience,
            )
        early_stopper = helper.EarlyStopping(
            patience=config.early_stopping_patience, min_delta=config.min_delta
        )

    # Activate LR Scheduler
    if config.lr_scheduler:
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                "Learning rate scheduler is activated with patience of ",
                config.lr_scheduler_patience,
            )
        lr_scheduler = helper.LRScheduler(
            optimizer=optimizer, patience=config.lr_scheduler_patience
        )

    # Training and Validation of the model
    train_loss_data = []
    val_loss_data = []
    train_loss = []
    val_loss = []
    start_time = time.time()

    # Registering hooks for activation extraction
    if config.activation_extraction and not is_ddp_active:
        hooks = model.store_hooks()

    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Beginning training for {config.epochs} epochs")

    for epoch in range(config.epochs):
        if is_ddp_active and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if not is_ddp_active or local_rank == 0:  # Using only rank 0 for logging
            print(f"Epoch {epoch + 1} / {config.epochs}")

        train_losses, train_epoch_loss, model = fit(
            config=config,
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            reg_param=config.reg_param,
            optimizer=optimizer,
            device=device,
            scaler=amp_scaler,
            is_ddp_active=is_ddp_active,
            local_rank=local_rank,
        )

        # For simplicity, rank 0 does most of the logging and state updates
        if not is_ddp_active or local_rank == 0:
            train_loss.append(train_epoch_loss.detach().cpu().numpy())
            train_loss_data.append(train_losses)

            if 1 - config.train_size > 0:
                val_losses, val_epoch_loss = validate(
                    config=config,
                    model=model,
                    dataloader=valid_dl,
                    loss_fn=loss_fn,
                    reg_param=config.reg_param,
                    device=device,
                    is_ddp_active=is_ddp_active,
                    local_rank=local_rank,
                )
                val_loss.append(val_epoch_loss.detach().cpu().numpy())
                val_loss_data.append(val_losses)
            else:
                val_epoch_loss = train_epoch_loss
                val_loss.append(val_epoch_loss.detach().cpu().numpy())
                val_loss_data.append(train_losses)

            if config.lr_scheduler:
                lr_scheduler(val_epoch_loss.item())

            if config.intermittent_model_saving:
                if epoch % config.intermittent_saving_patience == 0:
                    path = os.path.join(
                        output_path, "models", f"model_epoch_{epoch}.pt"
                    )
                    helper.save_model(
                        model, path, config
                    )  # Pass config for DDP awareness

            if config.early_stopping:
                early_stopper(val_epoch_loss.item())
                if early_stopper.early_stop:
                    if verbose:
                        print(
                            f"Rank {local_rank}: Early stopping at epoch {epoch + 1}; will signal other ranks."
                        )

        # Rank 0 determines if stopping is needed
        if local_rank == 0:
            stop_signal = (
                1.0 if config.early_stopping and early_stopper.early_stop else 0.0
            )
            stop_signal_tensor = torch.tensor(
                [stop_signal], dtype=torch.float32, device=device
            )
        else:
            # Other ranks prepare a tensor to receive the broadcast
            stop_signal_tensor = torch.empty(1, dtype=torch.float32, device=device)

        # All ranks participate in this broadcast. Rank 0 sends, others receive.
        dist.broadcast(stop_signal_tensor, src=0)

        # All ranks check the signal and break if needed
        if stop_signal_tensor.item() == 1.0:
            if verbose:  # Log on all ranks that are stopping
                print(
                    f"Rank {local_rank}: Early stopping signal received at epoch {epoch + 1}. Breaking training loop."
                )
            break  # All ranks break out of the epoch loop

    # model_to_save should be the DDP-wrapped model if DDP is active
    model_to_save = model

    if not is_ddp_active or local_rank == 0:
        final_model_for_eval = model.module if isinstance(model, DDP) else model
        final_model_for_eval.eval()  # Set the specific model for eval
        # Running final pass on full training data
        train_dl_final = DataLoader(
            train_ds_selected,
            sampler=None,
            shuffle=shuffle_train,
            **common_loader_args,
        )
        # Run a final forward pass on training data to get final latent space representation
        # Output Lists
        mu_data = []
        logvar_data = []
        z0_data = []
        zk_data = []
        log_det_jacobian_data = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(train_dl_final, desc="Final Pass")):
                inputs, labels = batch
                inputs = inputs.to(device)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(config.use_amp and torch.cuda.is_available()),
                ):
                    # Forward pass
                    out = helper.call_forward(final_model_for_eval, inputs)
                    recon, mu, logvar, ldj, z0, zk = out

                mu_data.append(mu.detach().cpu().numpy())
                logvar_data.append(logvar.detach().cpu().numpy())
                # Handling ldj extra carefully
                if hasattr(ldj, "detach"):
                    log_det_jacobian_data.append(ldj.detach().cpu().numpy())
                elif isinstance(ldj, (float, int, np.number)):
                    log_det_jacobian_data.append(np.array(ldj))
                else:  # Fallback for unexpected types
                    log_det_jacobian_data.append(np.array(0.0))
                z0_data.append(z0.detach().cpu().numpy())
                zk_data.append(zk.detach().cpu().numpy())

        # Save the final model using model_to_save
        helper.save_model(
            model_to_save, os.path.join(output_path, "models", "model.pt"), config
        )
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                f"Model saved to path: {os.path.join(output_path, 'models', 'model.pt')}"
            )
        # Save loss data
        save_dir = os.path.join(output_path, "results")
        np.save(
            os.path.join(save_dir, "train_epoch_loss_data.npy"),
            np.array(train_loss),
        )
        np.save(
            os.path.join(save_dir, "val_epoch_loss_data.npy"),
            np.array(val_loss),
        )

        # Convert all the data to numpy arrays
        (
            mu_data,
            logvar_data,
            z0_data,
            zk_data,
            log_det_jacobian_data,
        ) = [
            np.array(x)
            for x in [
                mu_data,
                logvar_data,
                z0_data,
                zk_data,
                log_det_jacobian_data,
            ]
        ]

        # Reshape the data if conv-models were used
        if "ConvVAE" in config.model_name or "ConvAE" in config.model_name:
            (mu_data, logvar_data, z0_data, zk_data) = [
                x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                for x in [mu_data, logvar_data, z0_data, zk_data]
            ]

        # Save all the data
        save_dir = os.path.join(output_path, "results")
        np.save(
            os.path.join(save_dir, "train_mu_data.npy"),
            mu_data,
        )
        np.save(
            os.path.join(save_dir, "train_logvar_data.npy"),
            logvar_data,
        )
        np.save(
            os.path.join(save_dir, "train_z0_data.npy"),
            z0_data,
        )
        np.save(
            os.path.join(save_dir, "train_zk_data.npy"),
            zk_data,
        )
        np.save(
            os.path.join(save_dir, "train_log_det_jacobian_data.npy"),
            log_det_jacobian_data,
        )

        helper.save_loss_components(
            loss_data=train_loss_data,
            component_names=loss_fn.component_names,
            suffix="train",
            save_dir=save_dir,
        )
        helper.save_loss_components(
            loss_data=val_loss_data,
            component_names=loss_fn.component_names,
            suffix="val",
            save_dir=save_dir,
        )

        if verbose and (not is_ddp_active or local_rank == 0):
            print("Loss data saved to path: ", save_dir)

    # If activations are extracted, this needs to be DDP aware or run post-training on a single GPU
    if config.activation_extraction and (not is_ddp_active or local_rank == 0):
        # model.module if DDP, else model
        actual_model = model.module if is_ddp_active else model
        activations = diagnostics.dict_to_square_matrix(actual_model.get_activations())
        model.detach_hooks(hooks)
        np.save(os.path.join(output_path, "models", "activations.npy"), activations)

    end_time = time.time()
    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Training the model took {(end_time - start_time) / 60:.3} minutes")

    return model
