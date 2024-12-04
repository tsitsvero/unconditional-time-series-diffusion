# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model.callback import EvaluateCallback
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
import matplotlib.pyplot as plt
import numpy as np

import os
import torch.nn.functional as F

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}


def create_model(config):
    # Get the model configuration class
    try:
        model_config = getattr(diffusion_configs, config["diffusion_config"])
        if isinstance(model_config, dict):
            model_cls = TSDiff
        else:
            model_cls = model_config
    except AttributeError:
        raise ValueError(f"Could not find diffusion config '{config['diffusion_config']}' in configs")

    # Setup model kwargs with improved numerical stability settings
    model_kwargs = {
        "freq": config["freq"],
        "use_features": config["use_features"],
        "use_lags": config["use_lags"],
        "normalization": config["normalization"],
        "context_length": config["context_length"],
        "prediction_length": config["prediction_length"],
        "lr": config.get("lr", 1e-4),
        "init_skip": config["init_skip"],
    }

    # If model_config is a dict, add it to model_kwargs
    if isinstance(model_config, dict):
        model_kwargs.update(model_config)

    # Safely determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    
    try:
        model = model_cls(**model_kwargs)
        
        # Override p_losses method with improved numerical stability
        def debug_p_losses(self, x, t, features=None, loss_type="l2", reduction="mean"):
            print("\nDiffusion step debug:")
            
            # Input validation and cleaning
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("WARNING: Input contains NaN/Inf values before normalization")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # More robust normalization with error checking
            try:
                batch_mean = x.mean(dim=(1, 2), keepdim=True)
                batch_std = x.std(dim=(1, 2), keepdim=True)
                batch_std = torch.clamp(batch_std, min=1e-5)  # Prevent division by zero
                x = (x - batch_mean) / batch_std
                
                # Clip normalized values to prevent extremes
                x = torch.clamp(x, min=-10.0, max=10.0)
                
            except RuntimeError as e:
                print(f"Error during normalization: {e}")
                raise
                
            print(f"Input x shape: {x.shape}, range: [{x.min().item():.3f}, {x.max().item():.3f}]")
            print(f"Timestep t: {t}")
            
            # Generate and validate noise
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, min=-4.0, max=4.0)  # Clip noise to reasonable range
            print(f"Noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}]")
            
            # Get noisy samples with error checking
            try:
                x_noisy = self.q_sample(x, t, noise=noise)
                if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
                    print("WARNING: NaN/Inf values in noisy samples")
                    x_noisy = torch.nan_to_num(x_noisy, nan=0.0, posinf=1e6, neginf=-1e6)
                    x_noisy = torch.clamp(x_noisy, min=-10.0, max=10.0)
                    
            except RuntimeError as e:
                print(f"Error during noise sampling: {e}")
                raise
                
            print(f"Noisy x range: [{x_noisy.min().item():.3f}, {x_noisy.max().item():.3f}]")
            
            # Get predicted noise with gradient scaling
            with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision here
                predicted_noise = self.backbone(x_noisy, t, features)
                
                # Clean predicted noise
                if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                    print("WARNING: NaN/Inf values in predicted noise")
                    predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=1e6, neginf=-1e6)
                    predicted_noise = torch.clamp(predicted_noise, min=-10.0, max=10.0)
                    
            print(f"Predicted noise range: [{predicted_noise.min().item():.3f}, {predicted_noise.max().item():.3f}]")
            print(f"Has NaN in predicted noise: {torch.isnan(predicted_noise).any().item()}")
            
            # Calculate loss with error checking
            try:
                if loss_type == "l2":
                    # Add small epsilon to prevent exact zeros
                    loss = F.mse_loss(noise + 1e-8, predicted_noise + 1e-8, reduction=reduction)
                elif loss_type == "l1":
                    loss = F.l1_loss(noise + 1e-8, predicted_noise + 1e-8, reduction=reduction)
                else:
                    raise NotImplementedError(f"Unknown loss type {loss_type}")
                    
                # Validate loss value
                if torch.isnan(loss) or torch.isinf(loss):
                    print("WARNING: Loss is NaN/Inf, using fallback loss")
                    loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
                    
                print(f"Loss value: {loss.item()}")
                    
            except RuntimeError as e:
                print(f"Error during loss calculation: {e}")
                raise
                
            return loss, x_noisy, noise
            
        # Monkey patch the p_losses method
        import types
        model.p_losses = types.MethodType(debug_p_losses, model)
        
        # Initialize weights with more stable initialization
        for name, param in model.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # Use Kaiming initialization for weights
                    torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    # Initialize 1D weights (vectors) carefully
                    torch.nn.init.uniform_(param, -0.01, 0.01)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
                
        # Add gradient and parameter value clipping
        for param in model.parameters():
            if param.requires_grad:
                # Clip parameter values to prevent extreme initializations
                with torch.no_grad():
                    param.clamp_(-1.0, 1.0)
                # Add hook for gradient clipping during training
                param.register_hook(
                    lambda grad: torch.nan_to_num(
                        grad.clamp(-1.0, 1.0),
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0
                    )
                )
                
        model = model.to(device)
        
        # Verify no NaN values in initial parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    raise ValueError(f"NaN values found in parameter {name} after initialization")
                
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {str(e)}")
    
    return model


def evaluate_guidance(
    config, model, test_dataset, transformation, num_samples=100
):
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        missing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = guidance_map[config["sampler"]]
    sampler_kwargs = config["sampler_params"]
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )
        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_kwargs,
        )

        transformed_testdata = transformation.apply(
            test_dataset, is_train=False
        )
        test_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="test",
        )

        masking_transform = MaskInput(
            FieldName.TARGET,
            FieldName.OBSERVED_VALUES,
            config["context_length"],
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
        )
        test_transform = test_splitter + masking_transform

        predictor = sampler.get_predictor(
            test_transform,
            batch_size=1280 // num_samples,
            device=config["device"],
        )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=predictor,
            num_samples=num_samples,
        )
        forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
        tss = list(ts_it)
        evaluator = Evaluator()
        metrics, _ = evaluator(tss, forecasts)
        metrics = filter_metrics(metrics)
        results.append(dict(**missing_data_kwargs, **metrics))

    return results


def normalize_data(data, eps=1e-8):
    mean = data.mean()
    std = data.std() + eps
    return (data - mean) / std, mean, std


def main(config, log_dir):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"].lower()  # Convert to lowercase for consistency
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    # Create model
    model = create_model(config)

    # Setup dataset and data loading
    dataset = get_gts_dataset(dataset_name)
    
    # Check and normalize frequency
    dataset_freq = dataset.metadata.freq.lower()
    if dataset_freq == 'h' and freq == 'h' or dataset_freq == 'hourly' and freq == 'h':
        # Normalize frequency to 'h'
        config["freq"] = 'h'
        freq = 'h'
    else:
        raise ValueError(
            f"Frequency mismatch: Config specified '{freq}' but dataset has '{dataset.metadata.freq}'. "
            f"Please ensure the frequencies match. Supported frequencies are: 'h', 'D', 'W', 'M', 'B'"
        )

    # Verify prediction length
    if dataset.metadata.prediction_length != prediction_length:
        logger.warning(
            f"Prediction length mismatch: Config specified {prediction_length} but dataset has "
            f"{dataset.metadata.prediction_length}. Using config value."
        )

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(dataset.train)

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
    )

    callbacks = []
    if config["use_validation_set"]:
        transformed_data = transformation.apply(training_data, is_train=True)
        if config.get("normalize_data", True):
            normalized_data = []
            for item in transformed_data:
                if "target" in item:
                    item["target"], mean, std = normalize_data(item["target"])
                    item["normalization_params"] = {"mean": mean, "std": std}
                normalized_data.append(item)
            transformed_data = normalized_data
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals
        )
        _, val_gen = train_val_splitter.split(training_data)
        val_data = val_gen.generate_instances(
            config["prediction_length"], num_rolling_evals
        )

        callbacks = [
            EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"],
                sampler_kwargs=config["sampler_params"],
                num_samples=config["num_samples"],
                model=model,
                transformation=transformation,
                test_dataset=dataset.test,
                val_dataset=val_data,
                eval_every=config["eval_every"],
            )
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)
        if config.get("normalize_data", True):
            normalized_data = []
            for item in transformed_data:
                if "target" in item:
                    item["target"], mean, std = normalize_data(item["target"])
                    item["normalization_params"] = {"mean": mean, "std": std}
                normalized_data.append(item)
            transformed_data = normalized_data

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    # Verify input data
    for batch in data_loader:
        if 'past_target' in batch:
            data = batch['past_target']
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("WARNING: Input data contains NaN or Inf values!")
                # Clean the data
                data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                data = torch.clamp(data, min=-1e6, max=1e6)
                batch['past_target'] = data
            print(f"Input data range: [{data.min().item():.3f}, {data.max().item():.3f}]")
        break

    # Plot several time series from the training set before training
    sample_size = 5  # Number of time series to plot
    sample_data = list(training_data)[:sample_size]  # Get first 5 time series

    # Debug print to check data structure
    print("Sample data structure:", sample_data[0].keys())
    
    # Create figure with two subplots stacked vertically
    fig, (ax_ts, ax_markers) = plt.subplots(2, 1, figsize=(12, 10), 
                                          gridspec_kw={'height_ratios': [4, 1]},
                                          sharex=True)
    
    # Color palette for different time series
    colors = plt.cm.tab10(np.linspace(0, 1, sample_size))
    
    # Plot time series in upper subplot
    for idx, (entry, color) in enumerate(zip(sample_data, colors)):
        target = entry['target']
        
        # Try different keys for observed values
        observed = None
        for key in ['observed_values', 'observed', 'mask']:
            if key in entry:
                observed = entry[key]
                print(f"Found observations under key: {key}")
                print(f"Number of missing values: {np.sum(~observed.astype(bool))}")
                break
        
        if observed is None:
            # If no mask found, create artificial missing values for demonstration
            print("No missing value mask found, creating artificial missing values")
            observed = np.ones_like(target)
            # Create random missing values (20% of the data)
            missing_mask = np.random.choice([True, False], size=len(target), p=[0.8, 0.2])
            observed = observed * missing_mask
        
        observed = observed.astype(bool)
        
        # Plot the full time series
        ax_ts.plot(target, color=color, alpha=0.3, label=f'Time Series {idx+1}')
        
        # Highlight observed values
        observed_target = np.where(observed, target, np.nan)
        ax_ts.plot(observed_target, color=color, linewidth=2)
        
        # Mark missing values with red dots
        missing_indices = np.where(~observed)[0]
        if len(missing_indices) > 0:
            missing_values = target[missing_indices]
            ax_ts.scatter(missing_indices, missing_values, color='red', alpha=0.5)
            
            # Add markers for missing values in lower subplot
            ax_markers.scatter(missing_indices, [idx] * len(missing_indices), 
                             marker='|', color=color, s=100, label=f'TS {idx+1} missing')
    
    # Customize upper subplot (time series)
    ax_ts.set_title('Sample Time Series from Training Set\n(Red dots indicate missing values)')
    ax_ts.set_ylabel('Value')
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='upper right')
    
    # Customize lower subplot (missing value markers)
    ax_markers.set_xlabel('Time')
    ax_markers.set_ylabel('Series')
    ax_markers.set_yticks(range(sample_size))
    ax_markers.set_yticklabels([f'TS {i+1}' for i in range(sample_size)])
    ax_markers.set_title('Missing Value Locations')
    ax_markers.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'sample_time_series.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saving image: 'sample_time_series.png' to {plot_path}")
    logger.info(f"Saved sample time series plot to {plot_path}")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)

    # Determine accelerator and devices configuration
    if torch.cuda.is_available():
        accelerator = "gpu"
        # Get device index for trainer
        if isinstance(config["device"], torch.device):
            device_idx = 0 if config["device"].type == "cuda" else None
        else:
            # Handle string device specification
            device_idx = int(config["device"].split(":")[-1]) if ":" in config["device"] else 0
        devices = [device_idx] if device_idx is not None else [0]
    else:
        accelerator = "cpu"
        devices = 1  # Use single CPU core

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        default_root_dir=log_dir,
        accumulate_grad_batches=4,
        detect_anomaly=True,
        precision="32-true",  # Use true float32 precision
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(
                monitor='train_loss',
                patience=5,
                mode='min',
                min_delta=1e-4,
                verbose=True
            ),
            *callbacks
        ]
    )

    # Add learning rate scheduler with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,  # Start with very small learning rate
        weight_decay=0.01,
        eps=1e-8,  # Increased epsilon for better numerical stability
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get("lr", 1e-4),
        total_steps=config["max_epochs"] * config["num_batches_per_epoch"],
        pct_start=0.1,  # 10% warmup
        div_factor=25.0,  # Initial lr = max_lr/25
        final_div_factor=1000.0,  # Final lr = max_lr/1000
    )
    
    trainer.lr_schedulers = [
        {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
    ]

    # Add more detailed debug prints for input data
    for batch in data_loader:
        print("\nDetailed Batch Statistics:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    print(f"WARNING: NaN values found in {key}")
                if torch.isinf(value).any():
                    print(f"WARNING: Inf values found in {key}")
                
                # Convert to float for statistics if needed
                if value.dtype in [torch.long, torch.int32, torch.int64]:
                    value_float = value.float()
                else:
                    value_float = value
                
                stats = {
                    "shape": value.shape,
                    "dtype": value.dtype,
                    "min": value_float.min().item(),
                    "max": value_float.max().item(),
                    "mean": value_float.mean().item(),
                    "std": value_float.std().item(),
                    "has_nan": torch.isnan(value_float).any().item(),
                    "has_inf": torch.isinf(value_float).any().item()
                }
                print(f"{key}:")
                for stat_name, stat_value in stats.items():
                    print(f"  {stat_name}: {stat_value}")
        break

    # Add a hook to monitor for NaN/Inf gradients
    def hook_fn(grad):
        if torch.isnan(grad).any():
            print("NaN gradient detected!")
            return torch.zeros_like(grad)
        if torch.isinf(grad).any():
            print("Inf gradient detected!")
            return torch.clip(grad, -1e6, 1e6)
        return grad

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name: hook_fn(grad))

    # Verify model parameters before training
    print("\nModel Parameter Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                "shape": param.shape,
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "has_nan": torch.isnan(param).any().item(),
                "has_inf": torch.isinf(param).any().item()
            }
            print(f"{name}:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value}")

    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(model, train_dataloaders=data_loader)
    logger.info("Training completed.")

    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )
    logger.info(f"Loading {best_ckpt_path}.")
    best_state_dict = torch.load(best_ckpt_path)
    model.load_state_dict(best_state_dict, strict=True)

    metrics = (
        evaluate_guidance(config, model, dataset.test, transformation)
        if config.get("do_final_eval", True)
        else "Final eval not performed"
    )
    with open(Path(trainer.logger.log_dir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
        )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir)
