"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Pytorch Lightning module for DAE
"""

from typing import Tuple

import lightning as pl
import torch
import torchmetrics.image as im_metrics
import torchvision


class LitDAE(pl.LightningModule):
    """
    Denoising Autoencoder (DAE) Pytorch Lightning module.
    """

    def __init__(self, model, cfg, optimizer, criterion, lr_scheduler) -> None:
        """
        Initialize the DAE.

        Args:
            model (nn.Module): DAE model.
            cfg (dict): Configuration dictionary.
            optimizer (optim.Optimizer): Configured optimizer.
            criterion (torchmetrics.Metric): Configured loss function.
            lr_scheduler (lr_scheduler.LRScheduler): Learning rate scheduler.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.loss = criterion

        self.ssim = im_metrics.StructuralSimilarityIndexMeasure()
        self.psnr = im_metrics.PeakSignalNoiseRatio()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)

        # denoising autoencoder metrics
        loss = self.loss(y_hat, y)
        ssim = self.ssim(y_hat, y)
        psnr = self.psnr(y_hat, y)

        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.log("ssim", ssim, sync_dist=True, prog_bar=True)
        self.log("psnr", psnr, sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure optimizers.

        Returns:
            Tuple[list, list]: Optimizers and schedulers.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
