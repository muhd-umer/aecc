"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Pytorch Lightning module for DAE
"""

from typing import Tuple

import lightning as pl
import torch
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

        samples = x[:6]
        reconstructions = y_hat[:6]
        originals = y[:6]

        # Log images last batch and ensure logger has add_image method
        if hasattr(self.logger.experiment, "log_image"):  # for wandb
            self.logger.experiment.log_image(
                key="samples", image=torchvision.utils.make_grid(samples)
            )
            self.logger.experiment.log_image(
                key="reconstructions",
                image=torchvision.utils.make_grid(reconstructions),
            )
            self.logger.experiment.log_image(
                key="originals", image=torchvision.utils.make_grid(originals)
            )
        elif hasattr(self.logger.experiment, "add_images"):  # for tensorboards
            self.logger.experiment.add_images(
                tag="samples", img_tensor=torchvision.utils.make_grid(samples)
            )
            self.logger.experiment.add_images(
                tag="reconstructions",
                img_tensor=torchvision.utils.make_grid(reconstructions),
            )
            self.logger.experiment.add_images(
                tag="originals", img_tensor=torchvision.utils.make_grid(originals)
            )
        else:
            pass

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
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure optimizers.

        Returns:
            Tuple[list, list]: Optimizers and schedulers.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
