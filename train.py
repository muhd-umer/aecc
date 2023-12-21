"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Training script for Pytorch models [Pytorch Lightning]
"""

import argparse
import os
import warnings

import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import torch
from termcolor import colored
from torchinfo import summary
from torchmetrics import MeanSquaredError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import get_config
from data import *
from models import DAEViT, LitDAE
from utils import EMACallback, SimplifiedProgressBar

# Common setup
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
plt.rcParams["font.family"] = "STIXGeneral"


def train(
    cfg,
    accelerator,
    devices,
    rich_progress,
    test_mode=False,
    resume=False,
    weights=None,
    logger_backend="tensorboard",
):
    if logger_backend == "tensorboard":
        logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=cfg.log_dir, name=".")

    elif logger_backend == "wandb":
        logger = pl.pytorch.loggers.WandbLogger(project="aecc", save_dir=cfg.log_dir)
    else:
        raise ValueError(
            colored(
                "Provide a valid logger (tensorboard, wandb)",
                "red",
            )
        )

    # Get the data loaders
    if cfg.dataset == "imagenette":
        train_transform, test_transform = get_imagenette_transform(cfg)
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            steps_per_epoch,
        ) = get_imagenette_loaders(
            cfg.data_dir,
            train_transform,
            test_transform,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_size=cfg.val_size,
            noise_factor=cfg.noise_factor,
            return_steps=True,
        )
    elif cfg.dataset == "mnist":
        raise NotImplementedError("Yet to be implemented!")
    elif cfg.dataset == "cifar10":
        train_transform, test_transform = get_cifar10_transform(cfg)
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            steps_per_epoch,
        ) = get_cifar10_loaders(
            cfg.data_dir,
            train_transform,
            test_transform,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_size=cfg.val_size,
            noise_factor=cfg.noise_factor,
            return_steps=True,
        )
    elif cfg.dataset == "cifar100":
        train_transform, test_transform = get_cifar100_transform(cfg)
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            steps_per_epoch,
        ) = get_cifar100_loaders(
            cfg.data_dir,
            train_transform,
            test_transform,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_size=cfg.val_size,
            noise_factor=cfg.noise_factor,
            return_steps=True,
        )
    else:
        raise ValueError(
            colored(
                "Provide a valid dataset (imagenette, mnist, cifar100, cifar10)",
                "red",
            )
        )

    # Get the model
    # model = dae_vit_models[cfg.model_name](
    #     in_channels=cfg.in_channels, img_size=cfg.img_size
    # )
    model = DAEViT(
        in_channels=cfg.in_channels,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        emb_dim=cfg.emb_dim,
        encoder_layer=cfg.encoder_layer,
        encoder_head=cfg.encoder_head,
        decoder_layer=cfg.decoder_layer,
        decoder_head=cfg.decoder_head,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=0, last_epoch=-1
    )

    if cfg.loss == "mse":
        loss = MeanSquaredError()
    elif cfg.loss == "lpips":
        loss = LearnedPerceptualImagePatchSimilarity(net_type="alex")
    else:
        raise ValueError(
            colored(
                "Provide a valid loss (mse, lpips)",
                "red",
            )
        )

    model = LitDAE(model, cfg, optimizer, loss, lr_scheduler)

    # Divide steps per epoch by number of GPUs
    if devices != "auto":
        steps_per_epoch = steps_per_epoch // devices

    cfg.steps_per_epoch = steps_per_epoch

    if os.getenv("LOCAL_RANK", "0") == "0":
        yaml_cfg = cfg.to_yaml()

        os.makedirs(cfg.log_dir, exist_ok=True)

        print(colored(f"Config:", "green", attrs=["bold"]))
        print(colored(yaml_cfg))
        model_title = cfg.model_name.replace("_", "-").upper()
        print(colored(f"Model: {model_title}", "green", attrs=["bold"]))
        summary(
            model,
            input_size=(3, cfg.img_size, cfg.img_size),
            depth=3,
            batch_dim=0,
            device="cpu",
        )

    # Load from checkpoint if weights are provided
    if weights is not None:
        model.load_state_dict(torch.load(weights)["state_dict"])

    if logger_backend == "wandb":
        logger.watch(model, log="all", log_freq=100)

    # Create a PyTorch Lightning trainer with the required callbacks
    if rich_progress:
        theme = pl_callbacks.progress.rich_progress.RichProgressBarTheme(
            description="black",
            progress_bar="cyan",
            progress_bar_finished="green",
            progress_bar_pulse="#6206E0",
            batch_progress="cyan",
            time="grey82",
            processing_speed="grey82",
            metrics="black",
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            # precision=16,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            check_val_every_n_epoch=cfg.val_freq,
            logger=logger,
            callbacks=[
                # pl_callbacks.RichModelSummary(max_depth=3),
                pl_callbacks.RichProgressBar(theme=theme),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename=f"{cfg.model_name}_best_model",
                ),
                EMACallback(decay=0.999),
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
            ],
        )
    else:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            # precision=16,
            max_epochs=cfg.num_epochs,
            enable_model_summary=False,
            check_val_every_n_epoch=cfg.val_freq,
            logger=logger,
            callbacks=[
                # pl_callbacks.ModelSummary(max_depth=3),
                SimplifiedProgressBar(),
                pl_callbacks.ModelCheckpoint(
                    dirpath=cfg.model_dir,
                    filename=f"{cfg.model_name}_best_model",
                ),
                EMACallback(decay=0.999),
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
            ],
        )

    # Train the model
    if not test_mode:
        if resume:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=weights)
        trainer.fit(model, train_dataloader, val_dataloader)

    # Evaluate the model on the test set
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    cfg = get_config()

    # Add argument parsing with cfg overrides
    parser = argparse.ArgumentParser(
        description="Train DAE-ViT using PyTorch Lightning"
    )
    parser.add_argument(
        "--data-dir", type=str, default=cfg.data_dir, help="Directory for the data"
    )
    parser.add_argument(
        "--model-dir", type=str, default=cfg.model_dir, help="Directory for the model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg.batch_size, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cfg.num_workers,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=cfg.num_epochs,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--lr", type=float, default=cfg.lr, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=cfg.val_size,
        help="Validation size for the data",
    )
    parser.add_argument(
        "--noise-factor",
        type=float,
        default=cfg.noise_factor,
        help="Noise factor for the data",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=cfg.loss,
        help="Loss function for training (mse, lpips)",
    )
    parser.add_argument(
        "--rich-progress", action="store_true", help="Use rich progress bar"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, tpu, etc.)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices to use for training (auto, cpu, gpu, etc.)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the weights file for the model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the provided weights",
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only test the model, do not train"
    )
    parser.add_argument(
        "--logger-backend",
        type=str,
        default="tensorboard",
        help="Logger backend (tensorboard, wandb)",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=cfg.val_freq,
        help="Validate every n epochs",
    )
    args = parser.parse_args()

    cfg.update(args.__dict__)

    # Set mean/std to get images in [-1, 1]
    cfg.mean = [0.5, 0.5, 0.5]
    cfg.std = [0.5, 0.5, 0.5]

    if args.devices != "auto":
        args.devices = int(args.devices)
    if (args.resume or args.test_only) and args.weights is None:
        raise ValueError(
            colored(
                "Provide the path to the weights file using --weights",
                "red",
            )
        )

    train(
        cfg,
        args.accelerator,
        args.devices,
        args.rich_progress,
        args.test_only,
        args.resume,
        args.weights if args.resume or args.test_only else None,
        args.logger_backend,
    )
