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
import torch.nn as nn
from termcolor import colored
from torchinfo import summary
from torchmetrics import MeanSquaredError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import get_cfg, get_defaults
from data import *
from models import DAEViT, LitDAE, dae_resnet_models, dae_vit_models
from utils import EMACallback, SimplifiedProgressBar

# Common setup
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
plt.rcParams["font.family"] = "STIXGeneral"

datasets = ["imagenette", "mnist", "cifar100", "cifar10"]  # Supported datasets
normalize_settings = ["default", "standard", "neg1to1"]  # Supported normalization


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
    # Get the data loaders
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        steps_per_epoch,
    ) = load_dataset(cfg)

    if cfg.normalize == "default":
        gate = nn.Identity
    elif cfg.normalize == "standard":
        gate = nn.Sigmoid
    elif cfg.normalize == "neg1to1":
        gate = nn.Tanh
    else:
        raise ValueError(
            colored(
                "Provide a valid normalization \n(default, standard, neg1to1)",
                "red",
            )
        )

    # Get the model
    if cfg.model_name in dae_vit_models:
        model = DAEViT(
            in_channels=cfg.in_channels,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            emb_dim=cfg.emb_dim,
            encoder_layer=cfg.encoder_layer,
            encoder_head=cfg.encoder_head,
            decoder_layer=cfg.decoder_layer,
            decoder_head=cfg.decoder_head,
            gate=gate,
        )
    elif cfg.model_name in dae_resnet_models:
        model = dae_resnet_models[cfg.model_name](
            in_channels=cfg.in_channels,
            gate=gate,
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
        loss = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True if cfg.normalize == "standard" else False
        )
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
            precision=16,
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
            precision=16,
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
    cfg = get_defaults()

    # Add argument parsing with cfg overrides
    parser = argparse.ArgumentParser(
        description="Train DAE-ViT using PyTorch Lightning"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (dae_vit_tiny, dae_vit_small, dae_vit_base, dae_vit_large, dae_vit_huge, dae_resnet18, dae_resnet34, dae_resnet50, dae_resnet101, dae_resnet152)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (imagenette, mnist, cifar100, cifar10)",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="./config/model_cfg.yaml",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--data-cfg",
        type=str,
        default="./config/data_cfg.yaml",
        help="Path to the data config file",
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
        "--normalize",
        type=str,
        default=cfg.normalize,
        help="Normalization type (default, standard, neg1to1)",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=cfg.val_freq,
        help="Validate every n epochs",
    )
    args = parser.parse_args()

    if args.devices != "auto":
        args.devices = int(args.devices)
    if (args.resume or args.test_only) and args.weights is None:
        raise ValueError(
            colored(
                "Provide the path to the weights file using --weights",
                "red",
            )
        )

    if (
        args.model_name not in dae_vit_models
        and args.model_name not in dae_resnet_models
    ):
        raise ValueError(
            colored(
                "Provide a valid model \n(dae_vit_tiny, dae_vit_small, dae_vit_base, "
                + "dae_vit_large, dae_vit_huge, dae_resnet18, dae_resnet34, dae_resnet50, "
                + "dae_resnet101, dae_resnet152)",
                "red",
            )
        )

    if args.dataset not in datasets:
        raise ValueError(
            colored(
                "Provide a valid dataset \n(imagenette, mnist, cifar100, cifar10)",
                "red",
            )
        )

    if args.logger_backend == "tensorboard":
        logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=cfg.log_dir, name=".")

    elif args.logger_backend == "wandb":
        logger = pl.pytorch.loggers.WandbLogger(project="aecc", save_dir=cfg.log_dir)
    else:
        raise ValueError(
            colored(
                "Provide a valid logger (tensorboard, wandb)",
                "red",
            )
        )

    cfg.update(args.__dict__)

    upd_cfg = get_cfg(
        cfg.model_name,
        cfg.dataset,
        args.model_cfg,
        args.data_cfg,
        cfg=cfg,
    )

    # Set cfg.normalize (default, standard, neg1to1)
    if upd_cfg.normalize not in normalize_settings:
        raise ValueError(
            colored(
                "Provide a valid normalization \n(default, standard, neg1to1)",
                "red",
            )
        )

    if upd_cfg.loss == "lpips" and upd_cfg.normalize not in [
        "standard",
        "neg1to1",
    ]:
        raise ValueError(
            colored(
                "LPIPS loss requires the data to be normalized in the range [0, 1] or [-1, 1]",
                "red",
            )
        )

    train(
        upd_cfg,
        args.accelerator,
        args.devices,
        args.rich_progress,
        args.test_only,
        args.resume,
        args.weights if args.resume or args.test_only else None,
        args.logger_backend,
    )
