"""
Default configuration file for AeCC.
"""

import os

from box import Box

# def get_config():
#     cfg = Box()

#     # Set root directories
#     cfg.root_dir = os.path.abspath(".")
#     cfg.log_dir = os.path.abspath(os.path.join(cfg.root_dir, "logs"))

#     # Misc
#     cfg.seed = 42

#     # Dataset
#     cfg.dataset = "imagenette"  # choose from "mnist", "cifar100", "imagenette"
#     cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
#     cfg.batch_size = 64
#     cfg.num_workers = 4
#     cfg.pin_memory = True
#     cfg.val_size = 0.1
#     cfg.in_channels = 3
#     cfg.img_size = 224  # desired image size, not actual image size
#     cfg.noise_factor = 0.2

#     # Training
#     cfg.loss = "lpips"  # choose from "mse", "lpips"
#     cfg.num_epochs = 1000
#     cfg.val_freq = 50
#     cfg.lr = 0.0005
#     cfg.weight_decay = 0.005
#     cfg.momentum = 0.9
#     cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))

#     # choose from "dae_vit_tiny", "dae_vit_small", "dae_vit_base", ...
#     # "dae_vit_large", "dae_vit_huge"
#     cfg.model_name = "dae_vit_small"

#     return cfg


def get_config():
    cfg = Box()

    # Set root directories
    cfg.root_dir = os.path.abspath(".")
    cfg.log_dir = os.path.abspath(os.path.join(cfg.root_dir, "logs"))

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.dataset = "cifar100"  # choose from "mnist", "cifar100", "imagenette"
    cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
    cfg.batch_size = 64
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.val_size = 0.1
    cfg.in_channels = 3
    cfg.img_size = 32  # desired image size, not actual image size
    cfg.noise_factor = 0.2

    # Model
    cfg.patch_size = 4
    cfg.emb_dim = 192
    cfg.encoder_layer = 12
    cfg.encoder_head = 3
    cfg.decoder_layer = 4
    cfg.decoder_head = 3

    # Training
    cfg.loss = "mse"  # choose from "mse", "lpips"
    cfg.num_epochs = 1000
    cfg.val_freq = 50
    cfg.lr = 0.0005
    cfg.weight_decay = 0.005
    cfg.momentum = 0.9
    cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))

    # choose from "dae_vit_tiny", "dae_vit_small", "dae_vit_base", ...
    # "dae_vit_large", "dae_vit_huge"
    cfg.model_name = "dae_vit_small"

    return cfg
