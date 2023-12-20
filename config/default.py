"""
Default configuration file for AeCC.
"""

import os

from box import Box


def get_config():
    cfg = Box()

    # Set root directories
    cfg.root_dir = os.path.abspath(".")
    cfg.log_dir = os.path.abspath(os.path.join(cfg.root_dir, "logs"))

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
    cfg.batch_size = 64
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.val_size = 0.1
    cfg.in_channels = 3
    cfg.img_size = 224  # desired image size, not actual image size
    cfg.noise_factor = 0.2

    # Training
    cfg.loss = "lpips"  # choose from "mse", "lpips"
    cfg.num_epochs = 100
    cfg.lr = 0.0005
    cfg.weight_decay = 0.005
    cfg.momentum = 0.9
    cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))

    # choose from "dae_vit_tiny", "dae_vit_small", "dae_vit_base", ...
    # "dae_vit_large", "dae_vit_huge"
    cfg.model_name = "dae_vit_small"

    return cfg
