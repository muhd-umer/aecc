"""
Default configuration file for ViTCC.
"""

import os

import ml_collections


def get_mnist_config():
    cfg = ml_collections.ConfigDict()

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_root = os.path.abspath("../data/")
    cfg.batch_size = 32
    cfg.num_workers = 4
    cfg.pin_memory = True

    # Training
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.lr_gamma = 0.1
    cfg.num_epochs = 30
    cfg.lr_step_size = 15
    cfg.noise_factor = 0.15
    cfg.model_dir = os.path.abspath("../weights/")

    return cfg


def get_imagenette_config():
    cfg = ml_collections.ConfigDict()

    # Misc
    cfg.seed = 42

    # Dataset
    cfg.data_root = os.path.abspath("../data/")
    cfg.batch_size = 8
    cfg.num_workers = 4
    cfg.pin_memory = True

    # Training
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.lr_gamma = 0.1
    cfg.num_epochs = 30
    cfg.lr_step_size = 15
    cfg.noise_factor = 0.2
    cfg.model_dir = os.path.abspath("../weights/")

    return cfg
