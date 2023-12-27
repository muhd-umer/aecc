"""
Default configuration file for AeCC.
"""

import os

import yaml
from box import Box


def get_defaults():
    """
    Returns a Box object containing the default configuration parameters.

    Returns:
        cfg (Box): default configuration parameters.
    """
    cfg = Box()

    # Set root directories
    cfg.root_dir = os.path.abspath(".")
    cfg.log_dir = os.path.abspath(os.path.join(cfg.root_dir, "logs"))

    # Misc
    cfg.seed = 42

    # Training
    cfg.num_epochs = 1000
    cfg.val_freq = 50
    cfg.val_size = 0.1
    cfg.noise_factor = 0.3
    cfg.lr = 0.0005
    cfg.weight_decay = 0.005
    cfg.momentum = 0.9
    cfg.model_dir = os.path.abspath(os.path.join(cfg.root_dir, "weights"))
    cfg.pretrained = False

    # Data
    cfg.data_dir = os.path.abspath(os.path.join(cfg.root_dir, "data"))
    cfg.batch_size = 64
    cfg.num_workers = 4
    cfg.pin_memory = True
    cfg.normalize = "default"  # choose from "default", "standard", "neg1to1"

    return cfg


def get_cfg(model, dataset, model_cfg_path, data_cfg_path, cfg=None):
    """
    Returns a Box object containing the configuration parameters updated with the provided arguments.

    Args:
        model (str): name of the model.
        dataset (str): name of the dataset.
        model_cfg_path (str): path to the YAML file containing the model configuration.
        data_cfg_path (str): path to the YAML file containing the dataset configuration.
        cfg (Box, optional): configuration parameters to update. If None, get defaults.

    Returns:
        cfg (Box): updated configuration parameters.
    """
    if cfg is None:
        cfg = get_defaults()

    # Load dataset and model configurations from YAML files
    with open(model_cfg_path, "r") as ymlfile:
        model_cfg_dict = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with open(data_cfg_path, "r") as ymlfile:
        data_cfg_dict = yaml.load(ymlfile, Loader=yaml.FullLoader)

    model_cfg = model_cfg_dict[model]
    data_cfg = data_cfg_dict[dataset]

    # Update cfg with the loaded configurations
    cfg.update(data_cfg)
    cfg.update(model_cfg)

    return cfg
