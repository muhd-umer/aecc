import ml_collections


def get_cfg():
    """
    Configure hyperparameters for training here.
    Data from ConfigDict can be accessed from the
    outside as any DictLike object.
    Example:
        >>> cfg = get_cfg()
        >>> print(cfg.learning_rate)
        >>> 0.1
    """
    config = ml_collections.ConfigDict()

    config.learning_rate = 3.5e-4
    config.batch_size = 32
    config.warmup_epochs = 2
    config.momentum = 0.9

    config.num_epochs = 10
    config.log_every_steps = 100

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    # configure input dataset keys
    config.dataset_name = "imagenette"
    config.data_shape = [224, 224]
    config.num_classes = 10
    config.split_keys = ["train", "validation"]

    return config
