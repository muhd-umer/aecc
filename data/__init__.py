from .cifar10 import *
from .cifar100 import *
from .imagenette import *
from .mnist import *
from .nploader import *


def load_dataset(cfg):
    dataset_functions = {
        "imagenette": {
            "transform": get_imagenette_transform,
            "loader": get_imagenette_loaders,
        },
        "mnist": {
            "transform": None,
            "loader": None,
        },
        "cifar10": {
            "transform": get_cifar10_transform,
            "loader": get_cifar10_loaders,
        },
        "cifar100": {
            "transform": get_cifar100_transform,
            "loader": get_cifar100_loaders,
        },
    }

    if cfg.dataset not in dataset_functions:
        raise ValueError(
            colored(
                "Provide a valid dataset (imagenette, mnist, cifar100, cifar10)",
                "red",
            )
        )

    if (
        dataset_functions[cfg.dataset]["transform"] is None
        or dataset_functions[cfg.dataset]["loader"] is None
    ):
        raise NotImplementedError(f"{cfg.dataset} is yet to be implemented!")

    train_transform, test_transform = dataset_functions[cfg.dataset]["transform"](cfg)
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        steps_per_epoch,
    ) = dataset_functions[cfg.dataset]["loader"](
        cfg.data_dir,
        train_transform,
        test_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_size=cfg.val_size,
        noise_factor=cfg.noise_factor,
        return_steps=True,
    )

    return train_dataloader, val_dataloader, test_dataloader, steps_per_epoch
