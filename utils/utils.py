import os
from typing import Union

import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from flax.training import checkpoints, train_state


class FlattenAndCast(object):
    """
    Returns contigious flattened array to make Numpy arrays.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """

    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class ReshapeAndCast(object):
    """
    Returns contigious flattened array to make Numpy arrays.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """

    def __call__(self, pic):
        return np.reshape(np.array(pic, dtype=jnp.float32), -1)


def numpy_collate(batch):
    """
    Collate function to use PyTorch datalaoders
        with JAX/Flax.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def save_checkpoint(
    target: train_state.TrainState, epoch: int, output_dir: Union[os.PathLike, str]
):
    """
    Args:
        target: TrainState to save as a checkpoint
        epoch: Training step number
        output_dir: Directory to save checkpoints to.
    Returns:
        None
    """
    save_dir = checkpoints.save_checkpoint(
        ckpt_dir=str(output_dir), target=target, step=epoch, overwrite=True
    )
    print(f"{Fore.MAGENTA}{' '*10} Saving checkpoint at {save_dir}{Style.RESET_ALL}")


def restore_checkpoint(
    state: train_state.TrainState, checkpoint_dir: Union[os.PathLike, str]
):
    """
    Args:
        checkpoint_dir: Directory to load checkpoint from.
    Returns:
        None
    """
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir, target=state
    )
    print(f"Restored state from {Fore.GREEN}{checkpoint_dir}{Style.RESET_ALL}")
    return restored_state
