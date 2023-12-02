from typing import Iterable

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def transform_images(image_bytes, shape):
    image = tf.image.resize_with_pad(image_bytes, shape[0], shape[1])
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image


def get_jnp_dataset(name, batch_size, img_shape, split):
    """
    Load "name" train and test data into memory;
    General Feature Structure:
        FeaturesDict({
            'image': Image(shape=(None, None, 3), dtype=tf.uint8),
            'image/filename': Text(shape=(), dtype=tf.string),
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
        })
    Note: This feature structure varies from dataset to dataset.
    For more information, refer to:
        https://www.tensorflow.org/datasets/catalog/overview
    Returns:
        Train and Test data with features_dict.
    """

    def decode_example(example):
        image = transform_images(example["image"], img_shape)
        return {"image": image, "label": example["label"]}

    dataset_builder = tfds.builder(name)
    dataset_builder.download_and_prepare()
    num_examples = dataset_builder.info.splits[split].num_examples
    split_size = num_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f"{split}[{start}:{start + split_size}]"

    dataset = dataset_builder.as_dataset(split=split)
    dataset = dataset.map(  # type: ignore
        decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.cache().repeat().batch(batch_size, drop_remainder=True)
    dataset.prefetch(10)

    return dataset, num_examples


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_iterator(name: str, batch_size: int, img_shape: Iterable[int], split: str):
    """
    Creates an iterator on the basis of split string
        and passes it onto device
    """
    data, examples = get_jnp_dataset(name, batch_size, img_shape, split)

    data_it = map(prepare_tf_data, data)
    data_it = jax_utils.prefetch_to_device(data_it, 2)

    return data_it, examples
