"""Preprocessing helpers shared by the notebook, training scripts and the app."""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["Cat", "Dog"]
LABEL_MAP = {"Cat": 0, "Dog": 1}          # Dog is the positive class
N_WORKERS = min(32, (os.cpu_count() or 2) * 4)


def load_image_array(path, size, gray=False):
    """Load -> RGB (or L) -> resize to size x size -> uint8 array."""
    with Image.open(path) as im:
        im = im.convert("L" if gray else "RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_image_batch(paths, size, gray=False, workers=N_WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(lambda p: load_image_array(p, size, gray), paths))
    return np.stack(arrs)


def normalize(x_uint8):
    """0-255 -> 0-1 (float32)."""
    return x_uint8.astype("float32") / 255.0


def flatten(x):
    return x.reshape(len(x), -1)


def make_dataset(paths, labels, img_size, batch_size=64, shuffle=False, seed=42, augment=None):
    """tf.data pipeline: file -> decode -> resize -> float32 [0,255] -> batch (-> augment, train only)."""
    import tensorflow as tf
    autotune = tf.data.AUTOTUNE

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((np.asarray(paths, dtype=str), np.asarray(labels, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=autotune).batch(batch_size)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=autotune)
    return ds.prefetch(autotune)
