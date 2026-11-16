"""Inference helpers: validate an uploaded image, preprocess exactly like training, predict."""
import io

import numpy as np
from PIL import Image

from src.data_cleaning import CLEAN_MAX_SIDE, cap_size, to_rgb

CLASS_NAMES = ["Cat", "Dog"]          # index 0 = Cat, 1 = Dog (probability output = P(Dog))
MAX_UPLOAD_MB = 10
MIN_SIDE = 32
_CACHE = {}


def load_trained_model(path):
    """Load the saved Keras model (no retraining). Cached per path."""
    import tensorflow as tf
    path = str(path)
    if path not in _CACHE:
        _CACHE[path] = tf.keras.models.load_model(path, compile=False)
    return _CACHE[path]


def open_uploaded_image(file_like, max_mb=MAX_UPLOAD_MB):
    """Validate an upload; returns a fully decoded PIL image or raises ValueError with a friendly message."""
    data = file_like.read() if hasattr(file_like, "read") else bytes(file_like)
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than {max_mb} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise ValueError("This file is not a valid / readable image (try JPG, PNG, WEBP, BMP or GIF).")
    if min(im.size) < MIN_SIDE:
        raise ValueError(f"Image is too small ({im.width}x{im.height}); the shorter side must be at least {MIN_SIDE}px.")
    return im


def preprocess_image(im, img_size):
    """PIL image -> float32 batch (1, img_size, img_size, 3) in [0, 255] (the model rescales internally)."""
    import tensorflow as tf
    rgb = cap_size(to_rgb(im), CLEAN_MAX_SIDE)
    arr = np.asarray(rgb, dtype=np.float32)
    x = tf.image.resize(arr[None, ...], (img_size, img_size), antialias=True)
    return x.numpy()


def predict_image(model, im, threshold=0.5, min_confidence=0.0):
    """Return label, P(Dog), P(Cat), confidence and an 'uncertain' flag."""
    img_size = int(model.input_shape[1])
    p_dog = float(model.predict(preprocess_image(im, img_size), verbose=0)[0, 0])
    label = CLASS_NAMES[int(p_dog >= threshold)]
    confidence = p_dog if label == "Dog" else 1.0 - p_dog
    return {"label": label, "prob_dog": p_dog, "prob_cat": 1.0 - p_dog,
            "confidence": confidence, "uncertain": confidence < min_confidence}
