"""Inference helpers: validate an uploaded image, preprocess exactly like training, predict."""
import io

import numpy as np
from PIL import Image

from src.data_cleaning import CLEAN_MAX_SIDE, cap_size, to_rgb

CLASS_NAMES = ["Cat", "Dog"]          # index 0 = Cat, 1 = Dog (probability output = P(Dog))
MAX_UPLOAD_MB = 10
MIN_SIDE = 32
_CACHE = {}


class DetectorError(RuntimeError):
    pass


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


def determine_predictions(model, batch, threshold=0.5, min_confidence=0.7,
                          detector=None, source_paths=None):
    """Predict a preprocessed batch, applying the same decision rules as uploads.

    A detector, when provided, must expose ``is_ood(batch, source_paths)``.
    Source paths allow its reference set to exclude the image being evaluated.
    Model and detector failures propagate rather than becoming image errors.
    """
    if not 0 <= threshold <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("Thresholds must be between 0 and 1.")
    count = len(batch)
    if source_paths is None:
        source_paths = [None] * count
    if len(source_paths) != count:
        raise ValueError("source_paths must have one entry per image.")
    probabilities = np.asarray(model.predict(batch, verbose=0))
    if probabilities.shape not in ((count,), (count, 1)):
        raise ValueError("Model must return one Dog probability per image.")
    probabilities = probabilities.reshape(-1)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Model returned invalid probabilities.")
    if detector is None:
        ood_flags = np.zeros(count, dtype=bool)
    else:
        try:
            ood_flags = np.asarray(detector.is_ood(batch, source_paths), dtype=bool)
            if ood_flags.shape != (count,):
                raise ValueError("Detector must return one OOD flag per image.")
        except Exception as error:
            raise DetectorError("The experimental reference-distance check failed.") from error
    results = []
    for p_dog, ood in zip(probabilities, ood_flags):
        p_dog = float(p_dog)
        candidate = CLASS_NAMES[int(p_dog >= threshold)]
        confidence = p_dog if candidate == "Dog" else 1.0 - p_dog
        uncertain = confidence < min_confidence or bool(ood)
        reasons = []
        if confidence < min_confidence:
            reasons.append("confidence is below the selected threshold")
        if ood:
            reasons.append("the embedding is distant from the provisional reference set")
        explanation = ("Uncertain: " + "; ".join(reasons) + "." if reasons else
                       "The candidate meets the selected confidence threshold" +
                       (" and provisional reference-distance check." if detector is not None else
                        "; no reference-distance check was performed."))
        results.append({"label": "Uncertain" if uncertain else candidate,
                        "candidate_label": candidate, "confidence": confidence,
                        "prob_dog": p_dog, "prob_cat": 1.0 - p_dog,
                        "uncertain": uncertain, "ood": bool(ood),
                        "explanation": explanation})
    return results


def predict_image(model, im, threshold=0.5, min_confidence=0.7,
                  detector=None, source_path=None):
    """Predict one image, returning a candidate and an uncertainty-aware label."""
    img_size = int(model.input_shape[1])
    batch = preprocess_image(im, img_size)
    return determine_predictions(model, batch, threshold, min_confidence,
                                 detector, [source_path])[0]
