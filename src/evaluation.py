"""Dataset evaluation and a provisional, unvalidated reference-distance check.

The detector uses at most 11 sorted images from each of root/cats and root/dogs.
Its small in-domain reference set and 95th-percentile leave-one-out cutoff are
not calibrated or validated on out-of-distribution images. It is a heuristic,
not evidence that an image is in or out of distribution.
"""
from pathlib import Path
import os

import numpy as np
from PIL import Image

from src.data_cleaning import to_rgb
from src.predict import (determine_predictions, open_uploaded_image,
                         preprocess_image)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
BATCH_SIZE = 16
REFERENCES_PER_CLASS = 11


def _path_key(path):
    return None if path is None else os.path.normcase(str(Path(path).resolve()))


def _make_embedding_model(model):
    """Return a Keras penultimate-layer model, or None for an unsuitable model."""
    try:
        import tensorflow as tf
        if len(model.layers) < 2:
            return None
        output = model.layers[-2].output
        shape = output.shape
        if len(shape) != 2 or shape[-1] is None or int(shape[-1]) < 2:
            return None
        return tf.keras.Model(inputs=model.inputs, outputs=output)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _normalise_embeddings(values, count):
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 2:
        raise ValueError("Penultimate layer must produce one feature vector per image.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Penultimate layer produced invalid or zero feature vectors.")
    return vectors / norms[:, None]


class ReferenceDetector:
    """Provisional cosine nearest-neighbour check; not a calibrated OOD detector."""

    def __init__(self, embedding_model, vectors, paths, cutoff):
        self.embedding_model = embedding_model
        self.vectors = vectors
        self.paths = [_path_key(path) for path in paths]
        self.cutoff = float(cutoff)

    def is_ood(self, batch, source_paths=None):
        if source_paths is None:
            source_paths = [None] * len(batch)
        if len(source_paths) != len(batch):
            raise ValueError("source_paths must have one entry per image.")
        vectors = _normalise_embeddings(self.embedding_model.predict(batch, verbose=0), len(batch))
        if vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match the reference set.")
        distances = np.clip(1.0 - vectors @ self.vectors.T, 0.0, 2.0)
        for row, path in enumerate(source_paths):
            key = _path_key(path)
            if key is not None:
                for col, reference_path in enumerate(self.paths):
                    if key == reference_path:
                        distances[row, col] = np.inf
        nearest = distances.min(axis=1)
        if not np.all(np.isfinite(nearest)):
            raise ValueError("No reference remains after excluding the source image.")
        return nearest > self.cutoff + 1e-7


def _open_image(path):
    """Fully decode a supported external image before preprocessing it."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported file extension (use JPG, JPEG, PNG, or WEBP)")
    with Image.open(path) as image:
        if image.format not in SUPPORTED_FORMATS:
            raise ValueError("unsupported image format (use JPEG, PNG, or WEBP)")
        image.load()
        return image.copy()


def _class_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files, errors = [], []
    for folder, label in (("cats", "Cat"), ("dogs", "Dog")):
        directory = root / folder
        if not directory.is_dir():
            errors.append(f"Missing {label} directory: {directory}")
            continue
        files.extend((path, label) for path in sorted(directory.rglob("*")) if path.is_file())
    return files, errors


def build_reference_detector(model, root, progress=None):
    """Build a provisional detector, or return None if missing/insufficient/unsuitable.

    Uses the first 11 supported paths per class in deterministic order; corrupt
    files among those candidates are skipped. At least two usable images per
    class are required. Missing roots also return None; missing TensorFlow and
    prediction errors propagate. Progress counts candidate files, including
    unreadable ones.
    """
    if not Path(root).is_dir():
        return None
    files, _ = _class_files(root)
    candidates = []
    for label in ("Cat", "Dog"):
        candidates.extend([(p, lab) for p, lab in files
                           if lab == label and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                          [:REFERENCES_PER_CLASS])
    embedding_model = _make_embedding_model(model)
    if embedding_model is None:
        return None
    paths, labels, vectors = [], [], []
    pending, pending_paths, pending_labels = [], [], []
    total = len(candidates)

    def flush():
        if pending:
            batch = np.concatenate(pending, axis=0)
            features = embedding_model.predict(batch, verbose=0)
            vectors.append(np.asarray(features))
            paths.extend(pending_paths)
            labels.extend(pending_labels)
            pending.clear()
            pending_paths.clear()
            pending_labels.clear()

    for done, (path, label) in enumerate(candidates, 1):
        try:
            image = _open_image(path)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            pass
        else:
            pending.append(preprocess_image(image, int(model.input_shape[1])))
            pending_paths.append(path)
            pending_labels.append(label)
            if len(pending) == BATCH_SIZE:
                flush()
        if done == total:
            flush()
        if progress:
            progress(done, total)
    if not total and progress:
        progress(0, 0)
    if labels.count("Cat") < 2 or labels.count("Dog") < 2:
        return None
    try:
        reference = _normalise_embeddings(np.concatenate(vectors, axis=0), len(paths))
    except (ValueError, TypeError):
        return None
    distances = np.clip(1.0 - reference @ reference.T, 0.0, 2.0)
    np.fill_diagonal(distances, np.inf)
    cutoff = np.percentile(distances.min(axis=1), 95)
    return ReferenceDetector(embedding_model, reference, paths, cutoff)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _class_report(confusion, index, name):
    tp = confusion[index][index]
    support = sum(confusion[index])
    fp = sum(confusion[row][index] for row in range(2) if row != index)
    fn = support - tp
    return {"class": name, "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, support), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "support": support}


def _metrics(results):
    confusion = [[0, 0, 0], [0, 0, 0]]
    for result in results:
        row = ("Cat", "Dog").index(result["actual"])
        col = ("Cat", "Dog", "Uncertain").index(result["prediction"])
        confusion[row][col] += 1
    total = len(results)
    correct = confusion[0][0] + confusion[1][1]
    uncertain = confusion[0][2] + confusion[1][2]
    incorrect = total - correct - uncertain
    report = [_class_report(confusion, i, name) for i, name in enumerate(("Cat", "Dog"))]
    return {"total": total, "correct": correct, "incorrect": incorrect,
            "uncertain": uncertain, "accuracy": _ratio(correct, total),
            "cat_accuracy": report[0]["recall"], "dog_accuracy": report[1]["recall"],
            "precision": report[1]["precision"], "recall": report[1]["recall"],
            "f1": report[1]["f1"], "coverage": _ratio(total - uncertain, total),
            "selective_accuracy": _ratio(correct, total - uncertain),
            "confusion": confusion, "report": report}


def evaluate_dataset(model, root, min_confidence=0.7, detector=None, progress=None):
    """Evaluate root/cats and root/dogs; accuracy includes uncertainty as wrong.

    The 2x3 confusion matrix is actual Cat/Dog by predicted Cat/Dog/Uncertain.
    Undefined ratios are None. Only image I/O and decoding errors are collected;
    preprocessing, model and detector errors propagate. Progress includes every
    file, including unsupported and unreadable files.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    files, errors = _class_files(root)
    results = []
    total = len(files)
    for start in range(0, total, BATCH_SIZE):
        chunk = files[start:start + BATCH_SIZE]
        images, valid = [], []
        for path, actual in chunk:
            try:
                image = _open_image(path)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            images.append(preprocess_image(image, int(model.input_shape[1])))
            valid.append((path, actual))
        if valid:
            batch = np.concatenate(images, axis=0)
            predictions = determine_predictions(model, batch, min_confidence=min_confidence,
                                               detector=detector,
                                               source_paths=[path for path, _ in valid])
            for (path, actual), prediction in zip(valid, predictions):
                label = prediction["label"]
                status = "Uncertain" if label == "Uncertain" else ("Correct" if label == actual else "Incorrect")
                results.append({"path": str(path), "filename": path.name, "actual": actual,
                                "prediction": label, "confidence": prediction["confidence"],
                                "status": status, "prob_cat": prediction["prob_cat"],
                                "prob_dog": prediction["prob_dog"], "ood": prediction["ood"]})
        if progress:
            for done in range(start + 1, start + len(chunk) + 1):
                progress(done, total)
    if not total and progress:
        progress(0, 0)
    return {"results": results, "errors": errors, "metrics": _metrics(results),
            "threshold": min_confidence}


def predict_uploaded_files(model, uploads, min_confidence=0.7, detector=None, progress=None):
    """Classify a sequence of uploaded images in bounded model batches."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    uploads = list(uploads)
    results, errors = [], []
    total = len(uploads)
    img_size = int(model.input_shape[1])
    accepted_formats = SUPPORTED_FORMATS | {"BMP", "GIF"}
    for start in range(0, total, BATCH_SIZE):
        chunk = uploads[start:start + BATCH_SIZE]
        images, valid = [], []
        for upload in chunk:
            filename = getattr(upload, "name", "uploaded image")
            try:
                image = open_uploaded_image(upload.getvalue())
                if image.format not in accepted_formats:
                    raise ValueError("unsupported image format (use JPG, JPEG, PNG, WEBP, BMP or GIF)")
                images.append(preprocess_image(image, img_size))
                thumbnail = to_rgb(image)
                thumbnail.thumbnail((128, 128))
                valid.append((filename, thumbnail))
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                errors.append({"filename": filename, "error": str(error)})
        if valid:
            batch = np.concatenate(images, axis=0)
            predictions = determine_predictions(model, batch, min_confidence=min_confidence,
                                               detector=detector)
            for (filename, image), prediction in zip(valid, predictions):
                results.append({"filename": filename, "image": image,
                                "prediction": prediction["label"],
                                "candidate_label": prediction["candidate_label"],
                                "confidence": prediction["confidence"],
                                "prob_cat": prediction["prob_cat"],
                                "prob_dog": prediction["prob_dog"],
                                "uncertain": prediction["uncertain"],
                                "ood": prediction["ood"],
                                "explanation": prediction["explanation"]})
        if progress:
            for done in range(start + 1, start + len(chunk) + 1):
                progress(done, total)
    if not total and progress:
        progress(0, 0)
    return {"results": results, "errors": errors, "total": total,
            "threshold": min_confidence}
