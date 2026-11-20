"""Focused tests for batched inference, metrics, and provisional OOD behavior."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.evaluation import (ReferenceDetector, _metrics, build_reference_detector,
                            evaluate_dataset, predict_uploaded_files)
from src.predict import DetectorError, predict_image


def fake_preprocess(image, size):
    return np.asarray(image.convert("RGB"), dtype=np.float32)[0, 0][None, :]


class FakeModel:
    input_shape = (None, 32, 32, 3)

    def __init__(self):
        self.batch_sizes = []

    def predict(self, batch, verbose=0):
        self.batch_sizes.append(len(batch))
        return np.asarray(batch)[:, :1] / 255.0


class FakeEmbeddingModel:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, batch, verbose=0):
        self.batch_sizes.append(len(batch))
        return np.asarray(batch, dtype=float)[:, :3]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "cats").mkdir()
        (self.root / "dogs").mkdir()
        self.preprocess_patch = patch("src.evaluation.preprocess_image", fake_preprocess)
        self.preprocess_patch.start()
        self.addCleanup(self.preprocess_patch.stop)

    def image(self, folder, name, rgb):
        path = self.root / folder / name
        Image.new("RGB", (32, 32), rgb).save(path)
        return path

    def test_metrics_errors_order_and_progress(self):
        self.image("cats", "a.png", (25, 50, 0))
        self.image("cats", "b.png", (230, 50, 0))
        self.image("dogs", "a.png", (204, 50, 0))
        self.image("dogs", "b.png", (153, 50, 0))
        self.image("dogs", "c.png", (25, 50, 0))
        (self.root / "cats" / "broken.jpg").write_bytes(b"not an image")
        (self.root / "dogs" / "note.txt").write_text("not an image")
        progress = []
        model = FakeModel()
        output = evaluate_dataset(model, self.root, progress=lambda d, t: progress.append((d, t)))
        metrics = output["metrics"]
        self.assertEqual([r["filename"] for r in output["results"]],
                         ["a.png", "b.png", "a.png", "b.png", "c.png"])
        self.assertEqual([r["status"] for r in output["results"]],
                         ["Correct", "Incorrect", "Correct", "Uncertain", "Incorrect"])
        self.assertEqual(metrics["confusion"], [[1, 1, 0], [1, 1, 1]])
        self.assertEqual((metrics["total"], metrics["correct"], metrics["incorrect"], metrics["uncertain"]),
                         (5, 2, 2, 1))
        self.assertAlmostEqual(metrics["accuracy"], .4)
        self.assertAlmostEqual(metrics["cat_accuracy"], .5)
        self.assertAlmostEqual(metrics["dog_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], 1 / 3)
        self.assertAlmostEqual(metrics["f1"], .4)
        self.assertAlmostEqual(metrics["coverage"], .8)
        self.assertAlmostEqual(metrics["selective_accuracy"], .5)
        self.assertEqual(metrics["report"][0]["support"], 2)
        self.assertEqual(len(output["errors"]), 2)
        self.assertEqual(progress, [(i, 7) for i in range(1, 8)])
        self.assertEqual(model.batch_sizes, [5])
        self.assertEqual(output["threshold"], .7)

    def test_empty_and_undefined_metrics(self):
        metrics = evaluate_dataset(FakeModel(), self.root)["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "coverage", "selective_accuracy"):
            self.assertIsNone(metrics[key])
        metrics = _metrics([{"actual": "Dog", "prediction": "Uncertain"}])
        self.assertEqual(metrics["recall"], 0)
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["f1"], 0)
        self.assertIsNone(metrics["selective_accuracy"])
        self.assertIsNone(metrics["cat_accuracy"])

    def test_single_prediction_candidate_and_low_confidence(self):
        image = Image.new("RGB", (32, 32), (153, 80, 0))
        with patch("src.predict.preprocess_image", fake_preprocess):
            result = predict_image(FakeModel(), image, min_confidence=.7)
        self.assertEqual(result["label"], "Uncertain")
        self.assertEqual(result["candidate_label"], "Dog")
        self.assertTrue(result["uncertain"])
        self.assertFalse(result["ood"])
        self.assertIn("confidence", result["explanation"])

    def test_reference_cutoff_and_self_match_exclusion(self):
        paths = [self.image("cats", "a.png", (255, 0, 0)),
                 self.image("cats", "b.png", (255, 20, 0)),
                 self.image("dogs", "a.png", (0, 255, 0)),
                 self.image("dogs", "b.png", (20, 255, 0))]
        embedder = FakeEmbeddingModel()
        progress = []
        with patch("src.evaluation._make_embedding_model", return_value=embedder):
            detector = build_reference_detector(FakeModel(), self.root,
                                                progress=lambda d, t: progress.append((d, t)))
        self.assertIsNotNone(detector)
        self.assertEqual(embedder.batch_sizes, [4])
        self.assertEqual(progress[-1], (4, 4))
        similarities = detector.vectors @ detector.vectors.T
        distances = np.clip(1 - similarities, 0, 2)
        np.fill_diagonal(distances, np.inf)
        self.assertAlmostEqual(detector.cutoff, np.percentile(distances.min(axis=1), 95))
        # A deliberately tighter cutoff makes the difference between matching
        # oneself and using the nearest distinct reference observable.
        detector.cutoff = 1e-5
        batch = np.array([[255, 0, 0]], dtype=float)
        self.assertFalse(detector.is_ood(batch)[0])
        self.assertTrue(detector.is_ood(batch, [paths[0]])[0])
        with patch("src.predict.preprocess_image", fake_preprocess):
            result = predict_image(FakeModel(), Image.new("RGB", (32, 32), (255, 0, 0)),
                                   detector=detector, source_path=paths[0])
        self.assertTrue(result["ood"])
        self.assertEqual(result["label"], "Uncertain")
        self.assertEqual(result["candidate_label"], "Dog")
        self.assertIn("provisional", result["explanation"])

    def test_reference_insufficient_and_unsuitable(self):
        with patch("src.evaluation._make_embedding_model", return_value=FakeEmbeddingModel()):
            self.assertIsNone(build_reference_detector(FakeModel(), self.root))
            self.assertIsNone(build_reference_detector(FakeModel(), self.root / "missing"))
        with patch("src.evaluation._make_embedding_model", return_value=None):
            self.assertIsNone(build_reference_detector(FakeModel(), self.root))

    def test_batch_detector_gets_each_source_path(self):
        path = self.image("cats", "a.png", (25, 50, 0))

        class Detector:
            def is_ood(self, batch, source_paths):
                self.paths = source_paths
                return [True] * len(batch)

        detector = Detector()
        output = evaluate_dataset(FakeModel(), self.root, detector=detector)
        self.assertEqual(detector.paths, [path])
        self.assertEqual(output["results"][0]["prediction"], "Uncertain")
        self.assertTrue(output["results"][0]["ood"])

    def test_detector_errors_are_distinct_from_model_errors(self):
        class BrokenDetector:
            def is_ood(self, batch, source_paths):
                raise RuntimeError("detector failed")

        with patch("src.predict.preprocess_image", fake_preprocess):
            with self.assertRaises(DetectorError):
                predict_image(FakeModel(), Image.new("RGB", (32, 32), (25, 50, 0)),
                              detector=BrokenDetector())

    def test_model_errors_are_not_reported_as_bad_images(self):
        self.image("cats", "a.png", (25, 50, 0))

        class BrokenModel(FakeModel):
            def predict(self, batch, verbose=0):
                raise RuntimeError("model failed")

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            evaluate_dataset(BrokenModel(), self.root)

    def test_multi_upload_predictions_continue_after_bad_file(self):
        class Upload(io.BytesIO):
            def __init__(self, name, data):
                super().__init__(data)
                self.name = name

        def encoded(color):
            data = io.BytesIO()
            Image.new("RGB", (32, 32), color).save(data, format="PNG")
            return data.getvalue()

        uploads = [Upload("cat.png", encoded((25, 0, 0))),
                   Upload("broken.png", b"not an image"),
                   Upload("dog.png", encoded((230, 0, 0)))]
        progress = []
        output = predict_uploaded_files(FakeModel(), uploads,
                                        progress=lambda done, total: progress.append((done, total)))
        self.assertEqual([result["filename"] for result in output["results"]], ["cat.png", "dog.png"])
        self.assertEqual([result["prediction"] for result in output["results"]], ["Cat", "Dog"])
        self.assertEqual(len(output["errors"]), 1)
        self.assertEqual(output["errors"][0]["filename"], "broken.png")
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual([result["image"].size for result in output["results"]], [(32, 32), (32, 32)])

    def test_batches_are_bounded(self):
        for i in range(35):
            self.image("cats", f"{i:02d}.png", (25, 50, 0))
        model = FakeModel()
        output = evaluate_dataset(model, self.root)
        self.assertEqual(model.batch_sizes, [16, 16, 3])
        self.assertEqual(output["metrics"]["total"], 35)


if __name__ == "__main__":
    unittest.main()
