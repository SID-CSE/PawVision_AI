"""Smoke tests and image-validation checks for PawVision AI."""

import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.predict import open_uploaded_image, predict_image

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
MODEL_PATH = ROOT / "models" / "final_model.keras"


class AppSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()

    def test_missing_model_shows_friendly_error(self) -> None:
        original_is_file = Path.is_file

        def is_file(path: Path) -> bool:
            if path == MODEL_PATH:
                return False
            return original_is_file(path)

        with patch.object(Path, "is_file", is_file):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn("final_model.keras", app.error[0].value)
        self.assertIn("models/", app.error[0].value)

    def test_model_present_and_navigation(self) -> None:
        self.assertTrue(MODEL_PATH.is_file(), "The bundled model is missing")
        app = AppTest.from_file(str(APP_PATH)).run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.radio(key="page").value, "Dashboard")
        for page in ["Image Classifier", "Prediction History", "Model Insights", "How It Works", "About", "Settings"]:
            app.radio(key="page").set_value(page).run(timeout=120)
            self.assertFalse(app.exception, page)
            self.assertFalse(app.error, page)
        app.radio(key="page").set_value("Settings").run(timeout=120)
        app.slider(key="threshold").set_value(0.90).run(timeout=120)
        self.assertEqual(app.session_state["threshold"], 0.90)
        app.selectbox(key="settings_theme").set_value("Dark").run(timeout=120)
        self.assertEqual(app.session_state["theme"], "Dark")


class PredictionTests(unittest.TestCase):
    def test_supported_image_formats(self) -> None:
        for image_format in ("JPEG", "PNG", "WEBP", "BMP", "GIF"):
            with self.subTest(image_format=image_format):
                data = io.BytesIO()
                Image.new("RGB", (40, 36), "white").save(data, format=image_format)
                image = open_uploaded_image(data.getvalue())
                self.assertEqual(image.size, (40, 36))
                self.assertEqual(image.format, image_format)

    def test_invalid_small_and_oversized_images(self) -> None:
        for data in (b"", b"not an image", b"x" * (10 * 1024 * 1024 + 1)):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                open_uploaded_image(data)
        data = io.BytesIO()
        Image.new("RGB", (31, 40)).save(data, format="PNG")
        with self.assertRaises(ValueError):
            open_uploaded_image(data.getvalue())

    def test_confidence_threshold_does_not_change_class(self) -> None:
        class Model:
            input_shape = (None, 224, 224, 3)

            def predict(self, batch, verbose=0):
                return np.array([[0.6]])

        with patch("src.predict.preprocess_image", return_value=np.zeros((1, 1))):
            confident = predict_image(Model(), Image.new("RGB", (32, 32)), min_confidence=0.59)
            uncertain = predict_image(Model(), Image.new("RGB", (32, 32)), min_confidence=0.70)
        self.assertEqual(confident["label"], "Dog")
        self.assertEqual(uncertain["label"], "Dog")
        self.assertFalse(confident["uncertain"])
        self.assertTrue(uncertain["uncertain"])
        self.assertAlmostEqual(uncertain["confidence"], 0.6)
        self.assertAlmostEqual(uncertain["prob_cat"], 0.4)


if __name__ == "__main__":
    unittest.main()
