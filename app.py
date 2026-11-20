"""PawVision AI: a session-based cat and dog classification workspace."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from src.evaluation import (REFERENCES_PER_CLASS, SUPPORTED_EXTENSIONS,
                            build_reference_detector, evaluate_dataset,
                            predict_uploaded_files)

from src.data_cleaning import to_rgb
from src.pawvision_styles import apply_styles
from src.predict import DetectorError, load_trained_model, open_uploaded_image, predict_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.keras"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"
DEFAULT_TEST_METRICS = {
    "accuracy": 0.9893,
    "precision": 0.988,
    "recall": 0.991,
    "f1": 0.989,
    "roc_auc": 0.9997,
}
FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "gif"]
IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}
PAGES = ["Dashboard", "Image Classifier", "Folder Evaluation", "Prediction History", "Model Insights", "How It Works", "About", "Settings"]
NAV_ICONS = {"Dashboard": "▦", "Image Classifier": "◎", "Folder Evaluation": "▧", "Prediction History": "◷", "Model Insights": "▥", "How It Works": "◇", "About": "ⓘ", "Settings": "⚙"}

st.set_page_config(page_title="PawVision AI", page_icon="🐾", layout="wide", initial_sidebar_state="expanded")

for key, value in {
    "page": "Dashboard", "theme": "Light", "threshold": 0.70,
    "show_probabilities": True, "history": [], "upload_version": 0,
    "last_upload": None, "last_result": None, "last_threshold": None,
    "last_history_index": None, "last_detector_used": False, "confirm_clear": False,
    "evaluation_result": None, "evaluation_threshold": None,
    "evaluation_detector_used": False, "evaluation_page_number": 1,
    "batch_result": None, "batch_page_number": 1, "batch_fingerprint": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value
for key in ("sidebar_theme", "settings_theme"):
    if key not in st.session_state:
        st.session_state[key] = st.session_state.theme


@st.cache_resource(show_spinner="Loading the model for the first time...")
def get_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    return load_trained_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_reference_detector(_model: Any, model_path: str, root: str, reference_version: tuple) -> Any:
    """Cache the experimental detector per model and reference directory."""
    return build_reference_detector(_model, Path(root))


def detector_notice() -> None:
    st.caption("Experimental feature-distance detector: built from the tiny bundled cats/dogs dataset, not validated for out-of-distribution (OOD) detection. Neither it nor the confidence threshold guarantees correctness or detects every unfamiliar image.")


def get_detector(model: Any) -> Any:
    try:
        with st.spinner("Preparing the experimental reference detector..."):
            references = []
            for folder in ("cats", "dogs"):
                directory = ROOT / folder
                paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)[:REFERENCES_PER_CLASS]
                for path in paths:
                    info = path.stat()
                    references.append((str(path), info.st_size, info.st_mtime_ns))
            detector = get_reference_detector(model, str(MODEL_PATH), str(ROOT), tuple(references))
        if detector is None:
            st.warning("Experimental feature-distance detector unavailable. Predictions will use confidence only; unfamiliar images may not be recognized.")
        return detector
    except Exception:
        st.warning("Experimental feature-distance detector could not be built. Predictions will use confidence only; unfamiliar images may not be recognized.")
        return None


@st.cache_data
def load_test_metrics() -> dict[str, float]:
    """Read recorded evaluation scores, with documented rounded values as fallback."""
    try:
        card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
        return card["test_metrics"]
    except (OSError, json.JSONDecodeError, KeyError):
        return DEFAULT_TEST_METRICS


def navigate(page: str) -> None:
    st.session_state.page = page


def navigate_to_batch_prediction() -> None:
    st.session_state.classifier_mode = "Batch prediction"
    st.session_state.page = "Image Classifier"


def sync_theme(source: str) -> None:
    theme = st.session_state[source]
    st.session_state.theme = theme
    st.session_state["settings_theme" if source == "sidebar_theme" else "sidebar_theme"] = theme


def clear_upload() -> None:
    st.session_state.upload_version += 1
    st.session_state.last_upload = None
    st.session_state.last_result = None
    st.session_state.last_threshold = None
    st.session_state.last_history_index = None


def clear_history() -> None:
    st.session_state.history = []
    st.session_state.last_history_index = None
    st.session_state.confirm_clear = False


def page_intro(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="page-intro"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def section(title: str, detail: str = "") -> None:
    st.markdown(f'<div class="section-header"><h2>{title}</h2><span>{detail}</span></div>', unsafe_allow_html=True)


def metrics() -> None:
    values = load_test_metrics()
    columns = st.columns(5)
    for col, (label, value) in zip(
        columns,
        [
            ("Accuracy", f'{values["accuracy"]:.2%}'),
            ("Precision", f'{values["precision"]:.3f}'),
            ("Recall", f'{values["recall"]:.3f}'),
            ("F1 score", f'{values["f1"]:.3f}'),
            ("ROC-AUC", f'{values["roc_auc"]:.4f}'),
        ],
    ):
        col.markdown(
            f'<div class="metric-card"><small>{label}</small><strong>{value}</strong></div>',
            unsafe_allow_html=True,
        )


def evaluation_chart() -> None:
    values = load_test_metrics()
    chart = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Score (%)": [
                values["accuracy"] * 100,
                values["precision"] * 100,
                values["recall"] * 100,
                values["f1"] * 100,
                values["roc_auc"] * 100,
            ],
        }
    )
    st.bar_chart(chart, x="Metric", y="Score (%)", horizontal=True, x_label="Score (%)")
    st.caption("Source: models/model_card.json · rounded held-out test metrics · 3,745 images.")


def confirmation(key: str) -> None:
    if st.button("Clear session history", key=f"clear_{key}"):
        st.session_state.confirm_clear = True
    if st.session_state.confirm_clear:
        st.warning("This will remove all predictions from this session. This cannot be undone.")
        a, b = st.columns(2)
        a.button("Yes, clear history", key=f"confirm_{key}", on_click=clear_history, type="primary")
        if b.button("Cancel", key=f"cancel_{key}"):
            st.session_state.confirm_clear = False
            st.rerun()


def dashboard() -> None:
    st.markdown("""<div class="hero-wrap"><div class="eyebrow">A LITTLE CLARITY, POWERED BY AI</div>
    <h1>Discover what's in your image.</h1><p>Meet your intelligent companion for understanding cat and dog photos. Simply upload an image and let PawVision take a closer look.</p>
    <div class="hero-art" aria-hidden="true"><svg viewBox="0 0 180 150" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="24" y="16" width="130" height="116" rx="15" fill="#fff" stroke="#ddd6ef" stroke-width="2" transform="rotate(-7 24 16)"/><rect x="48" y="29" width="112" height="99" rx="12" fill="#e5f1ec" stroke="#c9e2d9" stroke-width="2" transform="rotate(6 48 29)"/><circle cx="99" cy="77" r="25" fill="#b8a8db"/><path d="M78 66L80 45L94 55M104 55L118 45L120 66" fill="#b8a8db" stroke="#b8a8db" stroke-width="5" stroke-linejoin="round"/><circle cx="91" cy="76" r="2" fill="#514667"/><circle cx="107" cy="76" r="2" fill="#514667"/><path d="M96 85Q99 89 102 85" stroke="#514667" stroke-width="2" stroke-linecap="round"/><circle cx="149" cy="32" r="14" fill="#8472c7"/><path d="M143 32L147 36L155 28" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div></div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="small")
    for col in (a, b, c):
        with col:
            st.markdown('<div class="hero-action-marker"></div>', unsafe_allow_html=True)
    a.button("Classify one image  →", type="primary", on_click=navigate, args=("Image Classifier",), width="stretch")
    b.button("Evaluate local folders  →", on_click=navigate, args=("Folder Evaluation",), width="stretch")
    c.button("Explore model insights  ↗", on_click=navigate, args=("Model Insights",), width="stretch")
    section("Everything you need, at a glance", "BUILT FOR A CLOSER LOOK")
    features = [("✦", "AI-powered classification", "A fine-tuned model that knows its cats from its dogs."), ("◉", "Confidence analysis", "See the probabilities behind every prediction."), ("ϟ", "Instant predictions", "Go from image to insight in just a few moments."), ("◷", "Prediction history", "Keep track of your discoveries during this session.")]
    for col, (icon, title, description) in zip(st.columns(4), features):
        col.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
    section("Built on performance", "HELD-OUT TEST RESULTS")
    metrics()
    st.caption("Evaluated on 3,745 held-out test images · Results reflect model evaluation, not live prediction statistics.")
    st.markdown('<div class="panel"><h3>Thoughtful technology, simple experience.</h3><p>Built with transfer learning and a fine-tuned MobileNetV2 model, PawVision turns complex image analysis into a clear, easy-to-understand result.</p></div>', unsafe_allow_html=True)


def batch_classifier(model: Any) -> None:
    st.subheader("Batch prediction")
    st.caption("Upload multiple images to get a Cat, Dog, or Uncertain prediction for each one.")
    uploads = st.file_uploader("Upload images", type=FORMATS, accept_multiple_files=True,
                               key="batch_uploads", help="Select JPG, JPEG, PNG, WEBP, BMP, or GIF images.")
    threshold = st.session_state.threshold
    fingerprint = tuple((upload.name, sha256(upload.getvalue()).hexdigest()) for upload in uploads)
    st.caption(f"Confidence threshold: {threshold:.0%}. Each image is classified separately; results are not compared with folder labels.")
    if st.button("Predict all images", type="primary", disabled=not uploads):
        detector = get_detector(model)
        progress_bar = st.progress(0, text="Preparing batch predictions...")

        def update_progress(done: int, total: int) -> None:
            fraction = min(1.0, max(0.0, done / total)) if total else 0.0
            progress_bar.progress(fraction, text=f"Predicting images: {done} / {total}")

        try:
            try:
                output = predict_uploaded_files(model, uploads, min_confidence=threshold,
                                                detector=detector, progress=update_progress)
            except DetectorError:
                st.warning("The experimental detector failed. Retrying predictions with confidence only.")
                detector = None
                progress_bar.progress(0, text="Retrying without the experimental detector...")
                output = predict_uploaded_files(model, uploads, min_confidence=threshold,
                                                detector=None, progress=update_progress)
            st.session_state.batch_result = output
            st.session_state.batch_fingerprint = fingerprint
            st.session_state.batch_threshold = threshold
            st.session_state.batch_detector_used = detector is not None
            st.session_state.batch_page_number = 1
            progress_bar.progress(1.0, text="Batch predictions complete")
        except Exception as error:
            progress_bar.empty()
            st.error(f"Batch prediction failed: {error}")

    output = st.session_state.batch_result
    if output is None:
        st.info("Choose multiple images above, then select **Predict all images**. Predictions will appear here.")
        return
    if st.session_state.batch_fingerprint != fingerprint:
        st.info("The selected files differ from the last batch. Select **Predict all images** to update the results.")
        return
    if st.session_state.batch_threshold != threshold:
        st.info("The confidence setting has changed since this batch. Run predictions again to use the new threshold.")
    if not st.session_state.batch_detector_used:
        st.warning("The experimental reference detector was unavailable; unfamiliar images may not be recognized.")
    results = output["results"]
    errors = output["errors"]
    summary_columns = st.columns(3)
    summary_columns[0].metric("Uploaded", output["total"])
    summary_columns[1].metric("Predicted", len(results))
    summary_columns[2].metric("Could not read", len(errors))
    if errors:
        with st.expander(f"Files with errors ({len(errors)})"):
            for item in errors:
                st.text(f"{item['filename']}: {item['error']}")
    if not results:
        st.warning("No images could be predicted. Check the file formats and image dimensions, then try again.")
        return
    result_frame = pd.DataFrame(results)
    selected_filter = st.selectbox("Filter predictions", ["All", "Cat", "Dog", "Uncertain"], key="batch_filter")
    if selected_filter != "All":
        result_frame = result_frame[result_frame["prediction"] == selected_filter]
    if result_frame.empty:
        st.info("No images match this filter.")
        return
    page_size = 12
    pages = (len(result_frame) + page_size - 1) // page_size
    st.session_state.batch_page_number = min(st.session_state.batch_page_number, pages)
    if pages > 1:
        st.number_input("Batch results page", min_value=1, max_value=pages, step=1,
                        key="batch_page_number")
    current_page = st.session_state.batch_page_number
    st.caption(f"Showing {len(result_frame)} predictions · page {current_page} of {pages}")
    visible = result_frame.iloc[(current_page - 1) * page_size:current_page * page_size]
    records = visible.to_dict("records")
    for start in range(0, len(records), 4):
        for col, record in zip(st.columns(4), records[start:start + 4]):
            with col:
                with st.container(border=True):
                    st.markdown('<div class="batch-card-marker"></div>', unsafe_allow_html=True)
                    thumbnail = record["image"]
                    st.image(thumbnail, width=100)
                    st.text(str(record["filename"]))
                    st.markdown(f"**{record['prediction']}** · {record['confidence']:.1%} confidence")
                    if record["prediction"] == "Uncertain":
                        st.caption(f"Candidate: {record['candidate_label']}")
                    if record["ood"]:
                        st.caption("Potentially out of distribution · experimental flag")
                    st.caption(record["explanation"])


def classifier(model: Any) -> None:
    page_intro("IMAGE CLASSIFIER", "A closer look starts here.", "Upload one image or switch to batch prediction for multiple images.")
    detector_notice()
    mode = st.radio("Prediction mode", ["Single image", "Batch prediction"], horizontal=True,
                    key="classifier_mode")
    if mode == "Batch prediction":
        batch_classifier(model)
        return
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="classifier-column-marker"></div>', unsafe_allow_html=True)
        st.subheader("Your image")
        st.caption("Choose a clear photo containing one cat or dog.")
        uploaded = st.file_uploader("Upload an image", type=FORMATS, key=f"upload_{st.session_state.upload_version}", help="Maximum 10 MB; shortest side at least 32 pixels.")
        st.caption("JPG, JPEG, PNG, WEBP, BMP, GIF · Up to 10 MB · Minimum 32 × 32 px")
        image = None
        if uploaded is not None:
            try:
                uploaded.seek(0)
                image = open_uploaded_image(uploaded)
                if image.format not in IMAGE_FORMATS:
                    raise ValueError("Unsupported image format. Please upload a JPG, PNG, WEBP, BMP, or GIF.")
                display_image = to_rgb(image)
                st.image(display_image, width="stretch")
                st.write(uploaded.name)
                st.caption(f"{display_image.width} × {display_image.height} px  ·  {uploaded.size / 1024:.1f} KB")
                st.button("Remove image", on_click=clear_upload)
            except ValueError as error:
                st.session_state.last_result = None
                st.session_state.last_upload = None
                st.session_state.last_threshold = None
                st.error(str(error))
                image = None
                st.button("Remove image", on_click=clear_upload)
        else:
            st.session_state.last_upload = None
            st.session_state.last_result = None
            st.session_state.last_threshold = None
            st.session_state.last_history_index = None
    with right:
        st.markdown('<div class="classifier-column-marker"></div>', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.caption("A little insight into what your image contains.")
        if image is None:
            st.markdown('<div class="panel"><div class="feature-icon">◎</div><h3>Your result will appear here</h3><p>Upload an image to see the prediction, confidence, and probability breakdown.</p></div>', unsafe_allow_html=True)
        else:
            fingerprint = (sha256(uploaded.getvalue()).hexdigest(), uploaded.name, st.session_state.upload_version)
            is_new = st.session_state.last_upload != fingerprint
            if is_new or st.session_state.last_threshold != st.session_state.threshold:
                detector = get_detector(model)
                try:
                    with st.spinner("Taking a closer look..."):
                        try:
                            result = predict_image(model, image, min_confidence=st.session_state.threshold, detector=detector)
                        except DetectorError:
                            st.warning("The experimental detector failed during prediction. Retrying with confidence only; unfamiliar images may not be recognized.")
                            detector = None
                            result = predict_image(model, image, min_confidence=st.session_state.threshold, detector=None)
                    st.session_state.last_detector_used = detector is not None
                    st.session_state.last_upload = fingerprint
                    st.session_state.last_threshold = st.session_state.threshold
                    st.session_state.last_result = result
                    abstained = result["uncertain"] or result["label"] == "Uncertain"
                    entry = {"Filename": uploaded.name, "Prediction": "Uncertain" if abstained else result["label"], "Confidence": result["confidence"], "Cat probability": result["prob_cat"], "Dog probability": result["prob_dog"], "Threshold": st.session_state.threshold, "Status": "Uncertain" if abstained else "Threshold met", "Timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}
                    if is_new:
                        st.session_state.history.append(entry)
                        st.session_state.last_history_index = len(st.session_state.history) - 1
                    elif st.session_state.last_history_index is not None and st.session_state.last_history_index < len(st.session_state.history):
                        st.session_state.history[st.session_state.last_history_index] = entry
                except Exception:
                    st.session_state.last_result = None
                    st.session_state.last_upload = None
                    st.error("The image could not be analyzed. Please try another image.")
                    return
            result = st.session_state.last_result
            if not st.session_state.last_detector_used:
                st.warning("Experimental detector unavailable for this prediction. Unfamiliar images may not be recognized.")
            uncertain = result["uncertain"] or result["label"] == "Uncertain"
            status = "Uncertain" if uncertain else "Threshold met"
            badge_class = "badge warn" if uncertain else "badge"
            label = "Uncertain" if uncertain else result["label"]
            st.markdown(f'<div class="panel"><span class="{badge_class}">{status}</span><div class="result-title">{label}</div><div class="result-caption">MODEL CONFIDENCE</div><div class="result-confidence">{result["confidence"]:.1%}</div></div>', unsafe_allow_html=True)
            st.caption("The model abstained from a Cat/Dog result." if uncertain else "Confidence meets your selected threshold; this does not guarantee correctness.")
            if result.get("ood"):
                st.warning("Potentially out of distribution: this image differs from the small reference set. This experimental flag is not a definitive identification.")
            if result.get("explanation"):
                st.write(result["explanation"])
            if st.session_state.show_probabilities:
                st.markdown("**Probability breakdown**")
                st.progress(float(result["prob_cat"]), text=f'Cat  ·  {result["prob_cat"]:.1%}')
                st.progress(float(result["prob_dog"]), text=f'Dog  ·  {result["prob_dog"]:.1%}')
            st.caption("Probabilities reflect the model's Cat/Dog scores, including when it abstains. Confidence is not a guarantee of correctness.")


def reset_evaluation_page() -> None:
    st.session_state.evaluation_page_number = 1


def dataset_evaluation(model: Any) -> None:
    page_intro("FOLDER EVALUATION", "See the bigger picture.", "Evaluate supported images already stored in the repository's cats/ and dogs/ folders.")
    st.info("To upload your own group of images, go to Image Classifier and select Batch prediction.")
    st.button("Upload a batch of images", on_click=navigate_to_batch_prediction)
    detector_notice()
    counts = {
        label: sum(1 for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        if (ROOT / folder).is_dir() else 0
        for label, folder in (("Cats", "cats"), ("Dogs", "dogs"))
    }
    count_columns = st.columns(2)
    count_columns[0].metric("Cat images found", counts["Cats"])
    count_columns[1].metric("Dog images found", counts["Dogs"])
    st.info("Folder names supply actual labels. The bundled images also build the experimental reference detector, so these results are not an independent validation or a replacement for held-out test results.")
    threshold = st.session_state.threshold
    st.caption(f"Current confidence threshold: {threshold:.0%}. Change it in Settings; each run keeps its own threshold snapshot.")
    if st.button(f"Run batch predictions · {sum(counts.values())} images", type="primary", disabled=sum(counts.values()) == 0):
        detector = get_detector(model)
        progress_bar = st.progress(0, text="Preparing dataset evaluation...")

        def update_progress(done: int, total: int) -> None:
            fraction = min(1.0, max(0.0, done / total)) if total else 0.0
            progress_bar.progress(fraction, text=f"Evaluating images: {done} / {total}")

        try:
            try:
                evaluated = evaluate_dataset(model, ROOT, min_confidence=threshold, detector=detector, progress=update_progress)
            except DetectorError:
                st.warning("The experimental detector failed. Retrying without the detector; unfamiliar images may not be recognized.")
                detector = None
                progress_bar.progress(0, text="Retrying with confidence only...")
                evaluated = evaluate_dataset(model, ROOT, min_confidence=threshold, detector=None, progress=update_progress)
            st.session_state.evaluation_result = evaluated
            st.session_state.evaluation_threshold = threshold
            st.session_state.evaluation_detector_used = detector is not None
            reset_evaluation_page()
            progress_bar.progress(1.0, text="Evaluation complete")
        except Exception as error:
            progress_bar.empty()
            st.error(f"Dataset evaluation failed: {error}")
            st.caption("Check that the cats/ and dogs/ folders are accessible. Any previous successful run is shown below.")

    data = st.session_state.evaluation_result
    if data is None:
        st.info("Select **Run batch predictions** above. The app will process each image, then show its actual label, prediction, confidence, and evaluation metrics below.")
        return
    run_threshold = st.session_state.evaluation_threshold
    st.caption(f"Showing the last completed run · threshold {run_threshold:.0%} · " + ("experimental detector enabled" if st.session_state.evaluation_detector_used else "confidence only"))
    if not st.session_state.evaluation_detector_used:
        st.warning("The experimental detector was unavailable for this run. Unfamiliar images may not be recognized.")
    if run_threshold != threshold:
        st.info("Settings have changed since this run. Run evaluation again to apply the current threshold.")

    results = data.get("results", [])
    errors = data.get("errors", [])
    if errors:
        with st.expander(f"Skipped images and errors ({len(errors)})", expanded=not results):
            for error in errors:
                st.text(str(error))
    if not results:
        st.warning("No images were evaluated. Check the dataset folders and any errors above.")
        return

    summary = data["metrics"]
    section("Evaluation overview", "COMPLETED IMAGES")
    count_columns = st.columns(5)
    for col, (label, value) in zip(count_columns, [("Evaluated", summary["total"]), ("Correct", summary["correct"]), ("Incorrect", summary["incorrect"]), ("Uncertain", summary["uncertain"]), ("Potential OOD", sum(bool(result["ood"]) for result in results))]):
        col.metric(label, value)
    score_columns = st.columns(4)
    for col, (label, key) in zip(score_columns, [("Accuracy", "accuracy"), ("Coverage", "coverage"), ("Selective accuracy", "selective_accuracy"), ("F1 · Dog", "f1")]):
        value = summary.get(key)
        col.metric(label, f"{value:.1%}" if value is not None else "N/A")
    detail_columns = st.columns(4)
    for col, (label, key) in zip(detail_columns, [("Cat accuracy", "cat_accuracy"), ("Dog accuracy", "dog_accuracy"), ("Precision · Dog", "precision"), ("Recall · Dog", "recall")]):
        value = summary.get(key)
        col.metric(label, f"{value:.1%}" if value is not None else "N/A")
    st.caption("Only successfully evaluated images are included; skipped files are excluded. Accuracy = correct / all evaluated, so abstentions count as errors. Cat and Dog accuracy = correct for that class / all images of that class. Coverage = Cat/Dog predictions / all evaluated; selective accuracy = correct / Cat/Dog predictions. For Dog as the positive class, precision = true Dog predictions / all Dog predictions; recall = true Dog predictions / all actual Dogs, including abstentions; F1 is their harmonic mean. Undefined ratios are shown as N/A. These scores and confidence are not guarantees.")

    section("Confusion matrix", "ACTUAL LABEL × PREDICTION")
    confusion = pd.DataFrame(summary["confusion"], index=pd.Index(["Cat", "Dog"], name="Actual"), columns=["Cat", "Dog", "Uncertain"])
    st.dataframe(confusion, width="stretch")
    st.caption("Rows are folder labels; columns are the model's final predictions, including abstentions.")
    section("Class report", "PER-CLASS METRICS")
    st.dataframe(pd.DataFrame(summary["report"]), hide_index=True, width="stretch")

    frame = pd.DataFrame(results)
    section("Prediction patterns", "ALL EVALUATED IMAGES")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("**Prediction distribution**")
        counts = frame["prediction"].value_counts().reindex(["Cat", "Dog", "Uncertain"], fill_value=0)
        st.bar_chart(counts.rename_axis("Prediction").rename("Images"))
    with chart_b:
        st.markdown("**Confidence distribution**")
        bins = pd.cut(frame["confidence"], bins=[0, .5, .6, .7, .8, .9, 1.0], labels=["0–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"], include_lowest=True)
        st.bar_chart(bins.value_counts(sort=False).rename_axis("Confidence").rename("Images"))
    st.caption("Confidence is the score of the model's candidate class, even when the final prediction is Uncertain.")

    section("Image results", "FILTER AND EXPLORE")
    selected_filter = st.selectbox("Filter images", ["All", "Correct", "Incorrect", "Uncertain", "Actual Cat", "Actual Dog"], key="evaluation_filter", on_change=reset_evaluation_page)
    if selected_filter in ("Correct", "Incorrect", "Uncertain"):
        frame = frame[frame["status"] == selected_filter]
    elif selected_filter in ("Actual Cat", "Actual Dog"):
        frame = frame[frame["actual"] == selected_filter.removeprefix("Actual ")]
    if frame.empty:
        st.info("No images match this filter.")
        return
    page_size = 10
    pages = (len(frame) + page_size - 1) // page_size
    st.session_state.evaluation_page_number = min(st.session_state.evaluation_page_number, pages)
    if pages > 1:
        st.number_input("Page", min_value=1, max_value=pages, step=1, key="evaluation_page_number")
    current_page = st.session_state.evaluation_page_number
    st.caption(f"Page {current_page} of {pages} · {len(frame)} matching images")
    visible = frame.iloc[(current_page - 1) * page_size:current_page * page_size]
    records = visible.to_dict("records")
    for start in range(0, len(records), 5):
        for col, record in zip(st.columns(5), records[start:start + 5]):
            with col:
                with st.container(border=True):
                    st.markdown('<div class="evaluation-card-marker"></div>', unsafe_allow_html=True)
                    try:
                        with Image.open(ROOT / Path(record["path"])) as source:
                            thumbnail = to_rgb(source)
                            thumbnail.thumbnail((100, 100))
                            st.image(thumbnail, width=100)
                    except Exception:
                        st.caption("Image unavailable")
                    st.text(str(record["filename"]))
                    st.caption(f"Actual: {record['actual']} · Predicted: {record['prediction']}")
                    status_class = {"Correct": "correct", "Incorrect": "incorrect", "Uncertain": "uncertain"}[record["status"]]
                    st.markdown(f'<span class="evaluation-status {status_class}">{record["status"]}</span>', unsafe_allow_html=True)
                    st.caption(f"{record['confidence']:.0%} confidence")
                    if record.get("ood"):
                        st.caption("Experimental OOD flag")


def history_page() -> None:
    page_intro("YOUR ACTIVITY", "Prediction history", "Your discoveries in this session, all in one place. History is not saved permanently.")
    history = st.session_state.history
    cols = st.columns(3)
    cols[0].metric("Total predictions", len(history))
    cols[1].metric("Cat predictions", sum(item["Prediction"] == "Cat" for item in history))
    cols[2].metric("Dog predictions", sum(item["Prediction"] == "Dog" for item in history))
    section("Your predictions", "CURRENT SESSION")
    if not history:
        st.info("No predictions yet. Upload an image to start your history.")
        st.button("Start classification", on_click=navigate, args=("Image Classifier",), type="primary")
        return
    a, b, c = st.columns([2, 1, 1])
    search = a.text_input("Search filenames", placeholder="Search history")
    category = b.selectbox("Filter", ["All", "Cat", "Dog", "Uncertain"])
    sort = c.selectbox("Sort by", ["Newest first", "Oldest first", "Highest confidence", "Lowest confidence"])
    frame = pd.DataFrame(history)
    if "Status" not in frame:
        frame["Status"] = frame["Prediction"].map(lambda label: "Uncertain" if label == "Uncertain" else "Confident")
    else:
        frame["Status"] = frame["Status"].fillna(frame["Prediction"].map(lambda label: "Uncertain" if label == "Uncertain" else "Confident"))
    if search:
        frame = frame[frame["Filename"].str.contains(search, case=False, regex=False)]
    if category == "Uncertain":
        frame = frame[frame["Status"] == "Uncertain"]
    elif category != "All":
        frame = frame[frame["Prediction"] == category]
    frame = frame.sort_values("Confidence" if "confidence" in sort else "Timestamp", ascending=sort in ("Oldest first", "Lowest confidence"), kind="stable")
    display = frame.copy()
    for column in ("Confidence", "Cat probability", "Dog probability"):
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    display["Timestamp"] = display["Timestamp"].map(lambda timestamp: datetime.fromisoformat(timestamp).strftime("%b %d, %Y · %I:%M %p %Z"))
    st.dataframe(display, hide_index=True, width="stretch")
    if frame.empty:
        st.caption("No predictions match these filters.")
    export = frame.copy()
    export["Filename"] = export["Filename"].map(lambda name: "'" + name if name.startswith(("\t", "\r", "\n")) or name.lstrip().startswith(("=", "+", "-", "@")) else name)
    st.download_button("Download history as CSV", export.to_csv(index=False).encode("utf-8"), file_name="pawvision-history.csv", mime="text/csv", disabled=frame.empty)
    confirmation("history")


def insights() -> None:
    page_intro("MODEL INSIGHTS", "The intelligence behind the image.", "Explore how our fine-tuned model performs and what its predictions actually mean.")
    section("Held-out test performance", "3,745 IMAGES · DOG IS THE POSITIVE CLASS")
    metrics()
    st.caption("40 errors across 3,745 held-out test images. These are evaluation results, not live prediction statistics.")
    section("Score evidence", "REPORTED TEST METRICS")
    evaluation_chart()
    st.write("40 errors across 3,745 held-out test images · 21.7 MB model size.")
    section("Performance in perspective", "ROUNDED PROJECT EXPERIMENTS")
    benchmarks = [("Majority baseline", "~50%", 50), ("Classical ML · raw pixels", "57–65%", 65), ("HOG + SVM", "73%", 73), ("Custom CNN", "88.5%", 88.5), ("CNN + augmentation", "93.7%", 93.7), ("Fine-tuned MobileNetV2", "98.93%", 98.93)]
    for label, display, value in benchmarks:
        st.progress(value / 100, text=f"{label}  ·  {display}")
    st.caption("Rounded comparison figures from project experiments; the raw-pixel bar shows the upper end of the reported range.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><h3>Model at a glance</h3><p><strong>Architecture</strong> · MobileNetV2, ImageNet-pretrained and fine-tuned</p><p><strong>Framework</strong> · TensorFlow 2.21.0 / Keras</p><p><strong>Input</strong> · 224 × 224 × 3 RGB</p><p><strong>Output</strong> · Binary sigmoid probability, P(Dog)</p><p><strong>Model size</strong> · 21.7 MB</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>Understanding the output</h3><p>P(Cat) = 1 − P(Dog). A dog probability of at least 50% selects Dog; otherwise the model selects Cat. Confidence is the probability of the selected class.</p></div>', unsafe_allow_html=True)
        if st.session_state.last_result:
            result = st.session_state.last_result
            st.progress(result["prob_cat"], text=f'Latest image · Cat {result["prob_cat"]:.1%}')
            st.progress(result["prob_dog"], text=f'Latest image · Dog {result["prob_dog"]:.1%}')
        else:
            st.caption("Classify an image to see its Cat vs Dog probabilities here.")
        st.markdown('<div class="panel"><h3>Known limitations</h3><p>The model can abstain, but may still give a Cat or Dog result for other subjects. Multiple animals, occlusion, unusual images, and mislabelled examples can cause errors. The experimental feature-distance detector is trained from a tiny dataset and is not validated for out-of-distribution detection. Neither it nor confidence guarantees correctness.</p></div>', unsafe_allow_html=True)
    section("Inference pipeline")
    st.info("Image upload → validation → RGB / EXIF correction → resize → MobileNetV2 → sigmoid probability → candidate class → confidence and experimental detector → prediction or abstention")


def how_it_works() -> None:
    page_intro("THE PROCESS", "From pixels to prediction.", "A transparent look at the eight steps behind every PawVision result.")
    steps = [("Image upload", "Choose a cat or dog photo from your device."), ("Image validation", "Check image contents, size and minimum dimensions."), ("EXIF correction & RGB", "Correct orientation and convert to RGB; transparency is composited on white."), ("Resize to 224 × 224", "Cap the longest side at 256 pixels, then resize with antialiasing."), ("MobileNetV2 features", "The fine-tuned network extracts useful visual features."), ("Sigmoid probability", "The model outputs a probability for the Dog class."), ("Cat / Dog classification", "P(Dog) ≥ 0.5 selects Dog; otherwise Cat."), ("Confidence evaluation", "Compare the candidate class probability with your threshold and check the experimental detector; either may cause abstention.")]
    for start in range(0, len(steps), 4):
        for col, (index, (title, text)) in zip(st.columns(4), enumerate(steps[start:start + 4], start + 1)):
            col.markdown(f'<div class="step-card"><span class="step-number">{index:02d} /</span><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    section("Why MobileNetV2?")
    st.write("Transfer learning starts with features learned from ImageNet, then fine-tunes the network on cat and dog images. MobileNetV2 provides an efficient architecture for image classification while retaining strong performance on the project's held-out test set.")
    st.markdown('<div class="panel"><h3>A simplified architecture</h3><p>RGB image → MobileNetV2 feature extractor → classification head → sigmoid P(Dog) → Cat / Dog</p></div>', unsafe_allow_html=True)
    with st.expander("How is confidence different from class probability?"):
        st.write("P(Dog) is the model's sigmoid output; P(Cat) is its complement. Confidence is the probability of the candidate class. If confidence is below your threshold or the experimental detector flags the image, the result is Uncertain rather than a Cat/Dog prediction. Neither check guarantees correctness.")
    with st.expander("What happens to my image?"):
        st.write("The image is validated and processed in memory. Orientation is corrected, transparency is composited on white, and the image is resized for the model. It is not stored as a permanent image library.")


def about() -> None:
    page_intro("ABOUT PAWVISION", "A project with a purpose.", "Exploring how deep learning can turn an everyday image into an understandable prediction.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><h3>The project</h3><p>PawVision AI is a binary image classification project designed to distinguish cats from dogs. It compares classical and deep learning approaches and presents an accessible, interpretable prediction experience.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><h3>Dataset & preparation</h3><p>The project used approximately 25,000 cat and dog images. Cleaning addressed corrupt and tiny images, duplicate images, and label conflicts. The cleaned data was split using stratified 70% training, 15% validation, and 15% test partitions to limit leakage.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>Model development</h3><p>Classical models and custom CNNs were compared with transfer learning. The selected ImageNet-pretrained MobileNetV2 was fine-tuned after training its classification head on 17,474 cleaned training images.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><h3>Evaluation methodology</h3><p>Validation informed model selection; the held-out test set was reserved for final evaluation. Dog is the positive class for precision, recall, and F1. Reported comparisons are rounded project experiment results, not guarantees on new data.</p></div>', unsafe_allow_html=True)
    section("Technology stack")
    st.write("Python · Streamlit · TensorFlow · Keras · MobileNetV2 · NumPy · Pandas · Pillow · Scikit-learn · Matplotlib")
    st.caption("See the Difference. Understand the Prediction.")


def settings() -> None:
    page_intro("PREFERENCES", "Make PawVision yours.", "Adjust how results appear in your current session.")
    section("Prediction preferences")
    st.slider("Confidence threshold", 0.50, 0.99, step=0.01, format="%.2f", key="threshold")
    st.caption("Results below this threshold are Uncertain. The candidate class remains based on P(Dog) ≥ 0.5; the experimental detector can also trigger abstention. A threshold is not a guarantee of correctness. Changing it updates the current image; existing dataset runs keep their original threshold.")
    st.toggle("Show Cat and Dog probability bars", key="show_probabilities")
    section("Appearance")
    st.selectbox("Theme", ["Light", "Dark"], key="settings_theme", on_change=sync_theme, args=("settings_theme",))
    st.caption("Theme and preferences are kept for this session.")
    section("Upload requirements")
    st.write("JPG, JPEG, PNG, WEBP, BMP and GIF · maximum 10 MB · shortest side at least 32 pixels. Animated GIFs are analyzed using their first frame.")
    section("Session data")
    st.caption("Images and history are only kept in the current session; they are not saved as a permanent library.")
    confirmation("settings")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✣</div><div class="brand-name">PawVision <span>AI</span></div></div><div class="brand-tagline">See the Difference. Understand the Prediction.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", format_func=lambda page: f"{NAV_ICONS[page]}    {page}", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">APPEARANCE</div>', unsafe_allow_html=True)
    st.selectbox("Color mode", ["Light", "Dark"], key="sidebar_theme", on_change=sync_theme, args=("sidebar_theme",))
    st.markdown('<div class="sidebar-note"><strong>Made for curious minds.</strong>See the difference. Understand the prediction.</div>', unsafe_allow_html=True)

apply_styles(st.session_state.theme == "Dark")
st.markdown(f'<div class="topline"><span>WORKSPACE &nbsp; / &nbsp; <strong>{st.session_state.page}</strong></span><span><i class="status-dot"></i> MOBILE NET V2 &nbsp; · &nbsp; SESSION WORKSPACE</span></div>', unsafe_allow_html=True)

try:
    model = get_model()
except Exception:
    st.error("The model could not be loaded. Place a valid final_model.keras file in the models/ folder and restart the app.")
    st.stop()

page = st.session_state.page
if page == "Dashboard":
    dashboard()
elif page == "Image Classifier":
    classifier(model)
elif page == "Folder Evaluation":
    dataset_evaluation(model)
elif page == "Prediction History":
    history_page()
elif page == "Model Insights":
    insights()
elif page == "How It Works":
    how_it_works()
elif page == "About":
    about()
else:
    settings()

st.divider()
st.caption("PawVision AI  ·  Built for a closer look  ·  Cat & dog classification only")
