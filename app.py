"""PawVision AI: a session-based cat and dog classification workspace."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_cleaning import to_rgb
from src.pawvision_styles import apply_styles
from src.predict import load_trained_model, open_uploaded_image, predict_image

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
PAGES = ["Dashboard", "Image Classifier", "Prediction History", "Model Insights", "How It Works", "About", "Settings"]
NAV_ICONS = {"Dashboard": "▦", "Image Classifier": "◎", "Prediction History": "◷", "Model Insights": "▥", "How It Works": "◇", "About": "ⓘ", "Settings": "⚙"}

st.set_page_config(page_title="PawVision AI", page_icon="🐾", layout="wide", initial_sidebar_state="expanded")

for key, value in {
    "page": "Dashboard", "theme": "Light", "threshold": 0.70,
    "show_probabilities": True, "history": [], "upload_version": 0,
    "last_upload": None, "last_result": None, "confirm_clear": False,
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


def sync_theme(source: str) -> None:
    theme = st.session_state[source]
    st.session_state.theme = theme
    st.session_state["settings_theme" if source == "sidebar_theme" else "sidebar_theme"] = theme


def clear_upload() -> None:
    st.session_state.upload_version += 1
    st.session_state.last_upload = None
    st.session_state.last_result = None


def clear_history() -> None:
    st.session_state.history = []
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
    a, b = st.columns(2, gap="small")
    a.button("Start classification  →", type="primary", on_click=navigate, args=("Image Classifier",), width="stretch")
    b.button("Explore model insights  ↗", on_click=navigate, args=("Model Insights",), width="stretch")
    section("Everything you need, at a glance", "BUILT FOR A CLOSER LOOK")
    features = [("✦", "AI-powered classification", "A fine-tuned model that knows its cats from its dogs."), ("◉", "Confidence analysis", "See the probabilities behind every prediction."), ("ϟ", "Instant predictions", "Go from image to insight in just a few moments."), ("◷", "Prediction history", "Keep track of your discoveries during this session.")]
    for col, (icon, title, description) in zip(st.columns(4), features):
        col.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
    section("Built on performance", "HELD-OUT TEST RESULTS")
    metrics()
    st.caption("Evaluated on 3,745 held-out test images · Results reflect model evaluation, not live prediction statistics.")
    st.markdown('<div class="panel"><h3>Thoughtful technology, simple experience.</h3><p>Built with transfer learning and a fine-tuned MobileNetV2 model, PawVision turns complex image analysis into a clear, easy-to-understand result.</p></div>', unsafe_allow_html=True)


def classifier(model: Any) -> None:
    page_intro("IMAGE CLASSIFIER", "A closer look starts here.", "Upload a photo and discover what the model sees, along with the confidence behind its prediction.")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
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
                st.error(str(error))
                image = None
                st.button("Remove image", on_click=clear_upload)
        else:
            st.session_state.last_upload = None
            st.session_state.last_result = None
    with right:
        st.subheader("Prediction")
        st.caption("A little insight into what your image contains.")
        if image is None:
            st.markdown('<div class="panel"><div class="feature-icon">◎</div><h3>Your result will appear here</h3><p>Upload an image to see the prediction, confidence, and probability breakdown.</p></div>', unsafe_allow_html=True)
        else:
            fingerprint = (sha256(uploaded.getvalue()).hexdigest(), uploaded.name, st.session_state.upload_version)
            if st.session_state.last_upload != fingerprint:
                try:
                    with st.spinner("Taking a closer look..."):
                        result = predict_image(model, image, min_confidence=st.session_state.threshold)
                    st.session_state.last_upload = fingerprint
                    st.session_state.last_result = result
                    st.session_state.history.append({"Filename": uploaded.name, "Prediction": result["label"], "Confidence": result["confidence"], "Cat probability": result["prob_cat"], "Dog probability": result["prob_dog"], "Timestamp": datetime.now().astimezone().isoformat(timespec="seconds")})
                except Exception:
                    st.error("The image could not be analyzed. Please try another image.")
                    return
            result = st.session_state.last_result
            uncertain = result["confidence"] < st.session_state.threshold
            status = "Uncertain" if uncertain else "Confident"
            badge_class = "badge warn" if uncertain else "badge"
            ears = '<path d="M6 12 7 3l7 5M18 8l7-5 1 9"/>' if result["label"] == "Cat" else '<path d="M8 9C2 7 2 18 7 20M24 9c6-2 6 9 1 11"/>'
            icon = f'<svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="#8271c8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ears}<circle cx="16" cy="17" r="10"/><path d="M12 16h.01M20 16h.01M14 21q2 2 4 0"/></svg>'
            st.markdown(f'<div class="panel"><span class="{badge_class}">{status}</span><div class="result-title">{icon}{result["label"]}</div><div class="result-caption">PREDICTION CONFIDENCE</div><div class="result-confidence">{result["confidence"]:.1%}</div></div>', unsafe_allow_html=True)
            st.caption("Best guess shown; below your confidence threshold." if uncertain else "Confidence meets your selected threshold.")
            if st.session_state.show_probabilities:
                st.markdown("**Probability breakdown**")
                st.progress(float(result["prob_cat"]), text=f'Cat  ·  {result["prob_cat"]:.1%}')
                st.progress(float(result["prob_dog"]), text=f'Dog  ·  {result["prob_dog"]:.1%}')
            st.caption("The model always chooses Cat or Dog. Confidence is not a guarantee of correctness.")


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
    frame["Status"] = frame["Confidence"].map(lambda confidence: "Uncertain" if confidence < st.session_state.threshold else "Confident")
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
        st.markdown('<div class="panel"><h3>Known limitations</h3><p>Other subjects still receive a cat or dog label. Multiple animals, occlusion, unusual images, and mislabelled examples can cause errors. Confidence is not a guarantee or an out-of-distribution detector.</p></div>', unsafe_allow_html=True)
    section("Inference pipeline")
    st.info("Image upload → validation → RGB / EXIF correction → resize → MobileNetV2 → sigmoid probability → class & confidence")


def how_it_works() -> None:
    page_intro("THE PROCESS", "From pixels to prediction.", "A transparent look at the eight steps behind every PawVision result.")
    steps = [("Image upload", "Choose a cat or dog photo from your device."), ("Image validation", "Check image contents, size and minimum dimensions."), ("EXIF correction & RGB", "Correct orientation and convert to RGB; transparency is composited on white."), ("Resize to 224 × 224", "Cap the longest side at 256 pixels, then resize with antialiasing."), ("MobileNetV2 features", "The fine-tuned network extracts useful visual features."), ("Sigmoid probability", "The model outputs a probability for the Dog class."), ("Cat / Dog classification", "P(Dog) ≥ 0.5 selects Dog; otherwise Cat."), ("Confidence evaluation", "Compare the chosen class probability with your uncertainty threshold.")]
    for start in range(0, len(steps), 4):
        for col, (index, (title, text)) in zip(st.columns(4), enumerate(steps[start:start + 4], start + 1)):
            col.markdown(f'<div class="step-card"><span class="step-number">{index:02d} /</span><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    section("Why MobileNetV2?")
    st.write("Transfer learning starts with features learned from ImageNet, then fine-tunes the network on cat and dog images. MobileNetV2 provides an efficient architecture for image classification while retaining strong performance on the project's held-out test set.")
    st.markdown('<div class="panel"><h3>A simplified architecture</h3><p>RGB image → MobileNetV2 feature extractor → classification head → sigmoid P(Dog) → Cat / Dog</p></div>', unsafe_allow_html=True)
    with st.expander("How is confidence different from class probability?"):
        st.write("P(Dog) is the model's sigmoid output; P(Cat) is its complement. Confidence is the probability of whichever class was selected. The uncertainty threshold changes only the indicator, never the class decision.")
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
    st.caption("Results below this threshold are marked uncertain. The underlying class decision remains P(Dog) ≥ 0.5.")
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
