"""Streamlit interface for the cat and dog image classifier."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_trained_model, open_uploaded_image, predict_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.keras"
FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "gif"]

st.set_page_config(page_title="Cat & Dog Classifier", page_icon="🐾", layout="wide")
st.markdown(
    """
    <style>
    .hero {padding: 0.3rem 0 1rem;}
    .hero h1 {color: #1F2430; margin-bottom: 0.2rem;}
    .hero p {color: #596471; margin: 0;}
    .result-card {background: #f5f9f8; border: 1px solid #dcebe8;
                  border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;}
    .result-label {font-size: 2rem; font-weight: 700; color: #1F2430;}
    .badge {display: inline-block; border-radius: 999px; padding: 0.25rem 0.7rem;
            color: white; font-size: 0.8rem; font-weight: 600;}
    .confident {background: #1F8A7D;}
    .uncertain {background: #E9A23B; color: #1F2430;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model. This can take a moment on first launch...")
def get_model():
    """Load the bundled model once for all sessions."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    return load_trained_model(MODEL_PATH)


def show_result(result: dict) -> None:
    """Display the predicted class, confidence, and class probabilities."""
    uncertain = result["uncertain"]
    label = result["label"]
    icon = "🐱" if label == "Cat" else "🐶"
    badge = "Uncertain" if uncertain else "Confident"
    badge_class = "uncertain" if uncertain else "confident"
    st.markdown(
        f'<div class="result-card"><span class="badge {badge_class}">{badge}</span>'
        f'<div class="result-label">{icon} {label}</div>'
        f'<div>Confidence: <strong>{result["confidence"]:.1%}</strong></div></div>',
        unsafe_allow_html=True,
    )
    if uncertain:
        st.caption("Best guess shown; confidence is below the selected threshold.")
    st.progress(float(result["prob_cat"]), text=f'Cat · {result["prob_cat"]:.1%}')
    st.progress(float(result["prob_dog"]), text=f'Dog · {result["prob_dog"]:.1%}')


def show_history() -> None:
    """Show and export predictions from the current session."""
    st.subheader("Session history")
    history = st.session_state.history
    if not history:
        st.caption("Predictions will appear here during this session.")
        return
    frame = pd.DataFrame(history[::-1])
    st.dataframe(frame, hide_index=True)
    export = frame.copy()
    export["file"] = export["file"].map(
        lambda name: "'" + name
        if name.startswith(("\t", "\r", "\n")) or name.lstrip().startswith(("=", "+", "-", "@"))
        else name
    )
    download, clear = st.columns(2)
    with download:
        st.download_button(
            "Download history as CSV",
            export.to_csv(index=False).encode("utf-8"),
            file_name="prediction-history.csv",
            mime="text/csv",
        )
    with clear:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


st.markdown(
    '<div class="hero"><h1>Cat or dog?</h1>'
    '<p>A closer look at your photo, powered by a fine-tuned MobileNetV2.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Choose an image to see the model’s prediction and confidence.")

try:
    model = get_model()
except Exception:
    st.error(
        "The model could not be loaded. Place a valid final_model.keras file in the "
        "models/ folder and restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    min_confidence = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help=(
            "Predictions below this confidence are marked uncertain; the best guess remains visible."
        ),
    )
    st.caption("JPG, PNG, WEBP, BMP or GIF · up to 10 MB · minimum side 32 pixels")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

classify, about, how = st.tabs(["Classify", "About the model", "How it works"])
with classify:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Your image")
        uploaded = st.file_uploader("Choose a photo", type=FORMATS)
        st.caption("For best results, use a clear photo of one cat or dog.")
        image = None
        if uploaded is None:
            st.session_state.last_prediction = None
        else:
            try:
                uploaded.seek(0)
                image = open_uploaded_image(uploaded)
            except ValueError as error:
                st.error(str(error))
            if image is not None:
                st.image(
                    image,
                    caption=f"Uploaded image · {image.width} × {image.height} pixels",
                )
    with right:
        st.subheader("Prediction")
        if image is None:
            st.info("Upload a photo to see the prediction here.")
        else:
            with st.spinner("Analyzing the image..."):
                result = predict_image(model, image, min_confidence=min_confidence)
            show_result(result)
            entry = {
                "file": uploaded.name,
                "prediction": "Uncertain" if result["uncertain"] else result["label"],
                "P(Dog)": round(result["prob_dog"], 4),
                "confidence": round(result["confidence"], 4),
            }
            if st.session_state.last_prediction != entry:
                st.session_state.history.append(entry)
                st.session_state.last_prediction = entry
    show_history()

with about:
    st.subheader("Fine-tuned MobileNetV2")
    st.write(
        "An ImageNet-pretrained MobileNetV2 was fine-tuned on 17,474 cleaned cat and dog "
        "training images. Its single sigmoid output estimates the probability of a dog."
    )
    st.caption("Held-out test set · 3,745 images · Dog is the positive class")
    metrics = st.columns(5)
    for column, label, value in zip(
        metrics,
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["98.93%", "0.988", "0.991", "0.989", "0.9997"],
    ):
        column.metric(label, value)
    st.write("40 errors out of 3,745 images · model size: 21.7 MB")
    st.write(
        "The classifier only distinguishes cats from dogs. Other subjects are still assigned "
        "one of those labels; confidence is not a guarantee of correctness. Multiple animals, "
        "occlusion, and unusual images can also cause mistakes."
    )

with how:
    st.subheader("From photo to prediction")
    st.write(
        "Upload → validate → correct orientation and convert to RGB → cap the longest side "
        "at 256 pixels → resize to 224 × 224 → MobileNetV2 → probability → label."
    )
    st.write(
        "The model rescales pixel values internally. A dog probability of at least 0.5 "
        "selects Dog; otherwise it selects Cat. The confidence threshold controls whether "
        "the best guess is marked uncertain."
    )

st.divider()
st.caption("Only cats and dogs are supported; other subjects can still receive a prediction.")
