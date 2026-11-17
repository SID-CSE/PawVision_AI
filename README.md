# PawVision AI

PawVision AI is a Streamlit application that classifies cat and dog photos using a fine-tuned MobileNetV2 model. See the Difference. Understand the Prediction.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow 2.21.0](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

## Live demo

[Live demo](#live-demo) — https://paw-vision-ai.streamlit.app/

Screenshot placeholder: `docs/screenshot.png` (not yet available).

## Features

- Upload JPG, JPEG, PNG, WEBP, BMP, or GIF images up to 10 MB.
- View the predicted class, confidence, and Cat/Dog probabilities.
- Mark predictions below a configurable confidence threshold as uncertain.
- Review, download, or clear the current session's prediction history.
- Explore model results, the inference pipeline, and project details in dedicated workspace pages.
- Adjust the confidence threshold, probability display, and light or dark theme for the current session.

## Results

The held-out test set contains 3,745 images. Dog is the positive class.

| Metric | Result |
| --- | ---: |
| Accuracy | 98.93% |
| Precision | 0.988 |
| Recall | 0.991 |
| F1 | 0.989 |
| ROC-AUC | 0.9997 |
| Errors | 40 of 3,745 |
| Model size | 21.7 MB |

The comparison spans a majority baseline (~50%), classical machine learning on raw pixels
(57–65%), HOG + SVM (73%), a custom CNN (88.5%), a CNN with augmentation (93.7%), and the
fine-tuned MobileNetV2 (98.9%). These are rounded comparison figures, not estimates of
performance on every dataset.

## Repository structure

```text
.
├── app.py                         Streamlit entry point
├── src/
│   ├── data_cleaning.py           Shared image cleaning utilities
│   ├── predict.py                 Upload validation and inference
│   ├── preprocessing.py           Training preprocessing
│   ├── feature_engineering.py     Training features
│   ├── train_cnn.py               CNN training utilities
│   └── train_ml.py                Classical ML training utilities
├── models/
│   ├── final_model.keras          Model loaded by the app
│   ├── model_card.json            Model metadata and test metrics
│   └── ...                        Other training artifacts
├── .streamlit/config.toml         Upload limit and theme
├── app/app.py                     Legacy alternate app; not the deployment entry point
├── Cat_vs_Dog_Capstone.ipynb      Training and evaluation notebook
├── Cat vs Dog Image Classification, Final Capstone Project.docx
├── tests/test_app.py              Streamlit smoke tests
├── .gitignore
├── requirements.txt               App runtime dependencies
└── README.md
```

## Quick start

With Python 3.12 installed, substitute the repository's actual Git URL below:

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Model

The app loads `models/final_model.keras` from the repository and caches the loaded model.
The model is approximately 22 MB, so Git LFS is not required for GitHub's per-file limit.
If the file is absent or invalid, place the trained model at that path and restart the app.
The app requires no model download URL or secrets.

## Deployment on Streamlit Community Cloud

1. Commit and push the repository, including `models/final_model.keras`, to GitHub.
2. Create an app on Streamlit Community Cloud and select the repository and branch.
3. Set the main file path to `app.py` and select Python 3.12 in advanced settings.
4. Deploy the app. The first start includes dependency installation and model loading.

For an existing Git repository with an `origin` remote and a checked-out branch:

```bash
git add -A
git status
git commit -m "Update cat and dog classifier"
git push origin HEAD
```

The status should be reviewed before committing to ensure the model is included and no
secrets, datasets, debug models, or caches are staged.

## How it works

Upload → validate → correct EXIF orientation and convert to RGB → cap the longest side at
256 pixels → resize to 224 × 224 with antialiasing → MobileNetV2 → sigmoid probability →
label and confidence. Pixel rescaling is part of the model. Dog is selected when P(Dog) is
at least 0.5; otherwise Cat is selected. The separate confidence threshold controls the
uncertain indicator, not the underlying class decision.

Uploads must be nonempty, decodable images with a shorter side of at least 32 pixels and a
file size no greater than 10 MiB. Images with transparency or grayscale are converted to RGB.

## Data and methodology

The dataset consists of approximately 25,000 cat and dog photos. Download the
[dataset](https://drive.google.com/file/d/1erMgSlKsmCbmZe1dLrSVL5TNhe4JrWOY/view?usp=sharing).
Cleaning addressed corrupt and tiny images, duplicates, and label conflicts; images were
standardized to RGB. Data was split with stratification into 70% training, 15% validation,
and 15% test partitions, with no leakage between splits. Validation informed model selection,
while the held-out test set was reserved for final evaluation. The selected model was
pretrained on ImageNet and fine-tuned after training its classification head.

## Limitations

- Only cats and dogs are supported; other subjects are still assigned one of the two classes.
- Confidence is not a guarantee of correctness or a reliable out-of-distribution detector.
- Multiple animals, occlusion, unusual images, and mislabeled examples can cause errors.
- The comparison reflects one run per model and does not establish statistical significance.

## Troubleshooting

| Symptom | Suggested action |
| --- | --- |
| Model cannot be loaded | Place a valid `final_model.keras` in `models/` and restart. Confirm TensorFlow matches `requirements.txt`. |
| TensorFlow installation fails | Use Python 3.12 and install the root `requirements.txt`. |
| Image rejected | Check that it is readable, no larger than 10 MiB, and at least 32 pixels on its shorter side. |
| First load is slow | Allow time for TensorFlow and the model to load; the model is cached afterward. |
| Unexpected predictions | Try a clear image containing a single cat or dog and review the confidence and limitations. |

## Training

`Cat_vs_Dog_Capstone.ipynb` contains the training and evaluation workflow. Training uses
additional dependencies and data beyond the app-only requirements; the dataset is not
included in this repository.

## License and dataset attribution

No license is currently supplied for this repository. The source dataset belongs to its
original authors; its attribution and redistribution terms should be confirmed before use.
