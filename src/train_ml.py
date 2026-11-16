"""Training / evaluation helpers for the classical ML experiments."""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)


def get_scores(model, X):
    """Probability of the positive class (Dog); falls back to the SVM decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_score)}


def fit_timed(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def evaluate_model(model, X, y):
    pred = model.predict(X)
    score = get_scores(model, X)
    return compute_metrics(y, pred, score), pred, score


def tune_on_val(factory, values, Xtr, ytr, Xva, yva):
    """Fit one model per hyper-parameter value, keep the best on VALIDATION accuracy."""
    rows, best = [], None
    for v in values:
        model, t = fit_timed(factory(v), Xtr, ytr)
        m, _, _ = evaluate_model(model, Xva, yva)
        rows.append({"param": v, "val_accuracy": m["accuracy"], "val_f1": m["f1"], "train_time_s": t})
        if best is None or m["accuracy"] > best[0]:
            best = (m["accuracy"], v, model, t)
    return best[2], best[1], best[3], pd.DataFrame(rows)


def save_ml_model(model, path):
    joblib.dump(model, path, compress=3)
    import os
    return os.path.getsize(path) / 1e6
