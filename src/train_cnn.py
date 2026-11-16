"""CNN / transfer-learning builders, augmentation, training helpers."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# three architectures for Q46 (A = the brief's baseline)
CNN_VARIANTS = {
    "cnn_A_baseline": dict(filters=(32, 64, 128), kernel=3, convs_per_block=1, batch_norm=False,
                           gap=False, dense=128, dropout=0.4),
    "cnn_B_deep_bn": dict(filters=(32, 64, 128, 256), kernel=3, convs_per_block=2, batch_norm=True,
                          gap=True, dense=128, dropout=0.5),
    "cnn_C_k5_light": dict(filters=(16, 32, 64), kernel=5, convs_per_block=1, batch_norm=False,
                           gap=False, dense=64, dropout=0.3),
}


def set_seed(seed=42):
    keras.utils.set_random_seed(seed)


def build_cnn(cfg, name, img_size=128):
    """Input -> Rescaling(1/255) -> [Conv(+BN)+ReLU]xN -> MaxPool ... -> Flatten/GAP -> Dense -> Dropout -> sigmoid."""
    inputs = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)       # normalisation lives INSIDE the model
    for f in cfg["filters"]:
        for _ in range(cfg.get("convs_per_block", 1)):
            x = layers.Conv2D(f, cfg["kernel"], padding="same", use_bias=not cfg["batch_norm"])(x)
            if cfg["batch_norm"]:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x) if cfg["gap"] else layers.Flatten()(x)
    x = layers.Dense(cfg["dense"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid", name="p_dog")(x)
    return keras.Model(inputs, out, name=name)


def random_shear(images, max_shear=0.15):
    """Random horizontal shear of a float32 batch (N,H,W,3), reflect-filled."""
    n = tf.shape(images)[0]
    h = tf.cast(tf.shape(images)[1], tf.float32)
    s = tf.random.uniform([n], -max_shear, max_shear)
    ones, zeros = tf.ones([n]), tf.zeros([n])
    transforms = tf.stack([ones, s, -s * h / 2.0, zeros, ones, zeros, zeros, zeros], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images, transforms=transforms, output_shape=tf.shape(images)[1:3],
        fill_value=0.0, interpolation="BILINEAR", fill_mode="REFLECT")


def build_augmentation(rotation=0.08, zoom=0.2, shift=0.1, shear=0.15, flip=True):
    """Returns f(batch)->augmented batch (flip, rotation, zoom, shift, shear). Use on TRAIN data only."""
    seq = [layers.RandomFlip("horizontal")] if flip else []
    seq += [layers.RandomRotation(rotation, fill_mode="reflect"),
            layers.RandomZoom(zoom, fill_mode="reflect"),
            layers.RandomTranslation(shift, shift, fill_mode="reflect")]
    pipe = keras.Sequential(seq)

    def augment(x):
        x = pipe(x, training=True)
        return random_shear(x, shear) if shear else x
    return augment


def build_transfer_model(img_size=224, pretrained=True, dropout=0.2):
    """MobileNetV2 backbone (frozen) + GAP + Dropout + sigmoid head. Preprocessing is inside the model."""
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                          weights="imagenet" if pretrained else None)
    base.trainable = False
    inputs = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_-1_1")(inputs)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_dog")(x)
    return keras.Model(inputs, out, name="mobilenetv2_transfer"), base


def unfreeze_top(base, n_layers=30):
    """Fine-tuning: unfreeze the last n layers of the backbone (BatchNorm layers stay frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def get_callbacks(name, models_dir, logs_dir, patience=4):
    models_dir, logs_dir = Path(models_dir), Path(logs_dir)
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(models_dir / f"{name}.keras"), monitor="val_loss", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
            keras.callbacks.CSVLogger(str(logs_dir / f"{name}_history.csv"))]


def train_model(model, train_ds, val_ds, name, epochs, lr, models_dir, logs_dir, patience=4):
    """Compile + fit with EarlyStopping / ModelCheckpoint. Returns (history DataFrame, seconds, checkpoint path)."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    t0 = time.perf_counter()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  callbacks=get_callbacks(name, models_dir, logs_dir, patience), verbose=2)
    secs = time.perf_counter() - t0
    hist = pd.DataFrame(h.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    return hist, secs, Path(models_dir) / f"{name}.keras"


def predict_scores(model, ds):
    return model.predict(ds, verbose=0).ravel()


def n_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def parameter_table(model):
    """Per-layer parameter counts with the textbook formula for Conv2D / Dense / BatchNorm."""
    rows = []
    for l in model.layers:
        p = int(sum(np.prod(w.shape) for w in l.trainable_weights))
        if p == 0 and not l.weights:
            continue
        f = ""
        if isinstance(l, layers.Conv2D):
            kh, kw = l.kernel_size
            cin, cout = l.kernel.shape[2], l.filters
            f = f"({kh}x{kw}x{cin}) x {cout}" + (f" + {cout} bias" if l.use_bias else " (no bias, BN follows)")
        elif isinstance(l, layers.Dense):
            f = f"{l.kernel.shape[0]} x {l.units} + {l.units} bias"
        elif isinstance(l, layers.BatchNormalization):
            f = f"2 x {l.gamma.shape[0]} (gamma, beta)"
        rows.append({"layer": l.name, "type": l.__class__.__name__, "trainable_params": p, "formula": f})
    return pd.DataFrame(rows)


def plot_history(hist, title=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, (tr, va, lab) in zip(ax, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
        a.plot(hist["epoch"], hist[tr], marker="o", label="train")
        a.plot(hist["epoch"], hist[va], marker="o", label="validation")
        a.set_xlabel("epoch"); a.set_ylabel(lab); a.set_title(f"{title} – {lab}"); a.legend()
    plt.tight_layout(); plt.show()


def interpret_history(hist):
    """Auto-generated reading of the learning curves (markdown string)."""
    hist = hist.reset_index(drop=True)
    b = hist.loc[hist["val_loss"].idxmin()]
    last = hist.iloc[-1]
    gap_b, gap_l = b["accuracy"] - b["val_accuracy"], last["accuracy"] - last["val_accuracy"]
    if b["val_accuracy"] < 0.75 and b["accuracy"] < 0.80:
        verdict = "**Underfitting** – both curves are still low; the model needs more capacity, more epochs or better features."
    elif gap_l > 0.08 or last["val_loss"] > 1.10 * b["val_loss"]:
        verdict = "**Overfitting** – training accuracy keeps rising while validation loss stalls or increases (large train–val gap)."
    else:
        verdict = "**Good fit** – validation tracks training closely with a small gap; no strong over- or under-fitting."
    return (f"- Best epoch (lowest val-loss): **{int(b['epoch'])}** of {len(hist)}.\n"
            f"- At best epoch: train acc **{b['accuracy']:.3f}**, val acc **{b['val_accuracy']:.3f}** (gap {gap_b:+.3f}); val loss {b['val_loss']:.3f}.\n"
            f"- At last epoch: train acc {last['accuracy']:.3f}, val acc {last['val_accuracy']:.3f} (gap {gap_l:+.3f}); val loss {last['val_loss']:.3f}.\n"
            f"- Verdict: {verdict}")
