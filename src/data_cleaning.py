"""Data audit and cleaning utilities for the Cat vs Dog project."""
import hashlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
FORMAT_TO_EXT = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "GIF": {".gif"}, "BMP": {".bmp"},
                 "WEBP": {".webp"}, "TIFF": {".tif", ".tiff"}}
JUNK_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}
CLEAN_MAX_SIDE = 256
N_WORKERS = min(32, (os.cpu_count() or 2) * 4)


# ───────────────────────────── scanning ─────────────────────────────
def is_junk(path):
    p = Path(path)
    return p.name.startswith("._") or p.name.lower() in JUNK_NAMES or "__MACOSX" in p.parts


def label_from_path(path, root):
    """Class = nearest ancestor folder whose name mentions exactly one of 'cat' / 'dog'."""
    rel = Path(path).relative_to(root)
    for part in reversed(rel.parts[:-1]):
        p = part.lower()
        has_cat, has_dog = "cat" in p, "dog" in p
        if has_cat != has_dog:
            return "Cat" if has_cat else "Dog"
    return "Unknown"


def list_files(root):
    return sorted(str(p) for p in Path(root).rglob("*") if p.is_file())


def scan_image(path):
    """Return one metadata record; never raises. A file is `valid` only if it fully decodes."""
    p = Path(path)
    rec = {"path": str(p), "extension": p.suffix.lower(), "file_size": p.stat().st_size,
           "width": np.nan, "height": np.nan, "mode": None, "channels": np.nan,
           "format": None, "valid": False, "error": ""}
    if rec["file_size"] == 0:
        rec["error"] = "EmptyFile: 0 bytes"
        return rec
    try:
        with Image.open(p) as im:
            rec.update(width=im.width, height=im.height, mode=im.mode,
                       channels=len(im.getbands()), format=im.format)
            im.verify()
        with Image.open(p) as im:      # verify() does not decode pixels → force a full decode
            im.load()
        rec["valid"] = True
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:160]
    return rec


def scan_dataset(root, workers=N_WORKERS, limit_per_class=None):
    """Recursively scan `root` → DataFrame(path, label, width, height, channels, extension, ...)."""
    root = Path(root)
    all_files = list_files(root)
    files = [f for f in all_files if not is_junk(f)]
    labels = {f: label_from_path(f, root) for f in files}
    if limit_per_class:
        kept, seen = [], {}
        for f in files:
            lab = labels[f]
            seen[lab] = seen.get(lab, 0) + 1
            if lab == "Unknown" or seen[lab] <= limit_per_class:
                kept.append(f)
        files = kept
    with ThreadPoolExecutor(workers) as ex:
        records = list(tqdm(ex.map(scan_image, files), total=len(files), desc="scanning"))
    df = pd.DataFrame(records)
    df["label"] = df["path"].map(labels)
    df["is_image_ext"] = df["extension"].isin(IMAGE_EXTENSIONS)
    exp = df["format"].map(lambda f: FORMAT_TO_EXT.get(f, set()) if f else set())
    df["ext_mismatch"] = [bool(f) and e not in s for f, e, s in zip(df["format"], df["extension"], exp)]
    df.attrs["junk_files_ignored"] = len(all_files) - len(files) if not limit_per_class else None
    return df


# ───────────────────────── corruption handling ─────────────────────────
def detect_and_record_corrupted(df, report_csv=None, quarantine_dir=None):
    """Record every unreadable file (CSV report) and optionally MOVE it to a quarantine folder."""
    import shutil
    bad = df[~df["valid"]].copy()
    if report_csv is not None:
        bad.to_csv(report_csv, index=False)
    if quarantine_dir is not None:
        for p, lab in zip(bad["path"], bad["label"]):
            dst = Path(quarantine_dir) / str(lab)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(p, dst / Path(p).name)
    return bad


# ───────────────────────────── duplicates ─────────────────────────────
def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path, hash_size=16):
    """Difference hash: robust to re-encoding / mild resizing (256-bit for hash_size=16)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def _hash_one(path):
    try:
        m = md5_file(path)
    except Exception:
        m = None
    try:
        d = dhash(path)
    except Exception:
        d = None
    return m, d


def add_hashes(df, workers=N_WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(_hash_one, df["path"].tolist()), total=len(df), desc="hashing"))
    out = df.copy()
    out["md5"] = [r[0] for r in res]
    out["dhash"] = [r[1] for r in res]
    return out


# ─────────────────────── channel standardisation & cleaning ───────────────────────
def to_rgb(im):
    """Respect EXIF orientation and convert ANY mode (L, P, RGBA, CMYK …) to 3-channel RGB.
    Transparent pixels are composited on white instead of turning black."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def cap_size(im, max_side=CLEAN_MAX_SIDE):
    """Downscale (keeping aspect ratio) so the longest side is <= max_side."""
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return im


def standardize_image(src, max_side=CLEAN_MAX_SIDE):
    with Image.open(src) as im:
        out = cap_size(to_rgb(im), max_side)
        out.load()
    return out


def build_cleaned_dataset(df, out_dir, max_side=CLEAN_MAX_SIDE, quality=95, workers=N_WORKERS):
    """Write every row of `df` as an RGB JPEG into out_dir/<label>/ and return the manifest."""
    out_dir = Path(out_dir)
    df = df.copy().reset_index(drop=True)
    names = df["label"].str.lower() + "_" + df.groupby("label").cumcount().astype(str).str.zfill(5) + ".jpg"
    df["clean_path"] = [str(out_dir / lab / nm) for lab, nm in zip(df["label"], names)]
    for lab in df["label"].unique():
        (out_dir / lab).mkdir(parents=True, exist_ok=True)

    def _job(i):
        try:
            im = standardize_image(df.at[i, "path"], max_side)
            im.save(df.at[i, "clean_path"], "JPEG", quality=quality)
            return True, im.width, im.height
        except Exception:
            return False, np.nan, np.nan

    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(_job, range(len(df))), total=len(df), desc="cleaning"))
    df["clean_ok"] = [r[0] for r in res]
    df["clean_width"] = [r[1] for r in res]
    df["clean_height"] = [r[2] for r in res]
    return df
