"""Shared config, audio features and helper functions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# config 
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "genres_original"
FEATURES_CSV = ROOT / "data" / "features.csv"
MEL_NPY = ROOT / "data" / "mel.npy"
CNN_PATH = ROOT / "cnn_best.keras"
CLF_PATH = ROOT / "best_classical.joblib"

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

SR = 22050
SEG_SECONDS = 3            # 30 s track -> ten 3 s segments
MAX_SEGS = 10
N_MELS, N_FFT, HOP = 128, 2048, 512
N_FRAMES = 1 + (SR * SEG_SECONDS) // HOP     # 130

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10            # test = 0.20


# audio / features
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def segment_signal(y, max_segs=MAX_SEGS):
    n = int(SR * SEG_SECONDS)
    return [y[i * n:(i + 1) * n] for i in range(len(y) // n)][:max_segs]


def evenly_spaced_segments(y, k=10):
    """For long uploaded songs: k segments spread over the whole track."""
    segs = segment_signal(y, max_segs=10_000)
    if len(segs) <= k:
        return segs
    return [segs[i] for i in np.linspace(0, len(segs) - 1, k).astype(int)]


def mel_spectrogram(seg):
    """Log-mel spectrogram scaled to [0, 1], shape (N_MELS, N_FRAMES)."""
    m = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    db = (librosa.power_to_db(m, ref=np.max, top_db=80.0) + 80.0) / 80.0
    if db.shape[1] < N_FRAMES:
        db = np.pad(db, ((0, 0), (0, N_FRAMES - db.shape[1])))
    return db[:, :N_FRAMES].astype(np.float32)


def handcrafted_features(seg):
    """Mean and std of classic audio descriptors -> dict of 88 floats."""
    f = {}

    def add(name, arr):
        for i, row in enumerate(np.atleast_2d(arr)):
            f[f"{name}{i + 1}_mean"] = float(row.mean())
            f[f"{name}{i + 1}_std"] = float(row.std())

    add("mfcc", librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20))
    add("chroma", librosa.feature.chroma_stft(y=seg, sr=SR))
    add("contrast", librosa.feature.spectral_contrast(y=seg, sr=SR))
    add("centroid", librosa.feature.spectral_centroid(y=seg, sr=SR))
    add("bandwidth", librosa.feature.spectral_bandwidth(y=seg, sr=SR))
    add("rolloff", librosa.feature.spectral_rolloff(y=seg, sr=SR))
    add("zcr", librosa.feature.zero_crossing_rate(seg))
    add("rms", librosa.feature.rms(y=seg))
    return f


# evaluation helpers 
def track_level(probs, track_ids, y_true):
    """Average segment probabilities per track -> one prediction per track."""
    df = pd.DataFrame(probs)
    df["track"] = np.asarray(track_ids)
    df["y"] = np.asarray(y_true)
    g = df.groupby("track")
    return g["y"].first().values, g[list(range(probs.shape[1]))].mean().values.argmax(1)


def evaluate(y_true_seg, probs, track_ids):
    pred_seg = probs.argmax(1)
    y_tr, pred_tr = track_level(probs, track_ids, y_true_seg)
    return {
        "segment_acc": float(accuracy_score(y_true_seg, pred_seg)),
        "segment_macro_f1": float(f1_score(y_true_seg, pred_seg, average="macro")),
        "track_acc": float(accuracy_score(y_tr, pred_tr)),
        "track_macro_f1": float(f1_score(y_tr, pred_tr, average="macro")),
        "_y_tr": y_tr, "_pred_tr": pred_tr,
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(GENRES)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENRES)), GENRES)
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def save_json(obj, name):
    clean = {k: v for k, v in obj.items() if not k.startswith("_")}
    with open(ROOT / name, "w") as f:
        json.dump(clean, f, indent=2)