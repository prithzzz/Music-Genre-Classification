"""Step 1: python extract_features.py
The train/val/test split is per TRACK so segments of one song never leak across splits."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import *


def split_tracks(tracks):
    rng = np.random.RandomState(SEED)
    split = {}
    for g in GENRES:
        names = sorted(t for t, gg in tracks if gg == g)
        rng.shuffle(names)
        n_tr, n_va = int(len(names) * TRAIN_FRAC), int(len(names) * VAL_FRAC)
        for i, n in enumerate(names):
            split[n] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return split


files = []
for g in GENRES:
    files += [(p, g) for p in sorted((DATA_DIR / g).glob("*.wav"))]
if not files:
    raise SystemExit(f"No .wav files found under {DATA_DIR}")
split = split_tracks([(p.stem, g) for p, g in files])

rows, mels, skipped = [], [], []
for p, g in tqdm(files, desc="extracting"):
    try:
        y = load_audio(p)
    except Exception as e:          # Fixed bug ew: GTZAN's jazz.00054.wav is known to be corrupt
        skipped.append((p.name, str(e)[:60]))
        continue
    for k, seg in enumerate(segment_signal(y)):
        row = {"track": p.stem, "genre": g, "label": GENRES.index(g),
               "segment": k, "split": split[p.stem]}
        row.update(handcrafted_features(seg))
        rows.append(row)
        mels.append(mel_spectrogram(seg).astype(np.float16))

df = pd.DataFrame(rows)
df.to_csv(FEATURES_CSV, index=False)
np.save(MEL_NPY, np.stack(mels))
print(df.groupby("split").size().to_string())
print("segments:", len(df), "| tracks:", df.track.nunique(), "| skipped:", skipped)