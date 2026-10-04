"""Step 4 (live demo): 
pip install -r requirements.txt
python extract_features.py
python train_classical.py
python train_cnn.py
streamlit run app.py"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import librosa.display
import matplotlib.pyplot as plt
from common import *


@st.cache_resource
def get_cnn():
    import tensorflow as tf
    return tf.keras.models.load_model(CNN_PATH)


@st.cache_resource
def get_clf():
    return joblib.load(CLF_PATH)


def get_segments(path, k=10):
    segs = evenly_spaced_segments(load_audio(path), k)
    if not segs:
        raise ValueError("Audio must be at least 3 seconds long.")
    return segs


def predict_cnn(path):
    X = np.stack([mel_spectrogram(s) for s in get_segments(path)])[..., None]
    return dict(zip(GENRES, map(float, get_cnn().predict(X, verbose=0).mean(0))))


def predict_classical(path):
    b = get_clf()
    feats = pd.DataFrame([handcrafted_features(s) for s in get_segments(path)])[b["columns"]]
    return dict(zip(GENRES, map(float, b["model"].predict_proba(feats.values).mean(0))))


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")
st.title("🎵 Music Genre Classifier")
choice = st.radio("Model", ["CNN (mel-spectrogram)", "Classical ML (hand-crafted features)"],
                  horizontal=True)
up = st.file_uploader("Upload a song (wav / mp3 / au / ogg)", type=["wav", "mp3", "au", "ogg"])

if up:
    st.audio(up)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
        tmp.write(up.getbuffer())
        path = tmp.name
    try:
        with st.spinner("Analysing..."):
            probs = (predict_cnn if choice.startswith("CNN") else predict_classical)(path)
            y = load_audio(path)
        top = max(probs, key=probs.get)
        st.success(f"Predicted genre: **{top}** ({probs[top]:.0%})")
        st.bar_chart(pd.Series(probs).sort_values(ascending=False))

        seg = (segment_signal(y, max_segs=1) or [y])[0]
        fig, ax = plt.subplots(figsize=(7, 3))
        img = librosa.display.specshow(mel_spectrogram(seg) * 80 - 80, sr=SR, hop_length=HOP,
                                       x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel-spectrogram (first 3 s)")
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))
    finally:
        os.remove(path)