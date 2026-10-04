"""Step 3: python train_cnn.py   
(CNN on log-mel spectrograms)"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from common import *

tf.keras.utils.set_random_seed(SEED)
df = pd.read_csv(FEATURES_CSV, usecols=["track", "label", "split"])
X = np.load(MEL_NPY)[..., None]                  # (N, 128, 130, 1) float16


def part(s):
    m = (df.split == s).values
    return X[m], df.label.values[m], df.track.values[m]


(xtr, ytr, _), (xva, yva, _), (xte, yte, tte) = part("train"), part("val"), part("test")


def spec_augment(x, y):
    """SpecAugment: one random frequency mask and one random time mask."""
    h, w = N_MELS, N_FRAMES
    f = tf.random.uniform([], 0, 16, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, h - f, dtype=tf.int32)
    t = tf.random.uniform([], 0, 21, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, w - t, dtype=tf.int32)
    fm = tf.concat([tf.ones([f0]), tf.zeros([f]), tf.ones([h - f0 - f])], 0)
    tm = tf.concat([tf.ones([t0]), tf.zeros([t]), tf.ones([w - t0 - t])], 0)
    out = x * fm[:, None, None] * tm[None, :, None]
    out.set_shape([h, w, 1])
    return out, y


def make_ds(x, y, train):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(lambda a, b: (tf.cast(a, tf.float32), b))
    if train:
        ds = ds.shuffle(4096, seed=SEED).map(spec_augment)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def build_model():
    inp = layers.Input(shape=(N_MELS, N_FRAMES, 1))
    x = inp
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    return models.Model(inp, layers.Dense(len(GENRES), activation="softmax")(x))


model = build_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cbs = [callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
       callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
       callbacks.ModelCheckpoint(str(CNN_PATH), save_best_only=True, monitor="val_accuracy")]
hist = model.fit(make_ds(xtr, ytr, True), validation_data=make_ds(xva, yva, False),
                 epochs=60, callbacks=cbs)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].legend()
fig.tight_layout(); fig.savefig(ROOT / "cnn_curves.png", dpi=160)

probs = model.predict(xte.astype("float32"), batch_size=64)
m = evaluate(yte, probs, tte)
print({k: v for k, v in m.items() if not k.startswith("_")})
plot_confusion(m["_y_tr"], m["_pred_tr"], "CNN (track level)", ROOT / "cm_CNN.png")
save_json({"params": int(model.count_params()), **m}, "cnn_results.json")