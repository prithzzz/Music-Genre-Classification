"""Step 2: python train_classical.py   
(KNN, Logistic Regression, Random Forest, SVM)"""
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from common import *

df = pd.read_csv(FEATURES_CSV)
meta = ["track", "genre", "label", "segment", "split"]
cols = [c for c in df.columns if c not in meta]
dev = df[df.split.isin(["train", "val"])]       # CV on train+val, grouped by track
test = df[df.split == "test"]
Xd, yd, gd = dev[cols].values, dev.label.values, dev.track.values
Xt, yt, tt = test[cols].values, test.label.values, test.track.values

models = {
    "KNN": (Pipeline([("sc", StandardScaler()), ("pca", PCA(0.95)),
                      ("clf", KNeighborsClassifier(weights="distance"))]),
            {"clf__n_neighbors": [5, 10, 15]}),
    "LogReg": (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]),
               {"clf__C": [0.1, 1, 10]}),
    "RandomForest": (Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
                     {"clf__max_depth": [None, 20]}),
    "SVM-RBF": (Pipeline([("sc", StandardScaler()),
                          ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
                {"clf__C": [1, 10, 50], "clf__gamma": ["scale", 0.005]}),
}

results, best_name, best_cv, best_obj = {}, None, -1, None
for name, (pipe, grid) in models.items():
    print(f"== {name}")
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=5), scoring="accuracy", n_jobs=-1)
    gs.fit(Xd, yd, groups=gd)
    m = evaluate(yt, gs.predict_proba(Xt), tt)
    results[name] = {"best_params": gs.best_params_, "cv_acc": float(gs.best_score_),
                     **{k: v for k, v in m.items() if not k.startswith("_")}}
    print(results[name])
    plot_confusion(m["_y_tr"], m["_pred_tr"], f"{name} (track level)", ROOT / f"cm_{name}.png")
    if gs.best_score_ > best_cv:                # pick by CV score, never by test score
        best_cv, best_name, best_obj = gs.best_score_, name, gs.best_estimator_

joblib.dump({"model": best_obj, "columns": cols, "name": best_name}, CLF_PATH)
save_json({"models": results, "best_by_cv": best_name}, "classical_results.json")
print("Best classical model (by CV):", best_name)