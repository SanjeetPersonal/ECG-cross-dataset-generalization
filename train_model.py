import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

mit = pd.read_pickle("mitbih_beats.pkl")

FEATS = ["pre_rr", "post_rr", "loc_rr", "rr_ratio", "prepost_ratio", "amp", "qrs_width"]
mit3 = mit[mit.label.isin(["N", "S", "V"])].copy()
tr = mit3[mit3.split == "DS1_train"]

Xtr, ytr = tr[FEATS].values, tr.label.values
sc = StandardScaler().fit(Xtr)

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=5,
    random_state=0,
    n_jobs=-1
)
clf.fit(sc.transform(Xtr), ytr)

joblib.dump({"clf": clf, "scaler": sc, "feats": FEATS}, "model.joblib")
print("Model trained on", len(tr), "beats. Saved to model.joblib")
