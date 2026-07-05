import pandas as pd
import joblib
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report

mit = pd.read_pickle("mitbih_beats.pkl")
saved = joblib.load("model.joblib")
clf, sc, FEATS = saved["clf"], saved["scaler"], saved["feats"]

mit3 = mit[mit.label.isin(["N", "S", "V"])].copy()
te = mit3[mit3.split == "DS2_test"]

Xte, yte = te[FEATS].values, te.label.values
pred = clf.predict(sc.transform(Xte))

print("Macro-F1:", round(f1_score(yte, pred, average="macro"), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(yte, pred), 4))
print("Confusion matrix (rows=true, cols=pred) [N, S, V]:")
print(confusion_matrix(yte, pred, labels=["N", "S", "V"]))
print(classification_report(yte, pred, digits=3))
