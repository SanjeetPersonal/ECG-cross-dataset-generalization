import pandas as pd
import joblib
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report

ptb_beats = pd.read_pickle("ptbxl_beats.pkl")
saved = joblib.load("model.joblib")
clf, sc, FEATS = saved["clf"], saved["scaler"], saved["feats"]

Xc, yc = ptb_beats[FEATS].values, ptb_beats.label.values
pred = clf.predict(sc.transform(Xc))

print("Macro-F1:", round(f1_score(yc, pred, average="macro"), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(yc, pred), 4))
print("Confusion matrix (rows=true, cols=pred) [N, S, V]:")
print(confusion_matrix(yc, pred, labels=["N", "S", "V"]))
print(classification_report(yc, pred, digits=3))
