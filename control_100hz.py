import os
import json
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix

root = "data/mitbih"

AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
beat_syms = set(AAMI)

DS1 = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234']

def bandpass(sig, fs, lo=0.5, hi=40, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, sig)

def qrs_width_proxy(seg, fs):
    d = np.diff(seg)
    e = d**2
    thr = 0.1*e.max() if e.max() > 0 else 0
    idx = np.where(e > thr)[0]
    return (idx.max()-idx.min())/fs*1000 if len(idx) else 0.0

def extract_100hz(rec_id):
    r = wfdb.rdrecord(f"{root}/{rec_id}")
    fs0 = r.fs
    sig = r.p_signal[:,0]
    sig = resample_poly(sig, 100, fs0)
    fs = 100
    scale = fs/fs0
    sig = bandpass(sig, fs)
    ann = wfdb.rdann(f"{root}/{rec_id}", "atr")
    beats = [(int(s*scale), y) for s, y in zip(ann.sample, ann.symbol) if y in beat_syms]
    if len(beats) < 3:
        return []
    positions = np.array([b[0] for b in beats])
    labels = [AAMI[b[1]] for b in beats]
    rr = np.diff(positions)/fs*1000
    W = int(0.14*fs)
    rows = []
    for i in range(1, len(beats)-1):
        pos = positions[i]
        if pos-W < 0 or pos+W >= len(sig):
            continue
        pre_rr = rr[i-1]
        post_rr = rr[i]
        lo = max(0, i-5)
        hi = min(len(rr), i+5)
        loc_rr = np.mean(rr[lo:hi])
        seg = sig[pos-W:pos+W]
        rows.append(dict(rec=rec_id, label=labels[i], pre_rr=pre_rr, post_rr=post_rr, loc_rr=loc_rr,
                         rr_ratio=pre_rr/loc_rr if loc_rr > 0 else 1.0,
                         prepost_ratio=pre_rr/post_rr if post_rr > 0 else 1.0,
                         amp=seg.max()-seg.min(), qrs_width=qrs_width_proxy(seg, fs)))
    return rows

if __name__ == "__main__":
    rows = []
    for rid in DS1+DS2:
        rows.extend(extract_100hz(rid))
    mit100 = pd.DataFrame(rows)
    mit100["split"] = np.where(mit100.rec.isin(DS1), "DS1_train", "DS2_test")

    FEATS = ["pre_rr","post_rr","loc_rr","rr_ratio","prepost_ratio","amp","qrs_width"]
    m3 = mit100[mit100.label.isin(["N","S","V"])]
    tr = m3[m3.split=="DS1_train"]
    te = m3[m3.split=="DS2_test"]

    sc = StandardScaler().fit(tr[FEATS])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=5, random_state=0, n_jobs=-1)
    clf.fit(sc.transform(tr[FEATS]), tr.label.values)

    pred = clf.predict(sc.transform(te[FEATS]))
    yte = te.label.values

    ptb = pd.read_pickle("ptbxl_beats.pkl")
    pc_ = clf.predict(sc.transform(ptb[FEATS]))
    yc = ptb.label.values

    def perclass(y, p):
        return {c: f1_score(y==c, p==c) for c in ["N","S","V"]}

    f1_in100 = perclass(yte, pred)
    f1_out100 = perclass(yc, pc_)
    cmo = confusion_matrix(yc, pc_, labels=["N","S","V"])
    cmo_n = cmo / cmo.sum(1, keepdims=True)

    ctrl = {
        "train_fs": "100Hz_matched",
        "in_dataset": {"macro_f1": float(f1_score(yte, pred, average='macro')), "per_class": f1_in100},
        "cross_dataset": {"macro_f1": float(f1_score(yc, pc_, average='macro')), "per_class": f1_out100},
        "V_recall_cross_100hz": float(cmo_n[2,2]),
        "cm_cross_100hz": cmo.tolist()
    }

    print(json.dumps(ctrl, indent=2))
    json.dump(ctrl, open("control_100hz.json", "w"), indent=2)
