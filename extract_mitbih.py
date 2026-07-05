import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Downloads MIT-BIH directly from PhysioNet instead of relying on a local zip
os.makedirs("data/mitbih", exist_ok=True)
wfdb.dl_database('mitdb', dl_dir='data/mitbih')
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
    if len(idx) == 0: return 0.0
    return (idx.max()-idx.min())/fs*1000

def extract_record(rec_id):
    r = wfdb.rdrecord(f"{root}/{rec_id}")
    fs = r.fs
    sig = r.p_signal[:,0]
    sig = bandpass(sig, fs)
    ann = wfdb.rdann(f"{root}/{rec_id}", "atr")
    samp = ann.sample; sym = ann.symbol
    beats = [(s,y) for s,y in zip(samp,sym) if y in beat_syms]
    if len(beats) < 3: return []
    positions = np.array([b[0] for b in beats])
    labels = [AAMI[b[1]] for b in beats]
    rr = np.diff(positions)/fs*1000
    rows = []
    W = int(0.14*fs)
    for i in range(1, len(beats)-1):
        pre_rr = rr[i-1]; post_rr = rr[i]
        lo = max(0, i-5); hi = min(len(rr), i+5)
        loc_rr = np.mean(rr[lo:hi])
        pos = positions[i]
        if pos-W < 0 or pos+W >= len(sig): continue
        seg = sig[pos-W:pos+W]
        amp = seg.max()-seg.min()
        qw = qrs_width_proxy(seg, fs)
        rows.append(dict(rec=rec_id, label=labels[i],
                         pre_rr=pre_rr, post_rr=post_rr, loc_rr=loc_rr,
                         rr_ratio=pre_rr/loc_rr if loc_rr > 0 else 1.0,
                         prepost_ratio=pre_rr/post_rr if post_rr > 0 else 1.0,
                         amp=amp, qrs_width=qw))
    return rows

if __name__ == "__main__":
    all_rows = []
    for rid in DS1+DS2:
        all_rows.extend(extract_record(rid))
    mit = pd.DataFrame(all_rows)
    mit["split"] = np.where(mit["rec"].isin(DS1), "DS1_train", "DS2_test")
    print("total beats:", len(mit))
    print("\nclass distribution:")
    print(mit.groupby(["split","label"]).size().unstack(fill_value=0))
    mit.to_pickle("mitbih_beats.pkl")
