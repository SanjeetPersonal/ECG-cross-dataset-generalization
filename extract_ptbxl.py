import os
import ast
import urllib.request
import concurrent.futures as cf
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from wfdb import processing

base = "https://physionet.org/files/ptb-xl/1.0.3/"
os.makedirs("data/ptbxl", exist_ok=True)

for f in ["ptbxl_database.csv", "scp_statements.csv"]:
    urllib.request.urlretrieve(base + f, f"data/ptbxl/{f}")
    print("got", f, os.path.getsize(f"data/ptbxl/{f}"), "bytes")

ptb = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv("data/ptbxl/scp_statements.csv", index_col=0)
ptb["scp_codes"] = ptb["scp_codes"].apply(ast.literal_eval)

CROSSWALK = {
    'PVC': 'V', 'BIGU': 'V', 'TRIGU': 'V',
    'PAC': 'S', 'AFIB': 'S', 'AFLT': 'S', 'SVARR': 'S', 'SVTAC': 'S', 'PSVT': 'S', 'STACH': 'S',
    'SR': 'N', 'SBRAD': 'N', 'SARRH': 'N',
}
PRIORITY = ['V', 'S', 'N']

def assign_aami(codes):
    present = set()
    for c, lik in codes.items():
        if c in CROSSWALK and lik >= 0:
            present.add(CROSSWALK[c])
    if 'PACE' in codes:
        return None
    for cls in PRIORITY:
        if cls in present:
            return cls
    return None

ptb["aami"] = ptb["scp_codes"].apply(assign_aami)
lab = ptb.dropna(subset=["aami"])

V = lab[lab.aami == "V"]
S = lab[lab.aami == "S"]
N = lab[lab.aami == "N"].sample(2500, random_state=42)
sample = pd.concat([N, V, S])
files = sample["filename_lr"].tolist()
sample.to_pickle("ptbxl_sample.pkl")

os.makedirs("data/ptbxl/records100", exist_ok=True)

def fetch_one(fn):
    d = os.path.dirname(fn)
    os.makedirs(f"data/ptbxl/{d}", exist_ok=True)
    ok = True
    for ext in [".dat", ".hea"]:
        dst = f"data/ptbxl/{fn}{ext}"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            urllib.request.urlretrieve(base + fn + ext, dst)
        except Exception:
            ok = False
    return ok

with cf.ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(fetch_one, files))
print(f"downloaded {sum(results)}/{len(files)}")
