"""Run Z-PNN on every chip (full resolution) and on the reduced-resolution
inputs of evaluation chips. Outputs are written to <job>_zpnn.mat
(variables ZPNN and, for evaluation chips, ZPNN_RR)."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zpnn_run import zpnn_fuse  # noqa: E402

CH = ROOT / "data/processed/chips"
LOG = ROOT / "results/tables/zpnn_log.csv"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
for i, j in enumerate(jobs.itertuples()):
    out = CH / f"{j.job}_zpnn.mat"
    if out.exists():
        continue
    d = sio.loadmat(CH / f"{j.job}.mat", variable_names=["MS", "PAN", "SENSOR", "ZPNN"])
    sensor = str(np.atleast_1d(d["SENSOR"])[0]).strip()
    res, log = {}, []
    if "ZPNN" in d:  # produced by an earlier version of this script
        res["ZPNN"] = d["ZPNN"]
    else:
        t = time.time()
        z, hist = zpnn_fuse(np.moveaxis(d["MS"], -1, 0), d["PAN"], sensor)
        res["ZPNN"] = np.moveaxis(z, 0, -1).astype(np.float32)
        log.append(dict(job=j.job, res="FR", seconds=time.time() - t, loss0=hist[0], loss_min=hist.min()))
    rr = CH / f"{j.job}_RRin.mat"
    if rr.exists():
        r = sio.loadmat(rr)
        t = time.time()
        z, hist = zpnn_fuse(np.moveaxis(r["MS_LR"], -1, 0), r["PAN_LR"], sensor)
        res["ZPNN_RR"] = np.moveaxis(z, 0, -1).astype(np.float32)
        log.append(dict(job=j.job, res="RR", seconds=time.time() - t, loss0=hist[0], loss_min=hist.min()))
    tmp = CH / f"{j.job}_zpnn_part.mat"   # atomic hand-over to the fusion stage
    sio.savemat(tmp, res, do_compression=False)
    tmp.replace(out)
    if log:
        pd.DataFrame(log).to_csv(LOG, index=False, mode="a", header=not LOG.exists())
    print(i + 1, len(jobs), j.job, flush=True)
