"""Residual PAN-MS registration of each evaluation chip (k = 0 controlled chips and
field chips). PAN is block-averaged to the MS grid; a PAN-like intensity is formed
from the eight MS bands by least squares; both are converted to gradient magnitude
and registered by upsampled phase correlation (1/20 MS pixel). The shift is
estimated on the whole chip and on a 64 x 64 MS-pixel window centred on each marker."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "data/processed/chips"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")


def grad(a):
    return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))


def shift(a, b):
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=20, normalization=None)
    return s  # (row, col) shift that registers b to a, MS pixels


rows = []
for j in jobs[jobs.k == 0].itertuples():
    d = sio.loadmat(CH / f"{j.job}.mat", variable_names=["MS", "PAN"])
    M, P = d["MS"].astype(float), d["PAN"].astype(float)
    h = M.shape[0]
    Pl = P.reshape(h, 4, h, 4).mean((1, 3))
    X = np.column_stack([M.reshape(-1, 8), np.ones(h * h)])
    w = np.linalg.lstsq(X, Pl.ravel(), rcond=None)[0]
    I = (X @ w).reshape(h, h)
    gP, gI = grad(Pl), grad(I)
    r0, c0 = shift(gP, gI)
    rec = dict(job=j.job, kind=j.kind, catalog_id=j.catalog_id, ms_res=j.ms_res,
               dE_chip_m=c0 * j.ms_res, dN_chip_m=-r0 * j.ms_res)
    for m in mk[mk.job == j.job].itertuples():
        cc = int((m.E - j.x0) / j.ms_res)
        rr = int((j.y0 - m.N) / j.ms_res)
        a, b = max(rr - 32, 0), max(cc - 32, 0)
        win = (slice(a, a + 64), slice(b, b + 64))
        r1, c1 = shift(gP[win], gI[win])
        rows.append(dict(rec, marker=m.marker, ID=m.ID, dE_local_m=c1 * j.ms_res, dN_local_m=-r1 * j.ms_res))
df = pd.DataFrame(rows)
df["mag_chip_m"] = np.hypot(df.dE_chip_m, df.dN_chip_m)
df["mag_local_m"] = np.hypot(df.dE_local_m, df.dN_local_m)
df.to_csv(ROOT / "results/tables/registration.csv", index=False)
print(df.groupby("catalog_id")[["dE_chip_m", "dN_chip_m", "mag_chip_m", "mag_local_m"]].median().round(3))
print(df[["mag_chip_m", "mag_local_m"]].describe().round(3))
