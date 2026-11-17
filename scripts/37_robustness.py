"""Robustness experiments requested in the second review.
stage 'lists'        : job lists for the side-dependent misregistration run and the intensity subset
stage 'zpnn_side'    : adapted Z-PNN on the side-misregistered reduced-scale inputs
stage 'measure_side' : offsets on the side-misregistered products -> results/tables/rr_side_measurements.csv
stage 'zpnn_full'    : adapted Z-PNN with all eight bands kept for the intensity subset
stage 'intens'       : offsets on four intensity definitions -> results/tables/intensity_measurements.csv
stage 'xcorr'        : edge-free cross-correlation estimator on all controlled jobs -> results/tables/xcorr_measurements.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import Raster, measure_offset  # noqa: E402

CH = ROOT / "data/processed/chips"
SIDE = ROOT / "data/processed/rr_side"
FULL = ROOT / "data/processed/intensity"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
ctrl = jobs[jobs.kind == "controlled"]
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
stage = sys.argv[1]
RGB = [4, 2, 1]  # Red, Green, Blue (0-based band indices)


def subset_chips():
    """Up to five chips per acquisition, fixed seed."""
    rng = np.random.default_rng(23)
    out = []
    for cid, g in ctrl.drop_duplicates("chip").groupby("catalog_id"):
        c = sorted(g.chip)
        out += list(rng.choice(c, min(5, len(c)), replace=False))
    return sorted(out)


def intensities(F, ms, pan):
    """F: (8,H,W) product; ms: (8,h,w) original MS; pan: (H,W). Four single-channel images."""
    h = ms.shape[1]
    Pl = pan.reshape(h, 4, h, 4).mean((1, 3))
    X = np.column_stack([ms.reshape(8, -1).T, np.ones(h * h)])
    wts = np.linalg.lstsq(X, Pl.ravel(), rcond=None)[0]
    Z = ms.reshape(8, -1).T
    mu = Z.mean(0)
    _, _, vt = np.linalg.svd(Z - mu, full_matrices=False)
    v = vt[0] * np.sign(vt[0][RGB].sum())
    return dict(RGB=F[RGB].mean(0), PANW=np.tensordot(wts[:8], F, 1) + wts[8], NIR1=F[6],
                PC1=np.tensordot(v, F - mu[:, None, None], 1))


if stage == "lists":
    SIDE.mkdir(parents=True, exist_ok=True)
    FULL.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed/rr_side_list.txt").write_text(
        "\n".join(f"../data/processed/chips/{j.job}.mat;{j.orientation}" for j in ctrl.itertuples()), encoding="ascii")
    sub = subset_chips()
    (ROOT / "data/processed/intensity_list.txt").write_text(
        "\n".join(f"../data/processed/chips/{c}_k{k}.mat" for c in sub for k in range(4)), encoding="ascii")
    pd.Series(sub).to_csv(ROOT / "results/tables/v2/intensity_subset.csv", index=False)
    print(len(ctrl), len(sub))

elif stage == "zpnn_side":
    from zpnn_run import zpnn_fuse
    for f in sorted(SIDE.glob("*_s1.mat")):
        z = f.with_name(f.stem + "_zpnn.npz")
        if z.exists():
            continue
        d = sio.loadmat(f, variable_names=["MS_LR", "PAN_LR", "sensor"])
        sensor = str(np.atleast_1d(d["sensor"])[0]).strip()
        ad, _ = zpnn_fuse(np.moveaxis(d["MS_LR"], -1, 0), d["PAN_LR"], sensor, epochs=100)
        np.savez_compressed(z, ZPNN=ad[RGB].mean(0))

elif stage == "measure_side":
    rows = []
    for j in ctrl.itertuples():
        f = SIDE / f"{j.job}_s1.mat"
        z = f.with_name(f.stem + "_zpnn.npz")
        if not (f.exists() and z.exists()):
            continue
        r = sio.loadmat(f, squeeze_me=True, variable_names=["L", "NAMES"])
        prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
        prods["ZPNN"] = np.load(z)["ZPNN"].astype(np.float64)
        for mm in mk[mk.job == j.job].itertuples():
            for p, a in prods.items():
                o = measure_offset(Raster(a, j.x0, j.y0, j.ms_res), np.array([mm.E, mm.N]), mm.strike,
                                   mm.az_plus, mm.az_minus, aperture=0.0, mode="band")
                rows.append(dict(job=j.job, chip=j.chip, k=j.k, marker=mm.marker, product=p, H_true=mm.H_true,
                                 ms_res=j.ms_res, **o.flat()))
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/rr_side_measurements.csv", index=False)

elif stage == "zpnn_full":
    from zpnn_run import zpnn_fuse
    for line in (ROOT / "data/processed/intensity_list.txt").read_text().split():
        name = Path(line).stem
        z = FULL / f"{name}_zpnn_full.npy"
        if z.exists():
            continue
        d = sio.loadmat(CH / f"{name}.mat", variable_names=["MS", "PAN", "SENSOR"])
        sensor = str(np.atleast_1d(d["SENSOR"])[0]).strip()
        ad, _ = zpnn_fuse(np.moveaxis(d["MS"], -1, 0), d["PAN"], sensor, epochs=100)
        np.save(z, ad.astype(np.float32))

elif stage == "intens":
    jj = jobs.set_index("job")
    rows = []
    for line in (ROOT / "data/processed/intensity_list.txt").read_text().split():
        name = Path(line).stem
        jm = jj.loc[name]
        d = sio.loadmat(CH / f"{name}.mat", variable_names=["MS", "PAN"])
        ms, pan = np.moveaxis(d["MS"].astype(np.float64), -1, 0), d["PAN"].astype(np.float64)
        F = sio.loadmat(FULL / f"{name}_full.mat", squeeze_me=True, struct_as_record=False)["F"]
        prods = {p: np.moveaxis(getattr(F, p).astype(np.float64), -1, 0) for p in F._fieldnames}
        prods["ZPNN"] = np.load(FULL / f"{name}_zpnn_full.npy").astype(np.float64)
        for mm in mk[mk.job == name].itertuples():
            for p, A in prods.items():
                for iname, img in intensities(A, ms, pan).items():
                    o = measure_offset(Raster(img, jm.x0, jm.y0, jm.pan_res), np.array([mm.E, mm.N]), mm.strike,
                                       mm.az_plus, mm.az_minus, aperture=0.0, mode="band")
                    rows.append(dict(job=name, chip=jm.chip, k=jm.k, marker=mm.marker, product=p, intensity=iname,
                                     H_true=mm.H_true, **o.flat()))
        print(name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/intensity_measurements.csv", index=False)

elif stage == "xcorr":
    from offset_xcorr import measure_offset_xcorr
    rows = []
    for j in ctrl.itertuples():
        r = sio.loadmat(ROOT / f"data/processed/fused/{j.job}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
        prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
        v = CH / f"{j.job}_vis.npy"
        if v.exists():
            prods["VIS"] = np.load(v).astype(np.float64)
        for mm in mk[mk.job == j.job].itertuples():
            for p, a in prods.items():
                if np.isnan(a).all():
                    continue
                o = measure_offset_xcorr(Raster(a, j.x0, j.y0, j.pan_res), np.array([mm.E, mm.N]), mm.strike,
                                         mm.az_plus, mm.az_minus)
                rows.append(dict(job=j.job, chip=j.chip, k=j.k, marker=mm.marker, product=p, H_true=mm.H_true, **o))
        print(j.job, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/xcorr_measurements.csv", index=False)
