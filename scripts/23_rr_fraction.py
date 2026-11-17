"""Reduced-scale fractional-phase experiment, Python side.
stage 'list'   : write the MATLAB job list (controlled chips x misregistration m = 0, 1, 2)
stage 'zpnn'   : Z-PNN on each reduced-scale input, adapted (100 iterations) and pretrained only
stage 'measure': offsets on all reduced-scale products, including the latent reference image
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
OUT = ROOT / "data/processed/rr_fraction"
OUT.mkdir(parents=True, exist_ok=True)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
ctrl = jobs[jobs.kind == "controlled"]
MIS = (0, 1, 2)
stage = sys.argv[1]

if stage == "list":
    lines = [f"../data/processed/chips/{j}.mat;{m}" for j in ctrl.job for m in MIS]
    (ROOT / "data/processed/rr_fraction_list.txt").write_text("\n".join(lines), encoding="ascii")
    print(len(lines))

elif stage == "zpnn":
    from zpnn_run import zpnn_fuse
    for f in sorted(OUT.glob("*_m[0-9].mat")):
        z = f.with_name(f.stem + "_zpnn.npz")
        if z.exists():
            continue
        d = sio.loadmat(f, variable_names=["MS_LR", "PAN_LR", "sensor"])
        sensor = str(np.atleast_1d(d["sensor"])[0]).strip()
        ms, pan = np.moveaxis(d["MS_LR"], -1, 0), d["PAN_LR"]
        ad, _ = zpnn_fuse(ms, pan, sensor, epochs=100)
        pre, _ = zpnn_fuse(ms, pan, sensor, epochs=0)
        np.savez_compressed(z, ZPNN=ad[[4, 2, 1]].mean(0), ZPNN0=pre[[4, 2, 1]].mean(0),
                            ZPNN_full=ad.astype(np.float32) if f.stem.endswith("_k0_m0") else np.zeros(1))

elif stage == "measure":
    from offset import Raster, measure_offset
    mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
    rows = []
    for j in ctrl.itertuples():
        for m in MIS:
            f = OUT / f"{j.job}_m{m}.mat"
            z = f.with_name(f.stem + "_zpnn.npz")
            if not (f.exists() and z.exists()):
                continue
            r = sio.loadmat(f, squeeze_me=True, variable_names=["L", "NAMES"])
            prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
            zz = np.load(z)
            prods["ZPNN"], prods["ZPNN0"] = zz["ZPNN"].astype(np.float64), zz["ZPNN0"].astype(np.float64)
            for mm in mk[mk.job == j.job].itertuples():
                for p, a in prods.items():
                    o = measure_offset(Raster(a, j.x0, j.y0, j.ms_res), np.array([mm.E, mm.N]), mm.strike,
                                       mm.az_plus, mm.az_minus, aperture=0.0, mode="band")
                    rows.append(dict(job=j.job, chip=j.chip, k=j.k, m=m, marker=mm.marker, product=p,
                                     H_true=mm.H_true, ms_res=j.ms_res, **o.flat()))
        print(j.job, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/rr_fraction_measurements.csv", index=False)
