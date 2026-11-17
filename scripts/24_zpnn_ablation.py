"""Z-PNN ablation at full resolution: pretrained weights only (0 iterations) and
25, 100 and 200 target-adaptation iterations (seed 0), plus 100 iterations with
seed 1, on six controlled chips drawn with a fixed seed, at k = 0 and k = 2.
Offsets are measured with the same estimator; output results/tables/zpnn_ablation.csv"""
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
from offset import Raster, measure_offset  # noqa: E402
from zpnn_run import zpnn_fuse  # noqa: E402

CH = ROOT / "data/processed/chips"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
chips_all = sorted(jobs[(jobs.kind == "controlled")].chip.dropna().unique())
rng = np.random.default_rng(7)
sel = list(rng.choice(chips_all, 6, replace=False))
CONF = [(0, 0), (25, 0), (100, 0), (200, 0), (100, 1)]
rows = []
for ch in sel:
    for k in (0, 2):
        j = f"{ch}_k{k}"
        jm = jobs.loc[j]
        d = sio.loadmat(CH / f"{j}.mat", variable_names=["MS", "PAN", "SENSOR"])
        sensor = str(np.atleast_1d(d["SENSOR"])[0]).strip()
        for it, seed in CONF:
            t = time.time()
            z, hist = zpnn_fuse(np.moveaxis(d["MS"], -1, 0), d["PAN"], sensor, epochs=it, seed=seed)
            sec = time.time() - t
            lum = z[[4, 2, 1]].mean(0).astype(np.float64)
            img = Raster(lum, jm.x0, jm.y0, jm.pan_res)
            for m in mk[mk.job == j].itertuples():
                o = measure_offset(img, np.array([m.E, m.N]), m.strike, m.az_plus, m.az_minus, 0.0, "band")
                rows.append(dict(chip=ch, k=k, iterations=it, seed=seed, seconds=sec, marker=m.marker,
                                 H_true=m.H_true, loss_final=hist[-1] if len(hist) else np.nan, **o.flat()))
        print(ch, k, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/tables/zpnn_ablation.csv", index=False)
