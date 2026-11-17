"""Field case study: sensitivity of image-based offsets to the projected marker
length (aperture) and to the position of the field point.
Aperture: marker length L = 20, 40, 60, 80, 100 m on each side (nominal 40 m).
Position: field point displaced by -10, 0, +10 m along strike and -5, 0, +5 m
across strike (L = 40 m). Aperture: excluded zone A = 0, 10, 20, 40, 60 m across the
fault (stations start at max(A/2, 5 m)), L = 40 m. Output results/tables/field_sensitivity.csv"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import PARAMS, Raster, measure_offset, unit  # noqa: E402

jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
mk = mk[mk.kind == "field"]
CONFIGS = [("L", L, 0, 0) for L in (20, 40, 60, 80, 100)] + \
          [("pos", 40, du, dv) for du in (-10, 0, 10) for dv in (-5, 0, 5) if (du, dv) != (0, 0)] + \
          [("A", 40, A, 0) for A in (0, 10, 20, 40, 60)]  # aperture A (m) with L fixed; du holds A
ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # run one configuration family, e.g. A
if ONLY:
    CONFIGS = [c for c in CONFIGS if c[0] == ONLY]
rows = []
for j, g in mk.groupby("job"):
    jm = jobs.loc[j]
    r = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
    prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
    v = ROOT / f"data/processed/chips/{j}_vis.npy"
    if v.exists():
        prods["VIS"] = np.load(v).astype(np.float64)
    ras = {n: Raster(a, jm.x0, jm.y0, jm.pan_res) for n, a in prods.items() if not np.isnan(a).all()}
    for m in g.itertuples():
        north = unit(m.strike + 90.0)[1] > 0
        azp, azm = (m.az_N, m.az_S) if north else (m.az_S, m.az_N)
        s, n = unit(m.strike), unit(m.strike + 90.0)
        for kind, L, du, dv in CONFIGS:
            prm = dict(PARAMS, length=float(L))
            if kind == "A":
                p0, ap = np.array([m.E, m.N]), float(du)
            else:
                p0, ap = np.array([m.E, m.N]) + du * s + dv * n, m.aperture
            for pname, img in ras.items():
                o = measure_offset(img, p0, m.strike, azp, azm, aperture=ap, mode=m.mode, prm=prm)
                rows.append(dict(job=j, catalog_id=jm.catalog_id, ID=m.ID, marker=m.marker, product=pname,
                                 config=kind, L=L, du=du, dv=dv, H=o.H, ok=o.ok, H_ref=m.H_ref,
                                 H_min=m.H_min, H_max=m.H_max, aperture_field=m.aperture))
    print(j, flush=True)
out = ROOT / ("results/tables/field_sensitivity.csv" if not ONLY else f"results/tables/field_sensitivity_{ONLY}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
