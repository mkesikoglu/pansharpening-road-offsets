"""Field comparison with field angles converted from true north to UTM grid north.

The USGS field strikes and marker azimuths are referenced to true north, while the
estimator works in UTM zone 37N coordinates. This script adds the local grid azimuth
of true north (meridian convergence, about +1.0 to +1.2 deg at the field sites) to every
field-derived angle and repeats
  - the nominal field measurement of scripts/12_measure.py (field markers only), and
  - the reference-position, station-span and aperture runs of scripts/25_field_sensitivity.py.
Outputs (results/tables/v2):
  field_gridnorth_long.csv      nominal runs, both conventions side by side
  field_gridnorth_sens.csv      sensitivity runs with grid-north angles
  field_gridnorth_summary.csv   counts and changes quoted in the manuscript
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import PARAMS, Raster, measure_offset, unit  # noqa: E402

V = ROOT / "results/tables/v2"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
mk = mk[mk.kind == "field"].copy()

# grid azimuth of true north at each marker (UTM 37N)
to_ll = pyproj.Transformer.from_crs(32637, 4326, always_xy=True)
to_utm = pyproj.Transformer.from_crs(4326, 32637, always_xy=True)


def true_north_grid_azimuth(e, n):
    lon, lat = to_ll.transform(e, n)
    x0, y0 = to_utm.transform(lon, lat)
    x1, y1 = to_utm.transform(lon, lat + 0.01)
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)))


mk["gamma"] = [true_north_grid_azimuth(e, n) for e, n in zip(mk.E, mk.N)]
for c in ("strike", "az_N", "az_S"):
    mk[c + "_grid"] = (mk[c] + mk.gamma) % 360

CONFIGS = [("nominal", 40, 0, 0)] + [("L", L, 0, 0) for L in (20, 60, 80, 100)] + \
          [("pos", 40, du, dv) for du in (-10, 0, 10) for dv in (-5, 0, 5) if (du, dv) != (0, 0)] + \
          [("A", 40, A, 0) for A in (0, 10, 20, 40, 60)]


def products(job):
    r = sio.loadmat(ROOT / f"data/processed/fused/{job}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
    prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
    v = ROOT / f"data/processed/chips/{job}_vis.npy"
    if v.exists():
        prods["VIS"] = np.load(v).astype(np.float64)
    jm = jobs.loc[job]
    return {n: Raster(a, jm.x0, jm.y0, jm.pan_res) for n, a in prods.items() if not np.isnan(a).all()}


rows = []
for j, g in mk.groupby("job"):
    ras = products(j)
    for m in g.itertuples():
        for conv, phi, azn, azs in (("true", m.strike, m.az_N, m.az_S),
                                    ("grid", m.strike_grid, m.az_N_grid, m.az_S_grid)):
            north = unit(phi + 90.0)[1] > 0
            azp, azm = (azn, azs) if north else (azs, azn)
            s, n = unit(phi), unit(phi + 90.0)
            for kind, L, du, dv in CONFIGS:
                if conv == "true" and kind != "nominal":
                    continue  # sensitivity runs with true-north angles exist in field_sensitivity.csv
                prm = dict(PARAMS, length=float(L))
                if kind == "A":
                    p0, ap = np.array([m.E, m.N]), float(du)
                else:
                    p0, ap = np.array([m.E, m.N]) + du * s + dv * n, m.aperture
                for pname, img in ras.items():
                    o = measure_offset(img, p0, phi, azp, azm, aperture=ap, mode=m.mode, prm=prm)
                    rows.append(dict(job=j, catalog_id=jobs.loc[j].catalog_id, marker=m.marker, ID=m.ID,
                                     product=pname, convention=conv, config=kind, L=L, du=du, dv=dv,
                                     gamma=m.gamma, H=o.H, ok=bool(o.ok), H_min=m.H_min, H_max=m.H_max))
    print(j, flush=True)
d = pd.DataFrame(rows)

nom = d[d.config == "nominal"].pivot_table(index=["job", "marker", "ID", "product", "H_min", "H_max"],
                                            columns="convention", values=["H", "ok"], aggfunc="first")
nom.columns = [f"{a}_{b}" for a, b in nom.columns]
nom = nom.reset_index()
nom.to_csv(V / "field_gridnorth_long.csv", index=False)
d[(d.convention == "grid") & (d.config != "nominal")].to_csv(V / "field_gridnorth_sens.csv", index=False)

ok_t, ok_g = nom.ok_true.astype(bool), nom.ok_grid.astype(bool)
both = nom[ok_t & ok_g]
dh = (both.H_grid - both.H_true).abs()
g = nom[ok_g]
gnss = g[(g.marker == "M01") & (g.ID == 2)]
inside = ((gnss.H_grid >= gnss.H_min) & (gnss.H_grid <= gnss.H_max)).sum()

pos = d[(d.convention == "grid") & (d.config == "pos")].merge(
    nom[["job", "ID", "product", "H_grid", "ok_grid"]], on=["job", "ID", "product"])
pos_ok0 = pos.ok_grid.astype(bool)
pb = pos[pos.ok & pos_ok0]
summ = dict(
    gamma_min=mk.gamma.min(), gamma_max=mk.gamma.max(),
    n_attempts=len(nom), accepted_true=int(ok_t.sum()), accepted_grid=int(ok_g.sum()),
    accepted_both=len(both), lost=int((ok_t & ~ok_g).sum()), gained=int((~ok_t & ok_g).sum()),
    med_abs_change=dh.median(), max_abs_change=dh.max(), n_change_gt_005=int((dh > 0.05).sum()),
    n_change_gt_05=int((dh > 0.5).sum()),
    accepted_grid_M01=int((g.marker == "M01").sum()),
    gnss_accepted=len(gnss), gnss_inside=int(inside),
    pos_runs=int(pos_ok0.sum()), pos_rejected=int((pos_ok0 & ~pos.ok).sum()),
    pos_both=len(pb), pos_gt05=int(((pb.H - pb.H_grid).abs() > 0.5).sum()),
)
pd.Series(summ).to_csv(V / "field_gridnorth_summary.csv", header=["value"])
print(pd.Series(summ).round(3).to_string())
