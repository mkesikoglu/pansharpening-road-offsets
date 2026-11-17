"""Apply the frozen offset estimator to every marker on every product.
Products: PAN, EXP, GSA, BDSD-PC, MTF-GLP-HPM, AWLP, Z-PNN (luminance of R,G,B)
and the vendor visual product (VIS). Output: results/tables/measurements.csv"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import PARAMS, Raster, measure_offset, unit  # noqa: E402

CH = ROOT / "data/processed/chips"
RES = ROOT / "data/processed/fused"


def local_strike(E, N, lines, radius=300.0):
    """Length-weighted mean azimuth (deg, mod 180) of mapped rupture segments within radius."""
    pts = np.vstack([np.asarray(l) for l in lines])
    seg = []
    for l in lines:
        a = np.asarray(l)
        for p, q in zip(a[:-1], a[1:]):
            m = 0.5 * (p + q)
            if np.hypot(*(m - [E, N])) < radius:
                seg.append(q - p)
    if not seg:
        return np.nan
    ang = np.array([np.arctan2(v[0], v[1]) for v in seg]) * 2  # doubled-angle mean
    w = np.array([np.hypot(*v) for v in seg])
    return float(np.rad2deg(np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / 2) % 180)


def _first_num(x):
    """Parse field strike strings ('57-62' -> 59.5, circular for ranges across 0/360)."""
    import re
    if pd.isna(x):
        return np.nan
    v = [float(s) for s in re.findall(r"\d+(?:\.\d+)?", str(x))]
    if not v:
        return np.nan
    if len(v) == 2 and abs(v[0] - v[1]) > 180:
        v[1] += 360
    return float(np.mean(v)) % 360


# optional sensitivity run: python 12_measure.py <min_cnr>  -> measurements_cnr<min_cnr>.csv
PRM = dict(PARAMS)
SUFFIX = ""
if len(sys.argv) > 1:
    PRM["min_cnr"] = float(sys.argv[1])
    SUFFIX = f"_cnr{sys.argv[1]}"


def run_job(args):
    job, jm, mk = args
    f = RES / f"{job}_res.mat"
    if not f.exists():
        return []
    r = sio.loadmat(f, squeeze_me=True, variable_names=["L", "NAMES"])
    L, names = r["L"], [str(n) for n in r["NAMES"]]
    prods = {n: L[:, :, i].astype(np.float64) for i, n in enumerate(names)}
    vis = CH / f"{job}_vis.npy"
    if vis.exists():
        prods["VIS"] = np.load(vis).astype(np.float64)
    rasters = {n: Raster(a, jm["x0"], jm["y0"], jm["pan_res"]) for n, a in prods.items() if not np.isnan(a).all()}
    rows = []
    for m in mk:
        phi = m["strike"]
        if m["kind"] == "field":
            n_north = unit(phi + 90.0)[1] > 0
            azp, azm = (m["az_N"], m["az_S"]) if n_north else (m["az_S"], m["az_N"])
        else:
            azp, azm = m["az_plus"], m["az_minus"]
        for pname, img in rasters.items():
            o = measure_offset(img, np.array([m["E"], m["N"]]), phi, azp, azm,
                               aperture=m["aperture"], mode=m["mode"], prm=PRM)
            rows.append(dict(job=job, marker=m["marker"], ID=m["ID"], product=pname, **o.flat()))
    return rows


if __name__ == "__main__":
    jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
    mk = pd.read_csv(ROOT / "data/processed/markers.csv")
    # rupture-trace strike where the field strike is missing
    from pyproj import Transformer
    from shapely.geometry import shape
    tr = Transformer.from_crs(4326, 32637, always_xy=True)
    gj = json.loads((ROOT / "data/raw/usgs/duross2024/Rupture_lines_June2023.geojson").read_text(encoding="utf-8"))
    lines = []
    for ft in gj["features"]:
        geo = ft["geometry"]
        for cc in (geo["coordinates"] if geo["type"] == "MultiLineString" else [geo["coordinates"]]):
            if len(cc) > 1:
                x, y = tr.transform([c[0] for c in cc], [c[1] for c in cc])
                lines.append(list(zip(x, y)))
    miss = mk.strike.isna()
    mk.loc[miss, "strike"] = [local_strike(e, n, lines) for e, n in zip(mk.E[miss], mk.N[miss])]
    mk["strike_source"] = np.where(miss, "rupture map", np.where(mk.kind == "field", "field", "synthetic"))
    # no mapped rupture nearby: strike reported at the nearest field observation
    still = mk.strike.isna()
    if still.any():
        allobs = pd.read_csv(ROOT / "data/raw/usgs/duross2024/Displacement_observations_June2023.csv")
        from pyproj import Transformer as _T
        e_, n_ = _T.from_crs(4326, 32637, always_xy=True).transform(allobs.Longitude.values, allobs.Latitude.values)
        num = allobs.FaultStrike.map(lambda v: _first_num(v))
        for i in np.nonzero(still.values)[0]:
            dd = np.hypot(e_ - mk.E.iloc[i], n_ - mk.N.iloc[i])
            dd[(allobs.ID == mk.ID.iloc[i]).values | num.isna().values] = np.inf
            j = int(np.argmin(dd))
            mk.iloc[i, mk.columns.get_loc("strike")] = num.iloc[j]
            mk.iloc[i, mk.columns.get_loc("strike_source")] = f"nearest field obs. {int(allobs.ID.iloc[j])}"
    if not SUFFIX:
        mk.to_csv(ROOT / "data/processed/markers_measured.csv", index=False)
    tasks = [(j, jobs.loc[j].to_dict(), g.to_dict("records")) for j, g in mk.groupby("job")]
    rows = []
    for k, t_ in enumerate(tasks):
        try:
            rows += run_job(t_)
        except Exception as e:  # keep going, log the job
            print("failed", t_[0], repr(e), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/tables/measurements{SUFFIX}.csv", index=False)
    if not SUFFIX:
        (ROOT / "results/tables/offset_params.json").write_text(json.dumps(PARAMS, indent=1))
    print(out.groupby("product").ok.mean())
