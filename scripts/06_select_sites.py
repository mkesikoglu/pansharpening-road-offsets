"""A priori selection of field-referenced sites (DuRoss et al., 2024) and their
coverage by WV02 acquisitions. Rules use field metadata only (no image results)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chips  # noqa: E402

d = pd.read_csv(ROOT / "data/raw/usgs/duross2024/Displacement_observations_June2023.csv")
tr = Transformer.from_crs(4326, 32637, always_xy=True)
d["E"], d["N"] = tr.transform(d.Longitude.values, d.Latitude.values)


def first_num(x):
    """Parse '57-62' -> mean 59.5 ; '020-022' -> 21 ; '350-000' -> 355 (circular)."""
    if pd.isna(x):
        return np.nan
    v = [float(t) for t in re.findall(r"\d+(?:\.\d+)?", str(x))]
    if not v:
        return np.nan
    if len(v) == 2 and abs(v[0] - v[1]) > 180:
        v[1] += 360
    if len(v) == 2 and str(x).startswith("242-24"):
        v = [242.0, 244.0]  # '242-24' in source = 242-244
    return float(np.mean(v)) % 360


for c in ("FaultStrike", "FeatAzim_N", "FeatAzim_S"):
    d[c + "_deg"] = d[c].map(first_num)
d["Aperture_num"] = pd.to_numeric(d.Aperture_m.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")

LINEAR = {"Gravel road": "road", "Road edge": "road", "Field edge": "field edge",
          "Canal edge": "canal/ditch", "Irrigation ditch": "canal/ditch",
          "Concrete wall": "wall", "Cement and rock wall": "wall"}
d["marker_class"] = d.Feature.map(LINEAR)
d["eligible_type"] = d.marker_class.notna() & (d.Setting == "Cultural") & (d.Fault == "EAF")
d["eligible_size"] = d.H_pref_m >= 1.0
d["tier"] = np.where(d.MsmtType.isin(["Complete", "Likely complete"]), "primary", "sensitivity")
sel = d[d.eligible_type & d.eligible_size].copy()

# spatial clusters: observations closer than 30 m treated as one site
xy = sel[["E", "N"]].values
lab = -np.ones(len(sel), int)
k = 0
for i in range(len(sel)):
    if lab[i] >= 0:
        continue
    stack = [i]
    lab[i] = k
    while stack:
        j = stack.pop()
        dd = np.hypot(*(xy - xy[j]).T)
        for m in np.nonzero((dd < 30) & (lab < 0))[0]:
            lab[m] = k
            stack.append(m)
    k += 1
sel["cluster"] = lab

# coverage: valid unobscured pixels within 60 m in each WV02 acquisition
acq = chips.INV.drop_duplicates("catalog_id")[["catalog_id", "date", "platform", "off_nadir", "gsd"]]
rows = []
for _, s in sel.iterrows():
    for _, a in acq.iterrows():
        ch = chips.make_chip(a.catalog_id, s.E, s.N, size_pan=512)
        vf = 0.0 if ch is None else chips.valid_fraction(ch, s.E, s.N)
        rows.append(dict(ID=s.ID, catalog_id=a.catalog_id, date=a.date + "_" + a.platform, valid=vf))
cov = pd.DataFrame(rows)
cov.to_csv(ROOT / "data/processed/site_coverage.csv", index=False)
piv = cov.pivot(index="ID", columns="date", values="valid").round(2)
sel = sel.merge((cov.valid >= 0.95).groupby(cov.ID).sum().rename("n_acq"), left_on="ID", right_index=True)
sel.to_csv(ROOT / "data/processed/sites_selected.csv", index=False)
print(sel[["ID", "Feature", "marker_class", "tier", "H_pref_m", "H_min_m", "H_max_m", "FaultStrike_deg",
           "FeatAzim_N_deg", "FeatAzim_S_deg", "MsmtMethod", "cluster", "n_acq"]].to_string())
print(piv.to_string())
