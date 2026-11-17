"""Apparent strike-parallel offset produced by orthorectifying a post-event image
with a pre-event surface model across a fault scarp of height V:
    dH = V * tan(incidence) * cos(view_azimuth - strike)
(view azimuth: direction from the sub-satellite point to the target, STAC view
extension). Reported as magnitude per field observation and acquisition."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cat = json.loads((ROOT / "data/raw/maxar_catalog/Kahramanmaras.geojson").read_text(encoding="utf-8"))["features"]
geo = pd.DataFrame([f["properties"] for f in cat]).groupby("catalog_id").agg(
    incidence=("view:incidence_angle", "mean"), view_az=("view:azimuth", "mean"),
    off_nadir=("view:off_nadir", "mean"))
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
mk = mk[mk.kind == "field"]
d = pd.read_csv(ROOT / "data/raw/usgs/duross2024/Displacement_observations_June2023.csv")
d["V"] = pd.to_numeric(d.V_pref_m, errors="coerce")
x = mk.merge(d[["ID", "V", "ScarpFaceDir"]], on="ID").merge(geo, left_on="catalog_id", right_index=True)
# the Maxar ARD field view:incidence_angle holds the satellite elevation angle (90 deg minus incidence)
x["incidence"] = 90.0 - x.incidence
x["dH_parallax"] = (x.V * np.tan(np.deg2rad(x.incidence)) *
                    np.abs(np.cos(np.deg2rad(x.view_az - x.strike))))
x[["ID", "marker", "catalog_id", "V", "ScarpFaceDir", "incidence", "view_az", "strike", "dH_parallax"]].to_csv(
    ROOT / "results/tables/field_parallax.csv", index=False)
print(x[["ID", "catalog_id", "V", "incidence", "view_az", "strike", "dH_parallax"]].round(2).to_string())
