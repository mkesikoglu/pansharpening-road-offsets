"""Metadata manifest: for every ARD tile used by a chip, archive the official STAC
item JSON, record its acquisition fields and the actual raster grid and checksums.
Produces data/processed/manifest_tiles.csv and data/processed/manifest_acquisitions.csv,
from which Table 1 is generated."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import io_utils  # noqa: E402
import requests  # noqa: E402

STAC = ROOT / "data/raw/maxar_stac"
STAC.mkdir(parents=True, exist_ok=True)
inv = pd.read_csv(ROOT / "data/processed/tile_inventory.csv")
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
used = inv[inv.catalog_id.isin(jobs.catalog_id.unique())]
used = used[[io_utils.local_path(p).exists() and io_utils.local_path(m).exists() for p, m in zip(used.pan, used.ms)]]
rows = []
for r in used.itertuples():
    url = r.pan.rsplit("/", 1)[0] + f"/{r.catalog_id}.json"
    f = STAC / f"{r.catalog_id}_{r.quadkey}.json"
    if not f.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        f.write_text(resp.text, encoding="utf-8")
    item = json.loads(f.read_text(encoding="utf-8"))
    p = item["properties"]
    rec = dict(catalog_id=r.catalog_id, quadkey=str(r.quadkey), stac_url=url, datetime_utc=p.get("datetime"),
               platform=p.get("platform"), native_gsd_m=p.get("gsd"), off_nadir_deg=p.get("view:off_nadir"),
               view_azimuth_deg=p.get("view:azimuth"), sat_elevation_deg=p.get("view:incidence_angle"),
               sun_azimuth_deg=p.get("view:sun_azimuth"), sun_elevation_deg=p.get("view:sun_elevation"),
               epsg=p.get("proj:epsg"), clouds_pct=p.get("tile:clouds_percent"))
    for kind in ("pan", "ms", "visual"):
        lp = io_utils.local_path(getattr(r, kind))
        if lp.exists():
            with rasterio.open(lp) as s:
                rec[f"{kind}_grid_m"] = s.res[0]
                rec[f"{kind}_dtype"] = s.dtypes[0]
                rec[f"{kind}_bands"] = s.count
                rec[f"{kind}_transform"] = tuple(round(v, 6) for v in s.transform[:6])
            rec[f"{kind}_md5"] = hashlib.md5(lp.read_bytes()).hexdigest()
    rows.append(rec)
tiles = pd.DataFrame(rows)
tiles.to_csv(ROOT / "data/processed/manifest_tiles.csv", index=False)
acq = tiles.groupby("catalog_id").agg(
    datetime_utc=("datetime_utc", "min"), platform=("platform", "first"), native_gsd_m=("native_gsd_m", "mean"),
    pan_grid_m=("pan_grid_m", "first"), ms_grid_m=("ms_grid_m", "first"), visual_grid_m=("visual_grid_m", "first"),
    off_nadir_deg=("off_nadir_deg", "mean"), view_azimuth_deg=("view_azimuth_deg", "mean"),
    sat_elevation_deg=("sat_elevation_deg", "mean"), sun_azimuth_deg=("sun_azimuth_deg", "mean"),
    sun_elevation_deg=("sun_elevation_deg", "mean"), n_tiles=("quadkey", "nunique"))
acq.to_csv(ROOT / "data/processed/manifest_acquisitions.csv")
print(acq.round(2).to_string())
