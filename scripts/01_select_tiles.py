"""Select Maxar ARD tiles (post-event, separate PAN and MS assets) that overlap the
USGS field-displacement study area and record their metadata."""
import json
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import shape, box

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data/raw/maxar_catalog/Kahramanmaras.geojson"
OUT = ROOT / "data/processed/tile_inventory.csv"
# Field-study extent of DuRoss et al. (2024) data release (lon/lat, WGS84)
AOI = box(37.0428, 37.4378, 37.3307, 37.5895)

feats = json.loads(CAT.read_text(encoding="utf-8"))["features"]
rows = []
for f in feats:
    p = f["properties"]
    g = shape(f["geometry"])
    if not g.intersects(AOI):
        continue
    rows.append(dict(
        catalog_id=p["catalog_id"], date=p["datetime"][:10], datetime=p["datetime"],
        platform=p["platform"], gsd=p["gsd"], off_nadir=p.get("view:off_nadir"),
        azimuth=p.get("view:azimuth"), sun_az=p.get("view:sun_azimuth"),
        sun_el=p.get("view:sun_elevation"), epsg=p.get("proj:epsg"),
        quadkey=p["quadkey"], clouds_pct=p.get("tile:clouds_percent"),
        pan=p.get("pan_analytic"), ms=p.get("ms_analytic"), visual=p.get("visual"),
        mask=p.get("data-mask"), wkt=g.wkt))
df = pd.DataFrame(rows).sort_values(["date", "catalog_id", "quadkey"])
df = df[df.pan.notna() & df.ms.notna()]
df.to_csv(OUT, index=False)
print(df.groupby(["date", "platform", "catalog_id"]).agg(
    n=("quadkey", "size"), gsd=("gsd", "mean"), onad=("off_nadir", "mean"),
    cloud=("clouds_pct", "max")))
