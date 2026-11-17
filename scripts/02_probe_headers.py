"""Read remote GeoTIFF headers of one PAN/MS pair per acquisition."""
import os
from pathlib import Path

import pandas as pd
import rasterio

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
ROOT = Path(__file__).resolve().parents[1]
inv = pd.read_csv(ROOT / "data/processed/tile_inventory.csv")
for cid, g in inv.groupby("catalog_id"):
    r = g.iloc[0]
    for kind in ("pan", "ms"):
        with rasterio.open(r[kind]) as src:
            print(cid, r.date, kind, src.count, src.dtypes[0], src.res, src.shape,
                  src.crs.to_epsg(), src.descriptions, src.nodata)
            if kind == "ms":
                print("   tags:", {k: v for k, v in src.tags().items()})
                print("   band1 tags:", src.tags(1))
