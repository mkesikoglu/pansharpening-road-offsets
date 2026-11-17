"""Download PAN, MS, visual and data-mask assets of WV02/WV03 tiles that contain at
least one candidate field site (500 m buffer). The tile list is derived from
the field-site coordinates only."""
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform

import io_utils

inv = pd.read_csv(ROOT / "data/processed/tile_inventory.csv")
d = pd.read_csv(ROOT / "data/raw/usgs/duross2024/Displacement_observations_June2023.csv")
d = d[(d.Setting == "Cultural") & (d.Fault == "EAF")]
tr = Transformer.from_crs(4326, 32637, always_xy=True).transform
pts = [transform(tr, Point(x, y)).buffer(500) for x, y in zip(d.Longitude, d.Latitude)]
keep = []
for _, r in inv.iterrows():
    g = transform(tr, wkt.loads(r.wkt))
    if any(g.intersects(p) for p in pts):
        keep.append(r)
inv = pd.DataFrame(keep)
inv.to_csv(ROOT / "data/processed/tiles_needed.csv", index=False)
print(len(inv), "tiles needed")
urls = [u for k in ("ms", "pan", "mask", "visual") for u in inv[k].dropna()]


def job(u):
    try:
        p = io_utils.fetch(u)
        return u, p.stat().st_size, None
    except Exception as e:  # keep going; failures are logged
        return u, 0, repr(e)


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(job, urls))
rows = []
for u, size, err in res:
    p = io_utils.local_path(u)
    md5 = hashlib.md5(p.read_bytes()).hexdigest() if size else ""
    rows.append(dict(url=u, bytes=size, md5=md5, error=err))
pd.DataFrame(rows).to_csv(ROOT / "data/processed/download_log.csv", index=False)
print(sum(r["error"] is None for r in rows), "of", len(rows), "downloaded")
