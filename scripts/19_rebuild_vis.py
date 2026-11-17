"""Rebuild the vendor visual product on each chip's PAN grid by area averaging from
its native 0.30517578125 m ARD grid (the earlier chips used nearest-neighbour
resampling). For controlled chips the same whole-pixel dislocation (4k PAN pixels)
is applied after resampling. Writes <job>_vis.npy (luminance) and <job>_visrgb.npy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chips  # noqa: E402
import io_utils  # noqa: E402

CH = ROOT / "data/processed/chips"
MARGIN = 4  # MS pixels, as in 09_build_jobs.py
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")


def read_vis(cid, left, top, n, res):
    sub = chips.INV[chips.INV.catalog_id == cid]
    right, bottom = left + n * res, top - n * res
    paths = []
    for r in sub.itertuples():
        p = io_utils.local_path(r.visual) if isinstance(r.visual, str) else None
        if p is None or not p.exists():
            continue
        with rasterio.open(p) as s:
            b = s.bounds
        if b.left < right and b.right > left and b.bottom < top and b.top > bottom:
            paths.append(p)
    if not paths:
        return None
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, _ = merge(srcs, bounds=(left, bottom, right, top), res=res, nodata=0,
                       resampling=Resampling.average, indexes=[1, 2, 3])
    finally:
        for s in srcs:
            s.close()
    return arr[:, :n, :n].astype(np.float32)


def shift_vis(v, k, orientation):
    if not k:
        return v
    out = v.copy()
    _, H, W = v.shape
    if orientation == "NS":
        c = 4 * ((W // 4) // 2)
        out[:, :H - 4 * k, c:] = v[:, 4 * k:, c:]
    else:
        r = 4 * ((H // 4) // 2)
        out[:, r:, 4 * k:] = v[:, r:, :W - 4 * k]
    return out


done = 0
for j in jobs.itertuples():
    n = 1024
    if j.kind == "controlled":
        m = MARGIN * j.ms_res
        v = read_vis(j.catalog_id, j.x0 - m, j.y0 + m, n + 8 * MARGIN, j.pan_res)
        if v is None:
            continue
        v = shift_vis(v, int(j.k), j.orientation)[:, 4 * MARGIN:4 * MARGIN + n, 4 * MARGIN:4 * MARGIN + n]
    else:
        v = read_vis(j.catalog_id, j.x0, j.y0, n, j.pan_res)
        if v is None:
            continue
    np.save(CH / f"{j.job}_vis.npy", v.mean(0))
    np.save(CH / f"{j.job}_visrgb.npy", v.astype(np.uint8))
    done += 1
print(done, "visual chips rebuilt")
