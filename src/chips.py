"""Cut co-registered PAN/MS/visual chips from Maxar ARD tiles of one acquisition."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

import io_utils

ROOT = Path(__file__).resolve().parents[1]
INV = pd.read_csv(ROOT / "data/processed/tile_inventory.csv")
RATIO = 4
BAND_NAMES = ["Coastal", "Blue", "Green", "Yellow", "Red", "RedEdge", "NIR1", "NIR2"]
RGB = (4, 2, 1)  # zero-based indices of Red, Green, Blue in the 8-band stack


def tiles_for(catalog_id, E, N, half):
    sub = INV[INV.catalog_id == catalog_id]
    keep = []
    for _, r in sub.iterrows():
        p = io_utils.local_path(r.pan)
        if not (p.exists() and io_utils.local_path(r.ms).exists()):
            continue
        with rasterio.open(p) as s:
            b = s.bounds
        if b.left < E + half and b.right > E - half and b.bottom < N + half and b.top > N - half:
            keep.append(r)
    return keep


def _read(paths, bounds, res):
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, tr = merge(srcs, bounds=bounds, res=res, nodata=0, method="first")
    finally:
        for s in srcs:
            s.close()
    return arr, tr


def make_chip(catalog_id, E, N, size_pan=1024):
    """Return dict with MS (8,h,w), PAN (4h,4w), visual (3,4h,4w) or None, geometry."""
    tiles = tiles_for(catalog_id, E, N, half=size_pan)
    if not tiles:
        return None
    with rasterio.open(io_utils.local_path(tiles[0].ms)) as s:
        ms_res = s.res[0]
        gx, gy = s.bounds.left, s.bounds.top
    pan_res = ms_res / RATIO
    half = size_pan * pan_res / 2
    # snap chip origin to the MS grid so that each MS pixel covers exactly 4x4 PAN pixels
    left = gx + np.floor((E - half - gx) / ms_res) * ms_res
    top = gy - np.floor((gy - (N + half)) / ms_res) * ms_res
    right, bottom = left + size_pan * pan_res, top - size_pan * pan_res
    bounds = (left, bottom, right, top)
    ms, _ = _read([io_utils.local_path(t.ms) for t in tiles], bounds, ms_res)
    pan, _ = _read([io_utils.local_path(t.pan) for t in tiles], bounds, pan_res)
    vis = None
    vpaths = [io_utils.local_path(t.visual) for t in tiles if isinstance(t.visual, str)]
    if vpaths and all(p.exists() for p in vpaths):
        vis, _ = _read(vpaths, bounds, pan_res)
    h = size_pan // RATIO
    ms, pan = ms[:, :h, :h], pan[0, :size_pan, :size_pan].astype(np.float32)
    # isolated PAN no-data pixels (tile seams) inside MS coverage: nearest valid neighbour
    msu = np.repeat(np.repeat(ms.min(0) > 0, RATIO, 0), RATIO, 1)
    gap = (pan <= 0) & msu
    n_filled = int(gap.sum())
    if n_filled:
        from scipy import ndimage
        _, (ri, ci) = ndimage.distance_transform_edt(pan <= 0, return_indices=True)
        pan[gap] = pan[ri[gap], ci[gap]]
    if vis is not None:
        vis = vis[:3, :size_pan, :size_pan]
    meta = tiles[0]
    return dict(MS=ms.astype(np.float32), PAN=pan.astype(np.float32), VIS=vis, n_pan_filled=n_filled,
                x0=left, y0=top, pan_res=pan_res, ms_res=ms_res,
                catalog_id=catalog_id, date=meta.date, platform=meta.platform,
                SENSOR="WV2" if meta.platform == "WV02" else "WV3", off_nadir=float(meta.off_nadir),
                sun_el=float(meta.sun_el), sun_az=float(meta.sun_az),
                view_az=float(meta.azimuth), gsd=float(meta.gsd))


def valid_fraction(chip, E, N, radius=60.0):
    """Fraction of PAN pixels within radius of (E,N) that are valid and unobscured."""
    pan, ms = chip["PAN"], chip["MS"]
    n = pan.shape[0]
    xs = chip["x0"] + (np.arange(n) + 0.5) * chip["pan_res"]
    ys = chip["y0"] - (np.arange(n) + 0.5) * chip["pan_res"]
    X, Y = np.meshgrid(xs, ys)
    disk = (X - E) ** 2 + (Y - N) ** 2 <= radius ** 2
    msu = np.repeat(np.repeat(ms, RATIO, 1), RATIO, 2)
    vis_bright = (msu[1] > 2500) & (msu[2] > 2500) & (msu[4] > 2500)  # cloud / snow
    good = (pan > 0) & (msu.min(0) > 0) & ~vis_bright
    return float(good[disk].mean())
