"""Build all processing jobs (chips) for the field-referenced and the controlled
dislocation experiments. Selection uses metadata, OSM geometry and data
validity only; no fused product is inspected.

Outputs
  data/processed/chips/<job>.mat       MS [h w 8], PAN, ROADMASK (MS grid), SENSOR, EVAL
  data/processed/chips/<job>_vis.npy   luminance of the vendor visual product (PAN grid)
  data/processed/jobs.csv              one row per job
  data/processed/markers.csv           one row per marker to be measured
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chips  # noqa: E402
import dislocation  # noqa: E402

import collections
import os
DRY = os.environ.get("DRY_RUN") == "1"  # count the sampling flow without writing
COUNT = collections.Counter()
SEED = int(os.environ.get("SAMPLING_SEED", "20230206"))  # archived seed of the analysed set
RNG = np.random.default_rng(SEED)
OUT = ROOT / "data/processed/chips"
OUT.mkdir(parents=True, exist_ok=True)
NPAN = 1024
MARGIN = 4                 # MS pixels read beyond the chip on each side
K_SHIFTS = (0, 1, 2, 3)    # synthetic offsets in MS pixels
MAX_CHIPS = 20             # controlled chips per acquisition
ROAD_CLASSES = ("primary", "secondary", "tertiary", "unclassified")
NDVI_MAX = 0.35            # open terrain around a crossing (native MS, 25 m radius)
tr = Transformer.from_crs(4326, 32637, always_xy=True)

# ---------------------------------------------------------------- OSM roads
osm = json.loads((ROOT / "data/raw/osm/roads_aoi.json").read_text(encoding="utf-8"))["elements"]
roads = []
for w in osm:
    if "geometry" not in w or len(w["geometry"]) < 2:
        continue
    x, y = tr.transform([p["lon"] for p in w["geometry"]], [p["lat"] for p in w["geometry"]])
    roads.append(dict(id=w["id"], hw=w["tags"].get("highway"), geom=LineString(zip(x, y))))
road_union = unary_union([r["geom"] for r in roads])

# ---------------------------------------------------------------- exclusion zone (real rupture, field sites)
rup = json.loads((ROOT / "data/raw/usgs/duross2024/Rupture_lines_June2023.geojson").read_text(encoding="utf-8"))
rup_lines = []
for f in rup["features"]:
    geo = f["geometry"]
    parts = geo["coordinates"] if geo["type"] == "MultiLineString" else [geo["coordinates"]]
    for part in parts:
        cc = [c[:2] for c in part]
        if len(cc) < 2:
            continue
        x, y = tr.transform([c[0] for c in cc], [c[1] for c in cc])
        rup_lines.append(LineString(list(zip(x, y))))
sites = pd.read_csv(ROOT / "data/processed/sites_selected.csv")
excl = unary_union(rup_lines + [Point(e, n) for e, n in zip(sites.E, sites.N)]).buffer(400)


def to_mat(job, ch, roadmask, evalflag):
    if DRY:
        return
    if (OUT / f"{job}.mat").exists():  # keep existing chips (and their Z-PNN outputs)
        return
    sio.savemat(OUT / f"{job}.mat", dict(
        MS=np.moveaxis(ch["MS"], 0, -1).astype(np.float32), PAN=ch["PAN"].astype(np.float32),
        ROADMASK=roadmask.astype(np.uint8), SENSOR=ch["SENSOR"], EVAL=np.uint8(evalflag)),
        do_compression=True)
    if ch.get("VIS") is not None:
        np.save(OUT / f"{job}_vis.npy", ch["VIS"].astype(np.float32).mean(0))


def road_mask(ch):
    n = ch["MS"].shape[1]
    t = from_origin(ch["x0"], ch["y0"], ch["ms_res"], ch["ms_res"])
    shapes = [(r["geom"].buffer(5.0), 1) for r in roads]
    return rasterize(shapes, out_shape=(n, n), transform=t, fill=0, dtype="uint8")


def side_azimuth(line, p, sign_dir, length=50.0):
    """Azimuth of the road on one side of the crossing p, from the sub-line within
    `length` m; also returns max residual from a straight line."""
    s = line.project(p)
    a, b = (s, min(s + length, line.length)) if sign_dir > 0 else (max(s - length, 0), s)
    if b - a < 0.8 * length:
        return np.nan, np.inf
    pts = np.array([line.interpolate(t).coords[0] for t in np.linspace(a, b, 26)])
    c = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - c)
    d = vt[0]
    resid = np.abs((pts - c) @ np.array([-d[1], d[0]])).max()
    return float(np.rad2deg(np.arctan2(d[0], d[1])) % 180), float(resid)


def crossings(ch, orientation):
    phi, pc = dislocation.fault_geometry(ch["x0"], ch["y0"], ch["pan_res"], NPAN, orientation)
    half = NPAN * ch["pan_res"] / 2
    L = half - 60.0
    s = np.array([np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))])
    fl = LineString([pc - L * s, pc + L * s])
    out = []
    ndvi = (ch["MS"][6] - ch["MS"][4]) / (ch["MS"][6] + ch["MS"][4] + 1e-6)
    nms = ndvi.shape[0]
    gx = ch["x0"] + (np.arange(nms) + 0.5) * ch["ms_res"]
    gy = ch["y0"] - (np.arange(nms) + 0.5) * ch["ms_res"]
    GX, GY = np.meshgrid(gx, gy)
    for r in roads:
        if r["hw"] not in ROAD_CLASSES:
            continue
        inter = r["geom"].intersection(fl)
        if inter.is_empty:
            continue
        COUNT["x_candidate"] += 1
        for p in (inter.geoms if inter.geom_type.startswith("Multi") else [inter]):
            if p.geom_type != "Point":
                continue
            azA, rA = side_azimuth(r["geom"], p, +1)
            azB, rB = side_azimuth(r["geom"], p, -1)
            if not (np.isfinite(azA) and np.isfinite(azB)) or max(rA, rB) > 1.0:
                COUNT["x_not_straight"] += 1
                continue
            ang = abs(((azA - phi) + 90) % 180 - 90)
            if ang < 30:
                COUNT["x_low_angle"] += 1
                continue
            # no other road within 25 m of the crossing (junctions, parallel roads)
            others = [o for o in roads if o["id"] != r["id"] and o["geom"].distance(p) < 25]
            if others:
                COUNT["x_other_road"] += 1
                continue
            near = (GX - p.x) ** 2 + (GY - p.y) ** 2 <= 25.0 ** 2
            if ndvi[near].mean() > NDVI_MAX:
                COUNT["x_vegetated"] += 1
                continue
            COUNT["x_pass"] += 1
            # which half-line lies on the + side (right of strike)?
            n_vec = np.array([s[1], -s[0]])
            q = np.array(r["geom"].interpolate(r["geom"].project(p) + 20).coords[0])
            plus_is_A = np.dot(q - np.array(p.coords[0]), n_vec) > 0
            out.append(dict(osm_id=r["id"], highway=r["hw"], E=p.x, N=p.y, strike=phi,
                            az_plus=azA if plus_is_A else azB, az_minus=azB if plus_is_A else azA,
                            angle=ang))
    out = sorted(out, key=lambda c: (c["N"], c["E"]))
    keep = []
    for c in out:
        if all(np.hypot(c["E"] - k["E"], c["N"] - k["N"]) > 60 for k in keep):
            keep.append(c)
    return keep


jobs, markers = [], []
inv = chips.INV.drop_duplicates("catalog_id").set_index("catalog_id")

# ================================================================= field-referenced jobs
cov = pd.read_csv(ROOT / "data/processed/site_coverage.csv")
scr = pd.read_csv(ROOT / "data/processed/field_marker_screening.csv")
fsites = sites.merge(scr, on="ID")
fsites = fsites[fsites.include == 1]
for cl, g in fsites.groupby("cluster"):
    okc = cov[cov.ID.isin(g.ID) & (cov.valid >= 0.95)].groupby("catalog_id").ID.nunique()
    for cid in okc[okc == g.ID.nunique()].index:
        E, N = g.E.mean(), g.N.mean()
        ch = chips.make_chip(cid, E, N, size_pan=NPAN)
        job = f"F{cl:02d}_{cid}"
        to_mat(job, ch, road_mask(ch), 1)
        jobs.append(dict(job=job, kind="field", cluster=cl, catalog_id=cid, k=0, orientation="",
                         x0=ch["x0"], y0=ch["y0"], pan_res=ch["pan_res"], ms_res=ch["ms_res"],
                         sensor=ch["SENSOR"], date=ch["date"], off_nadir=ch["off_nadir"],
                         view_az=ch["view_az"], sun_el=ch["sun_el"], gsd=ch["gsd"]))
        for _, s in g.iterrows():
            markers.append(dict(job=job, kind="field", marker=s.marker, ID=int(s.ID), mode=s["mode"], az_N=s.az_init_N, az_S=s.az_init_S,
                                E=s.E, N=s.N, strike=s.FaultStrike_deg, az_plus=np.nan, az_minus=np.nan,
                                aperture=s.Aperture_num if np.isfinite(s.Aperture_num) else 0.0,
                                H_true=np.nan, H_ref=s.H_pref_m, H_min=s.H_min_m, H_max=s.H_max_m,
                                tier=s.tier, marker_class=s.marker_class, catalog_id=cid))
print("field jobs:", len(jobs))

# ================================================================= controlled jobs
import io_utils  # noqa: E402
tiles_needed = chips.INV[[io_utils.local_path(a).exists() and io_utils.local_path(b).exists()
                          for a, b in zip(chips.INV.pan, chips.INV.ms)]]
for cid, tg in tiles_needed.groupby("catalog_id"):
    from shapely import wkt
    foot = unary_union([wkt.loads(w) for w in tg.wkt])
    xs, ys = tr.transform(*foot.exterior.xy) if foot.geom_type == "Polygon" else tr.transform(
        *max(foot.geoms, key=lambda g: g.area).exterior.xy)
    from shapely.geometry import Polygon
    fp = Polygon(zip(xs, ys)).buffer(-NPAN * 0.4)
    cand_roads = [r for r in roads if r["geom"].intersects(fp) and r["hw"] in ROAD_CLASSES]
    centres, n_try = [], 0
    if not cand_roads:
        print(cid, "no candidate roads")
        continue
    while len(centres) < MAX_CHIPS and n_try < 1500:
        n_try += 1
        r = cand_roads[RNG.integers(len(cand_roads))]
        p = r["geom"].interpolate(RNG.uniform(0, r["geom"].length))
        COUNT["c_draw"] += 1
        if not fp.contains(p) or excl.contains(p):
            COUNT["c_outside_or_excluded"] += 1
            continue
        if any(np.hypot(p.x - c[0], p.y - c[1]) < 700 for c in centres):
            COUNT["c_too_close"] += 1
            continue
        orientation = "NS" if RNG.uniform() < 0.5 else "EW"
        # read chip with margin; validity of the whole chip
        base = chips.make_chip(cid, p.x, p.y, size_pan=NPAN + 8 * MARGIN)
        if base is None:
            continue
        msb, panb, visb = base["MS"], base["PAN"], base["VIS"]
        ms0, pan0, vis0 = dislocation.apply(msb, panb, visb, 0, orientation, MARGIN)
        ch = dict(base, MS=ms0, PAN=pan0, VIS=vis0,
                  x0=base["x0"] + MARGIN * base["ms_res"], y0=base["y0"] - MARGIN * base["ms_res"])
        if chips.valid_fraction(ch, p.x, p.y, radius=NPAN * ch["pan_res"] / 2) < 0.98:
            COUNT["c_invalid_data"] += 1
            continue
        cr = [c for c in crossings(ch, orientation) if not excl.contains(Point(c["E"], c["N"]))]
        if len(cr) < 1:
            COUNT["c_no_crossing"] += 1
            continue
        COUNT["c_accepted"] += 1
        centres.append((p.x, p.y))
        rm = road_mask(ch)
        for k in K_SHIFTS:
            ms_k, pan_k, vis_k = dislocation.apply(msb, panb, visb, k, orientation, MARGIN)
            chk = dict(ch, MS=ms_k, PAN=pan_k, VIS=vis_k)
            job = f"C{len(centres):02d}_{cid}_k{k}"
            to_mat(job, chk, rm, 1 if k == 0 else 0)
            jobs.append(dict(job=job, kind="controlled", cluster=-1, catalog_id=cid, k=k,
                             orientation=orientation, x0=ch["x0"], y0=ch["y0"], pan_res=ch["pan_res"],
                             ms_res=ch["ms_res"], sensor=ch["SENSOR"], date=ch["date"],
                             off_nadir=ch["off_nadir"], view_az=ch["view_az"], sun_el=ch["sun_el"],
                             gsd=ch["gsd"], chip=f"C{len(centres):02d}_{cid}"))
            for j, c in enumerate(cr):
                markers.append(dict(job=job, kind="controlled", marker=f"R{j:02d}", ID=-1, mode="band", E=c["E"], N=c["N"],
                                    strike=c["strike"], az_plus=c["az_plus"], az_minus=c["az_minus"],
                                    aperture=0.0, H_true=k * ch["ms_res"], H_ref=np.nan, H_min=np.nan,
                                    H_max=np.nan, tier="controlled", marker_class=f"road:{c['highway']}",
                                    catalog_id=cid, angle=c["angle"], osm_id=c["osm_id"]))
    print(cid, inv.loc[cid, "date"], inv.loc[cid, "platform"], "chips:", len(centres), "tries:", n_try)

if DRY:
    name = "sampling_flow_counts.csv" if SEED == 20230206 else f"sampling_flow_counts_seed{SEED}.csv"
    pd.Series(COUNT).to_csv(ROOT / "results/tables/v2" / name)
    print(dict(COUNT))
else:
    pd.DataFrame(jobs).to_csv(ROOT / "data/processed/jobs.csv", index=False)
    pd.DataFrame(markers).to_csv(ROOT / "data/processed/markers.csv", index=False)
print("jobs", len(jobs), "markers", len(markers))
