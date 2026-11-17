"""Figure 1: study area, rupture, field markers, acquisitions and controlled chips."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pyproj import Transformer
from shapely import wkt
from shapely.ops import unary_union, transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402

fs.setup()
plt = fs.plt
tr = Transformer.from_crs(4326, 32637, always_xy=True)
km = lambda a: np.asarray(a) / 1000.0  # noqa: E731

rup = json.loads((ROOT / "data/raw/usgs/duross2024/Rupture_lines_June2023.geojson").read_text(encoding="utf-8"))
lines = []
for ft in rup["features"]:
    geo = ft["geometry"]
    for cc in (geo["coordinates"] if geo["type"] == "MultiLineString" else [geo["coordinates"]]):
        if len(cc) > 1:
            x, y = tr.transform([c[0] for c in cc], [c[1] for c in cc])
            lines.append((x, y))
osm = json.loads((ROOT / "data/raw/osm/roads_aoi.json").read_text(encoding="utf-8"))["elements"]
roads = []
for w in osm:
    if "geometry" in w and w["tags"].get("highway") in ("primary", "secondary", "tertiary", "trunk"):
        x, y = tr.transform([p["lon"] for p in w["geometry"]], [p["lat"] for p in w["geometry"]])
        roads.append((x, y))
tiles = pd.read_csv(ROOT / "data/processed/tiles_needed.csv")
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
sites = pd.read_csv(ROOT / "data/processed/sites_selected.csv")
scr = pd.read_csv(ROOT / "data/processed/field_marker_screening.csv")
sites = sites.merge(scr, on="ID")

ACQ = {"10300500D9F8D200": "WV-2, 8 Feb", "10300100E291D100": "WV-2, 11 Feb", 
       "10300100E3154100": "WV-2, 28 Feb", "1040010082B6D600": "WV-3, 28 Feb", "10300100E383D900": "WV-2, 11 Mar"}
ACOL = dict(zip(ACQ, ["#2a78d6", "#eb6834", "#eda100", "#6250d6", "#1baf7a"]))

fig, axes = plt.subplots(1, 2, figsize=(fs.COL2, 0.47 * fs.COL2), gridspec_kw=dict(width_ratios=[1.15, 1]))
for ax, (xl, yl) in zip(axes, [((318, 358), (4141, 4170)), ((325.6, 332.6), (4148.6, 4153.4))]):
    for x, y in roads:
        ax.plot(km(x), km(y), color="#bdbcb7", lw=0.5, zorder=1)
    for cid, g in tiles[tiles.catalog_id != "10300100E2BF8000"].groupby("catalog_id"):
        u = unary_union([transform(tr.transform, wkt.loads(s)) for s in g.wkt])
        for p in (u.geoms if u.geom_type == "MultiPolygon" else [u]):
            xs, ys = p.exterior.xy
            ax.plot(km(xs), km(ys), color=ACOL[cid], lw=0.9, zorder=2,
                    label=ACQ[cid] if ax is axes[0] and p is (u.geoms[0] if u.geom_type == "MultiPolygon" else u)
                    else None)
    for x, y in lines:
        ax.plot(km(x), km(y), color="#d03b3b", lw=1.0, zorder=3)
    cj = jobs[(jobs.kind == "controlled") & (jobs.k == 0)]
    for r in cj.itertuples():
        size = 1024 * r.pan_res
        ax.add_patch(Rectangle((km(r.x0), km(r.y0 - size)), size / 1000, size / 1000, fill=False,
                               ec=ACOL[r.catalog_id], lw=0.6, ls="--", zorder=4))
    inc, exc = sites[sites.include == 1], sites[sites.include == 0]
    ax.scatter(km(inc.E), km(inc.N), s=14, marker="o", fc="#0b0b0b", ec="white", lw=0.4, zorder=5)
    ax.scatter(km(exc.E), km(exc.N), s=14, marker="x", c="#52514e", lw=0.8, zorder=5)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 37N)")
    ax.set_ylabel("Northing (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
# labels of markers in the zoom panel (one label per marker)
z = axes[1]
for i, (m, g) in enumerate(sites[sites.include == 1].groupby("marker")):
    r = g.iloc[0]
    if 325600 < r.E < 332600 and 4148600 < r.N < 4153400:
        z.annotate(m, (km(r.E), km(r.N)), xytext=(-22, 12) if i % 2 == 0 else (10, -16), textcoords="offset points", fontsize=6, color=fs.INK, arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.INK2, shrinkA=0, shrinkB=2))
zx, zy = (325.6, 332.6), (4148.6, 4153.4)
axes[0].add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=fs.INK, lw=0.8))
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color=ACOL[k], lw=0.9) for k in ACQ]
l = list(ACQ.values())
h += [Line2D([], [], color="#d03b3b", lw=1.0), Line2D([], [], color="#52514e", lw=0.6, ls="--"),
      Line2D([], [], marker="o", ls="", mfc="#0b0b0b", mec="white", ms=4),
      Line2D([], [], marker="x", ls="", color="#52514e", ms=4), Line2D([], [], color="#bdbcb7", lw=0.8)]
l += ["Field-mapped rupture", "Controlled-test chip", "Field marker (used)", "Field marker (excluded)",
      "Main road (OSM)"]
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.2)
fs.panel_label(axes[0], "(a)")
fs.panel_label(axes[1], "(b)")
fig.subplots_adjust(bottom=0.2, wspace=0.28, left=0.07, right=0.99, top=0.95)
out = ROOT / "results/figures"
fig.savefig(out / "fig01_study_area.pdf")
fig.savefig(out / "fig01_study_area.png", dpi=300)
print("ok")
