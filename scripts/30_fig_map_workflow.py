"""Figure 1 (multi-scale study area) and Figure 2 (workflow)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402

fs.setup()
plt = fs.plt
OUT = ROOT / "results/figures"
tr = Transformer.from_crs(4326, 32637, always_xy=True)
km = lambda a: np.asarray(a) / 1000.0  # noqa: E731
ACQ = {"10300500D9F8D200": "WV-2, 8 Feb", "10300100E291D100": "WV-2, 11 Feb", "10300100E3154100": "WV-2, 28 Feb",
       "1040010082B6D600": "WV-3, 28 Feb", "10300100E383D900": "WV-2, 11 Mar"}
ACOL = dict(zip(ACQ, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]))  # Okabe-Ito


def scalebar(ax, x, y, length, label, lw=2):
    ax.plot([x, x + length], [y, y], color="k", lw=lw, solid_capstyle="butt")
    ax.text(x + length / 2, y, label, ha="center", va="bottom", fontsize=7.5)


def north(ax, x, y, h):
    ax.annotate("N", xy=(x, y + h), xytext=(x, y), ha="center", fontsize=8.0,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k"))


# ------------------------------------------------------------------ data
countries = json.loads((ROOT / "data/raw/context/ne_10m_countries.geojson").read_text(encoding="utf-8"))["features"]
faults = json.loads((ROOT / "data/raw/context/gem_active_faults.geojson").read_text(encoding="utf-8"))["features"]
ev = json.loads((ROOT / "data/raw/context/usgs_epicenters.json").read_text(encoding="utf-8"))
REG = box(35.3, 35.9, 39.6, 38.7)
rup = json.loads((ROOT / "data/raw/usgs/duross2024/Rupture_lines_June2023.geojson").read_text(encoding="utf-8"))
rlines = []
for ft in rup["features"]:
    geo = ft["geometry"]
    for cc in (geo["coordinates"] if geo["type"] == "MultiLineString" else [geo["coordinates"]]):
        if len(cc) > 1:
            rlines.append(np.array([c[:2] for c in cc]))
tiles = pd.read_csv(ROOT / "data/processed/manifest_tiles.csv")
inv = pd.read_csv(ROOT / "data/processed/tile_inventory.csv")
tiles = tiles.astype({"quadkey": str}).merge(inv[["catalog_id", "quadkey", "wkt"]].astype({"quadkey": str}), on=["catalog_id", "quadkey"])
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
cross = mk[(mk.kind == "controlled") & mk.job.str.endswith("_k0")]
sel = pd.read_csv(ROOT / "data/processed/sites_selected.csv").merge(
    pd.read_csv(ROOT / "data/processed/field_marker_screening.csv"), on="ID")

fig = plt.figure(figsize=(fs.COL2, 0.50 * fs.COL2))
axA = fig.add_axes([0.085, 0.36, 0.25, 0.60])
axB = fig.add_axes([0.445, 0.36, 0.33, 0.60])
axC = fig.add_axes([0.83, 0.36, 0.16, 0.60])

# (a) regional context, geographic coordinates
for c in countries:
    g = shape(c["geometry"])
    if not g.intersects(REG):
        continue
    for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
        x, y = p.exterior.xy
        axA.fill(x, y, fc="#f2f1ee" if c["properties"].get("ADMIN") == "Turkey" else "#e4e3df", ec="#9a9994",
                 lw=0.4, zorder=1)
for f in faults:
    g = shape(f["geometry"])
    if g.is_empty or not g.intersects(REG):
        continue
    for ls in (g.geoms if g.geom_type.startswith("Multi") else [g]):
        x, y = ls.xy
        axA.plot(x, y, color="#6d6c68", lw=0.5, zorder=2)
for eid, e in ev.items():
    axA.plot(e["lon"], e["lat"], marker="*", ms=9, mfc="#D55E00", mec="k", mew=0.4, zorder=5)
    axA.text(e["lon"] - 0.12 if e["mag"] > 7.7 else e["lon"] + 0.12, e["lat"] - 0.05 if e["mag"] > 7.7 else e["lat"] - 0.18, f"M {e['mag']:.1f}", fontsize=7.5, zorder=6, ha="right" if e["mag"] > 7.7 else "left")
bb = unary_union([wkt.loads(w) for w in tiles.wkt]).bounds
axA.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="k", lw=1.0, zorder=6))
for name, (x, y) in {"Kahramanmaraş": (36.93, 37.58), "Gaziantep": (37.38, 37.07), "Malatya": (38.31, 38.35),
                     "Adıyaman": (38.28, 37.76), "Antakya": (36.16, 36.20)}.items():  # noqa
    axA.plot(x, y, "s", ms=2.5, color="k", zorder=6)
    left = name in ("Kahramanmaraş", "Antakya")
    axA.text(x - 0.06 if left else x + 0.06, y + 0.04, name, fontsize=7.5, zorder=6, ha="right" if left else "left")
axA.text(36.25, 36.85, "EAF", fontsize=7.5, style="italic", color="#3d3c39", rotation=40)
axA.set_xlim(35.3, 39.6)
axA.set_ylim(35.9, 38.7)
axA.set_aspect(1 / np.cos(np.deg2rad(37.3)))
axA.set_xlabel("Longitude (°E)")
axA.set_ylabel("Latitude (°N)")
scalebar(axA, 38.2, 36.05, 100 / (111.32 * np.cos(np.deg2rad(37.3))), "100 km")
north(axA, 35.65, 38.1, 0.35)
fs.panel_label(axA, "(a)")

# (b) data coverage, UTM
for cid, g in tiles.groupby("catalog_id"):
    u = unary_union([transform(tr.transform, wkt.loads(s)) for s in g.wkt])
    for p in (u.geoms if u.geom_type == "MultiPolygon" else [u]):
        x, y = p.exterior.xy
        axB.fill(km(x), km(y), fc=ACOL[cid], alpha=0.12, ec=ACOL[cid], lw=0.8, zorder=1)
for ln in rlines:
    x, y = tr.transform(ln[:, 0], ln[:, 1])
    axB.plot(km(x), km(y), color="#D55E00", lw=1.2, zorder=3)
for cid, g in cross.groupby("catalog_id"):
    axB.scatter(km(g.E), km(g.N), s=10, marker="^", fc=ACOL[cid], ec="k", lw=0.3, zorder=4)
inc, exc = sel[sel.include == 1], sel[sel.include == 0]
axB.scatter(km(inc.E), km(inc.N), s=14, marker="o", fc="k", ec="white", lw=0.4, zorder=5)
axB.scatter(km(exc.E), km(exc.N), s=14, marker="x", c="#52514e", lw=0.8, zorder=5)
zx, zy = (326.6, 329.2), (4149.6, 4151.2)
axB.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="k", lw=0.8, zorder=6))
axB.set_xlim(320, 358)
axB.set_ylim(4142, 4170)
axB.set_aspect("equal")
axB.set_xlabel("Easting (km, UTM 37N)")
axB.set_ylabel("Northing (km)")
scalebar(axB, 322, 4143.5, 5, "5 km")
north(axB, 355.5, 4164.5, 3)
fs.panel_label(axB, "(b)")

# (c) local detail with marker labels
for ln in rlines:
    x, y = tr.transform(ln[:, 0], ln[:, 1])
    axC.plot(km(x), km(y), color="#D55E00", lw=1.2, zorder=3)
for i, (m, g) in enumerate(inc.groupby("marker")):
    r = g.iloc[0]
    if zx[0] * 1000 < r.E < zx[1] * 1000 and zy[0] * 1000 < r.N < zy[1] * 1000:
        axC.plot(km(r.E), km(r.N), "o", ms=4, mfc="k", mec="white", mew=0.4, zorder=5)
        off = {"M01": (-24, 6), "M03": (6, -14), "M05": (-6, 12), "M06": (2, -18), "M07": (-30, 6), "M61": (-8, -16)}
        axC.annotate(m, (km(r.E), km(r.N)), xytext=off.get(m, (6, 6)), textcoords="offset points",
                     fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.INK2))
for _, r in exc.iterrows():
    if zx[0] * 1000 < r.E < zx[1] * 1000 and zy[0] * 1000 < r.N < zy[1] * 1000:
        axC.plot(km(r.E), km(r.N), "x", ms=4, color="#52514e", zorder=5)
axC.set_xlim(*zx)
axC.set_ylim(*zy)
axC.set_aspect("equal")
axC.set_xlabel("Easting (km)")
scalebar(axC, zx[0] + 0.35, zy[1] - 0.22, 0.5, "500 m")
fs.panel_label(axC, "(c)")
h = [Line2D([], [], marker="*", ls="", mfc="#D55E00", mec="k", ms=8), Line2D([], [], color="#6d6c68", lw=0.6),
     Line2D([], [], color="#D55E00", lw=1.4)] + \
    [Line2D([], [], marker="s", ls="", mfc=ACOL[k], alpha=0.5, mec=ACOL[k], ms=7) for k in ACQ] + \
    [Line2D([], [], marker="^", ls="", mfc="#bbbbbb", mec="k", ms=5),
     Line2D([], [], marker="o", ls="", mfc="k", mec="white", ms=5), Line2D([], [], marker="x", ls="", color="#52514e", ms=5)]
l = ["2023 mainshock (USGS)", "Active fault (GEM)", "Field-mapped 2023 rupture"] + \
    [f"{v} tiles" for v in ACQ.values()] + ["Controlled crossing", "Field marker, used", "Field marker, excluded"]
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0, fontsize=7.5)
fig.savefig(OUT / "fig01_study_area.pdf")
fig.savefig(OUT / "fig01_study_area.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2 design and estimands
fig, ax = plt.subplots(figsize=(fs.COL2, 0.62 * fs.COL2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 72)
ax.axis("off")
C = dict(data="#f0f0f0", arm="#ffffff", prod="#f0f0f0", est="#dbe7f3", stat="#ffffff")  # grey boxes, one accent


def bx(x, y, w, h, text, col, bold=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.4", fc=col, ec="#3d3c39",
                                lw=0.7))
    if bold:
        ax.text(x + w / 2, y + h - 1.0, bold, ha="center", va="top", fontsize=8.0, fontweight="bold")
        ax.text(x + w / 2, y + (h - 4.0) / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.2)


def ar(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.7, color="#3d3c39"))


bx(2, 63, 30, 7, "Maxar ARD: PAN, 8-band MS,\nvendor RGB, STAC metadata", C["data"])
bx(36, 63, 28, 7, "OpenStreetMap roads\n(candidate crossings only)", C["data"])
bx(68, 63, 30, 7, "USGS field offsets,\nrupture traces", C["data"])
bx(2, 38, 30, 20, "Integer shift, k = 0-3 MS px\nFixed PAN-MS phase\nReference: exact offset\nQuantity: translation\nrecovery, acceptance",
   C["arm"], bold="Full-resolution test")
bx(35, 38, 30, 20, "Wald degradation\nMS phase, grid position\nUniform, one-sided shifts\nReference: original MS\nQuantity: deviation",
   C["arm"], bold="Reduced-resolution test")
bx(68, 38, 30, 20, "16 observations, 12 markers\nAperture, span, position\nReference: field offset\nQuantity: agreement",
   C["arm"], bold="Field case study")
bx(14, 24, 72, 9, "Image products: PAN, EXP, GSA, BDSD-PC, MTF-GLP-HPM,\nAWLP, Z-PNN, vendor RGB and λ-PNN (full resolution only)", C["prod"])
bx(14, 12.5, 72, 7.5, "Fixed automated estimator on the RGB mean\nchecks: cross-correlation estimator, four intensities", C["est"])
bx(2, 1, 30, 7, "Acceptance rate\nand failure reasons", C["stat"])
bx(35, 1, 30, 7, "Conditional error, repeatability,\nlarge deviations", C["stat"])
bx(68, 1, 30, 7, "Quality-task associations,\nacceptance and error", C["stat"])
ar(17, 63, 17, 58.6)
ar(17, 63, 48, 58.6)
ar(50, 63, 22, 58.6)
ar(83, 63, 83, 58.6)
for x in (17, 50, 83):
    ar(x, 38, 50 + (x - 50) * 0.45, 33.6)
ar(50, 24, 50, 20.6)
ar(40, 12.5, 17, 8.6)
ar(50, 12.5, 50, 8.6)
ar(60, 12.5, 83, 8.6)
fig.savefig(OUT / "fig02_workflow.pdf")
fig.savefig(OUT / "fig02_workflow.png", dpi=300)
print("ok")
