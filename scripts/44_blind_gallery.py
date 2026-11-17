"""Blinded, randomly ordered gallery of the 94 baseline-corrected deviations above 0.5 m
for independent manual adjudication.

Each event gets a random code B001-B094 (fixed seed). For every code two PNG files are
written: one with the fitted marker lines and edge pairs, one without overlays. Both
show the baseline image (left) and the dislocated image (right) with the same extent and
the same grey stretch. Product, displacement level, measured offset and automated label
are not shown. The code-to-event key is written outside the gallery folder.

Outputs
  validation/blind_gallery/img/Bxxx_lines.png, Bxxx_raw.png
  validation/blind_gallery/events.js          (code list read by index.html)
  validation/gallery_blind_key.csv            (do not give to raters)
Images are derived from Maxar Open Data (CC BY-NC 4.0) and are not for redistribution.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import unit  # noqa: E402

SEED = 20260214
HALF = 38.0          # half-width of the displayed window (m)
PX = 760             # output size of each panel (px)
T = ROOT / "results/tables/v2"
VAL = ROOT / "validation"
GAL = VAL / "blind_gallery"
IMG = GAL / "img"
IMG.mkdir(parents=True, exist_ok=True)

c = pd.read_csv(T / "controlled_long_v2.csv")
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ev = pd.read_csv(T / "gallery_index.csv")
ev = ev[["event", "unit", "product", "k", "inc_ae", "cls"]].copy()

rng = np.random.default_rng(SEED)
ev["code"] = [f"B{i:03d}" for i in rng.permutation(len(ev)) + 1]
ev = ev.sort_values("code").reset_index(drop=True)
_cache = {}


def image(j, product):
    if (j, product) not in _cache:
        if product == "VIS":
            _cache[(j, product)] = np.load(ROOT / f"data/processed/chips/{j}_vis.npy")
        else:
            r = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True,
                            variable_names=["L", "NAMES"])
            names = [str(n) for n in r["NAMES"]]
            _cache[(j, product)] = r["L"][:, :, names.index(product)]
    return _cache[(j, product)]


def window(j, marker, product):
    """Return the image crop, its extent and the marker row for one job."""
    jm = jobs.loc[j]
    m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
    img = image(j, product)
    n = img.shape[0]
    cols = jm.x0 + (np.arange(n) + 0.5) * jm.pan_res
    rows = jm.y0 - (np.arange(n) + 0.5) * jm.pan_res
    ci = np.where(np.abs(cols - m.E) <= HALF + jm.pan_res)[0]
    ri = np.where(np.abs(rows - m.N) <= HALF + jm.pan_res)[0]
    crop = img[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
    ext = [cols[ci[0]] - jm.pan_res / 2, cols[ci[-1]] + jm.pan_res / 2,
           rows[ri[-1]] - jm.pan_res / 2, rows[ri[0]] + jm.pan_res / 2]
    row = c[(c.job == j) & (c.marker == marker) & (c["product"] == product)].iloc[0]
    return crop, ext, m, row


def overlay(ax, m, row):
    p0, s_, nn = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
    ax.plot(*np.c_[p0 - 45 * s_, p0 + 45 * s_], color="#FF8C00", lw=1.0, ls=(0, (6, 4)))
    for side, sg, col in (("p", 1, "#00E5FF"), ("m", -1, "#FF4FD8")):
        if not np.isfinite(row[f"{side}_u0"]):
            continue
        d = unit(row[f"{side}_az"])
        if np.dot(d, nn) * sg < 0:
            d = -d
        a = p0 + row[f"{side}_u0"] * s_
        ax.plot([a[0], a[0] + 40 * d[0]], [a[1], a[1] + 40 * d[1]], color=col, lw=0.8, ls=(0, (2, 3)))
        w = row[f"{side}_width"]
        if np.isfinite(w):
            q = np.array([-d[1], d[0]]) * w / 2
            for sgn in (1, -1):
                b = a + sgn * q
                ax.plot([b[0], b[0] + 40 * d[0]], [b[1], b[1] + 40 * d[1]], color=col, lw=1.3)


def render(e, lines):
    chip, marker = e.unit.split("/")
    crops = [window(f"{chip}_k{kk}", marker, e["product"]) for kk in (0, int(e.k))]
    lo, hi = np.nanpercentile(crops[0][0], [1, 99])
    dpi = 100
    fig, axs = plt.subplots(1, 2, figsize=(2 * PX / dpi, PX / dpi + 0.45), dpi=dpi)
    for ax, (crop, ext, m, row), title in zip(axs, crops, ("referans (kaydırma yok)", "kaydırılmış")):
        ax.imshow(np.clip((crop - lo) / (hi - lo), 0, 1), cmap="gray", extent=ext,
                  interpolation="nearest", vmin=0, vmax=1)
        if lines:
            overlay(ax, m, row)
        ax.set_xlim(m.E - HALF, m.E + HALF)
        ax.set_ylim(m.N - HALF, m.N + HALF)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{e.code}  {title}", fontsize=13)
        x0, y0 = m.E - HALF + 3, m.N - HALF + 3
        ax.plot([x0, x0 + 10], [y0, y0], color="white", lw=3, solid_capstyle="butt")
        ax.text(x0 + 5, y0 + 1.5, "10 m", color="white", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.94, wspace=0.01)
    fig.savefig(IMG / f"{e.code}_{'lines' if lines else 'raw'}.png", dpi=dpi)
    plt.close(fig)


for _, e in ev.iterrows():
    render(e, True)
    render(e, False)

ev[["code", "event", "unit", "product", "k", "inc_ae", "cls"]].to_csv(VAL / "gallery_blind_key.csv", index=False)
(GAL / "events.js").write_text("const EVENTS = [" + ",".join(f'"{x}"' for x in ev.code) + "];\n",
                               encoding="utf-8")
print(len(ev), "events written to", GAL)
