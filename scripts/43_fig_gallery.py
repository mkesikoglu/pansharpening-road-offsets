"""Supplementary gallery of all baseline-corrected deviations above 0.5 m, for visual
adjudication of the automated edge-pair-change proxy. Each event shows the product at
k = 0 and at the dislocated level with the fitted marker lines.
Output results/figures/figS_gallery_pXX.pdf and results/tables/v2/gallery_index.csv
(an empty 'adjudication' column is provided for two independent raters)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from offset import unit  # noqa: E402

fs.setup()
plt = fs.plt
T = ROOT / "results/tables/v2"
OUT = ROOT / "results/figures"
c = pd.read_csv(T / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ev = pd.read_csv(T / "deviation_events.csv")
ev = ev[ev.cls != "none"].sort_values(["product", "unit", "k"]).reset_index(drop=True)
ev["event"] = np.arange(1, len(ev) + 1)
ev["adjudication_rater1"] = ""
ev["adjudication_rater2"] = ""
ev.to_csv(T / "gallery_index.csv", index=False)
_cache = {}


def image(j, product):
    if (j, product) not in _cache:
        if product == "VIS":
            _cache[(j, product)] = np.load(ROOT / f"data/processed/chips/{j}_vis.npy")
        else:
            r = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
            names = [str(n) for n in r["NAMES"]]
            _cache[(j, product)] = r["L"][:, :, names.index(product)]
    return _cache[(j, product)]


def draw(ax, j, marker, product):
    jm = jobs.loc[j]
    m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
    img = image(j, product)
    lo, hi = np.nanpercentile(img, [1, 99])
    n = img.shape[0]
    ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray",
              extent=[jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0])
    row = c[(c.job == j) & (c.marker == marker) & (c["product"] == product)].iloc[0]
    p0, s_, nn = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
    ax.plot(*np.c_[p0 - 45 * s_, p0 + 45 * s_], color="#D55E00", lw=0.6, ls="--")
    for side, sg in (("p", 1), ("m", -1)):
        if not np.isfinite(row[f"{side}_u0"]):
            continue
        d = unit(row[f"{side}_az"])
        if np.dot(d, nn) * sg < 0:
            d = -d
        a = p0 + row[f"{side}_u0"] * s_
        ax.plot([a[0], a[0] + 40 * d[0]], [a[1], a[1] + 40 * d[1]], color="#56B4E9", lw=0.9)
    ax.set_xlim(m.E - 38, m.E + 38)
    ax.set_ylim(m.N - 38, m.N + 38)
    ax.set_xticks([])
    ax.set_yticks([])
    return row


per = 12
pages = int(np.ceil(len(ev) / per))
for pg in range(pages):
    sub = ev.iloc[pg * per:(pg + 1) * per]
    fig, axs = plt.subplots(4, 6, figsize=(fs.COL2, 1.05 * fs.COL2))
    for a in axs.ravel():
        a.axis("off")
    for i, e in enumerate(sub.itertuples()):
        r_, c_ = divmod(i, 3)
        chip, marker = e.unit.split("/")
        for q, kk in enumerate((0, int(e.k))):
            ax = axs[r_, 2 * c_ + q]
            ax.axis("on")
            row = draw(ax, f"{chip}_k{kk}", marker, e.product)
            ax.set_title(f"#{e.event} {fs.LABEL[e.product]}, k={kk}" if q == 0 else f"k={kk}, H={row.H:.2f} m",
                         fontsize=6.8)
        axs[r_, 2 * c_].text(0.02, 0.03, "flagged" if e.cls == "different edge pair" else "not flagged", color="white",
                             fontsize=6.5, transform=axs[r_, 2 * c_].transAxes,
                             bbox=dict(fc="black", alpha=0.6, pad=0.8, ec="none"))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.05, hspace=0.25)
    fig.savefig(OUT / f"figS_gallery_p{pg + 1:02d}.pdf")
    plt.close(fig)
print(len(ev), pages)
