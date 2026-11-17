"""Figure 3: controlled displacement and measurement geometry."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.patches import Rectangle
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from offset import PARAMS, Raster, _Side, unit  # noqa: E402

fs.setup()
plt = fs.plt
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ms = pd.read_csv(ROOT / "results/tables/measurements.csv")
chip, marker = "C09_10300100E3154100", "R00"


def stretch(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def lines(ax, m, row, col):
    p0, s, n = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
    ax.plot(*np.c_[p0 - 60 * s, p0 + 60 * s], color="#D55E00", lw=0.9, ls="--")
    for side, sg in (("p", 1), ("m", -1)):
        d = unit(row[f"{side}_az"])
        if np.dot(d, n) * sg < 0:
            d = -d
        a = p0 + row[f"{side}_u0"] * s
        ax.plot([a[0], a[0] + 50 * d[0]], [a[1], a[1] + 50 * d[1]], color=col, lw=1.1)
        ax.plot(*a, "o", ms=2.5, color=col)


fig = plt.figure(figsize=(fs.COL2, 0.62 * fs.COL2))
gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 0.65], wspace=0.6, hspace=0.42)
axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 0:3]),
       fig.add_subplot(gs[1, 3:6]), fig.add_subplot(gs[0, 4:6])]
for ax, k, lab in zip(axs[:2], (0, 2), ("(a)", "(b)")):
    j = f"{chip}_k{k}"
    jm = jobs.loc[j]
    P = sio.loadmat(ROOT / f"data/processed/chips/{j}.mat", variable_names=["PAN"])["PAN"]
    n = P.shape[0]
    ax.imshow(stretch(P), cmap="gray", extent=[jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0])
    m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
    row = ms[(ms.job == j) & (ms.marker == marker) & (ms["product"] == "PAN")].iloc[0]
    lines(ax, m, row, "#0072B2")
    ax.set_xlim(m.E - 50, m.E + 50)
    ax.set_ylim(m.N - 50, m.N + 50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([m.E + 22, m.E + 42], [m.N - 45, m.N - 45], color="white", lw=2)
    ax.text(m.E + 32, m.N - 43, "20 m", color="white", ha="center", fontsize=7.5)
    ax.set_title(f"PAN, k = {k}" + (f", H = {k * jm.ms_res:.2f} m" if k else ", H = 0"), fontsize=8.0)
    fs.panel_label(ax, lab)
    if k == 2:
        p0, s, n = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
        for side, sg, col in (("p", 1, "#E69F00"), ("m", -1, "#CC79A7")):
            d = unit(row[f"{side}_az"])
            if np.dot(d, n) * sg < 0:
                d = -d
            dv = abs(np.dot(d, n))
            a = p0 + row[f"{side}_u0"] * s
            # support from t_min = max(A/2, 5 m) to t_min + L along the marker (A = 0 here)
            q0, q1 = a + (5.0 / dv) * d, a + (45.0 / dv) * d
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=col, lw=4, alpha=0.55, solid_capstyle="butt")
        q = p0 + row["p_u0"] * s + (25.0 / abs(np.dot(unit(row["p_az"]), n))) * unit(row["p_az"]) * np.sign(np.dot(unit(row["p_az"]), n))
        ax.annotate("L = 40 m", q, xytext=(-30, 22), textcoords="offset points", fontsize=7.5, color="white",
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
        ax.annotate("t$_{min}$", p0 + row["p_u0"] * s, xytext=(10, -26), textcoords="offset points", fontsize=7.5,
                    color="white", arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
# (c) whole-pixel shift on both grids; (d) reduced scale: MS phase
for ax, title, lab, frac in ((axs[2], "Full scale: k MS pixels = 4k PAN pixels", "(c)", False),
                             (axs[3], "Reduced scale: k PAN pixels = k/4 MS pixels", "(d)", True)):
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.7, 4.2)
    ax.axis("off")
    for x in np.arange(0, 8.01, 0.5 if not frac else 1):
        ax.plot([x, x], [2.3, 4.0], color="#9a9994", lw=0.4)
    for x in np.arange(0, 8.01, 2 if not frac else 4):
        ax.plot([x, x], [0.3, 2.0], color="#3d3c39", lw=0.8)
    ax.text(-0.1, 3.15, "PAN", fontsize=7.5, ha="right", va="center")
    ax.text(-0.1, 1.15, "MS", fontsize=7.5, ha="right", va="center")
    sh = 2.0 if not frac else 1.0
    for y0, y1 in ((2.3, 4.0), (0.3, 2.0)):
        ax.add_patch(Rectangle((1.0, y0), 1.0, y1 - y0, fc="#0072B2", alpha=0.35, ec="none"))
        ax.add_patch(Rectangle((1.0 + sh, y0), 1.0, y1 - y0, fc="#E69F00", alpha=0.45, ec="none"))
    ax.annotate("", xy=(1.0 + sh, 1.15), xytext=(1.0, 1.15), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(8.0, -0.35, "MS phase 0" if not frac else "MS phase 1/4", fontsize=7.5, va="center", ha="right")
    ax.set_title(title, fontsize=8.0)
    fs.panel_label(ax, lab, x=-0.02, y=1.13)
# (e) profile and derivative at one station, + side, k = 2
j = f"{chip}_k2"
jm = jobs.loc[j]
m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
L = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L"])["L"][:, :, 0]
side = _Side(Raster(L.astype(float), jm.x0, jm.y0, jm.pan_res), (m.E, m.N), m.strike, m.az_plus, +1, 0.0, PARAMS)
w = side.w
pt = side.p0 + side.base_u * side.s
d, e = side.d, side.e
st = 10
q = pt + ((side.t[st] / side.dv) * d)[None, :] + w[:, None] * e[None, :]
prof = side.img.sample(q)
g = ndimage.gaussian_filter1d(prof, PARAMS["sigma_d"] / PARAMS["dw"], order=1)
ax = axs[4]
ax.plot(w, (prof - prof.min()) / (prof.max() - prof.min()), color="#0072B2", lw=1, label="intensity (scaled)")
ax.plot(w, g / np.abs(g).max(), color="#E69F00", lw=1, label="Gaussian derivative")
ia, ib = np.argmax(g), np.argmin(g)
for i_ in (ia, ib):
    ax.axvline(w[i_], color=fs.INK2, lw=0.6, ls=":")
ax.axhline(0, color=fs.GRID, lw=0.5)
ax.set_xlim(-20, 20)
ax.set_xlabel("Position across the road (m)")

ax.set_ylim(-1.1, 1.1)
ax.set_yticks([-1, 0, 1])
ax.legend(loc="lower left", fontsize=7.5)
ax.set_title(f"Normalised profiles, station at {side.t[st]:.0f} m", fontsize=8.0)
fs.panel_label(ax, "(e)")
fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.04)
fig.savefig(ROOT / "results/figures/fig03_geometry.pdf")
fig.savefig(ROOT / "results/figures/fig03_geometry.png", dpi=300)
print("ok")
