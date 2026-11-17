"""Figure 2: controlled dislocation and measurement geometry.
Figure 3: products at a field site (true-colour composites, common stretch)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from offset import PARAMS, unit  # noqa: E402

fs.setup()
plt = fs.plt
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ms = pd.read_csv(ROOT / "results/tables/measurements.csv")
OUT = ROOT / "results/figures"


def extent(jm, n):
    return [jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0]


def draw_lines(ax, m, row, color, half=None):
    """Fault trace and fitted marker lines from a measurement row."""
    p0 = np.array([m.E, m.N])
    s = unit(m.strike)
    L = 70
    ax.plot([p0[0] - L * s[0], p0[0] + L * s[0]], [p0[1] - L * s[1], p0[1] + L * s[1]], color="#d03b3b",
            lw=0.8, ls="--")
    n = unit(m.strike + 90)
    for side, sg in (("p", 1), ("m", -1)):
        u0, az = row[f"{side}_u0"], row[f"{side}_az"]
        if not np.isfinite(u0):
            continue
        d = unit(az)
        if np.dot(d, n) * sg < 0:
            d = -d
        a = p0 + u0 * s
        t = np.linspace(0, 50, 2)
        ax.plot(a[0] + t * d[0], a[1] + t * d[1], color=color, lw=1.1)
        ax.plot(a[0], a[1], "o", ms=2.5, color=color)


def stretch(img, lo, hi):
    return np.clip((img - lo) / (hi - lo), 0, 1)


# --------------------------------------------------------------- Figure 2
c = mk[(mk.kind == "controlled")]
ok = ms[(ms["product"] == "PAN") & ms.ok.astype(bool)].merge(c, on=["job", "marker"])
ok = ok[ok.job.str.endswith("_k2")]
ex = ok.sort_values("p_cnr", ascending=False).iloc[0] if len(ok) else None
if ex is not None:
    chip = ex.job[:-3]
    fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 0.36 * fs.COL2))
    for ax, k, lab in zip(axes[:2], (0, 2), ("(a)", "(b)")):
        j = f"{chip}_k{k}"
        jm = jobs.loc[j]
        P = sio.loadmat(ROOT / f"data/processed/chips/{j}.mat", variable_names=["PAN"])["PAN"]
        lo, hi = np.percentile(P, [1, 99])
        ax.imshow(stretch(P, lo, hi), cmap="gray", extent=extent(jm, P.shape[0]), interpolation="nearest")
        m = mk[(mk.job == j) & (mk.marker == ex.marker)].iloc[0]
        row = ms[(ms.job == j) & (ms.marker == ex.marker) & (ms["product"] == "PAN")].iloc[0]
        draw_lines(ax, m, row, "#2a78d6")
        ax.set_xlim(m.E - 55, m.E + 55)
        ax.set_ylim(m.N - 55, m.N + 55)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"PAN, k = {k} (H = {k * jm.ms_res:.2f} m)" if k else "PAN, k = 0", fontsize=7.5)
        fs.panel_label(ax, lab)
    # (c) profile geometry sketch: stations and profiles on the + side of k2
    ax = axes[2]
    j = f"{chip}_k2"
    jm = jobs.loc[j]
    P = sio.loadmat(ROOT / f"data/processed/chips/{j}.mat", variable_names=["PAN"])["PAN"]
    lo, hi = np.percentile(P, [1, 99])
    ax.imshow(stretch(P, lo, hi), cmap="gray", extent=extent(jm, P.shape[0]), alpha=0.55)
    m = mk[(mk.job == j) & (mk.marker == ex.marker)].iloc[0]
    row = ms[(ms.job == j) & (ms.marker == ex.marker) & (ms["product"] == "PAN")].iloc[0]
    p0, s, n = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
    for side, sg, col in (("p", 1, "#2a78d6"), ("m", -1, "#eb6834")):
        d = unit(row[f"{side}_az"])
        if np.dot(d, n) * sg < 0:
            d = -d
        e = np.array([d[1], -d[0]])
        a = p0 + row[f"{side}_u0"] * s
        dv = abs(np.dot(d, n))
        for t in np.arange(PARAMS["t_min_floor"], PARAMS["t_min_floor"] + PARAMS["length"] + 1, 5):
            q = a + (t / dv) * d
            ax.plot([q[0] - 8 * e[0], q[0] + 8 * e[0]], [q[1] - 8 * e[1], q[1] + 8 * e[1]], color=col, lw=0.5)
    draw_lines(ax, m, row, fs.INK)
    ax.annotate("", xy=p0 + row.p_u0 * s, xytext=p0 + row.m_u0 * s,
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=fs.INK))
    ax.set_xlim(m.E - 55, m.E + 55)
    ax.set_ylim(m.N - 55, m.N + 55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Profiles and fitted marker lines", fontsize=7.5)
    fs.panel_label(ax, "(c)")
    fig.subplots_adjust(left=0.03, right=0.99, wspace=0.08, top=0.9, bottom=0.03)
    fig.savefig(OUT / "fig02_controlled_example.pdf")
    fig.savefig(OUT / "fig02_controlled_example.png", dpi=300)
    print("fig2", chip, ex.marker)

# --------------------------------------------------------------- Figure 3
fj = "F08_1040010082B6D600"
if (ROOT / f"data/processed/fused/{fj}_res.mat").exists():
    r = sio.loadmat(ROOT / f"data/processed/fused/{fj}_res.mat", squeeze_me=True, variable_names=["RGBs", "L"])
    rgbs = r["RGBs"]
    jm = jobs.loc[fj]
    m = mk[(mk.job == fj)].iloc[0]
    names = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
    ref = rgbs["EXP"].item().astype(float)
    los = [np.percentile(ref[..., i], 1) for i in range(3)]
    his = [np.percentile(ref[..., i], 99) for i in range(3)]
    fig, axes = plt.subplots(2, 4, figsize=(fs.COL2, 0.42 * fs.COL2))
    for ax, nm, lab in zip(axes.ravel(), names, "abcdefgh"):
        if nm == "PAN":
            img = r["L"][:, :, 0]
            lo, hi = np.percentile(img, [1, 99])
            show = stretch(img, lo, hi)
            ax.imshow(show, cmap="gray", extent=extent(jm, img.shape[0]))
        elif nm == "VIS":
            v = np.load(ROOT / f"data/processed/chips/{fj}_vis.npy")
            lo, hi = np.percentile(v, [1, 99])
            ax.imshow(stretch(v, lo, hi), cmap="gray", extent=extent(jm, v.shape[0]))
        else:
            img = rgbs[nm].item().astype(float)
            show = np.dstack([stretch(img[..., i], los[i], his[i]) for i in range(3)])
            ax.imshow(show, extent=extent(jm, img.shape[0]))
        ax.set_xlim(m.E - 45, m.E + 45)
        ax.set_ylim(m.N - 35, m.N + 35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(fs.LABEL[nm] + (" (luminance)" if nm == "VIS" else ""), fontsize=7.5)
        ax.text(0.03, 0.95, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                fontweight="bold", color=fs.INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.04, hspace=0.16, top=0.94, bottom=0.01)
    fig.savefig(OUT / "fig03_products_field.pdf")
    fig.savefig(OUT / "fig03_products_field.png", dpi=300)
    print("fig3")
