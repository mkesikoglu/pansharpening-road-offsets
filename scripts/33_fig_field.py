"""Figure 10: field case study. (a) PAN and Z-PNN crops of marker M01 on 11 Feb with
fitted edge lines; (b) M01 offset against the projected marker length; (c) outcome
matrix for all field observations and products at the nominal 40 m aperture."""
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
P = fs.PRODUCTS
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
fl = pd.read_csv(ROOT / "results/tables/field_long.csv")
sens = pd.read_csv(ROOT / "results/tables/field_sensitivity.csv")
DATE = {"10300500D9F8D200": "8 Feb", "10300100E291D100": "11 Feb", "10300100E3154100": "28 Feb (WV-2)",
        "1040010082B6D600": "28 Feb (WV-3)", "10300100E383D900": "11 Mar"}


def stretch(a):
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1)


fig = plt.figure(figsize=(fs.COL2, 0.72 * fs.COL2))
gs = fig.add_gridspec(2, 2, width_ratios=[0.75, 1.0], wspace=0.35, hspace=0.38)
j = "F00_10300100E291D100"
jm = jobs.loc[j]
res = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
imgs = {str(n): res["L"][:, :, i] for i, n in enumerate(res["NAMES"])}
n = imgs["PAN"].shape[0]
ext = [jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0]
for r_, p in enumerate(("PAN", "ZPNN")):
    ax = fig.add_subplot(gs[r_, 0])
    ax.imshow(stretch(imgs[p]), cmap="gray", extent=ext)
    row = fl[(fl.job == j) & (fl.marker == "M01") & (fl["product"] == p)].iloc[0]
    p0, s, nn = np.array([row.E, row.N]), unit(row.strike), unit(row.strike + 90)
    ax.plot(*np.c_[p0 - 60 * s, p0 + 60 * s], color="#D55E00", lw=0.7, ls="--")
    if row.ok:
        for side, sg in (("p", 1), ("m", -1)):
            d = unit(row[f"{side}_az"])
            if np.dot(d, nn) * sg < 0:
                d = -d
            a = p0 + row[f"{side}_u0"] * s
            ax.plot([a[0], a[0] + 40 * d[0]], [a[1], a[1] + 40 * d[1]], color="#56B4E9", lw=1.0)
    ax.set_xlim(row.E - 50, row.E + 50)
    ax.set_ylim(row.N - 50, row.N + 50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([row.E + 20, row.E + 40], [row.N - 44, row.N - 44], color="white", lw=2)
    ax.text(row.E + 30, row.N - 41, "20 m", color="white", ha="center", fontsize=7.5)
    lab = f"{row.H:.2f} m" if row.ok else "failed"
    ax.set_title(f"{fs.LABEL[p]}, 11 Feb: {lab}", fontsize=7.5)
    fs.panel_label(ax, "(a)" if r_ == 0 else "(b)")
# support length (L) and aperture (A) curves for the GNSS observation of M01
ls = {"10300500D9F8D200": ":", "10300100E291D100": "-", "10300100E3154100": "--"}
fa_path = ROOT / "results/tables/field_sensitivity_A.csv"
sensA = pd.read_csv(fa_path) if fa_path.exists() else None
for r_, (cfg, xcol, xl, lab) in enumerate((("L", "L", "Station span L (m)", "(c)"),
                                          ("A", "du", "Aperture A (m)", "(d)"))):
    ax = fig.add_subplot(gs[r_, 1])
    src = sens if cfg == "L" else sensA
    if src is not None:
        m01 = src[(src.config == cfg) & (src.marker == "M01") & (src.ID == 2)]
        for p in P:
            for cid, g in m01[m01["product"] == p].groupby("catalog_id"):
                g = g.sort_values(xcol)
                y = np.where(g.ok.astype(bool), g.H, np.nan)
                ax.plot(g[xcol], y, ls=ls.get(cid, "-"), marker=fs.MARKER[p], ms=3, lw=0.8, color=fs.COLOR[p])
    for H, lo, hi, txt in ((1.75, 1.5, 2.0, "obs. 1, tape"), (4.81, 3.1, 6.9, "obs. 2, GNSS 96 m")):
        ax.axhspan(lo, hi, color="#9a9994", alpha=0.18, lw=0)
        ax.axhline(H, color="#6d6c68", lw=0.7)
        fs.halo(ax.text(0.02, H, txt, fontsize=7.5, color=fs.INK, transform=ax.get_yaxis_transform(), va="bottom", zorder=10))
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel("M01 offset (m)")
    ax.grid(True)
    fs.panel_label(ax, lab)
    if r_ == 0:
        h = [plt.Line2D([], [], color="k", ls=v, lw=0.8, label=DATE[k]) for k, v in ls.items()]
        ax.legend(handles=h, fontsize=7.0, loc="lower right")
fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.08)
fig.savefig(ROOT / "results/figures/fig10_field.pdf")
fig.savefig(ROOT / "results/figures/fig10_field.png", dpi=300)
plt.close(fig)
# outcome matrix at the nominal setting (supplementary figure)
fig, ax = plt.subplots(figsize=(0.62 * fs.COL2, 0.95 * fs.COL2))
ACQL = {"10300500D9F8D200": "8 Feb, WV-2", "10300100E291D100": "11 Feb, WV-2", "10300100E3154100": "28 Feb, WV-2",
        "1040010082B6D600": "28 Feb, WV-3", "10300100E383D900": "11 Mar, WV-2"}
fl["obs"] = fl.marker + " obs. " + fl.ID.astype(str) + ", " + fl.catalog_id.map(ACQL)
fl["code"] = np.where(fl.ok, np.where(fl.inside.astype(str) == "True", 2, 1), 0)
mat = fl.pivot_table(index="obs", columns="product", values="code", aggfunc="max").reindex(columns=P)
order = sorted(mat.index, key=lambda s: (int(s.split("obs. ")[1].split(",")[0]), s))
mat = mat.loc[order]
cmap = plt.matplotlib.colors.ListedColormap(["#ece9e1", "#E69F00", "#0072B2"])
ax.imshow(np.ma.masked_invalid(mat.values.astype(float)), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto",
          interpolation="nearest")
ax.set_xticks(range(len(P)))
ax.set_xticklabels([fs.LABEL[p] for p in P], rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(mat)))
ax.set_yticklabels(mat.index, fontsize=7.5)
ax.tick_params(length=0)
h = [plt.matplotlib.patches.Patch(color=c, label=l) for c, l in
     zip(["#ece9e1", "#E69F00", "#0072B2"], ["failed", "accepted, outside field range", "accepted, inside field range"])]
ax.legend(handles=h, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, frameon=False)
fig.savefig(ROOT / "results/figures/figS_field_matrix.pdf")
fig.savefig(ROOT / "results/figures/figS_field_matrix.png", dpi=300)
print(mat.shape)
