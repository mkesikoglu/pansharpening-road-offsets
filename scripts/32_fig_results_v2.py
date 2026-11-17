"""Revision figures from the full-resolution controlled experiment:
Fig. 4 products and fitted edges for three rule-selected cases; Fig. 5 measurability
and failures; Fig. 6 error distributions; Fig. 7 accuracy-coverage-runtime; Fig. 9
quality-task models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from offset import PARAMS, Raster, _Side, unit  # noqa: E402

fs.setup()
plt = fs.plt
T = ROOT / "results/tables/v2"
OUT = ROOT / "results/figures"
P = fs.PRODUCTS
c = pd.read_csv(T / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")


def stretch(a):
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1)


# ------------------------------------------------------------------ Fig. 4: three rule-selected cases at k = 2
k2 = c[c.k == 2]
wide = k2.pivot_table(index="unit", columns="product", values="H")
okw = k2.pivot_table(index="unit", columns="product", values="ok", aggfunc="max").fillna(False)
allok = okw[P].all(axis=1)
spread = (wide[P].max(axis=1) - wide[P].min(axis=1))
typical = spread[allok].idxmin()
shift = spread[allok & (spread < 1.0)].idxmax()
mixed = okw.index[(~allok) & (okw[P].sum(axis=1) >= 3)]
switch = spread[mixed].idxmax() if len(mixed) else spread.idxmax()
cases = [("(a) all accepted,\nsmallest spread", typical),
         ("(b) all accepted,\nlargest spread\nbelow 1 m", shift),
         ("(c) largest spread,\nthree or more\naccepted", switch)]
fig, axes = plt.subplots(3, 9, figsize=(fs.COL2, 0.46 * fs.COL2), gridspec_kw=dict(width_ratios=[1] * 8 + [1.5]))
for r_, (title, u) in enumerate(cases):
    chip, marker = u.split("/")
    j = f"{chip}_k2"
    jm = jobs.loc[j]
    m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
    res = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
    imgs = {str(n): res["L"][:, :, i] for i, n in enumerate(res["NAMES"])}
    v = ROOT / f"data/processed/chips/{j}_vis.npy"
    if v.exists():
        imgs["VIS"] = np.load(v)
    n = 1024
    ext = [jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0]
    prof_ax = axes[r_, 8]
    for col, p in enumerate(P):
        ax = axes[r_, col]
        ax.set_xticks([])
        ax.set_yticks([])
        if p not in imgs or np.isnan(imgs[p]).all():
            ax.axis("off")
            continue
        ax.imshow(stretch(imgs[p]), cmap="gray", extent=ext)
        row = c[(c.job == j) & (c.marker == marker) & (c["product"] == p)].iloc[0]
        p0, s, nn = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
        ax.plot(*np.c_[p0 - 40 * s, p0 + 40 * s], color="#D55E00", lw=0.6, ls="--")
        if row.ok:
            for side, sg in (("p", 1), ("m", -1)):
                d = unit(row[f"{side}_az"])
                if np.dot(d, nn) * sg < 0:
                    d = -d
                a = p0 + row[f"{side}_u0"] * s
                ax.plot([a[0], a[0] + 35 * d[0]], [a[1], a[1] + 35 * d[1]], color="#56B4E9", lw=0.9)
            lab = f"{row.err:+.2f} m"
        else:
            lab = "failed"
        ax.set_xlim(m.E - 35, m.E + 35)
        ax.set_ylim(m.N - 35, m.N + 35)
        ax.text(0.03, 0.04, lab, transform=ax.transAxes, color="white", fontsize=7.0,
                bbox=dict(fc="black", alpha=0.5, pad=0.8, ec="none"))
        if r_ == 0:
            ax.set_title({"BDSD_PC": "BDSD-PC", "MTF_GLP_HPM": "HPM", "VIS": "Vendor"}.get(p, fs.LABEL[p]), fontsize=7.5)
        # profile across the road at 15 m from the fault on the + side
        side = _Side(Raster(imgs[p].astype(float), jm.x0, jm.y0, jm.pan_res), (m.E, m.N), m.strike, m.az_plus, +1,
                     0.0, PARAMS)
        q = side.p0 + side.base_u * side.s + (15 / side.dv) * side.d
        pts = q[None, :] + side.w[:, None] * side.e[None, :]
        pr = side.img.sample(pts)
        prof_ax.plot(side.w, (pr - pr.min()) / (pr.max() - pr.min() + 1e-9), color=fs.COLOR[p], lw=0.7)
    axes[r_, 0].set_ylabel(title, fontsize=7.0, rotation=0, ha="right", va="center", labelpad=4)
    prof_ax.set_xlim(-15, 15)
    prof_ax.set_yticks([])
    prof_ax.tick_params(labelsize=6)
    if r_ == 2:
        prof_ax.set_xlabel("Across road (m)", fontsize=7.5)
    if r_ == 0:
        prof_ax.set_title("Profiles at 15 m", fontsize=7.5)
fig.subplots_adjust(left=0.105, right=0.99, top=0.93, bottom=0.12, wspace=0.05, hspace=0.08)
h_ = [plt.Line2D([], [], color=fs.COLOR[p], lw=1.4, label=fs.LABEL[p]) for p in P]
fig.legend(handles=h_, loc="lower center", ncol=8, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06), handlelength=1.2, columnspacing=0.8)
fig.savefig(OUT / "fig04_cases.pdf")
fig.savefig(OUT / "fig04_cases.png", dpi=300)
plt.close(fig)
pd.Series(dict(typical=typical, shift=shift, switch=switch)).to_csv(T / "fig4_cases.csv")

# ------------------------------------------------------------------ Fig. 5: measurability and failures
meas = pd.read_csv(T / "measurability.csv").set_index("product")
fr = pd.read_csv(T / "failure_reasons.csv").set_index("product").reindex(P)
order = c.groupby("unit").ok.mean().sort_values().index
hm = c.pivot_table(index="unit", columns="product", values="ok", aggfunc="mean").reindex(order)[P]
fig = plt.figure(figsize=(fs.COL2, 0.52 * fs.COL2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.8], wspace=0.95)
SHORT = dict(PAN="PAN", EXP="EXP", GSA="GSA", BDSD_PC="BDSD", MTF_GLP_HPM="HPM", AWLP="AWLP", ZPNN="Z-PNN", VIS="RGB")
ax = fig.add_subplot(gs[0])
im = ax.imshow(hm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(P)))
ax.set_xticklabels([SHORT[p] for p in P], rotation=70, ha="right", fontsize=7.5)
ax.set_yticks([])
ax.set_ylabel(f"Crossings (n = {hm.shape[0]}), sorted")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
ax.set_title("share of k = 0-3 accepted", fontsize=7.5, pad=8, loc="right")
cb.ax.tick_params(labelsize=6)
fs.panel_label(ax, "(a)")
ax = fig.add_subplot(gs[1])
for i, p in enumerate(P):
    r = meas.loc[p]
    ax.plot([r.lo_x, r.hi_x], [i, i], color=fs.COLOR[p], lw=1.3)
    ax.plot(r.success_pct, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=5, mec="white", mew=0.4)
    ax.text(r.hi_x + 2, i, f"{int(r.n_success)}/{int(r.n_total)}", va="center", fontsize=7.0, color=fs.INK2)
ax.set_yticks(range(len(P)))
ax.set_yticklabels([fs.LABEL[p] for p in P], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(25, 95)
ax.set_xlabel("Accepted (%) with 95% CI")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
fs.panel_label(ax, "(b)")
ax = fig.add_subplot(gs[2])
cats = ["low contrast", "edge not tracked"]
cols = ["#E69F00", "#56B4E9"]
left = np.zeros(len(P))
tot = fr.sum(axis=1).values
for cat, col in zip(cats, cols):
    if cat not in fr.columns:
        continue
    v = fr[cat].values / tot * 100
    ax.barh(range(len(P)), v, left=left, color=col, height=0.7, label=cat, edgecolor="white", linewidth=0.4)
    left += v
ax.set_yticks(range(len(P)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel("Failed attempts (%)")
ax.set_xlim(0, 85)
ax.legend(fontsize=7.0, loc="lower center", bbox_to_anchor=(0.62, 1.03), handlelength=1, ncol=1)
fs.panel_label(ax, "(c)")
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.2)
fig.savefig(OUT / "fig05_measurability.pdf")
fig.savefig(OUT / "fig05_measurability.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Fig. 6: error distributions
okk = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
common = set(okk.index[okk[P].all(axis=1)])
s = c[(c.k > 0) & c.ok].copy()
s["common"] = [(u, k) in common for u, k in zip(s.unit, s.k)]
pr = pd.read_csv(T / "paired_vs_pan.csv").set_index("product")
fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 0.36 * fs.COL2), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
for ax, sub, lab, ttl in ((axes[0], s[s.common], "(a)", f"Complete cases (n = {len(common)} per product)"),
                          (axes[1], s, "(b)", "All successful estimates")):
    for p in P:
        v = np.sort(sub[sub["product"] == p].ae.values)
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=fs.COLOR[p], lw=1.0,
                    label=f"{fs.LABEL[p]} ({len(v)})" if ax is axes[1] else fs.LABEL[p])
    ax.set_xscale("log")
    ax.set_xlim(0.02, 20)
    ax.set_xlabel(r"$|\hat{H}-H|$ (m)")
    ax.set_title(ttl, fontsize=8.0)
    ax.grid(True, which="major")
    fs.panel_label(ax, lab)
axes[0].set_ylabel("Cumulative share")
axes[1].legend(fontsize=7.0, loc="lower right")
ax = axes[2]
order = [p for p in P if p != "PAN"]
for i, p in enumerate(order):
    r = pr.loc[p]
    ax.plot([r.lo, r.hi], [i, i], color=fs.COLOR[p], lw=1.3)
    ax.plot(r.med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=5, mec="white", mew=0.4)
ax.axvline(0, color=fs.INK2, lw=0.6)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{fs.LABEL[p]} ({int(pr.loc[p].n_crossings)})" for p in order], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Crossing-level paired\ndifference vs PAN (m)")
ax.xaxis.grid(True)
fs.panel_label(ax, "(c)")
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.24, wspace=0.42)
fig.savefig(OUT / "fig06_errors.pdf")
fig.savefig(OUT / "fig06_errors.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Fig. 7: Pareto
par = pd.read_csv(T / "pareto.csv").set_index("product")
fig, axes = plt.subplots(1, 2, figsize=(fs.COL2, 0.36 * fs.COL2))
for ax, ycol, ylab, lab in ((axes[0], "common_med", "Complete-case median error (m)", "(a)"),
                            (axes[1], "common_p90", "Complete-case 90th percentile (m)", "(b)")):
    for p in P:
        r = par.loc[p]
        rt = r.runtime_s if np.isfinite(r.runtime_s) else 0.5
        ax.scatter(r.success_pct, r[ycol], s=18 + 18 * np.log10(1 + rt), marker=fs.MARKER[p], color=fs.COLOR[p],
                   edgecolor="k" if p != "VIS" else "none", lw=0.3, zorder=3)
        ax.annotate(fs.LABEL[p], (r.success_pct, r[ycol]), xytext=(4, -9) if p == "ZPNN" and ycol == "common_med" else (4, 3),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Accepted (%)")
    ax.set_ylabel(ylab)
    ax.grid(True)
    fs.panel_label(ax, lab)
axes[0].text(0.98, 0.03, "symbol size: run time per chip", transform=axes[0].transAxes, ha="right", fontsize=7.5,
             color=fs.INK2)
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.18, wspace=0.3)
fig.savefig(OUT / "fig07_pareto.pdf")
fig.savefig(OUT / "fig07_pareto.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Fig. 9: quality-task models
qm = pd.read_csv(T / "quality_task_models.csv")
NAME = {"Q2n": "Q2$^n$ (8 bands, RR)", "SAM": "SAM (8 bands, RR)", "ERGAS": "ERGAS (8 bands, RR)",
        "SCC": "SCC (8 bands, RR)", "Q2n_B7": "Q2$^n$ (no Coastal, RR)", "ERGAS_B7": "ERGAS (no Coastal, RR)",
        "Q2n_RGB": "Q2$^n$ (RGB, RR)", "SAM_RGB": "SAM (RGB, RR)", "ERGAS_RGB": "ERGAS (RGB, RR)",
        "SCC_RGB": "SCC (RGB, RR)", "SAM_road": "SAM, road corridor (RR)", "D_lambda": r"$D_\lambda^K$ (FR)",
        "D_s": r"$D_S$ (FR)", "HQNR": "HQNR (FR)", "edge_disp_m": "Edge displacement vs reference"}
qm = qm[qm["index"].isin(NAME)].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(fs.COL2, 0.42 * fs.COL2), sharey=True)
for ax, pre, xl, lab in ((axes[0], "err", "Effect on log median error per SD", "(a)"),
                         (axes[1], "succ", "Effect on measured share per SD", "(b)")):
    for i, r in qm.iterrows():
        sig = r[f"{pre}_p_holm"] < 0.05
        ax.plot([r[f"{pre}_lo"], r[f"{pre}_hi"]], [i, i], color="#0072B2" if sig else "#9a9994", lw=1.2)
        ax.plot(r[f"{pre}_coef"], i, "o", ms=4, color="#0072B2" if sig else "#6d6c68")
    ax.axvline(0, color=fs.INK2, lw=0.6)
    ax.set_xlabel(xl)
    ax.xaxis.grid(True)
    fs.panel_label(ax, lab)
axes[0].set_yticks(range(len(qm)))
axes[0].set_yticklabels([NAME[x] for x in qm["index"]], fontsize=7.5)
axes[0].invert_yaxis()
fig.subplots_adjust(left=0.27, right=0.99, top=0.93, bottom=0.14, wspace=0.08)
fig.savefig(OUT / "fig09_quality_task.pdf")
fig.savefig(OUT / "fig09_quality_task.png", dpi=300)
print("ok", typical, shift, switch)
