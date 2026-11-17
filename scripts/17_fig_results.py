"""Figures 4-7 from the analysis tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402

fs.setup()
plt = fs.plt
T = ROOT / "results/tables"
OUT = ROOT / "results/figures"
P = fs.PRODUCTS
c = pd.read_csv(T / "controlled_long.csv")
c["ok"] = c.ok.astype(bool)
pr = pd.read_csv(T / "controlled_paired_vs_pan.csv").set_index("product")

# ---------------------------------------------------------------- Figure 4: controlled errors
fig, axes = plt.subplots(1, 2, figsize=(fs.COL2, 0.42 * fs.COL2), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
for i, p in enumerate(P):
    for kk, k in enumerate((0, 1, 2, 3)):
        v = c[(c["product"] == p) & (c.k == k) & c.ok].err.values
        if len(v) == 0:
            continue
        x = i + (kk - 1.5) * 0.18
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lo, hi = np.percentile(v, [5, 95])
        ax.plot([x, x], [lo, hi], color=fs.COLOR[p], lw=0.7)
        ax.add_patch(plt.Rectangle((x - 0.07, q1), 0.14, q3 - q1, fc=fs.COLOR[p], ec="white", lw=0.4,
                                   alpha=0.35 + 0.18 * kk))
        ax.plot([x - 0.07, x + 0.07], [med, med], color=fs.INK, lw=0.8)
ax.axhline(0, color=fs.INK2, lw=0.5)
ax.set_xticks(range(len(P)))
ax.set_xticklabels([fs.LABEL[p] for p in P], rotation=35, ha="right")
ax.set_ylabel(r"Signed error $\hat{H}-H$ (m)")
ax.set_ylim(-1.6, 1.6)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
fs.panel_label(ax, "(a)")
ax = axes[1]
order = [p for p in P if p != "PAN"]
for i, p in enumerate(order):
    r = pr.loc[p]
    ax.plot([r.d_lo, r.d_hi], [i, i], color=fs.COLOR[p], lw=1.4)
    ax.plot(r.d_median, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=5, mec="white", mew=0.5)
    ax.text(0.99, i, f"{r.frac_better:.0f}%", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=6.5, color=fs.INK2)
ax.axvline(0, color=fs.INK2, lw=0.6)
ax.text(0.99, -0.75, "better\nthan PAN", transform=ax.get_yaxis_transform(), ha="right", va="center",
        fontsize=6, color=fs.INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([fs.LABEL[p] for p in order])
ax.invert_yaxis()
ax.set_xlabel(r"Median paired difference in $|\hat{H}-H|$ vs PAN (m)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
fs.panel_label(ax, "(b)")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.93, wspace=0.42)
fig.savefig(OUT / "fig04_controlled_errors.pdf")
fig.savefig(OUT / "fig04_controlled_errors.png", dpi=300)

# ---------------------------------------------------------------- Figure 5: field comparison
f = pd.read_csv(T / "field_long.csv")
f["ok"] = f.ok.astype(bool)
f = f[f.ok]
obs = f.groupby("ID").first().sort_values("H_ref")
ids = list(obs.index)
fig, ax = plt.subplots(figsize=(fs.COL2, 0.42 * fs.COL2))
for i, idn in enumerate(ids):
    o = obs.loc[idn]
    lo = o.H_min if np.isfinite(o.H_min) else o.H_ref
    hi = o.H_max if np.isfinite(o.H_max) else o.H_ref
    ax.add_patch(plt.Rectangle((i - 0.42, lo), 0.84, hi - lo, fc="#e6e5e1", ec="none", zorder=0))
    ax.plot([i - 0.42, i + 0.42], [o.H_ref, o.H_ref], color=fs.INK, lw=1.0, zorder=1)
    g = f[f.ID == idn]
    for j, p in enumerate(P):
        v = g[g["product"] == p].H.values
        x = i + (j - 3.5) * 0.09
        ax.scatter(np.full(len(v), x), v, s=9, marker=fs.MARKER[p], color=fs.COLOR[p], lw=0, zorder=3,
                   label=fs.LABEL[p] if i == 0 else None)
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f"{int(i)}\n{f[f.ID == i].marker.iloc[0]}" for i in ids], fontsize=6.5)
ax.set_xlabel("Field observation ID and marker")
ax.set_ylabel("Left-lateral offset (m)")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(ncol=8, loc="upper left", bbox_to_anchor=(0, 1.13), handletextpad=0.2, columnspacing=0.8)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.86)
fig.savefig(OUT / "fig05_field_offsets.pdf")
fig.savefig(OUT / "fig05_field_offsets.png", dpi=300)

# ---------------------------------------------------------------- Figure 6: quality vs offset error
q = pd.read_csv(T / "quality_long.csv")
FU = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
qq = q.groupby("product")[["Q2n", "SAM", "ERGAS", "HQNR"]].median().reindex(FU)
ae = c[c.ok & (c.k > 0)].groupby("product").ae.median().reindex(FU)
fig, axes = plt.subplots(1, 4, figsize=(fs.COL2, 0.3 * fs.COL2), sharey=True)
for ax, ind, lab in zip(axes, ["Q2n", "SAM", "ERGAS", "HQNR"], ["(a)", "(b)", "(c)", "(d)"]):
    for p in FU:
        ax.scatter(qq.loc[p, ind], ae.loc[p], marker=fs.MARKER[p], color=fs.COLOR[p], s=22, lw=0,
                   label=fs.LABEL[p])
    ax.set_xlabel({"Q2n": r"$Q2^n$ (RR)", "SAM": "SAM (deg, RR)", "ERGAS": "ERGAS (RR)", "HQNR": "HQNR (FR)"}[ind])
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fs.panel_label(ax, lab)
axes[0].set_ylabel("Median $|\\hat{H}-H|$ (m)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.82, wspace=0.12)
fig.savefig(OUT / "fig06_quality_vs_error.pdf")
fig.savefig(OUT / "fig06_quality_vs_error.png", dpi=300)


# ---------------------------------------------------------------- Figure 7: conditions
desc = pd.read_csv(T / "controlled_delta_covariates.csv")[["unit", "width"]].drop_duplicates("unit")
g = c[c.ok & (c.k > 0)].merge(desc, on="unit", how="left")
g["angle_bin"] = pd.cut(g.angle, [30, 50, 70, 90], labels=["30–50", "50–70", "70–90"])
g["width_bin"] = pd.cut(g.width, [0, 4, 7, 100], labels=["< 4", "4–7", "> 7"])
nunit = g.groupby("acq").unit.nunique()
acq_order = (c.drop_duplicates("acq").sort_values("off_nadir")[["acq", "off_nadir"]])
acq_order = acq_order[acq_order.acq.map(nunit).fillna(0) >= 3]  # acquisitions with >= 3 crossings
ACQLAB = {a: f"{a}\n({o:.0f}°)" for a, o in zip(acq_order.acq, acq_order.off_nadir)}
g["acq_lab"] = pd.Categorical(g.acq, categories=list(acq_order.acq), ordered=True)
rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 0.34 * fs.COL2), sharey=True,
                         gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
for ax, var, xl, lab in zip(axes, ["angle_bin", "width_bin", "acq_lab"],
                            ["Angle between marker and fault (deg)", "Road width on PAN (m)",
                             "Acquisition (increasing off-nadir angle)"], ["(a)", "(b)", "(c)"]):
    cats = list(g[var].cat.categories)
    for i, cat in enumerate(cats):
        for j, p in enumerate(P):
            v = g[(g[var] == cat) & (g["product"] == p)].ae.values
            if len(v) < 4:
                continue
            bs = [np.median(rng.choice(v, len(v))) for _ in range(1000)]
            x = i + (j - 3.5) * 0.09
            ax.plot([x, x], np.percentile(bs, [2.5, 97.5]), color=fs.COLOR[p], lw=0.8)
            ax.plot(x, np.median(v), marker=fs.MARKER[p], color=fs.COLOR[p], ms=3.5, mec="white", mew=0.3,
                    ls="", label=fs.LABEL[p] if (i == 0 and ax is axes[0]) else None)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([ACQLAB.get(str(s), str(s)) for s in cats], fontsize=6.5)
    ax.set_xlabel(xl)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fs.panel_label(ax, lab)
axes[0].set_ylabel(r"Median $|\hat{H}-H|$ (m)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.02),
           handletextpad=0.2, columnspacing=0.9)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.84, wspace=0.08)
fig.savefig(OUT / "fig07_conditions.pdf")
fig.savefig(OUT / "fig07_conditions.png", dpi=300)
print("done")
