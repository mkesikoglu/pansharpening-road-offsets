"""Figure 8: reduced-scale experiment. (a) median |delta| (product minus latent
reference) by MS phase without misregistration; (b) median |delta| by misregistration;
(c) share of attempts measured by misregistration. Intervals: crossing bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402

fs.setup()
plt = fs.plt
RNG = np.random.default_rng(13)
r = pd.read_csv(ROOT / "results/tables/v2/rr_long.csv")
r["ok"] = r.ok.astype(bool)
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]
LAB = dict(fs.LABEL, ZPNN0="Z-PNN (pretrained)")
COL = dict(fs.COLOR, ZPNN0="#b3a7e8")
MRK = dict(fs.MARKER, ZPNN0="x")


def boot(df, col, fun=np.median, nb=1000):
    if df.empty:
        return np.nan, np.nan, np.nan
    idx = df.groupby("unit").indices
    keys = list(idx)
    v = df[col].values
    bs = [fun(v[np.concatenate([idx[keys[i]] for i in RNG.integers(len(keys), size=len(keys))])]) for _ in range(nb)]
    return fun(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


d = r[r.ok & r.delta.notna() & r["product"].isin(P)].assign(ad=lambda x: x.delta.abs())
fig, axs = plt.subplots(2, 2, figsize=(fs.COL2, 0.78 * fs.COL2))
axes = axs.ravel()
ax = axes[0]
phases = [0, 0.25, 0.5, 0.75]
for i, p in enumerate(P):
    xs, ys, lo, hi = [], [], [], []
    for ph in phases:
        e = boot(d[(d["product"] == p) & (d.m == 0) & np.isclose(d.phase, ph)], "ad")
        xs.append(ph + (i - 3.5) * 0.012)
        ys.append(e[0]), lo.append(e[1]), hi.append(e[2])
    ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=COL[p], marker=MRK[p], ms=3.5,
                lw=0.9, capsize=0, label=LAB[p])
ax.set_xticks(phases)
ax.set_xticklabels(["0", "1/4", "1/2", "3/4"])
ax.set_xlabel("Translation phase on the MS grid")
ax.set_ylabel(r"Median $|\delta|$ (m)")
ax.grid(True)
fs.panel_label(ax, "(a)")
ax = axes[1]
s = r[(r.k > 0) & r.ok & r.inc_ae.notna() & (r.m == 0)]
for i, p in enumerate(["REF"] + P):
    ys, lo, hi = [], [], []
    for ph in phases[1:]:
        e = boot(s[(s["product"] == p) & np.isclose(s.phase, ph)], "inc_ae")
        ys.append(e[0]), lo.append(e[1]), hi.append(e[2])
    xs = np.array(phases[1:]) + (i - 4) * 0.012
    ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=COL.get(p, "k"), marker=MRK.get(p, "s"),
                ms=3.5, lw=0.9, capsize=0, ls="--" if p == "REF" else "-", label=LAB.get(p, "Reference (original MS)"))
ax.set_yscale("log")
ax.set_xticks(phases[1:])
ax.set_xticklabels(["1/4", "1/2", "3/4"])
ax.set_xlabel("Translation phase on the MS grid")
ax.set_ylabel("Median baseline-corrected |error| (m)")
ax.grid(True, which="major")
fs.panel_label(ax, "(b)")
ax = axes[2]
pan = r[r["product"] == "PAN"].set_index(["unit", "k", "m"])
r["d_pan"] = [abs(h - pan.H.get((u, k, m), np.nan)) if o and pan.ok.get((u, k, m), False) else np.nan
              for u, k, m, h, o in zip(r.unit, r.k, r.m, r.H, r.ok)]
for i, p in enumerate(["REF"] + P[1:]):
    ys, lo, hi = [], [], []
    for m in (0, 1, 2):
        e = boot(r[(r["product"] == p) & (r.m == m) & r.d_pan.notna()], "d_pan")
        ys.append(e[0]), lo.append(e[1]), hi.append(e[2])
    xs = np.array([0, 0.25, 0.5]) + (i - 4) * 0.012
    ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=COL.get(p, "k"), marker=MRK.get(p, "s"),
                ms=3.5, lw=0.9, capsize=0, ls="--" if p == "REF" else "-")
ax.set_yscale("log")
ax.set_xticks([0, 0.25, 0.5])
ax.set_xticklabels(["0", "1/4", "1/2"])
ax.set_xlabel("PAN–MS misregistration (MS pixels)")
ax.set_ylabel(r"Median $|\hat{H}_p-\hat{H}_{\mathrm{PAN}}|$ (m)")
ax.grid(True, which="major")
fs.panel_label(ax, "(c)")
ax = axes[3]
ss = pd.read_csv(ROOT / "results/tables/v2/side_summary.csv").set_index("product")
order = ["REF", "PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
for i, p in enumerate(order):
    r_ = ss.loc[p]
    ax.plot([r_.lo, r_.hi], [i, i], color=COL.get(p, "k"), lw=1.4)
    ax.plot(r_.med_frac, i, marker=MRK.get(p, "s"), color=COL.get(p, "k"), ms=6, mec="k", mew=0.4, ls="")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([LAB.get(p, "Reference (original MS)") for p in order])
ax.invert_yaxis()
ax.axvline(0, color=fs.INK2, lw=0.6)
ax.axvline(1, color=fs.INK2, lw=0.6, ls=":")
ax.set_xlim(-0.1, 1.1)
ax.set_xlabel("Offset change / MS shift")
ax.set_title("one-sided MS shift")
ax.xaxis.grid(True)
fs.panel_label(ax, "(d)")
axes[1].set_ylabel("Baseline-corrected |error| (m)")
axes[2].set_ylabel(r"$|\hat{H}_p-\hat{H}_{\mathrm{PAN}}|$ (m)")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.0, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.17, wspace=0.6, hspace=0.45)
fig.savefig(ROOT / "results/figures/fig08_rr.pdf")
fig.savefig(ROOT / "results/figures/fig08_rr.png", dpi=300)
print("ok")
