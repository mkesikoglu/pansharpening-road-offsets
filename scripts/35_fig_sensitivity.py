"""Figure 11: sensitivity of the controlled results. (a) share measured and (b) median
absolute error of measured attempts (k > 0) at CNR thresholds 2, 3 and 4; (c) mixed-model
error factors relative to PAN for four values of epsilon and with acquisition fixed
effects. Writes results/tables/v2/sensitivity_cnr_v2.csv."""
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
P = fs.PRODUCTS
RNG = np.random.default_rng(11)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")


def load(f):
    d = pd.read_csv(f).merge(mk[["job", "marker", "ID", "kind", "H_true"]], on=["job", "marker", "ID"]).merge(
        jobs[["job", "k", "chip"]], on="job")
    d = d[d.kind == "controlled"].copy()
    d["unit"] = d.chip + "/" + d.marker
    d["ok"] = d.ok.astype(bool)
    d["ae"] = (d.H - d.H_true).abs()
    return d


def boot(df, fun, nb=1000):
    idx = df.groupby("unit").indices
    keys = list(idx)
    bs = [fun(df.iloc[np.concatenate([idx[keys[i]] for i in RNG.integers(len(keys), size=len(keys))])])
          for _ in range(nb)]
    return fun(df), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


rows = []
for thr, f in ((2, "measurements_cnr2.csv"), (3, "measurements.csv"), (4, "measurements_cnr4.csv")):
    d = load(T / f)
    d = d[d.k > 0]
    for p in P:
        g = d[d["product"] == p]
        s = g[g.ok]
        a = boot(g, lambda x: x.ok.mean() * 100)
        b = boot(s, lambda x: x.ae.median())
        rows.append(dict(cnr=thr, product=p, succ=a[0], succ_lo=a[1], succ_hi=a[2], med=b[0], med_lo=b[1],
                         med_hi=b[2], n=len(s)))
S = pd.DataFrame(rows)
S.to_csv(T / "v2/sensitivity_cnr_v2.csv", index=False)
lm = pd.read_csv(T / "v2/lmm_eps.csv")

fig, axs = plt.subplots(2, 3, figsize=(fs.COL2, 0.85 * fs.COL2), sharey="row")
axes = axs.ravel()
off = {2: -0.22, 3: 0.0, 4: 0.22}
mk_ = {2: "v", 3: "o", 4: "^"}
for ax, (col, lo, hi), xl, lab in ((axes[0], ("succ", "succ_lo", "succ_hi"), "Accepted (%)", "(a)"),
                                   (axes[1], ("med", "med_lo", "med_hi"), "Median |error| (m)", "(b)")):
    for thr in (2, 3, 4):
        for i, p in enumerate(P):
            r = S[(S.cnr == thr) & (S["product"] == p)].iloc[0]
            y = i + off[thr]
            ax.plot([r[lo], r[hi]], [y, y], color=fs.COLOR[p], lw=0.9, alpha=0.4 + 0.2 * (thr - 2))
            ax.plot(r[col], y, marker=mk_[thr], color=fs.COLOR[p], ms=3.4, mec="white", mew=0.3)
    ax.set_xlabel(xl)
    ax.xaxis.grid(True)
    fs.panel_label(ax, lab)
h = [plt.Line2D([], [], color="k", marker=mk_[t], ls="", ms=3.5, label=f"CNR ≥ {t}") for t in (2, 3, 4)]
axes[0].legend(handles=h, fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, handletextpad=0.1, columnspacing=0.6)
ax = axes[2]
cfg = [("primary", 0.001, "s", "ε = 0.001"), ("primary", 0.01, "o", "ε = 0.01"), ("primary", 0.05, "D", "ε = 0.05"),
       ("primary", 0.1, "^", "ε = 0.1"), ("acquisition fixed effect", 0.01, "x", "acq. FE")]
for j, (mdl, eps, m_, lab_) in enumerate(cfg):
    for i, p in enumerate(P):
        if p == "PAN":
            continue
        r = lm[(lm.model == mdl) & np.isclose(lm.eps, eps) & (lm["product"] == p)].iloc[0]
        y = i + (j - 2) * 0.13
        ax.plot([r.lo, r.hi], [y, y], color=fs.COLOR[p], lw=0.7, alpha=0.6)
        ax.plot(r.ratio, y, marker=m_, color=fs.COLOR[p], ms=3, mec="white" if m_ != "x" else fs.COLOR[p], mew=0.3)
ax.axvline(1, color=fs.INK2, lw=0.6)
ax.set_xscale("log")
ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6, 2])
ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6", "2"])
ax.set_xlabel("Multiplicative effect vs PAN")
ax.xaxis.grid(True)
h = [plt.Line2D([], [], color="k", marker=m_, ls="", ms=3.5, label=l_) for _, _, m_, l_ in cfg]
ax.legend(handles=h, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, handletextpad=0.1, columnspacing=0.5)
fs.panel_label(ax, "(c)")
axes[0].set_yticks(range(len(P)))
axes[0].set_yticklabels([fs.LABEL[p] for p in P], fontsize=7.5)
axes[0].invert_yaxis()
# (d, e) edge-free cross-correlation estimator
V = T / "v2"
xs = pd.read_csv(V / "xcorr_summary.csv").set_index("product")
for i, p in enumerate(P):
    r = xs.loc[p]
    axes[3].plot(r.success_pct, i, marker="o", color=fs.COLOR[p], ms=5, mec="k", mew=0.4)
    axes[3].plot(r.main_success_pct, i, marker="o", color="white", ms=5, mec=fs.COLOR[p], mew=1.0)
    axes[3].plot([r.main_success_pct, r.success_pct], [i, i], color=fs.COLOR[p], lw=0.8, alpha=0.6)
    axes[4].plot([r.cc_lo, r.cc_hi], [i, i], color=fs.COLOR[p], lw=1.2)
    axes[4].plot(r.cc_med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=5, mec="k", mew=0.4)
axes[3].set_xlabel("Accepted (%)")
axes[3].legend(handles=[plt.Line2D([], [], marker="o", ls="", color="k", label="cross-correlation"),
                        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="profile (main)")],
               fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
axes[4].set_xlabel("Jointly accepted median (m)")
for ax, lab in ((axes[3], "(d)"), (axes[4], "(e)")):
    ax.xaxis.grid(True)
    fs.panel_label(ax, lab)
axes[3].set_yticks(range(len(P)))
axes[3].set_yticklabels([fs.LABEL[p] for p in P], fontsize=7.5)
axes[3].invert_yaxis()
# (f) intensity definitions
ax = axes[5]
ip = V / "intensity_summary.csv"
IM = {"RGB": "o", "PANW": "s", "NIR1": "^", "PC1": "D"}
if ip.exists():
    it = pd.read_csv(ip)
    for j, (iname, m_) in enumerate(IM.items()):
        for i, p in enumerate(P):
            g = it[(it["product"] == p) & (it.intensity == iname)]
            if g.empty:
                continue
            ax.plot(g.success_pct.iloc[0], i + (j - 1.5) * 0.15, marker=m_, color=fs.COLOR[p], ms=4.5, mec="k", mew=0.3, ls="")
    ax.legend(handles=[plt.Line2D([], [], marker=m_, ls="", color="k", label=n) for n, m_ in IM.items()], fontsize=7,
              loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=4, handletextpad=0.1, columnspacing=0.6)
ax.set_xlabel("Accepted (%), subset")
ax.xaxis.grid(True)
fs.panel_label(ax, "(f)")
fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.08, wspace=0.12, hspace=0.42)
fig.savefig(ROOT / "results/figures/fig11_sensitivity.pdf")
fig.savefig(ROOT / "results/figures/fig11_sensitivity.png", dpi=300)
print(S.pivot_table(index="product", columns="cnr", values=["succ", "med"]).round(2).reindex(P))
