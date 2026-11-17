"""Revised result figures (second review):
fig05_errors.pdf      conditional error: crossing-balanced medians, complete-case ECDF, paired differences
                      with practical margins, coverage against complete-case error
fig06_repeat.pdf      baseline-corrected repeatability, large deviations by class, and an example atlas
fig08_quality.pdf     quality indices against measurability, complete-case error and failure-penalised loss"""
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
P = fs.PRODUCTS
RNG = np.random.default_rng(37)
c = pd.read_csv(T / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")

# ------------------------------------------------------------------ Fig. 5 conditional error
cb = pd.read_csv(T / "crossing_balanced.csv").set_index("product")
cbv = pd.read_csv(T / "crossing_balanced_values.csv", index_col=0)
pr = pd.read_csv(T / "paired_vs_pan.csv").set_index("product")
par = pd.read_csv(T / "pareto.csv").set_index("product")
meas = pd.read_csv(T / "measurability.csv").set_index("product")
fig, axes = plt.subplots(2, 2, figsize=(fs.COL2, 0.80 * fs.COL2))
ax = axes[0, 0]
for i, p in enumerate(P):
    v = cbv[p].dropna()
    ax.scatter(v, i + RNG.uniform(-0.18, 0.18, len(v)), s=7, color=fs.COLOR[p], alpha=0.35, lw=0)
    r = cb.loc[p]
    ax.plot([r.lo, r.hi], [i, i], color="k", lw=1.4)
    ax.plot(r.med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5)
ax.set_xscale("log")
ax.set_xlim(0.02, 10)
ax.set_yticks(range(len(P)))
ax.set_yticklabels([fs.LABEL[p] for p in P])
ax.invert_yaxis()
ax.set_xlabel("Crossing median |error| (m), k = 1-3")
ax.set_title(f"Complete crossings (n = {int(cb.loc['PAN'].n_crossings)})")
ax.xaxis.grid(True)
fs.panel_label(ax, "(a)")
ax = axes[0, 1]
okk = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
common = set(okk.index[okk[P].all(axis=1)])
s = c[(c.k > 0) & c.ok]
s = s[[(u, k) in common for u, k in zip(s.unit, s.k)]]
for p in P:
    v = np.sort(s[s["product"] == p].ae.values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=fs.COLOR[p], lw=1.1, label=fs.LABEL[p])
ax.set_xscale("log")
ax.set_xlim(0.02, 10)
ax.set_xlabel("|error| (m)")
ax.set_ylabel("Cumulative share")
ax.set_title(f"Complete cases ({len(common)} cases, {len({u for u, _ in common})} crossings)")
ax.grid(True)
ax.legend(fontsize=7, loc="upper left", ncol=1)
fs.panel_label(ax, "(b)")
ax = axes[1, 0]
order = P[1:]
ax.axvspan(-0.25, 0.25, color="#d9d8d4", alpha=0.5, lw=0)
for i, p in enumerate(order):
    r = pr.loc[p]
    ax.plot([r.lo, r.hi], [i, i], color=fs.COLOR[p], lw=1.6)
    ax.plot(r.med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5)
ax.axvline(0, color=fs.INK2, lw=0.7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{fs.LABEL[p]} ({int(pr.loc[p].n_crossings)})" for p in order])
ax.invert_yaxis()
ax.set_xlabel("Paired difference in |error| vs PAN (m)")
ax.set_xlim(-0.5, 0.5)
ax.xaxis.grid(True)
fs.panel_label(ax, "(c)")
ax = axes[1, 1]
for p in P:
    r = par.loc[p]
    m = meas.loc[p]
    ax.errorbar(r.success_pct, r.common_med, xerr=[[r.success_pct - m.lo_x], [m.hi_x - r.success_pct]], fmt="none",
                ecolor=fs.COLOR[p], lw=0.8, alpha=0.6)
    ax.plot(r.success_pct, r.common_med, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5, ls="")
    dx, dy = {"ZPNN": (6, -12), "BDSD_PC": (6, 5), "MTF_GLP_HPM": (-20, -14), "PAN": (6, 4)}.get(p, (5, 4))
    ax.annotate(fs.LABEL[p], (r.success_pct, r.common_med), xytext=(dx, dy), textcoords="offset points", fontsize=7.5)
ax.set_xlabel("Measured attempts (%)")
ax.set_ylabel("Complete-case median |error| (m)")
ax.grid(True)
fs.panel_label(ax, "(d)")
fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.08, wspace=0.42, hspace=0.38)
fig.savefig(OUT / "fig05_errors.pdf")
fig.savefig(OUT / "fig05_errors.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Fig. 6 repeatability and deviations
ev = pd.read_csv(T / "deviation_events.csv")
dcls = pd.read_csv(T / "deviation_classes.csv").set_index("product")


def stretch(a):
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def draw(ax, j, marker, product, title):
    jm = jobs.loc[j]
    m = mk[(mk.job == j) & (mk.marker == marker)].iloc[0]
    res = sio.loadmat(ROOT / f"data/processed/fused/{j}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
    names = [str(n) for n in res["NAMES"]]
    img = np.load(ROOT / f"data/processed/chips/{j}_vis.npy") if product == "VIS" else res["L"][:, :, names.index(product)]
    n = img.shape[0]
    ax.imshow(stretch(img), cmap="gray", extent=[jm.x0, jm.x0 + n * jm.pan_res, jm.y0 - n * jm.pan_res, jm.y0])
    row = c[(c.job == j) & (c.marker == marker) & (c["product"] == product)].iloc[0]
    p0, s_, nn = np.array([m.E, m.N]), unit(m.strike), unit(m.strike + 90)
    ax.plot(*np.c_[p0 - 45 * s_, p0 + 45 * s_], color="#D55E00", lw=0.8, ls="--")
    for side, sg in (("p", 1), ("m", -1)):
        if not np.isfinite(row[f"{side}_u0"]):
            continue
        d = unit(row[f"{side}_az"])
        if np.dot(d, nn) * sg < 0:
            d = -d
        a = p0 + row[f"{side}_u0"] * s_
        ax.plot([a[0], a[0] + 40 * d[0]], [a[1], a[1] + 40 * d[1]], color="#56B4E9" if row[f"{side}_ok"] else "#E69F00",
                lw=1.2)
    ax.set_xlim(m.E - 38, m.E + 38)
    ax.set_ylim(m.N - 38, m.N + 38)
    ax.set_xticks([])
    ax.set_yticks([])
    lab = f"error {row.err:+.2f} m" if row.ok else "failed"
    ax.text(0.03, 0.04, lab, transform=ax.transAxes, color="white", fontsize=7.5,
            bbox=dict(fc="black", alpha=0.55, pad=1.0, ec="none"))
    ax.set_title(title, fontsize=7.0)


fig = plt.figure(figsize=(fs.COL2, 0.95 * fs.COL2))
gs0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.9], hspace=0.32)
gt = gs0[0].subgridspec(1, 2, wspace=0.55)
gs = gs0[1].subgridspec(2, 4, hspace=0.28, wspace=0.1)
ax = fig.add_subplot(gt[0, 0])
sinc = c[(c.k > 0) & c.ok & c.inc_ae.notna()]
for p in P:
    v = np.sort(sinc[sinc["product"] == p].inc_ae.values) + 1e-4
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=fs.COLOR[p], lw=1.1, label=fs.LABEL[p])
ax.axvline(0.5, color=fs.INK2, lw=0.7, ls=":")
ax.set_xscale("log")
ax.set_xlim(1e-4, 10)
ax.legend(fontsize=7, ncol=2, loc="lower right", handlelength=1.2)
ax.set_xlabel("Baseline-corrected |error| (m)")
ax.set_ylabel("Cumulative share")
ax.grid(True)
fs.panel_label(ax, "(a)")
ax = fig.add_subplot(gt[0, 1])
it = pd.read_csv(T / "inc_tail.csv").set_index("product").reindex(P)
y = np.arange(len(P))
tot = dcls["measured_pairs"]
a1 = dcls["different edge pair"] / tot * 100
a2 = (dcls["events"] - dcls["different edge pair"]) / tot * 100
ax.barh(y, a1, color="#0072B2", height=0.65, label="different edge pair")
ax.barh(y, a2, left=a1, color="#9ec9e2", height=0.65, label="same pair, line change")
ax.errorbar(it.share_gt05, y, xerr=[it.share_gt05 - it.lo, it.hi - it.share_gt05], fmt="none", ecolor="k", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([fs.LABEL[p] for p in P])
ax.invert_yaxis()
ax.set_xlabel("Deviations > 0.5 m (% of measured pairs)")
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
ax.xaxis.grid(True)
fs.panel_label(ax, "(b)")
# atlas: rule-selected examples
e1 = ev[ev.cls == "different edge pair"].sort_values("inc_ae").iloc[len(ev[ev.cls == "different edge pair"]) // 2]
e2 = ev[ev.cls == "same pair, line position change"].sort_values("inc_ae").iloc[-1]
fails = c[(c.k == 2) & ~c.ok]
f1 = fails[fails.failure == "low contrast"].iloc[0]
f2 = fails[fails.failure == "edge not tracked"].iloc[0]
exs = [(e1.unit, e1["product"], int(e1.k), "(c) other edge pair"),
       (e2.unit, e2["product"], int(e2.k), "(d) line change"),
       (f1.unit, f1["product"], 2, "(e) low contrast"), (f2.unit, f2["product"], 2, "(f) not tracked")]
for i, (u, p, k, lab) in enumerate(exs):
    chip, marker = u.split("/")
    for r_, kk in enumerate((0, k)):
        ax = fig.add_subplot(gs[r_, i])
        draw(ax, f"{chip}_k{kk}", marker, p, f"{fs.LABEL[p]}, k = {kk}")
        if r_ == 0:
            ax.text(0.0, 1.17, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.03)
fig.savefig(OUT / "fig06_repeat.pdf")
fig.savefig(OUT / "fig06_repeat.png", dpi=300)
plt.close(fig)
pd.DataFrame(exs, columns=["unit", "product", "k", "label"]).to_csv(T / "fig6_examples.csv", index=False)

# ------------------------------------------------------------------ Fig. 8 quality-task associations
qa = pd.read_csv(T / "quality_task_models.csv").set_index("index")
qe = pd.read_csv(T / "quality_task_ext.csv").set_index("index")
NAME = {"Q2n": "Q2$^n$, 8 bands", "SAM": "SAM, 8 bands", "ERGAS": "ERGAS, 8 bands", "SCC": "SCC, 8 bands",
        "Q2n_B7": "Q2$^n$, no Coastal", "ERGAS_B7": "ERGAS, no Coastal", "Q2n_RGB": "Q2$^n$, RGB",
        "SAM_RGB": "SAM, RGB", "ERGAS_RGB": "ERGAS, RGB", "SCC_RGB": "SCC, RGB", "SAM_road": "SAM, road corridor",
        "D_lambda": r"$D_\lambda^K$ (FR)", "D_s": r"$D_S$ (FR)", "HQNR": "HQNR (FR)",
        "edge_disp_m": "Edge displacement"}
idx = [i for i in NAME if i in qa.index]
fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 0.62 * fs.COL2), sharey=True)
spec = [(qa, "succ", "Acceptance rate", "(a)"), (qe, "cc", "log |error|, jointly accepted", "(b)"),
        (qe, "loss", "Penalised loss (m)", "(c)")]
for ax, (d, pre, xl, lab) in zip(axes, spec):
    for i, ind in enumerate(idx):
        r = d.loc[ind]
        sig = r[f"{pre}_p_holm"] < 0.05
        col = "#0072B2" if sig else "#8a8984"
        ax.plot([r[f"{pre}_lo"], r[f"{pre}_hi"]], [i, i], color=col, lw=1.3)
        ax.plot(r[f"{pre}_coef"], i, "o", ms=5.5, color=col, mfc=col if sig else "white", mew=1.2)
        if sig:
            ax.text(r[f"{pre}_hi"], i - 0.15, " *", color=col, fontsize=9, va="center")
    ax.axvline(0, color=fs.INK2, lw=0.7)
    ax.set_xlabel(xl)
    ax.xaxis.grid(True)
    fs.panel_label(ax, lab)
axes[0].set_yticks(range(len(idx)))
axes[0].set_yticklabels([NAME[i] for i in idx])
axes[0].invert_yaxis()
fig.subplots_adjust(left=0.22, right=0.98, top=0.94, bottom=0.12, wspace=0.12)
fig.savefig(OUT / "fig08_quality.pdf")
fig.savefig(OUT / "fig08_quality.png", dpi=300)
print("ok", exs)
