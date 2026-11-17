"""Main-text result figures of the third version:
fig04_success.pdf  measurement success with intervals, failure reasons and rule-selected failure examples
fig05_error.pdf    crossing-balanced error, paired difference to PAN, baseline-corrected repeatability and
                   large deviations classified by the edge-pair-change proxy
figS_heatmap.pdf, figS_pareto.pdf  supplementary versions of the removed panels"""
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
RNG = np.random.default_rng(41)
c = pd.read_csv(T / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
SHORT = dict(PAN="PAN", EXP="EXP", GSA="GSA", BDSD_PC="BDSD-PC", MTF_GLP_HPM="MTF-GLP-HPM", AWLP="AWLP",
             ZPNN="Z-PNN", VIS="Vendor RGB")


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
                lw=1.3)
    ax.set_xlim(m.E - 38, m.E + 38)
    ax.set_ylim(m.N - 38, m.N + 38)
    ax.set_xticks([])
    ax.set_yticks([])
    lab = f"error {row.err:+.2f} m" if row.ok else "failed"
    ax.text(0.03, 0.04, lab, transform=ax.transAxes, color="white", fontsize=7.5,
            bbox=dict(fc="black", alpha=0.6, pad=1.0, ec="none"))
    ax.set_title(title, fontsize=7.5)


# ------------------------------------------------------------------ Fig. 4 measurement success and failures
meas = pd.read_csv(T / "measurability.csv").set_index("product")
fr = pd.read_csv(T / "failure_reasons.csv").set_index("product").reindex(P)
ev = pd.read_csv(T / "deviation_events.csv")
fig = plt.figure(figsize=(fs.COL2, 0.95 * fs.COL2))
gs0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75], hspace=0.3)
gt = gs0[0].subgridspec(1, 2, wspace=0.08, width_ratios=[1.2, 1.0])
gb = gs0[1].subgridspec(2, 4, hspace=0.3, wspace=0.08)
ax = fig.add_subplot(gt[0, 0])
for i, p in enumerate(P):
    r = meas.loc[p]
    ax.plot([r.lo_x, r.hi_x], [i, i], color=fs.COLOR[p], lw=1.8)
    ax.plot(r.success_pct, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5, ls="")
    ax.text(r.hi_x + 1.5, i, f"{int(r.n_success)}/{int(r.n_total)}", va="center", fontsize=7.5, color=fs.INK2)
ax.set_yticks(range(len(P)))
ax.set_yticklabels([SHORT[p] for p in P])
ax.invert_yaxis()
ax.set_xlim(30, 90)
ax.set_xlabel("Accepted estimates (%), 95% interval")
ax.xaxis.grid(True)
fs.panel_label(ax, "(a)")
ax = fig.add_subplot(gt[0, 1], sharey=fig.axes[0])
tot = fr.sum(axis=1).values
left = np.zeros(len(P))
for cat, col in (("low contrast", "#E69F00"), ("edge not tracked", "#56B4E9")):
    v = fr[cat].values / tot * 100
    ax.barh(range(len(P)), v, left=left, color=col, height=0.62, edgecolor="white", lw=0.4,
            label={"low contrast": "CNR below threshold", "edge not tracked": "other criteria"}[cat])
    left += v
plt.setp(ax.get_yticklabels(), visible=False)
ax.set_xlim(0, 75)
ax.set_xlabel("Failed attempts (%)")
ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.xaxis.grid(True)
fs.panel_label(ax, "(b)")
e1 = ev[ev.cls == "different edge pair"].sort_values("inc_ae").iloc[len(ev[ev.cls == "different edge pair"]) // 2]
e2 = ev[ev.cls == "same pair, line position change"].sort_values("inc_ae").iloc[-1]
fails = c[(c.k == 2) & ~c.ok]
f1 = fails[fails.failure == "low contrast"].iloc[0]
f2 = fails[fails.failure == "edge not tracked"].iloc[0]
exs = [(e1.unit, e1["product"], int(e1.k), "(c) edge-pair change"),
       (e2.unit, e2["product"], int(e2.k), "(d) not flagged"),
       (f1.unit, f1["product"], 2, "(e) low contrast"), (f2.unit, f2["product"], 2, "(f) other criteria")]
for i, (u, p, k, lab) in enumerate(exs):
    chip, marker = u.split("/")
    for r_, kk in enumerate((0, k)):
        ax = fig.add_subplot(gb[r_, i])
        draw(ax, f"{chip}_k{kk}", marker, p, f"{SHORT[p]}, k = {kk}")
        if r_ == 0:
            ax.text(0.0, 1.2, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.03)
fig.savefig(OUT / "fig04_success.pdf")
fig.savefig(OUT / "fig04_success.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Fig. 5 conditional error and repeatability
cb = pd.read_csv(T / "crossing_balanced.csv").set_index("product")
cbv = pd.read_csv(T / "crossing_balanced_values.csv", index_col=0)
pr = pd.read_csv(T / "paired_vs_pan.csv").set_index("product")
it = pd.read_csv(T / "inc_tail.csv").set_index("product").reindex(P)
dcls = pd.read_csv(T / "deviation_classes.csv").set_index("product")
fig, axs = plt.subplots(2, 2, figsize=(fs.COL2, 0.82 * fs.COL2))
ax = axs[0, 0]
for i, p in enumerate(P):
    v = cbv[p].dropna()
    ax.scatter(v, i + RNG.uniform(-0.18, 0.18, len(v)), s=9, color=fs.COLOR[p], alpha=0.4, lw=0)
    r = cb.loc[p]
    ax.plot([r.lo, r.hi], [i, i], color="k", lw=1.4)
    ax.plot(r.med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5, ls="")
ax.set_xscale("log")
ax.set_xlim(0.02, 10)
ax.set_yticks(range(len(P)))
ax.set_yticklabels([SHORT[p] for p in P])
ax.invert_yaxis()
ax.set_xlabel("Crossing median |error| (m)")
ax.set_title(f"{int(cb.loc['PAN'].n_crossings)} jointly accepted crossings")
ax.xaxis.grid(True)
fs.panel_label(ax, "(a)")
ax = axs[0, 1]
order = P[1:]
ax.axvspan(-0.5, 0.5, color="#ecebe7", lw=0)
for x in (-0.25, 0.25):
    ax.axvline(x, color=fs.INK2, lw=0.6, ls=":")
for i, p in enumerate(order):
    r = pr.loc[p]
    ax.plot([r.lo, r.hi], [i, i], color=fs.COLOR[p], lw=1.8)
    ax.plot(r.med, i, marker=fs.MARKER[p], color=fs.COLOR[p], ms=7, mec="k", mew=0.5, ls="")
ax.axvline(0, color=fs.INK2, lw=0.7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{SHORT[p]} ({int(pr.loc[p].n_crossings)})" for p in order])
ax.invert_yaxis()
ax.set_xlim(-0.6, 0.6)
ax.set_xlabel("Paired difference in |error| vs PAN (m)")
ax.set_title("paired, per crossing")
ax.xaxis.grid(True)
fs.panel_label(ax, "(b)")
ax = axs[1, 0]
sinc = c[(c.k > 0) & c.ok & c.inc_ae.notna()]
for p in P:
    v = np.sort(sinc[sinc["product"] == p].inc_ae.values) + 1e-4
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=fs.COLOR[p], lw=1.2, label=SHORT[p])
ax.axvline(0.5, color=fs.INK2, lw=0.7, ls=":")
ax.set_xscale("log")
ax.set_xlim(1e-4, 10)
ax.set_xlabel("Baseline-corrected |error| (m)")
ax.set_ylabel("Cumulative share")
ax.grid(True)
ax.legend(fontsize=7, ncol=2, loc="lower right", handlelength=1.2, frameon=True, framealpha=0.95, edgecolor="none")
fs.panel_label(ax, "(c)")
ax = axs[1, 1]
y = np.arange(len(P))
tt = dcls["measured_pairs"]
a1 = dcls["different edge pair"] / tt * 100
a2 = (dcls["events"] - dcls["different edge pair"]) / tt * 100
ax.barh(y, a1, color="#0072B2", height=0.62, label="edge-pair change")
ax.barh(y, a2, left=a1, color="#a6cbe3", height=0.62, label="not flagged")
ax.errorbar(it.share_gt05, y, xerr=[it.share_gt05 - it.lo, it.hi - it.share_gt05], fmt="none", ecolor="k", lw=0.9)
ax.set_yticks(y)
ax.set_yticklabels([SHORT[p] for p in P])
ax.invert_yaxis()
ax.set_xlabel("Deviations > 0.5 m (% of paired estimates)")
ax.set_xlim(0, 55)
ax.legend(fontsize=7.5, title="Automated indicator", title_fontsize=7.5, loc="lower right", frameon=True, framealpha=0.95, edgecolor="none")
ax.xaxis.grid(True)
fs.panel_label(ax, "(d)")
fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.08, wspace=0.62, hspace=0.42)
fig.savefig(OUT / "fig05_error.pdf")
fig.savefig(OUT / "fig05_error.png", dpi=300)
plt.close(fig)
print("ok", exs)
