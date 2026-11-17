"""LaTeX table bodies for the revised manuscript (manuscript/tables/*.tex)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
V = T / "v2"
O = ROOT / "manuscript/tables"
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
LAB = {"PAN": "PAN", "EXP": "EXP", "GSA": "GSA", "BDSD_PC": "BDSD-PC", "MTF_GLP_HPM": "MTF-GLP-HPM",
       "AWLP": "AWLP", "ZPNN": "Z-PNN", "VIS": "Vendor RGB", "MTF_GLP_HPM_stab": "MTF-GLP-HPM$^{*}$"}


def w(name, lines):
    (O / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(x, d=2):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci(e, lo, hi, d=2):
    return f"{f(e, d)} [{f(lo, d)}, {f(hi, d)}]"


def ph(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")


# ------------------------------------------------------------------ T1 acquisitions
acq = pd.read_csv(ROOT / "data/processed/manifest_acquisitions.csv")
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
nch = jobs[(jobs.kind == "controlled") & (jobs.k == 0)].groupby("catalog_id").size()
nfo = mk[mk.kind == "field"].groupby("catalog_id").ID.nunique()
acq["t"] = pd.to_datetime(acq.datetime_utc.str.rstrip("Z"))
rows = []
for r in acq.sort_values("t").itertuples():
    rows.append(f"{r.t:%d %b %Y} {r.t:%H:%M} & {'WV-3' if r.platform == 'WV03' else 'WV-2'} & \\texttt{{{r.catalog_id}}} & "
                f"{r.native_gsd_m:.2f} & {r.pan_grid_m:.2f} & {r.ms_grid_m:.2f} & {r.off_nadir_deg:.1f} & "
                f"{r.view_azimuth_deg:.0f} & {r.sun_azimuth_deg:.0f} & {r.sun_elevation_deg:.1f} & "
                f"{int(nfo.get(r.catalog_id, 0))} & {int(nch.get(r.catalog_id, 0))} \\\\")
w("tab1_acquisitions.tex", rows)

# ------------------------------------------------------------------ T4 quality, median [IQR]
qe = pd.read_csv(T / "quality_ext.csv")
ql = pd.read_csv(T / "quality_long.csv")
ql = ql[ql.kind == "controlled"] if "kind" in ql else ql
qe = qe[qe.job.isin(ql.job.unique())] if len(set(qe.job) & set(ql.job)) else qe
cols = []


def mi(s, d):
    s = s.dropna()
    return "--" if s.empty else f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}--{s.quantile(.75):.{d}f}]"


rows = []
prods = [p for p in P if p not in ("PAN", "VIS")] + ["MTF_GLP_HPM_stab"]
ed = pd.read_csv(V / "edge_displacement.csv") if (V / "edge_displacement.csv").exists() else None
for p in prods:
    e8 = qe[(qe["product"] == p) & (qe.bandset == "B8")]
    e7 = qe[(qe["product"] == p) & (qe.bandset == "B7")]
    eR = qe[(qe["product"] == p) & (qe.bandset == "RGB")]
    l = ql[ql["product"] == p]
    cells = [mi(e8.Q2n, 3), mi(e8.ERGAS, 1), mi(e7.ERGAS, 1), mi(eR.SAM, 2), mi(l.SAM_road, 2),
             mi(l.HQNR, 3)]
    if ed is not None:
        cells.append(mi(ed[ed["product"] == p].edge_disp_m, 2))
    rows.append(f"{LAB[p]} & " + " & ".join(cells) + " \\\\")
w("tab4_quality.tex", rows)

# ------------------------------------------------------------------ T5 failure-aware controlled results
me = pd.read_csv(V / "measurability.csv").set_index("product")
ac = pd.read_csv(V / "accuracy.csv").set_index("product")
cbt = pd.read_csv(V / "crossing_balanced.csv").set_index("product")
itl = pd.read_csv(V / "inc_tail.csv").set_index("product")
fl = pd.read_csv(V / "failure_loss.csv")
fl2 = fl[fl.penalty_m == 2.0].set_index("product")
rows = []
for p in P:
    m, a, cb, it, l = me.loc[p], ac.loc[p], cbt.loc[p], itl.loc[p], fl2.loc[p]
    rows.append(f"{LAB[p]} & {int(m.n_success)}/{int(m.n_total)} & {ci(m.success_pct, m.lo_x, m.hi_x, 1)} & "
                f"{int(a.n)} & {ci(a.med_ae, a.med_lo, a.med_hi)} & {ci(cb.med, cb.lo, cb.hi)} & {f(a.common_p90)} & "
                f"{f(a.inc_med, 3)} & {ci(it.share_gt05, it.lo, it.hi, 1)} & {ci(l.loss, l.lo, l.hi)} \\\\")
w("tab5_controlled.tex", rows)  # full version, Table S14
rows = []
for p in P:
    m, cb, it, l = me.loc[p], cbt.loc[p], itl.loc[p], fl2.loc[p]
    rows.append(f"{LAB[p]} & {int(m.n_success)}/{int(m.n_total)} ({m.success_pct:.1f}) & {ci(cb.med, cb.lo, cb.hi)} & "
                f"{ci(it.share_gt05, it.lo, it.hi, 1)} & {ci(l.loss, l.lo, l.hi)} \\\\")
w("tab5_main.tex", rows)

# ------------------------------------------------------------------ T6 model effects
lm = pd.read_csv(V / "lmm_eps.csv")
prim = lm[(lm.model == "primary") & np.isclose(lm.eps, 0.01)].set_index("product")
acqfe = lm[(lm.model == "acquisition fixed effect") & np.isclose(lm.eps, 0.01)].set_index("product")
pr = pd.read_csv(V / "paired_vs_pan.csv").set_index("product")
sh = pd.read_csv(V / "shift_vs_pan.csv").set_index("product")
rows = []
for p in P[1:]:
    m, q, r, s = me.loc[p], prim.loc[p], pr.loc[p], sh.loc[p]
    rng = lm[(lm.model == "primary") & (lm["product"] == p)].p_holm
    eqm = "0.25" if (r.lo > -0.25 and r.hi < 0.25) else ("0.5" if (r.lo > -0.5 and r.hi < 0.5) else "--")
    rows.append(f"{LAB[p]} & {ci(m.OR, m.lo_y, m.hi_y)} & {ph(m.p_holm)} & {ci(q.ratio, q.lo, q.hi)} & {ph(q.p_holm)} & "
                f"{ph(rng.min())}--{ph(rng.max())} & {int(r.n_crossings)} & "
                f"{ci(r.med, r.lo, r.hi)} & {ph(r.p_holm_wilcoxon)} & {eqm} & {ci(s.med_shift, s.lo, s.hi)} \\\\")
w("tab6_models.tex", rows)

# ------------------------------------------------------------------ T3 sampling flow
sf = V / "sampling_flow_counts.csv"
if sf.exists():
    s = pd.read_csv(sf, index_col=0).iloc[:, 0]
    n_x = int(s.x_not_straight + s.x_low_angle + s.x_other_road + s.x_vegetated + s.x_pass)
    items = [("Random points drawn on candidate roads", s.c_draw),
             ("\\quad outside the footprint or within 400~m of the rupture or a field site", s.c_outside_or_excluded),
             ("\\quad within 700~m of an accepted chip", s.c_too_close),
             ("\\quad less than 98\\% valid pixels", s.c_invalid_data),
             ("\\quad no admissible crossing", s.c_no_crossing),
             ("Accepted chips", s.c_accepted),
             ("Road--fault intersections examined", n_x),
             ("\\quad road not straight to 1~m over 50~m", s.x_not_straight),
             ("\\quad road--fault angle below 30$^\\circ$", s.x_low_angle),
             ("\\quad another road within 25~m", s.x_other_road),
             ("\\quad mean NDVI above 0.35", s.x_vegetated),
             ("Admissible crossings", s.x_pass),
             ("Crossings in accepted chips (one per chip)", s.c_accepted),
             ("Attempts per product (57 crossings $\\times$ 4 offset levels)", 228),
             ("\\quad vendor RGB (no visual product on 11 March)", 224)]
    fitems = [("Field observations in the release", 68), ("Eligible linear markers with $H\\geq1$~m", 21),
              ("\\quad periodic markers (onion rows)", 3), ("\\quad walls not resolved on PAN", 2),
              ("Observations used", 16), ("Markers used", 12), ("Field chips (site cluster $\\times$ acquisition)", 23),
              ("Observation--acquisition combinations (attempts per product)", 36)]
    lines = ["\\multicolumn{2}{l}{\\textit{Controlled experiment}} \\\\"] + \
        [f"{a_} & {int(b_)} \\\\" for a_, b_ in items] + ["\\midrule", "\\multicolumn{2}{l}{\\textit{Field case study}} \\\\"] + \
        [f"{a_} & {int(b_)} \\\\" for a_, b_ in fitems]
    w("tab3_sampling.tex", lines)

# ------------------------------------------------------------------ T8 reduced-scale experiment
rl = V / "rr_long.csv"
if rl.exists():
    d = pd.read_csv(rl)
    d["ok"] = d.ok.astype(bool)
    rng = np.random.default_rng(17)
    rows = []
    for p in ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]:
        for m in (0, 1, 2):
            g = d[(d["product"] == p) & (d.m == m)]
            s = g[g.ok & g.delta.notna()].assign(ad=lambda x: x.delta.abs())
            idx = s.groupby("unit").indices
            keys = list(idx)
            bs = [s.ad.values[np.concatenate([idx[keys[i]] for i in rng.integers(len(keys), size=len(keys))])]
                  for _ in range(1000)]
            med = [np.median(x) for x in bs]
            rows.append(f"{LAB.get(p, 'Z-PNN, pretrained')} & {m}/4 & {int(g.ok.sum())}/{len(g)} & {g.ok.mean() * 100:.1f} & "
                        f"{ci(s.ad.median(), np.percentile(med, 2.5), np.percentile(med, 97.5))} & "
                        f"{f(s.ad.quantile(.9))} & {f((s.ad > 0.5).mean() * 100, 1)} \\\\")
    w("tab8_rr.tex", rows)
print("tables written")
