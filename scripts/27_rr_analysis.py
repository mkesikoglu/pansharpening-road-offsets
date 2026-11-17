"""Analysis of the reduced-scale fractional-phase and misregistration experiment.
REF is the luminance of the latent (original) MS image at the reduced PAN grid, an
ideal fused product; delta = H(product) - H(REF) isolates the fusion-induced part
of the offset error from road geometry. Writes results/tables/v2/rr_*.csv and
the edge-displacement index used in the quality-task analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "results/tables/v2"
O.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(5)
r = pd.read_csv(ROOT / "results/tables/rr_fraction_measurements.csv")
r["unit"] = r.chip + "/" + r.marker
r["ok"] = r.ok.astype(bool)
r["phase"] = (r.k % 4) / 4.0
r["err"] = r.H - r.H_true
r["ae"] = r.err.abs()
ref = r[r["product"] == "REF"].set_index(["unit", "k", "m"])
r["H_ref_img"] = [ref.H.get((u, k, m), np.nan) if ref.ok.get((u, k, m), False) else np.nan
                  for u, k, m in zip(r.unit, r.k, r.m)]
r["delta"] = r.H - r.H_ref_img
base = r[r.k == 0].set_index(["unit", "product", "m"])
r["H0"] = [base.H.get((u, p, m), np.nan) if base.ok.get((u, p, m), False) else np.nan
           for u, p, m in zip(r.unit, r["product"], r.m)]
r["inc_ae"] = ((r.H - r.H0) - r.H_true).abs()
r.to_csv(O / "rr_long.csv", index=False)


def cmed(df, col):
    g = {u: x[col].values for u, x in df.groupby("unit")}
    keys = list(g)
    est = np.median(np.concatenate(list(g.values()))) if keys else np.nan
    bs = [np.median(np.concatenate([g[k] for k in RNG.choice(keys, len(keys))])) for _ in range(2000)] if keys else [np.nan]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


ORDER = ["REF", "PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]
rows = []
for p in ORDER:
    for m in (0, 1, 2):
        g = r[(r["product"] == p) & (r.m == m)]
        s = g[g.ok & (g.k > 0)]
        dd = g[g.ok & g.delta.notna()]
        a = cmed(s, "ae")
        dl = cmed(dd.assign(ad=dd.delta.abs()), "ad") if p != "REF" else (np.nan,) * 3
        rows.append(dict(product=p, m=m, success_pct=g.ok.mean() * 100, n_success=int(g.ok.sum()), n_total=len(g),
                         med_ae=a[0], ae_lo=a[1], ae_hi=a[2], inc_med=s.inc_ae.median(),
                         med_abs_delta=dl[0], delta_lo=dl[1], delta_hi=dl[2], med_delta=dd.delta.median(),
                         n_delta=len(dd)))
summ = pd.DataFrame(rows)
summ.to_csv(O / "rr_summary.csv", index=False)
# by MS phase (m = 0)
ph = r[(r.m == 0) & r.ok & r.delta.notna() & (r["product"] != "REF")].groupby(["product", "phase"]).delta.agg(
    med_delta="median", med_abs=lambda v: np.median(np.abs(v)), n="size").reset_index()
ph.to_csv(O / "rr_by_phase.csv", index=False)
# Friedman-type check of a phase effect on |delta| per product (crossing level)
pt = []
for p in [x for x in ORDER if x != "REF"]:
    g = r[(r["product"] == p) & (r.m == 0) & r.ok & r.delta.notna()]
    w = g.pivot_table(index="unit", columns="phase", values="delta", aggfunc=lambda v: np.abs(v).mean()).dropna()
    if len(w) > 4 and w.shape[1] == 4:
        st = stats.friedmanchisquare(*[w[c] for c in w.columns])
        pt.append(dict(product=p, n=len(w), chi2=st.statistic, p=st.pvalue))
pd.DataFrame(pt).to_csv(O / "rr_phase_test.csv", index=False)
# misregistration effect on |delta|: crossing-level paired difference m=2 vs m=0
mr = []
for p in [x for x in ORDER if x != "REF"]:
    g = r[(r["product"] == p) & r.ok & r.delta.notna()]
    w = g.pivot_table(index=["unit", "k"], columns="m", values="delta", aggfunc=lambda v: float(np.abs(v).mean()))
    for mm in (1, 2):
        if mm not in w.columns or 0 not in w.columns:
            continue
        x = (w[mm] - w[0]).dropna().groupby(level="unit").mean()
        if len(x) > 5:
            t = stats.wilcoxon(x)
            mr.append(dict(product=p, m=mm, n=len(x), med_change=x.median(), p=t.pvalue))
pd.DataFrame(mr).to_csv(O / "rr_misreg_test.csv", index=False)
# edge displacement index: mean |u0(product) - u0(REF)| over the two sides, k = 0, m = 0
e = r[(r.k == 0) & (r.m == 0)].set_index(["unit", "product"])
ed = []
for (u, p), row in e.iterrows():
    if p == "REF" or (u, "REF") not in e.index:
        continue
    rf = e.loc[(u, "REF")]
    vals = []
    for s in ("p", "m"):
        if row[f"{s}_ok"] and rf[f"{s}_ok"]:
            vals.append(abs(row[f"{s}_u0"] - rf[f"{s}_u0"]))
    if vals:
        ed.append(dict(job=u.split("/")[0] + "_k0", product=p, edge_disp_m=float(np.mean(vals))))
edf = pd.DataFrame(ed).groupby(["job", "product"]).edge_disp_m.mean().reset_index()
edf.to_csv(O / "edge_displacement.csv", index=False)

# offsets relative to PAN and baseline-corrected errors at fractional phase
pan = r[r["product"] == "PAN"].set_index(["unit", "k", "m"])
r["d_pan"] = [abs(h - pan.H.get((u, k, m), np.nan)) if o and pan.ok.get((u, k, m), False) else np.nan
              for u, k, m, h, o in zip(r.unit, r.k, r.m, r.H, r.ok)]
s = r[(r.k > 0) & r.ok & r.inc_ae.notna()]
inc = s.groupby(["product", "m"]).inc_ae.agg(inc_med="median", inc_gt05=lambda v: (v > 0.5).mean() * 100,
                                            inc_p90=lambda v: v.quantile(.9), n="size").reset_index()
inc = inc.merge(r.groupby(["product", "m"]).d_pan.median().rename("med_abs_vs_pan").reset_index(), on=["product", "m"])
inc.to_csv(O / "rr_inc_vs_pan.csv", index=False)
s[s.m == 0].groupby(["product", "phase"]).inc_ae.median().unstack().to_csv(O / "rr_inc_by_phase.csv")
print(inc.round(3).to_string())
K = dict(rr_crossings=int(r.unit.nunique()), rr_jobs=int(r.groupby(["chip", "k", "m"]).ngroups))
(O / "key_numbers_rr.json").write_text(json.dumps(K, indent=1), encoding="utf-8")
print(summ.round(3).to_string())
print(ph.round(3).to_string())
print(pd.DataFrame(pt).round(4).to_string())
print(pd.DataFrame(mr).round(4).to_string())
print(edf.groupby("product").edge_disp_m.median().round(3))
