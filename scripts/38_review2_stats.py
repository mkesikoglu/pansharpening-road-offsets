"""Additional statistics for the second review (results/tables/v2):
deviation_classes.csv   automated classification of baseline-corrected deviations > 0.5 m
equivalence.csv         crossing-level paired differences vs PAN against practical margins
crossing_balanced.csv   one value per crossing: median |e| over k = 1-3, complete crossings
quality_task_ext.csv    quality indices against measurability, failure-penalised loss and complete-case error
registration_partial.csv  registration magnitude vs product-PAN difference, partial on CNR and off-nadir
attempt_flow.csv        attempted / measured / failed per product (consistency check)"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
O = T / "v2"
RNG = np.random.default_rng(29)
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
FU = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
c = pd.read_csv(O / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)


def cboot(df, fun, nb=2000):
    df = df.reset_index(drop=True)
    idx = df.groupby("unit").indices
    keys = list(idx)
    bs = [fun(df.iloc[np.concatenate([idx[keys[i]] for i in RNG.integers(len(keys), size=len(keys))])])
          for _ in range(nb)]
    return fun(df), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


# ------------------------------------------------------------------ attempt flow
fl = c.groupby("product").agg(attempted=("ok", "size"), measured=("ok", "sum"), crossings=("unit", "nunique"))
fl["failed"] = fl.attempted - fl.measured
assert (fl.attempted == fl.measured + fl.failed).all()
fl.reindex(P).to_csv(O / "attempt_flow.csv")

# ------------------------------------------------------------------ classification of large deviations
base = c[c.k == 0].set_index(["unit", "product"])
rows = []
for r in c[(c.k > 0) & c.ok & c.inc_ae.notna()].itertuples():
    b = base.loc[(r.unit, r.product)]
    dwp, dwm = abs(r.p_width - b.p_width), abs(r.m_width - b.m_width)
    tolp, tolm = max(1.0, 0.2 * b.p_width), max(1.0, 0.2 * b.m_width)
    pair_change = (dwp > tolp) or (dwm > tolm) or (r.p_polarity != b.p_polarity)
    dup = (r.p_u0 - b.p_u0) - r.H_true          # + side moves by H_true
    dum = r.m_u0 - b.m_u0                       # - side stays
    daz = max(abs(((r.p_az - b.p_az) + 90) % 180 - 90), abs(((r.m_az - b.m_az) + 90) % 180 - 90))
    if r.inc_ae <= 0.5:
        cls = "none"
    elif pair_change:
        cls = "different edge pair"
    elif daz > 2.0:
        cls = "same pair, azimuth change"
    else:
        cls = "same pair, line position change"
    rows.append(dict(unit=r.unit, product=r.product, k=r.k, inc_ae=r.inc_ae, cls=cls, dw_plus=dwp, dw_minus=dwm,
                     du_plus=dup, du_minus=dum, daz=daz))
dc = pd.DataFrame(rows)
dc.to_csv(O / "deviation_events.csv", index=False)
tab = dc[dc.cls != "none"].groupby(["product", "cls"]).size().unstack(fill_value=0).reindex(P).fillna(0).astype(int)
tab["events"] = tab.sum(axis=1)
tab["measured_pairs"] = dc.groupby("product").size().reindex(P)
tab.to_csv(O / "deviation_classes.csv")

# ------------------------------------------------------------------ equivalence against practical margins
pr = pd.read_csv(O / "paired_vs_pan.csv").set_index("product")
eq = []
for p in P[1:]:
    r = pr.loc[p]
    eq.append(dict(product=p, med=r.med, lo=r.lo, hi=r.hi,
                   within_025=bool(r.lo > -0.25 and r.hi < 0.25), within_050=bool(r.lo > -0.5 and r.hi < 0.5)))
pd.DataFrame(eq).to_csv(O / "equivalence.csv", index=False)

# ------------------------------------------------------------------ crossing-balanced conditional error
okk = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
comp = okk[P].all(axis=1)
cu = comp.groupby(level="unit").sum()
units_any = set(cu.index[cu > 0])
s = c[(c.k > 0) & c.ok & c.unit.isin(units_any)].copy()
s = s[[(u, k) in set(comp.index[comp]) for u, k in zip(s.unit, s.k)]]
per = s.groupby(["unit", "product"]).ae.median().unstack()[P]
cb = []
for p in P:
    v = per[p].dropna()
    bs = [np.median(v.values[RNG.integers(len(v), size=len(v))]) for _ in range(2000)]
    d = (per[p] - per["PAN"]).dropna()
    bd = [np.median(d.values[RNG.integers(len(d), size=len(d))]) for _ in range(2000)]
    cb.append(dict(product=p, n_crossings=len(v), med=v.median(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                   worst=v.max(), d_pan=d.median(), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5)))
pd.DataFrame(cb).to_csv(O / "crossing_balanced.csv", index=False)
per.to_csv(O / "crossing_balanced_values.csv")

# ------------------------------------------------------------------ quality indices vs further task outcomes
qq = pd.read_csv(O / "quality_task_data.csv")
loss = c[(c.k > 0) & c["product"].isin(FU)].assign(l2=lambda x: np.where(x.ok, x.ae, 2.0)).groupby(
    ["chip", "product"]).l2.mean().rename("loss2").reset_index()
cc_err = s[s["product"].isin(FU)].groupby(["chip", "product"]).ae.median().rename("cc_med").reset_index()
qq = qq.merge(loss, on=["chip", "product"], how="left").merge(cc_err, on=["chip", "product"], how="left")
IND = ["Q2n", "SAM", "ERGAS", "SCC", "SAM_road", "D_lambda", "D_s", "HQNR", "Q2n_RGB", "SAM_RGB", "ERGAS_RGB",
       "SCC_RGB", "Q2n_B7", "ERGAS_B7", "edge_disp_m"]
qr = []
for ind in [i for i in IND if i in qq.columns]:
    z = qq.dropna(subset=[ind]).copy()
    z["Qz"] = (z[ind] - z[ind].mean()) / z[ind].std()
    rec = dict(index=ind)
    for out, lab in (("loss2", "loss"), ("cc_med", "cc")):
        zz = z.dropna(subset=[out]).copy()
        if out == "cc_med":
            zz["y"] = np.log(zz[out] + 0.01)
        else:
            zz["y"] = zz[out]
        try:
            f = smf.ols("y ~ Qz + C(product)", zz).fit(cov_type="cluster",
                                                     cov_kwds={"groups": zz.chip.astype("category").cat.codes})
            rec.update({f"{lab}_n": len(zz), f"{lab}_chips": zz.chip.nunique(), f"{lab}_coef": f.params["Qz"],
                        f"{lab}_lo": f.conf_int().loc["Qz", 0], f"{lab}_hi": f.conf_int().loc["Qz", 1],
                        f"{lab}_p": f.pvalues["Qz"]})
        except Exception:
            pass
    qr.append(rec)
qr = pd.DataFrame(qr)
for lab in ("loss", "cc"):
    p_ = qr[f"{lab}_p"].fillna(1).values
    o = np.argsort(p_)
    adj, run = np.empty_like(p_), 0.0
    for rnk, i in enumerate(o):
        run = max(run, (len(p_) - rnk) * p_[i])
        adj[i] = min(run, 1.0)
    qr[f"{lab}_p_holm"] = adj
qr.to_csv(O / "quality_task_ext.csv", index=False)

# ------------------------------------------------------------------ registration: partial association
reg = pd.read_csv(T / "registration.csv")
rr = reg[reg.kind == "controlled"].copy()
rr["unit"] = rr.job.str.replace("_k0", "", regex=False) + "/" + rr.marker
ae = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ae")
dd = ae.sub(ae["PAN"], axis=0).groupby(level="unit").mean()
cov = c[(c.k == 0) & (c["product"] == "PAN")].set_index("unit")
cnr = np.fmin(cov.p_cnr, cov.m_cnr)
rows = []
for p in P[1:]:
    df = pd.DataFrame(dict(x=rr.set_index("unit").mag_local_m, y=dd[p], cnr=cnr, on=cov.off_nadir)).dropna()
    rk = df.rank()
    res = {}
    for v in ("x", "y"):
        res[v] = smf.ols(f"{v} ~ cnr + on", rk).fit().resid
    rho0 = stats.spearmanr(df.x, df.y)
    rho1 = stats.pearsonr(res["x"], res["y"])
    rows.append(dict(product=p, n=len(df), rho=rho0.statistic, p=rho0.pvalue, partial_rho=rho1.statistic,
                     partial_p=rho1.pvalue))
rg = pd.DataFrame(rows)
p_ = rg.partial_p.values
o = np.argsort(p_)
adj, run = np.empty_like(p_), 0.0
for rnk, i in enumerate(o):
    run = max(run, (len(p_) - rnk) * p_[i])
    adj[i] = min(run, 1.0)
rg["partial_p_holm"] = adj
rg.to_csv(O / "registration_partial.csv", index=False)
print(fl.reindex(P))
print(tab)
print(pd.DataFrame(eq))
print(pd.DataFrame(cb).round(3))
print(qr.round(3).to_string())
print(rg.round(3))
