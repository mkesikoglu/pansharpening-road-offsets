"""Statistical analysis of the controlled-dislocation and field-referenced experiments
and of the fusion quality indices. Writes CSV tables to results/tables."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
RNG = np.random.default_rng(1)
NB = 5000
PRODUCTS = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
FUSED = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]

jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ms = pd.read_csv(T / "measurements.csv")
ACQ = {"10300500D9F8D200": "WV-2 08 Feb", "10300100E291D100": "WV-2 11 Feb", "10300100E3154100": "WV-2 28 Feb",
       "1040010082B6D600": "WV-3 28 Feb", "10300100E383D900": "WV-2 11 Mar"}
d = ms.merge(mk, on=["job", "marker", "ID"]).merge(
    jobs[["job", "kind", "k", "chip", "date", "sensor", "off_nadir", "gsd", "pan_res", "ms_res", "cluster"]].assign(
        acq=jobs.catalog_id.map(ACQ)),
    on=["job", "kind"])


def cluster_boot(values, clusters, fun=np.median, nb=NB):
    """Percentile CI of fun(values) resampling clusters with replacement."""
    v, c = np.asarray(values), np.asarray(clusters)
    u = np.unique(c)
    idx = {k: np.nonzero(c == k)[0] for k in u}
    est = fun(v)
    bs = np.empty(nb)
    for b in range(nb):
        pick = RNG.choice(u, len(u), replace=True)
        bs[b] = fun(v[np.concatenate([idx[k] for k in pick])])
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ============================================================ controlled experiment
c = d[d.kind == "controlled"].copy()
c["unit"] = c.chip + "/" + c.marker          # crossing identifier
c["err"] = c.H - c.H_true
c["ae"] = c.err.abs()
c["ok"] = c.ok.astype(bool)
c.to_csv(T / "controlled_long.csv", index=False)

rows = []
for p in PRODUCTS:
    for kset, lab in ((c.k > 0, "k>0"), (c.k == 0, "k=0")):
        g = c[(c["product"] == p) & kset]
        if g.empty:
            continue
        v = g[g.ok]
        med, lo, hi = cluster_boot(v.ae.values, v.chip.values) if len(v) else (np.nan,) * 3
        rows.append(dict(product=p, set=lab, n=len(g), measurable=g.ok.mean() * 100,
                         bias=v.err.mean(), sd=v.err.std(), rmse=np.sqrt((v.err ** 2).mean()),
                         mae_median=med, mae_lo=lo, mae_hi=hi, mae_mean=v.ae.mean(),
                         p_gt_05=(v.ae > 0.5).mean() * 100, p_gt_1=(v.ae > 1.0).mean() * 100))
summ = pd.DataFrame(rows)
summ.to_csv(T / "controlled_summary.csv", index=False)

# paired differences against PAN (same crossing and offset, both measurable)
w = c[c.k > 0].pivot_table(index=["unit", "chip", "k"], columns="product", values="ae")
okw = c.pivot_table(index=["unit", "chip", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
pr = []
for p in [q for q in PRODUCTS if q != "PAN"]:
    m = (okw["PAN"].astype(bool) & okw[p].astype(bool)).reindex(w.index).fillna(False)
    dd = (w[p] - w["PAN"])[m.values].dropna()
    chipv = dd.index.get_level_values("chip")
    med, lo, hi = cluster_boot(dd.values, chipv)
    mean, mlo, mhi = cluster_boot(dd.values, chipv, fun=np.mean)
    wil = stats.wilcoxon(dd.values) if len(dd) > 5 else None
    pr.append(dict(product=p, n=len(dd), d_median=med, d_lo=lo, d_hi=hi, d_mean=mean, d_mean_lo=mlo,
                   d_mean_hi=mhi, frac_better=(dd < 0).mean() * 100, p_wilcoxon=wil.pvalue if wil else np.nan))
pr = pd.DataFrame(pr)
pr["p_holm"] = holm(pr.p_wilcoxon.fillna(1))
pr.to_csv(T / "controlled_paired_vs_pan.csv", index=False)

# linear mixed model on log absolute error: product fixed, chip random, crossing variance component
try:
    import statsmodels.formula.api as smf
    g = c[(c.k > 0) & c.ok].copy()
    g["lae"] = np.log(g.ae + 0.01)
    g["product"] = pd.Categorical(g["product"], categories=PRODUCTS)
    md = smf.mixedlm("lae ~ C(product) + C(k)", g, groups=g["chip"], vc_formula={"unit": "0 + C(unit)"})
    fit = md.fit(reml=True, method="lbfgs")
    ci = fit.conf_int()
    lm = []
    for nm in fit.params.index:
        if nm.startswith("C(product)"):
            lm.append(dict(term=nm.split("T.")[1].rstrip("]"), ratio=np.exp(fit.params[nm]),
                           lo=np.exp(ci.loc[nm, 0]), hi=np.exp(ci.loc[nm, 1]), p=fit.pvalues[nm]))
    lm = pd.DataFrame(lm)
    lm["p_holm"] = holm(lm.p.values)
    lm.to_csv(T / "controlled_lmm.csv", index=False)
    (T / "controlled_lmm.txt").write_text(str(fit.summary()))
except Exception as e:  # report but continue
    print("LMM failed:", e)

# by offset size and acquisition
c[c.ok & (c.k > 0)].groupby(["product", "k"]).ae.median().unstack().to_csv(T / "controlled_by_k.csv")
c[c.ok & (c.k > 0)].groupby(["product", "acq"]).ae.median().unstack().to_csv(T / "controlled_by_acq.csv")
c[c.ok & (c.k > 0)].groupby("acq").unit.nunique().to_csv(T / "controlled_units_by_acq.csv")

# offset change caused by the product: H(product) - H(PAN) on the same crossing and offset
hh = c.pivot_table(index=["unit", "chip", "k"], columns="product", values="H")
dl = []
for p in [q for q in PRODUCTS if q != "PAN"]:
    m_ = (okw["PAN"].astype(bool) & okw[p].astype(bool)).reindex(hh.index).fillna(False)
    dd_ = (hh[p] - hh["PAN"])[m_.values].dropna()
    med, lo, hi = cluster_boot(dd_.abs().values, dd_.index.get_level_values("chip"))
    dl.append(dict(product=p, n=len(dd_), med_abs_shift=med, lo=lo, hi=hi, mean_shift=dd_.mean(),
                   p90_abs_shift=np.percentile(dd_.abs(), 90), frac_gt_05=(dd_.abs() > 0.5).mean() * 100))
pd.DataFrame(dl).to_csv(T / "controlled_shift_vs_pan.csv", index=False)

# crossing descriptors from the PAN measurement at k=0 (width, contrast) and geometry
desc = c[(c["product"] == "PAN") & (c.k == 0)][["unit", "p_width", "m_width", "p_cnr", "m_cnr", "angle"]]
desc = desc.assign(width=desc[["p_width", "m_width"]].mean(1), cnr=desc[["p_cnr", "m_cnr"]].min(1))[
    ["unit", "width", "cnr", "angle"]]
dd = (w.sub(w["PAN"], axis=0)).reset_index().merge(desc, on="unit")
dd.to_csv(T / "controlled_delta_covariates.csv", index=False)

# ============================================================ field experiment
f = d[d.kind == "field"].copy()
f["ok"] = f.ok.astype(bool)
f["err_pref"] = f.H - f.H_ref
lo_, hi_ = f.H_min.fillna(f.H_ref), f.H_max.fillna(f.H_ref)
f["err_unc"] = np.where(f.H < lo_, f.H - lo_, np.where(f.H > hi_, f.H - hi_, 0.0))
f["inside"] = (f.H >= lo_) & (f.H <= hi_)
f.to_csv(T / "field_long.csv", index=False)
fr = []
for p in PRODUCTS:
    g = f[f["product"] == p]
    v = g[g.ok]
    prim = v[v.tier == "primary"]
    fr.append(dict(product=p, n=len(g), measurable=g.ok.mean() * 100, n_ok=len(v),
                   bias=prim.err_pref.mean(), mae=prim.err_pref.abs().mean(),
                   medae=prim.err_pref.abs().median(), mae_unc=prim.err_unc.abs().mean(),
                   inside=prim.inside.mean() * 100))
pd.DataFrame(fr).to_csv(T / "field_summary.csv", index=False)
# repeatability across acquisitions (markers with >= 2 measurable acquisitions)
rep = []
for p in PRODUCTS:
    g = f[(f["product"] == p) & f.ok]
    sds, ns = [], []
    for (_, idn), h in g.groupby(["marker", "ID"]):
        if h.catalog_id.nunique() >= 2:
            sds.append(h.H.var(ddof=1))
            ns.append(len(h) - 1)
    rep.append(dict(product=p, n_obs=len(sds),
                    pooled_sd=np.sqrt(np.sum(np.array(sds) * ns) / np.sum(ns)) if ns else np.nan))
pd.DataFrame(rep).to_csv(T / "field_repeatability.csv", index=False)

# ============================================================ quality indices
qrows = []
for j in jobs[(jobs.k == 0)].job:
    fpath = ROOT / f"data/processed/fused/{j}_res.mat"
    if not fpath.exists():
        continue
    r = sio.loadmat(fpath, squeeze_me=True, variable_names=["FR", "RR", "T"])
    for i, p in enumerate(FUSED):
        qrows.append(dict(job=j, product=p, D_lambda=r["FR"][i, 0], D_s=r["FR"][i, 1], HQNR=r["FR"][i, 2],
                          Q2n=r["RR"][i, 0], SAM=r["RR"][i, 1], ERGAS=r["RR"][i, 2], SCC=r["RR"][i, 3],
                          SAM_road=r["RR"][i, 4], ERGAS_road=r["RR"][i, 5], SAM_bg=r["RR"][i, 6],
                          ERGAS_bg=r["RR"][i, 7]))
q = pd.DataFrame(qrows).merge(jobs[["job", "kind", "chip", "date"]], on="job")
q.to_csv(T / "quality_long.csv", index=False)
qs = q.groupby("product")[["Q2n", "SAM", "ERGAS", "SCC", "SAM_road", "ERGAS_road", "SAM_bg", "ERGAS_bg",
                           "D_lambda", "D_s", "HQNR"]].quantile([0.25, 0.5, 0.75]).unstack()
qs.to_csv(T / "quality_summary.csv")

# per-chip rank agreement between quality indices and offset error (controlled chips)
chip_err = c[c.ok & (c.k > 0) & c["product"].isin(FUSED)].groupby(["chip", "product"]).ae.median()
qc = q[q.kind == "controlled"].set_index(["chip", "product"])
rho_rows = []
for ind, better_high in (("Q2n", True), ("SAM", False), ("ERGAS", False), ("SCC", True), ("HQNR", True),
                         ("D_lambda", False), ("D_s", False)):
    rhos = []
    for ch in qc.index.get_level_values(0).unique():
        try:
            a = qc.loc[ch][ind].reindex(FUSED)
            b = chip_err.loc[ch].reindex(FUSED)
        except KeyError:
            continue
        ok = a.notna() & b.notna()
        if ok.sum() >= 4:
            # sign convention: positive rho = better index goes with smaller offset error
            r_ = stats.spearmanr(a[ok] * (1 if better_high else -1), -b[ok]).statistic
            rhos.append(r_)
    rhos = np.array(rhos)
    if len(rhos):
        bs = [np.nanmean(RNG.choice(rhos, len(rhos))) for _ in range(NB)]
        rho_rows.append(dict(index=ind, n_chips=len(rhos), rho_mean=np.nanmean(rhos),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
pd.DataFrame(rho_rows).to_csv(T / "quality_vs_error_rho.csv", index=False)

# runtime
tr = []
for j in jobs.job:
    fpath = ROOT / f"data/processed/fused/{j}_res.mat"
    if fpath.exists():
        r = sio.loadmat(fpath, squeeze_me=True, variable_names=["T"])["T"]
        tr.append({n: float(r[n]) for n in r.dtype.names})
zl = pd.read_csv(T / "zpnn_log.csv") if (T / "zpnn_log.csv").exists() else pd.DataFrame()
rt = pd.DataFrame(tr).mean().to_dict()
if not zl.empty:
    rt["ZPNN"] = zl[zl.res == "FR"].seconds.mean()
pd.Series(rt, name="seconds_per_1024_chip").to_csv(T / "runtime.csv")
print(summ.round(3).to_string())
print(pr.round(3).to_string())

# ============================================================ sensitivity to the contrast threshold
sens = []
for suf, thr in (("_cnr2", 2.0), ("", 3.0), ("_cnr4", 4.0)):
    fpath = T / f"measurements{suf}.csv"
    if not fpath.exists():
        continue
    s_ = pd.read_csv(fpath).merge(mk, on=["job", "marker", "ID"]).merge(jobs[["job", "k", "chip"]], on="job")
    s_["ok"] = s_.ok.astype(bool)
    for kind, g in s_.groupby("kind"):
        for p, h in g.groupby("product"):
            v = h[h.ok]
            e = (v.H - v.H_true).abs() if kind == "controlled" else (v.H - v.H_ref).abs()
            sens.append(dict(min_cnr=thr, kind=kind, product=p, measurable=h.ok.mean() * 100,
                             n_ok=len(v), med_ae=e.median(), p_gt_1=(e > 1).mean() * 100))
pd.DataFrame(sens).to_csv(T / "sensitivity_cnr.csv", index=False)
