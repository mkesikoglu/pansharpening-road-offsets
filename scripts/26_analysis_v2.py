"""Failure-aware analysis (revision). All inference uses the road crossing as the
independent unit. Outputs CSV tables in results/tables/v2 and key numbers in
results/tables/v2/key_numbers_v2.json."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
O = T / "v2"
O.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(11)
NB = 5000
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
FU = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
K = {}
ACQ = {"10300500D9F8D200": "WV-2 08 Feb", "10300100E291D100": "WV-2 11 Feb", "10300100E3154100": "WV-2 28 Feb",
       "1040010082B6D600": "WV-3 28 Feb", "10300100E383D900": "WV-2 11 Mar"}


def cboot(df, unit, fun, nb=2000):
    """Cluster bootstrap over `unit`: estimate and 95% percentile interval of fun(df)."""
    df = df.reset_index(drop=True)
    idx = df.groupby(unit).indices
    keys = list(idx)
    est = fun(df)
    if not keys:
        return est, np.nan, np.nan
    bs = np.empty(nb)
    for b in range(nb):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        bs[b] = fun(df.iloc[np.concatenate([idx[keys[i]] for i in pick])])
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def signflip(x, nperm=20000):
    x = np.asarray(x)
    obs = abs(x.mean())
    s = RNG.choice([-1, 1], size=(nperm, len(x)))
    return float((np.abs((s * x).mean(1)) >= obs).mean())


def classify(row):
    if row.ok:
        return "measured"
    r = str(row.reason) if isinstance(row.reason, str) else ""
    if "no_candidate_pair" in r:
        return "no edge pair"
    if "low_cnr" in r:
        return "low contrast"
    if "too_few_stations" in r or "low_valid_fraction" in r or "lost_marker" in r:
        return "edge not tracked"
    if "high_rmse" in r:
        return "poor line fit"
    return "offset out of range"


# ------------------------------------------------------------------ data
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
ms = pd.read_csv(T / "measurements.csv")
reg = pd.read_csv(T / "registration.csv")
d = ms.merge(mk, on=["job", "marker", "ID"]).merge(
    jobs[["job", "k", "chip", "off_nadir", "pan_res", "ms_res"]].assign(acq=jobs.catalog_id.map(ACQ)), on="job")
c = d[d.kind == "controlled"].copy()
c["unit"] = c.chip + "/" + c.marker
c["ok"] = c.ok.astype(bool)
c["err"] = c.H - c.H_true
c["ae"] = c.err.abs()
c["failure"] = c.apply(classify, axis=1)
# baseline (k = 0) estimate of the same product and crossing
base = c[c.k == 0].set_index(["unit", "product"])
c["H0"] = [base.H.get((u, p), np.nan) if base.ok.get((u, p), False) else np.nan for u, p in zip(c.unit, c["product"])]
c["inc_err"] = (c.H - c.H0) - c.H_true
c["inc_ae"] = c.inc_err.abs()
c.to_csv(O / "controlled_long_v2.csv", index=False)
K["N_crossings"] = int(c.unit.nunique())
K["N_attempts_per_product"] = c.groupby("product").size().to_dict()

# ------------------------------------------------------------------ 1 measurability
rows = []
for p in P:
    g = c[c["product"] == p]
    est, lo, hi = cboot(g, "unit", lambda x: x.ok.mean() * 100)
    rows.append(dict(product=p, n_success=int(g.ok.sum()), n_total=len(g), success_pct=est, lo=lo, hi=hi))
meas = pd.DataFrame(rows)
fr = c.groupby(["product", "failure"]).size().unstack(fill_value=0)
fr.to_csv(O / "failure_reasons.csv")
# GEE logistic, exchangeable within crossing
g = c.copy()
g["product"] = pd.Categorical(g["product"], categories=P)
g["kf"] = g.k.astype(str)
g["okint"] = g.ok.astype(int)
gee = smf.gee("okint ~ C(product) + C(kf)", "unit", g, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
orr = []
for nm in gee.params.index:
    if nm.startswith("C(product)"):
        b, se = gee.params[nm], gee.bse[nm]
        orr.append(dict(product=nm.split("T.")[1].rstrip("]"), OR=np.exp(b), lo=np.exp(b - 1.96 * se),
                        hi=np.exp(b + 1.96 * se), p=gee.pvalues[nm]))
orr = pd.DataFrame(orr)
orr["p_holm"] = holm(orr.p)
meas = meas.merge(orr, on="product", how="left")
meas.to_csv(O / "measurability.csv", index=False)
# covariates of measurability (crossing angle and acquisition off-nadir), pooled over products
g2 = g.assign(angle_z=(g.angle - g.angle.mean()) / g.angle.std(),
              offnadir_z=(g.off_nadir - g.off_nadir.mean()) / g.off_nadir.std())
gee2 = smf.gee("okint ~ C(product) + angle_z + offnadir_z", "unit", g2, family=sm.families.Binomial(),
               cov_struct=sm.cov_struct.Exchangeable()).fit()
K["meas_cov"] = {v: dict(OR=float(np.exp(gee2.params[v])), lo=float(np.exp(gee2.params[v] - 1.96 * gee2.bse[v])),
                         hi=float(np.exp(gee2.params[v] + 1.96 * gee2.bse[v])), p=float(gee2.pvalues[v]))
                 for v in ("angle_z", "offnadir_z")}

# ------------------------------------------------------------------ 2 conditional accuracy, common sets
okk = c[(c.k > 0)].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
complete = okk[P].all(axis=1)
complete_idx = set(okk.index[complete])
K["complete_case_n"] = int(complete.sum())
K["complete_case_crossings"] = int(len({u for u, _ in complete_idx}))
acc = []
for p in P:
    s = c[(c["product"] == p) & (c.k > 0) & c.ok]
    sc = s[[(u, k) in complete_idx for u, k in zip(s.unit, s.k)]]
    si = s.dropna(subset=["inc_ae"])
    sci = sc.dropna(subset=["inc_ae"])
    med, lo, hi = cboot(s, "unit", lambda x: x.ae.median())
    cm, clo, chi = cboot(sc, "unit", lambda x: x.ae.median()) if len(sc) else (np.nan,) * 3
    im, ilo, ihi = cboot(si, "unit", lambda x: x.inc_ae.median()) if len(si) else (np.nan,) * 3
    cim, cilo, cihi = cboot(sci, "unit", lambda x: x.inc_ae.median()) if len(sci) else (np.nan,) * 3
    rm, rlo, rhi = cboot(s, "unit", lambda x: np.sqrt((x.err ** 2).mean()))
    acc.append(dict(product=p, n=len(s), med_ae=med, med_lo=lo, med_hi=hi, iqr_lo=s.ae.quantile(.25),
                    iqr_hi=s.ae.quantile(.75), rmse=rm, rmse_lo=rlo, rmse_hi=rhi, p90=s.ae.quantile(.9),
                    max_ae=s.ae.max(), n_common=len(sc), common_med=cm, common_lo=clo, common_hi=chi,
                    n_inc=len(si), inc_med=im, inc_lo=ilo, inc_hi=ihi, common_inc_med=cim, common_inc_lo=cilo,
                    common_inc_hi=cihi, common_p90=sc.ae.quantile(.9) if len(sc) else np.nan,
                    bias=s.err.median()))
acc = pd.DataFrame(acc)
acc.to_csv(O / "accuracy.csv", index=False)
k0 = c[(c.k == 0) & c.ok].groupby("product").H.agg(lambda v: np.median(np.abs(v)))
K["null_offset_median_abs"] = k0.round(3).to_dict()

# ------------------------------------------------------------------ 3 crossing-level paired inference vs PAN
ae = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ae")
pr = []
for p in [q for q in P if q != "PAN"]:
    m = (okk["PAN"] & okk[p]).reindex(ae.index).fillna(False)
    dd = (ae[p] - ae["PAN"])[m.values].dropna()
    per = dd.groupby(level="unit").mean()          # one value per crossing
    est, lo, hi = cboot(per.reset_index(name="v"), "unit", lambda x: x.v.median())
    mest, mlo, mhi = cboot(per.reset_index(name="v"), "unit", lambda x: x.v.mean())
    w = stats.wilcoxon(per.values) if len(per) > 5 else None
    pr.append(dict(product=p, n_crossings=len(per), n_cases=len(dd), med=est, lo=lo, hi=hi, mean=mest,
                   mean_lo=mlo, mean_hi=mhi, frac_better=(per < 0).mean() * 100,
                   p_wilcoxon=w.pvalue if w else np.nan, p_signflip=signflip(per.values)))
pr = pd.DataFrame(pr)
pr["p_holm_wilcoxon"] = holm(pr.p_wilcoxon.fillna(1))
pr["p_holm_signflip"] = holm(pr.p_signflip.fillna(1))
pr.to_csv(O / "paired_vs_pan.csv", index=False)
# same with incremental error
inc = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="inc_ae")
pri = []
for p in [q for q in P if q != "PAN"]:
    dd = (inc[p] - inc["PAN"]).dropna()
    per = dd.groupby(level="unit").mean()
    est, lo, hi = cboot(per.reset_index(name="v"), "unit", lambda x: x.v.median())
    w = stats.wilcoxon(per.values) if len(per) > 5 else None
    pri.append(dict(product=p, n_crossings=len(per), med=est, lo=lo, hi=hi,
                    p_wilcoxon=w.pvalue if w else np.nan))
pri = pd.DataFrame(pri)
pri["p_holm"] = holm(pri.p_wilcoxon.fillna(1))
pri.to_csv(O / "paired_vs_pan_incremental.csv", index=False)
# change of the measured offset caused by the product
hh = c.pivot_table(index=["unit", "k"], columns="product", values="H")
okall = c.pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
sh = []
for p in [q for q in P if q != "PAN"]:
    m = (okall["PAN"] & okall[p]).reindex(hh.index).fillna(False)
    dd = (hh[p] - hh["PAN"])[m.values].dropna().abs()
    per = dd.groupby(level="unit").median()
    est, lo, hi = cboot(per.reset_index(name="v"), "unit", lambda x: x.v.median())
    sh.append(dict(product=p, n_crossings=len(per), med_shift=est, lo=lo, hi=hi,
                   frac_gt_05=(dd > 0.5).mean() * 100, p90=dd.quantile(.9)))
pd.DataFrame(sh).to_csv(O / "shift_vs_pan.csv", index=False)

# ------------------------------------------------------------------ 4 mixed model (single crossing random effect)
lm_rows = []
s = c[(c.k > 0) & c.ok].copy()
s["product"] = pd.Categorical(s["product"], categories=P)
SPECS = [(eps, "y ~ C(product) + C(k)", s, "primary") for eps in (0.001, 0.01, 0.05, 0.1)] + \
        [(0.01, "y ~ C(product) + C(k) + C(acq)", s[s.acq != "WV-2 11 Mar"], "acquisition fixed effect")]
for eps, form, data, label in SPECS:
    data = data.copy()
    data["y"] = np.log(data.ae + eps)
    fits = []
    for meth in ("powell", "nm"):
        try:
            fits.append(smf.mixedlm(form, data, groups=data["unit"]).fit(reml=True, method=meth))
        except Exception:
            pass
    fit = fits[0]  # Powell; L-BFGS stopped at a spurious zero-variance boundary
    ci = fit.conf_int()
    for nm in fit.params.index:
        if nm.startswith("C(product)"):
            lm_rows.append(dict(model=label, eps=eps, product=nm.split("T.")[1].rstrip("]"), ratio=np.exp(fit.params[nm]),
                                lo=np.exp(ci.loc[nm, 0]), hi=np.exp(ci.loc[nm, 1]), p=fit.pvalues[nm],
                                re_var=float(fit.cov_re.iloc[0, 0]), resid_var=float(fit.scale),
                                converged=bool(fit.converged), n=int(fit.nobs)))
lmm = pd.DataFrame(lm_rows)
lmm["p_holm"] = lmm.groupby(["model", "eps"]).p.transform(holm)
lmm.to_csv(O / "lmm_eps.csv", index=False)

# ------------------------------------------------------------------ 5 failure-penalised loss and Pareto inputs
rt = pd.read_csv(T / "runtime.csv", index_col=0).iloc[:, 0].to_dict()
loss = []
for p in P:
    g_ = c[(c["product"] == p) & (c.k > 0)]
    for pen in (1.0, 2.0, 5.0):
        est, lo, hi = cboot(g_, "unit", lambda x: np.where(x.ok, x.ae, pen).mean())
        loss.append(dict(product=p, penalty_m=pen, loss=est, lo=lo, hi=hi))
pd.DataFrame(loss).to_csv(O / "failure_loss.csv", index=False)
par = meas[["product", "success_pct"]].merge(acc[["product", "common_med", "common_p90", "med_ae", "p90"]],
                                            on="product")
par["runtime_s"] = par["product"].map(rt)
par.to_csv(O / "pareto.csv", index=False)

# ------------------------------------------------------------------ 6 registration and error
rr = reg[reg.kind == "controlled"].copy()
rr["unit"] = rr.job.str.replace("_k0", "", regex=False) + "/" + rr.marker
dd = (ae.sub(ae["PAN"], axis=0)).groupby(level="unit").mean()
regrows = []
for p in [q for q in P if q != "PAN"]:
    x = rr.set_index("unit").mag_local_m.reindex(dd.index)
    ok_ = x.notna() & dd[p].notna()
    rho = stats.spearmanr(x[ok_], dd[p][ok_])
    regrows.append(dict(product=p, n=int(ok_.sum()), rho=rho.statistic, p=rho.pvalue))
pd.DataFrame(regrows).to_csv(O / "registration_vs_delta.csv", index=False)
K["registration"] = dict(median_m=float(reg.mag_local_m.median()), p90_m=float(reg.mag_local_m.quantile(.9)),
                         max_m=float(reg.mag_local_m.max()),
                         median_pan_px=float((reg.mag_local_m / (reg.ms_res / 4)).median()))

# ------------------------------------------------------------------ 7 quality-task relation (chip level)
q = pd.read_csv(T / "quality_long.csv")
qx = pd.read_csv(T / "quality_ext.csv") if (T / "quality_ext.csv").exists() else pd.DataFrame()
if not qx.empty:
    w_ = qx.pivot_table(index=["job", "product"], columns="bandset", values=["Q2n", "SAM", "ERGAS", "SCC"])
    w_.columns = [f"{a}_{b}" for a, b in w_.columns]
    q = q.merge(w_.reset_index(), on=["job", "product"], how="left")
edge_p = T / "v2" / "edge_displacement.csv"
if edge_p.exists():
    q = q.merge(pd.read_csv(edge_p), on=["job", "product"], how="left")
q["chip"] = q.job.str.replace("_k0", "", regex=False)
err_chip = c[(c.k > 0) & c.ok & c["product"].isin(FU)].groupby(["chip", "product"]).ae.median().rename("med_ae")
ok_chip = c[(c.k > 0) & c["product"].isin(FU)].groupby(["chip", "product"]).ok.mean().rename("succ")
qq = q[q.kind == "controlled"].merge(err_chip.reset_index(), on=["chip", "product"], how="left").merge(
    ok_chip.reset_index(), on=["chip", "product"], how="left")
qrows = []
IND = [c_ for c_ in ["Q2n", "SAM", "ERGAS", "SCC", "SAM_road", "D_lambda", "D_s", "HQNR", "Q2n_RGB", "SAM_RGB",
                     "ERGAS_RGB", "SCC_RGB", "Q2n_B7", "ERGAS_B7", "edge_disp_m"] if c_ in qq.columns]
for ind in IND:
    z = qq.dropna(subset=[ind]).copy()
    z["Qz"] = (z[ind] - z[ind].mean()) / z[ind].std()
    zz = z.dropna(subset=["med_ae"]).copy()
    zz["y"] = np.log(zz.med_ae + 0.01)
    try:
        f1 = smf.mixedlm("y ~ Qz + C(product)", zz, groups=zz["chip"]).fit(reml=True)
        b, lo, hi, pv = f1.params["Qz"], *f1.conf_int().loc["Qz"], f1.pvalues["Qz"]
    except Exception:
        b = lo = hi = pv = np.nan
    try:
        f2 = smf.ols("succ ~ Qz + C(product)", z.dropna(subset=["succ"])).fit(
            cov_type="cluster", cov_kwds={"groups": z.dropna(subset=["succ"]).chip.astype("category").cat.codes})
        b2, (lo2, hi2), pv2 = f2.params["Qz"], f2.conf_int().loc["Qz"], f2.pvalues["Qz"]
    except Exception:
        b2 = lo2 = hi2 = pv2 = np.nan
    qrows.append(dict(index=ind, n_err=len(zz), err_coef=b, err_lo=lo, err_hi=hi, err_p=pv,
                      n_succ=int(z.succ.notna().sum()), succ_coef=b2, succ_lo=lo2, succ_hi=hi2, succ_p=pv2))
qr = pd.DataFrame(qrows)
qr["err_p_holm"] = holm(qr.err_p.fillna(1))
qr["succ_p_holm"] = holm(qr.succ_p.fillna(1))
qr.to_csv(O / "quality_task_models.csv", index=False)
qq.to_csv(O / "quality_task_data.csv", index=False)

(O / "key_numbers_v2.json").write_text(json.dumps(K, indent=1, default=float), encoding="utf-8")
print(meas.round(3).to_string())
print(acc[["product", "n", "med_ae", "n_common", "common_med", "inc_med", "common_inc_med", "rmse", "p90"]].round(3))
print(pr.round(4).to_string())
print(lmm[lmm.eps == 0.01].round(3).to_string())
print(qr.round(3).to_string())
