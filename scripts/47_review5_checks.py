"""Checks for the fifth review round, computed from the existing measurement tables.
All intervals are 95% percentile intervals of a cluster bootstrap over road crossings
(B = 2000, fixed seed). One crossing per chip, so crossing and chip are the same unit.

Outputs (results/tables/v2):
review5_rr_misreg.csv          1  uniform misregistration (m = 1/4, 1/2 MS px), paired per crossing and k
review5_rr_misreg_flow.csv     1  acceptance counts per misregistration condition
review5_side.csv               2  fault-side misregistration, signed and absolute response
review5_rr_abs_error.csv       3  reduced-resolution |H - H_true| per product, misregistration and phase
review5_quality_fe.csv         4  quality indices vs task outcomes with chip (crossing) fixed effects
review5_quality_missing.csv    4  complete-case chip-product rows without edge displacement, and why
review5_acq_loao.csv           5  measured share and contrasts vs PAN, leaving one acquisition out
review5_acq_stratified.csv     5  acquisition-stratified bootstrap of the seven contrasts vs PAN
review5_xcorr.csv              6  agreement of the primary and the cross-correlation estimator
review5_baseline.csv           7  baseline-corrected error and its pairing with k = 0
review5_parallax.csv           8  orthorectification parallax bound with incidence and off-nadir angles
review5_hpm_stab.csv           9  availability of stabilised MTF-GLP-HPM images
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pyproj import Transformer
from shapely.geometry import Point, shape
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
V = T / "v2"
B = 2000
RNG = np.random.default_rng(47)
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
FU = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
RR = ["REF", "PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]


def boot(df, col, stat=np.median, unit="unit", nb=B):
    """Estimate and 95% interval of stat(df[col]), resampling crossings with replacement."""
    df = df.reset_index(drop=True)
    if df.empty:
        return np.nan, np.nan, np.nan
    v = df[col].to_numpy(float)
    idx = list(df.groupby(unit).indices.values())
    bs = np.empty(nb)
    for b in range(nb):
        pick = RNG.integers(len(idx), size=len(idx))
        bs[b] = stat(v[np.concatenate([idx[i] for i in pick])])
    return float(stat(v)), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def pct_above(t):
    return lambda v: float((v > t).mean() * 100)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


# ================================================================== data
c = pd.read_csv(V / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
r = pd.read_csv(V / "rr_long.csv")
r["ok"] = r.ok.astype(bool)

# ================================================================== 1 uniform misregistration
rows, flow = [], []
for p in RR:
    g = r[r["product"] == p].set_index(["unit", "k", "m"])
    for mm in (1, 2):
        a = g.xs(0, level="m")
        b = g.xs(mm, level="m").reindex(a.index)
        flow.append(dict(product=p, m=mm, n_ok_m0=int(a.ok.sum()), n_ok_m=int(b.ok.sum()),
                         n_ok_both=int((a.ok & b.ok).sum()), n_lost=int((a.ok & ~b.ok).sum()),
                         n_gained=int((~a.ok & b.ok).sum()), n_total=len(a)))
        if p == "REF":
            continue
        both = a.ok & b.ok & a.delta.notna() & b.delta.notna()
        d = pd.DataFrame(dict(sgn=(b.delta.abs() - a.delta.abs())[both],
                              absch=(b.delta - a.delta).abs()[both])).reset_index()
        e1 = boot(d, "sgn")
        e2 = boot(d, "absch")
        e3 = boot(d, "absch", pct_above(0.5))
        # crossing-level statistic of rr_misreg_test.csv: mean change per crossing, median over crossings
        per = d.groupby("unit").sgn.mean()
        rows.append(dict(product=p, m=mm, n_cells=len(d), n_crossings=d.unit.nunique(),
                         med_signed=e1[0], signed_lo=e1[1], signed_hi=e1[2],
                         med_abs=e2[0], abs_lo=e2[1], abs_hi=e2[2],
                         pct_abs_gt05=e3[0], pct_lo=e3[1], pct_hi=e3[2], p90_abs=d.absch.quantile(.9),
                         crossing_mean_med=per.median(), pct_signed_abs_lt004=float((d.sgn.abs() < 0.04).mean() * 100)))
mis = pd.DataFrame(rows)
mis.to_csv(V / "review5_rr_misreg.csv", index=False)
pd.DataFrame(flow).to_csv(V / "review5_rr_misreg_flow.csv", index=False)

# ================================================================== 2 fault-side misregistration
sd = pd.read_csv(T / "rr_side_measurements.csv")
sd["unit"] = sd.chip + "/" + sd.marker
sd["ok"] = sd.ok.astype(bool)
r0 = r[r.m == 0]
j = sd.merge(r0[["unit", "k", "product", "H", "ok"]], on=["unit", "k", "product"], suffixes=("", "_0"))
j = j[j.ok & j.ok_0].copy()
j["dH"] = j.H - j.H_0
j["frac"] = j.dH / j.ms_res            # one latent MS pixel = one reduced PAN pixel = ms_res
j["afrac"] = j.frac.abs()
rows = []
for p in ["REF", "PAN"] + FU:
    g = j[j["product"] == p]
    s1, s2 = boot(g, "frac"), boot(g, "afrac")
    rows.append(dict(product=p, n_cells=len(g), n_crossings=g.unit.nunique(),
                     n_attempted=int((sd["product"] == p).sum()), n_ok_side=int(sd[sd["product"] == p].ok.sum()),
                     med_frac=s1[0], frac_lo=s1[1], frac_hi=s1[2], med_absfrac=s2[0], absfrac_lo=s2[1],
                     absfrac_hi=s2[2], med_dH=g.dH.median(), med_abs_dH=g.dH.abs().median(),
                     pct_frac_gt05=(g.frac > 0.5).mean() * 100, pct_absfrac_gt05=(g.afrac > 0.5).mean() * 100,
                     pct_abs_gt_half_rr_ms_px=(g.afrac > 2.0).mean() * 100,
                     pct_abs_dH_gt05m=(g.dH.abs() > 0.5).mean() * 100))
pd.DataFrame(rows).to_csv(V / "review5_side.csv", index=False)

# ================================================================== 3 reduced-resolution absolute error
ok_all = r.pivot_table(index=["unit", "k", "m"], columns="product", values="ok", aggfunc="max")[RR].fillna(False)
common = set(ok_all.index[ok_all.all(axis=1)])
r["common"] = [(u, k, m) in common for u, k, m in zip(r.unit, r.k, r.m)]
rows = []
for p in RR:
    for mm in (0, 1, 2):
        for setname in ("own", "common"):
            g = r[(r["product"] == p) & (r.m == mm) & r.ok]
            if setname == "common":
                g = g[g.common]
            for ph, sub in [("k>0", g[g.k > 0])] + [(f"{k}/4", g[g.k == k]) for k in range(4)]:
                e = boot(sub, "ae")
                rows.append(dict(product=p, m=mm, set=setname, phase=ph, n=len(sub), n_crossings=sub.unit.nunique(),
                                 med_ae=e[0], lo=e[1], hi=e[2], med_err=sub.err.median() if len(sub) else np.nan))
rra = pd.DataFrame(rows)
rra.to_csv(V / "review5_rr_abs_error.csv", index=False)

# ================================================================== 4 quality indices with chip fixed effects
qq = pd.read_csv(V / "quality_task_data.csv")
okk = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
comp = set(okk.index[okk[P].all(axis=1)])
s = c[(c.k > 0) & c.ok].copy()
s = s[[(u, k) in comp for u, k in zip(s.unit, s.k)]]
cc_err = s[s["product"].isin(FU)].groupby(["chip", "product"]).ae.median().rename("cc_med").reset_index()
loss = c[(c.k > 0) & c["product"].isin(FU)].assign(l2=lambda d: np.where(d.ok, d.ae, 2.0)).groupby(
    ["chip", "product"]).l2.mean().rename("loss2").reset_index()
qq = qq.merge(cc_err, on=["chip", "product"], how="left").merge(loss, on=["chip", "product"], how="left")
IND =["Q2n", "SAM", "ERGAS", "SCC", "SAM_road", "D_lambda", "D_s", "HQNR", "Q2n_RGB", "SAM_RGB", "ERGAS_RGB",
       "SCC_RGB", "Q2n_B7", "ERGAS_B7", "edge_disp_m"]
# outcome: (column, log scale). err_all: median |e| of all measured k > 0 attempts per chip and product;
# err_cc: the same on complete cases; succ: measured share; loss: mean error with 2 m per failed attempt
OUTC = {"err_all": ("med_ae", True), "err_cc": ("cc_med", True), "succ": ("succ", False), "loss": ("loss2", False)}
SPEC = {"product FE": "y ~ Qz + C(product)", "chip FE": "y ~ Qz + C(chip)",
        "chip + product FE": "y ~ Qz + C(product) + C(chip)"}
rows = []
for ind in IND:
    z = qq.dropna(subset=[ind]).copy()
    z["Qz"] = (z[ind] - z[ind].mean()) / z[ind].std()   # standardised as in scripts 26 and 38
    for oname, (col, logy) in OUTC.items():
        zz = z.dropna(subset=[col]).copy()
        zz["y"] = np.log(zz[col] + 0.01) if logy else zz[col]
        for sname, form in SPEC.items():
            f = smf.ols(form, zz).fit(cov_type="cluster", cov_kwds={"groups": zz.chip.astype("category").cat.codes})
            b, (lo, hi), pv = f.params["Qz"], f.conf_int().loc["Qz"], f.pvalues["Qz"]
            nper = zz.groupby("chip").size()
            rows.append(dict(index=ind, outcome=oname, spec=sname, n_rows=len(zz), n_chips=zz.chip.nunique(),
                             n_chips_within=int((nper > 1).sum()), coef=b, lo=lo, hi=hi, p=pv,
                             ratio=np.exp(b) if logy else np.nan, ratio_lo=np.exp(lo) if logy else np.nan,
                             ratio_hi=np.exp(hi) if logy else np.nan))
qfe = pd.DataFrame(rows)
qfe["p_holm"] = qfe.groupby(["outcome", "spec"]).p.transform(holm)
qfe.to_csv(V / "review5_quality_fe.csv", index=False)
# complete-case rows without edge displacement: inspect the k = 0, m = 0 reduced-resolution fits
miss = qq[qq.cc_med.notna() & qq.edge_disp_m.isna()][["chip", "product"]].copy()
e0 = r[(r.k == 0) & (r.m == 0)].set_index(["chip", "product"])
info = []
for chip, p in zip(miss.chip, miss["product"]):
    a, rf = e0.loc[(chip, p)], e0.loc[(chip, "REF")]
    sides = [s_ for s_ in ("p", "m") if a[f"{s_}_ok"] and rf[f"{s_}_ok"]]
    info.append(dict(chip=chip, product=p, prod_p_ok=bool(a.p_ok), prod_m_ok=bool(a.m_ok), ref_p_ok=bool(rf.p_ok),
                     ref_m_ok=bool(rf.m_ok), prod_reason=a.reason, ref_reason=rf.reason,
                     common_side=",".join(sides) or "none"))
pd.DataFrame(info).to_csv(V / "review5_quality_missing.csv", index=False)

# ================================================================== 5 acquisition sensitivity of measurement success
g = c.groupby(["unit", "product"]).ok.agg(["sum", "count"]).unstack("product")
succ, att = g["sum"].fillna(0), g["count"].fillna(0)
acq_of = c.groupby("unit").acq.first().reindex(succ.index)


def contrast(rows_idx, p, sv=succ, nv=att):
    S, N = sv.iloc[rows_idx], nv.iloc[rows_idx]
    m = N[p] > 0                   # crossings where the product was attempted (vendor: not on 11 March)
    return S[p][m].sum() / N[p][m].sum() - S["PAN"][m].sum() / N["PAN"][m].sum()


rows = []
for left in ["none"] + sorted(acq_of.unique()):
    keep = np.where(acq_of.values != left)[0]
    bidx = RNG.integers(0, len(keep), size=(B, len(keep)))
    rec = dict(left_out=left, n_crossings=len(keep))
    for p in P:
        S, N = succ.iloc[keep][p], att.iloc[keep][p]
        rec[f"rate_{p}"] = 100 * S.sum() / N.sum()
    for p in ["EXP", "MTF_GLP_HPM"]:
        bs = np.array([contrast(keep[i], p) for i in bidx])
        rec[f"{p}-PAN"] = 100 * contrast(keep, p)
        rec[f"{p}-PAN_lo"], rec[f"{p}-PAN_hi"] = 100 * np.quantile(bs, .025), 100 * np.quantile(bs, .975)
    rows.append(rec)
pd.DataFrame(rows).to_csv(V / "review5_acq_loao.csv", index=False)
# stratified bootstrap: crossings resampled within each acquisition
groups = [np.where(acq_of.values == a)[0] for a in sorted(acq_of.unique())]
strat = [np.concatenate([gi[RNG.integers(len(gi), size=len(gi))] for gi in groups]) for _ in range(B)]
plain = RNG.integers(0, len(succ), size=(B, len(succ)))
allidx = np.arange(len(succ))
rows = []
for p in P[1:]:
    est = contrast(allidx, p)
    for lab, draws in (("stratified", strat), ("unstratified", plain)):
        bs = np.array([contrast(i, p) for i in draws])
        rows.append(dict(product=p, bootstrap=lab, diff_pp=100 * est, lo=100 * np.quantile(bs, .025),
                         hi=100 * np.quantile(bs, .975), p_boot=min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))))
acs = pd.DataFrame(rows)
acs["p_holm"] = acs.groupby("bootstrap").p_boot.transform(holm)
acs.to_csv(V / "review5_acq_stratified.csv", index=False)

# ================================================================== 6 cross-correlation estimator
x = pd.read_csv(T / "xcorr_measurements.csv")
x["unit"] = x.chip + "/" + x.marker
x["ok"] = x.ok.astype(bool)
x["ae"] = (x.H - x.H_true).abs()
mx = c[c.k > 0][["unit", "k", "product", "H", "ok"]].merge(
    x[x.k > 0][["unit", "k", "product", "H", "ok"]], on=["unit", "k", "product"], suffixes=("_m", "_x"))
both = mx[mx.ok_m & mx.ok_x].assign(ad=lambda d: (d.H_m - d.H_x).abs())
xok = x[x.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
xcomp = set(xok.index[xok[P].all(axis=1)])
rows = []
for p in ["all"] + P:
    g = both if p == "all" else both[both["product"] == p]
    e = boot(g, "ad")
    rec = dict(product=p, n_both=len(g), n_crossings=g.unit.nunique(), med_absdiff=e[0], lo=e[1], hi=e[2],
               pct_lt05=(g.ad < 0.5).mean() * 100)
    if p != "all":
        sx = x[(x["product"] == p) & (x.k > 0) & x.ok]
        sx = sx[[(u, k) in xcomp for u, k in zip(sx.unit, sx.k)]]
        rec.update(n_cc=len(sx), cc_crossings=sx.unit.nunique(), cc_med=sx.ae.median())
    rows.append(rec)
pd.DataFrame(rows).to_csv(V / "review5_xcorr.csv", index=False)

# ================================================================== 7 baseline-corrected error
base_ok = c[c.k == 0].set_index(["unit", "product"]).ok
rows = []
for p in P:
    g = c[(c["product"] == p) & (c.k > 0) & c.ok].copy()
    g["base_ok"] = [bool(base_ok.get((u, p), False)) for u in g.unit]
    gi = g.dropna(subset=["inc_ae"])
    assert gi.base_ok.all() and len(gi) == int(g.base_ok.sum())   # inc_ae exists only for paired attempts
    e = boot(gi, "inc_ae")
    rows.append(dict(product=p, n_measured_k_gt0=len(g), n_paired=len(gi), n_crossings=gi.unit.nunique(),
                     med_inc=e[0], lo=e[1], hi=e[2], pct_lt005=(gi.inc_ae < 0.05).mean() * 100))
bl = pd.DataFrame(rows)
allinc = c[(c.k > 0) & c.ok].inc_ae.dropna()
bl = pd.concat([bl, pd.DataFrame([dict(product="all", n_paired=len(allinc), med_inc=allinc.median(),
                                       pct_lt005=(allinc < 0.05).mean() * 100)])], ignore_index=True)
bl.to_csv(V / "review5_baseline.csv", index=False)

# ================================================================== 8 parallax bound
feats = json.loads((ROOT / "data/raw/maxar_catalog/Kahramanmaras.geojson").read_text(encoding="utf-8"))["features"]
props = pd.DataFrame([f["properties"] for f in feats])
props["geom"] = [shape(f["geometry"]) for f in feats]
mean_geo = props.groupby("catalog_id")[["view:incidence_angle", "view:off_nadir", "view:azimuth"]].mean()
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
mk = mk[mk.kind == "field"]
du = pd.read_csv(ROOT / "data/raw/usgs/duross2024/Displacement_observations_June2023.csv")
du["V"] = pd.to_numeric(du.V_pref_m, errors="coerce")
fx = mk.merge(du[["ID", "V"]], on="ID").dropna(subset=["V"]).drop_duplicates(["ID", "catalog_id"])
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
epsg = int(props["proj:epsg"].mode()[0])
to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)


def bound(V_, theta, az, strike):
    return V_ * np.tan(np.deg2rad(theta)) * np.abs(np.cos(np.deg2rad(az - strike)))


rows = []
for f in fx.itertuples():
    mg = mean_geo.loc[f.catalog_id]
    lon, lat = to_ll.transform(f.E, f.N)
    tiles = props[(props.catalog_id == f.catalog_id) & props.geom.apply(lambda gm: gm.contains(Point(lon, lat)))]
    loc_el = tiles["view:incidence_angle"].mean() if len(tiles) else np.nan
    loc_on = tiles["view:off_nadir"].mean() if len(tiles) else np.nan
    loc_az = tiles["view:azimuth"].mean() if len(tiles) else np.nan
    inc = 90.0 - mg["view:incidence_angle"]      # ARD field holds the satellite elevation
    rows.append(dict(ID=f.ID, marker=f.marker, catalog_id=f.catalog_id, V=f.V, strike=f.strike,
                     elev_field_mean=mg["view:incidence_angle"], incidence_mean=inc, off_nadir_mean=mg["view:off_nadir"],
                     view_az_mean=mg["view:azimuth"], n_tiles_local=len(tiles), incidence_local=90.0 - loc_el,
                     off_nadir_local=loc_on, view_az_local=loc_az,
                     off_nadir_job=jobs[(jobs.kind == "field") & (jobs.catalog_id == f.catalog_id)].off_nadir.median(),
                     dH_incidence_mean=bound(f.V, inc, mg["view:azimuth"], f.strike),
                     dH_offnadir_mean=bound(f.V, mg["view:off_nadir"], mg["view:azimuth"], f.strike),
                     dH_incidence_local=bound(f.V, 90.0 - loc_el, loc_az, f.strike),
                     dH_offnadir_local=bound(f.V, loc_on, loc_az, f.strike),
                     dH_max_tan=f.V * np.tan(np.deg2rad(inc))))
pd.DataFrame(rows).to_csv(V / "review5_parallax.csv", index=False)

# ================================================================== 9 stabilised MTF-GLP-HPM images
hits = [p_ for p_ in (ROOT / "data").rglob("*") if p_.is_file() and ("stab" in p_.name.lower() or "hpm" in p_.name.lower())]
pd.DataFrame(dict(file=[str(h.relative_to(ROOT)) for h in hits] or ["none found"])).to_csv(
    V / "review5_hpm_stab.csv", index=False)

# ================================================================== console summary
pd.set_option("display.width", 250)
print("1", mis.round(3).to_string(), pd.DataFrame(flow).to_string(), sep="\n")
print("2", pd.read_csv(V / "review5_side.csv").round(3).to_string(), sep="\n")
print("3", rra[(rra.m == 0)].round(3).to_string(), sep="\n")
print("4", qfe[qfe.outcome != "err_all"].round(3).to_string(), pd.read_csv(V / "review5_quality_missing.csv").to_string(), sep="\n")
print("5", pd.read_csv(V / "review5_acq_loao.csv").round(1).to_string(), acs.round(3).to_string(), sep="\n")
print("6", pd.read_csv(V / "review5_xcorr.csv").round(4).to_string(), sep="\n")
print("7", bl.round(4).to_string(), sep="\n")
print("8", pd.read_csv(V / "review5_parallax.csv").round(3).T.to_string(), sep="\n")
print("9", len(hits), "files")
