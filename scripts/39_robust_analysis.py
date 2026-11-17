"""Analysis of the second-review robustness experiments (results/tables/v2):
xcorr_summary.csv     edge-free estimator: measurability, conditional and baseline-corrected error, agreement with main
side_summary.csv      fault-side misregistration: offset change relative to the unshifted run, per product
intensity_summary.csv four intensity definitions on the stratified subset
field_aperture.csv    field offsets against aperture A (L fixed) and support length L (A fixed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
O = T / "v2"
RNG = np.random.default_rng(31)
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
what = sys.argv[1:] or ["xcorr", "side", "intens", "field"]


def boot_med(df, col, nb=2000):
    if df.empty:
        return np.nan, np.nan, np.nan
    idx = df.groupby("unit").indices
    keys = list(idx)
    v = df[col].values
    bs = [np.median(v[np.concatenate([idx[keys[i]] for i in RNG.integers(len(keys), size=len(keys))])]) for _ in range(nb)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if "xcorr" in what:
    x = pd.read_csv(T / "xcorr_measurements.csv")
    x["unit"] = x.chip + "/" + x.marker
    x["ok"] = x.ok.astype(bool)
    x["ae"] = (x.H - x.H_true).abs()
    b = x[x.k == 0].set_index(["unit", "product"])
    x["H0"] = [b.H.get((u, p), np.nan) if b.ok.get((u, p), False) else np.nan for u, p in zip(x.unit, x["product"])]
    x["inc_ae"] = ((x.H - x.H0) - x.H_true).abs()
    okk = x[x.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").fillna(False)
    comp = set(okk.index[okk[P].all(axis=1)])
    main = pd.read_csv(O / "controlled_long_v2.csv")
    main["ok"] = main.ok.astype(bool)
    rows = []
    for p in P:
        g = x[x["product"] == p]
        s = g[g.ok & (g.k > 0)]
        sc = s[[(u, k) in comp for u, k in zip(s.unit, s.k)]]
        si = s.dropna(subset=["inc_ae"])
        a, cc, inc = boot_med(s, "ae"), boot_med(sc, "ae"), boot_med(si, "inc_ae")
        rows.append(dict(product=p, n_success=int(g.ok.sum()), n_total=len(g), success_pct=g.ok.mean() * 100,
                         med=a[0], lo=a[1], hi=a[2], n_cc=len(sc), cc_med=cc[0], cc_lo=cc[1], cc_hi=cc[2],
                         inc_med=inc[0], inc_gt05=(si.inc_ae > 0.5).mean() * 100,
                         main_success_pct=main[main["product"] == p].ok.mean() * 100))
    xs = pd.DataFrame(rows)
    xs.to_csv(O / "xcorr_summary.csv", index=False)
    # agreement of the two estimators where both measured
    m = main[(main.k > 0)][["unit", "k", "product", "H", "ok"]].merge(
        x[x.k > 0][["unit", "k", "product", "H", "ok"]], on=["unit", "k", "product"], suffixes=("_m", "_x"))
    both = m[m.ok_m & m.ok_x]
    agr = dict(n_both=len(both), med_absdiff=float((both.H_m - both.H_x).abs().median()),
               share_lt05=float(((both.H_m - both.H_x).abs() < 0.5).mean() * 100),
               rank_rho_success=stats.spearmanr(xs.success_pct, xs.main_success_pct).statistic,
               n_cc=len(comp), n_cc_crossings=len({u for u, _ in comp}))
    pd.Series(agr).to_csv(O / "xcorr_agreement.csv")
    print(xs.round(3).to_string())
    print(agr)

if "side" in what and (T / "rr_side_measurements.csv").exists():
    sd = pd.read_csv(T / "rr_side_measurements.csv")
    r0 = pd.read_csv(O / "rr_long.csv")
    r0 = r0[r0.m == 0]
    sd["unit"] = sd.chip + "/" + sd.marker
    sd["ok"] = sd.ok.astype(bool)
    r0["ok"] = r0.ok.astype(bool)
    j = sd.merge(r0[["unit", "k", "product", "H", "ok"]], on=["unit", "k", "product"], suffixes=("", "_0"))
    j = j[j.ok & j.ok_0]
    j["dH"] = j.H - j.H_0
    exp_shift = j.ms_res  # one latent MS pixel along strike on the displaced side
    j["frac"] = j.dH / exp_shift
    rows = []
    for p in ["REF", "PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]:
        g = j[j["product"] == p]
        a = boot_med(g.assign(v=g.frac), "v")
        rows.append(dict(product=p, n=len(g), med_frac=a[0], lo=a[1], hi=a[2], med_dH=g.dH.median(),
                         share_follow_ms=float((g.frac > 0.5).mean() * 100),
                         success_pct=sd[sd["product"] == p].ok.mean() * 100))
    ss = pd.DataFrame(rows)
    ss.to_csv(O / "side_summary.csv", index=False)
    print(ss.round(3).to_string())

if "intens" in what and (T / "intensity_measurements.csv").exists():
    it = pd.read_csv(T / "intensity_measurements.csv")
    it["unit"] = it.chip + "/" + it.marker
    it["ok"] = it.ok.astype(bool)
    it["ae"] = (it.H - it.H_true).abs()
    b = it[it.k == 0].set_index(["unit", "product", "intensity"])
    it["H0"] = [b.H.get((u, p, i), np.nan) if b.ok.get((u, p, i), False) else np.nan
                for u, p, i in zip(it.unit, it["product"], it.intensity)]
    it["inc_ae"] = ((it.H - it.H0) - it.H_true).abs()
    rows = []
    for (p, i), g in it.groupby(["product", "intensity"]):
        s = g[g.ok & (g.k > 0)]
        si = s.dropna(subset=["inc_ae"])
        a = boot_med(s, "ae")
        rows.append(dict(product=p, intensity=i, n_total=len(g), success_pct=g.ok.mean() * 100, med=a[0], lo=a[1],
                         hi=a[2], inc_gt05=(si.inc_ae > 0.5).mean() * 100 if len(si) else np.nan, n_inc=len(si)))
    isu = pd.DataFrame(rows)
    isu.to_csv(O / "intensity_summary.csv", index=False)
    print(isu.pivot_table(index="product", columns="intensity", values=["success_pct", "med"]).round(2))

if "field" in what:
    fs_ = pd.read_csv(T / "field_sensitivity.csv")
    fs_["ok"] = fs_.ok.astype(bool)
    fa_ = pd.read_csv(T / "field_sensitivity_A.csv")
    fa_["ok"] = fa_.ok.astype(bool)
    A = fa_[fa_.config == "A"].rename(columns={"du": "A"})
    L = fs_[fs_.config == "L"]
    ra = A[A.ok].groupby(["marker", "ID", "catalog_id", "A"]).H.agg(["min", "median", "max", "count"]).reset_index()
    rl = L[L.ok].groupby(["marker", "ID", "catalog_id", "L"]).H.agg(["min", "median", "max", "count"]).reset_index()
    ra.assign(kind="aperture").rename(columns={"A": "value"}).pipe(
        lambda d: pd.concat([d, rl.assign(kind="support").rename(columns={"L": "value"})])).to_csv(
        O / "field_aperture.csv", index=False)
    print(ra[ra.marker == "M01"].round(2).to_string())
    print(A.groupby("A").ok.mean().round(3))
