"""Offset measurement on the numerically stabilised MTF-GLP-HPM (HPM*) of the
controlled crossings, paired with the plain MTF-GLP-HPM.

Inputs
  data/processed/fused_hpmstab/<job>_hpmstab.mat  from matlab/hpm_stab_fr.m
      (run: cd matlab; matlab -batch "hpm_stab_fr('../data/processed/hpmstab_list.txt','../data/processed/fused_hpmstab')")
      L(:,:,1) = recomputed MTF-GLP-HPM luminance, L(:,:,2) = HPM* luminance,
      FLOOR / FLOOR_ANY = share of pixels where the 1e-4 reflectance floor is active
  data/processed/fused/<job>_res.mat   original products (reproduction check)
  data/processed/jobs.csv, data/processed/markers_measured.csv
  results/tables/v2/controlled_long_v2.csv   original measurements (reproduction check)

The frozen estimator (src/offset.py, PARAMS) and the acceptance rules are
applied exactly as in scripts/12_measure.py; error definitions follow
scripts/26_analysis_v2.py (baseline-corrected error uses the k = 0 estimate of
the same product and crossing) and scripts/38_review2_stats.py (large deviation:
baseline-corrected |error| > 0.5 m). Intervals: crossing-cluster percentile
bootstrap, B = 2000.

Outputs
  results/tables/v2/hpm_stab_long.csv     one row per attempt and product
  results/tables/v2/hpm_stab_summary.csv  comparison table (metric, product, n, N, value, lo, hi)
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import PARAMS, Raster, measure_offset  # noqa: E402

FU = ROOT / "data/processed/fused"
HS = ROOT / "data/processed/fused_hpmstab"
O = ROOT / "results/tables/v2"
PRODS = ["MTF_GLP_HPM", "MTF_GLP_HPM_stab"]
RGB = {"R": 4, "G": 2, "B": 1}          # zero-based band index (bands 5, 3, 2)
NB = 2000
RNG = np.random.default_rng(48)
ACQ = {"10300500D9F8D200": "WV-2 08 Feb", "10300100E291D100": "WV-2 11 Feb", "10300100E3154100": "WV-2 28 Feb",
       "1040010082B6D600": "WV-3 28 Feb", "10300100E383D900": "WV-2 11 Mar"}


def classify(row):
    """Failure class, identical to scripts/26_analysis_v2.py."""
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


def run_job(args):
    """Measure all markers of one chip on both luminances; also return chip diagnostics."""
    job, jm, mk = args
    r = sio.loadmat(HS / f"{job}_hpmstab.mat", squeeze_me=True)
    names = [str(n) for n in r["NAMES"]]
    L = r["L"].astype(np.float64)
    ref = sio.loadmat(FU / f"{job}_res.mat", squeeze_me=True, variable_names=["L", "NAMES"])
    lref = ref["L"][:, :, [str(n) for n in ref["NAMES"]].index("MTF_GLP_HPM")].astype(np.float64)
    fl, fa = np.atleast_2d(r["FLOOR"]), np.ravel(r["FLOOR_ANY"])
    diag = dict(job=job, repro_max_abs=float(np.nanmax(np.abs(L[:, :, 0] - lref))),
                repro_ref_max=float(np.nanmax(np.abs(lref))),
                lum_max_hpm=float(np.nanmax(L[:, :, 0])), lum_max_stab=float(np.nanmax(L[:, :, 1])),
                lum_med_absdiff=float(np.nanmedian(np.abs(L[:, :, 1] - L[:, :, 0]))))
    for b, i in RGB.items():
        diag.update({f"floor_ms_{b}": fl[0, i], f"floor_pan_{b}": fl[1, i], f"floor_panlp_{b}": fl[2, i],
                     f"floor_any_{b}": fa[i]})
    diag["floor_any_RGBmean"] = float(np.mean([fa[i] for i in RGB.values()]))
    diag["floor_any_allbands"] = float(np.mean(fa))
    rows = []
    for i, pname in enumerate(names):
        img = Raster(L[:, :, i], jm["x0"], jm["y0"], jm["pan_res"])
        for m in mk:
            o = measure_offset(img, np.array([m["E"], m["N"]]), m["strike"], m["az_plus"], m["az_minus"],
                               aperture=m["aperture"], mode=m["mode"], prm=PARAMS)
            rows.append(dict(job=job, marker=m["marker"], ID=m["ID"], product=pname, **o.flat()))
    return rows, diag


def cboot(df, fun, nb=NB):
    """Crossing-cluster bootstrap: estimate and 95% percentile interval of fun(df)."""
    df = df.reset_index(drop=True)
    idx = df.groupby("unit").indices
    keys = list(idx)
    est = fun(df)
    if not keys:
        return est, np.nan, np.nan
    bs = np.array([fun(df.iloc[np.concatenate([idx[keys[i]] for i in RNG.integers(len(keys), size=len(keys))])])
                   for _ in range(nb)])
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def main():
    jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
    mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
    mk = mk[mk.kind == "controlled"]
    jm = jobs.set_index("job")
    tasks = [(j, jm.loc[j].to_dict(), g.to_dict("records")) for j, g in mk.groupby("job")]
    missing = [t[0] for t in tasks if not (HS / f"{t[0]}_hpmstab.mat").exists()]
    if missing:
        sys.exit(f"{len(missing)} fused HPM* files missing (first: {missing[0]}); run matlab/hpm_stab_fr.m")
    rows, diags = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, (r_, d_) in enumerate(ex.map(run_job, tasks, chunksize=4), 1):
            rows += r_
            diags.append(d_)
            if n % 25 == 0:
                print(f"measured {n}/{len(tasks)} chips", flush=True)
    ms = pd.DataFrame(rows)
    dg = pd.DataFrame(diags)

    # ---- long table with the column conventions of controlled_long_v2.csv
    d = ms.merge(mk, on=["job", "marker", "ID"]).merge(
        jobs[["job", "k", "chip", "off_nadir", "pan_res", "ms_res"]].assign(acq=jobs.catalog_id.map(ACQ)), on="job")
    c = d.copy()
    c["unit"] = c.chip + "/" + c.marker
    c["ok"] = c.ok.astype(bool)
    c["err"] = c.H - c.H_true
    c["ae"] = c.err.abs()
    c["failure"] = c.apply(classify, axis=1)
    base = c[c.k == 0].set_index(["unit", "product"])
    c["H0"] = [base.H.get((u, p), np.nan) if base.ok.get((u, p), False) else np.nan for u, p in zip(c.unit, c["product"])]
    c["inc_err"] = (c.H - c.H0) - c.H_true
    c["inc_ae"] = c.inc_err.abs()
    c = c.sort_values(["job", "marker", "product"]).reset_index(drop=True)
    c.to_csv(O / "hpm_stab_long.csv", index=False)

    out = []

    def add(metric, product, n, N, value, lo=np.nan, hi=np.nan, note=""):
        out.append(dict(metric=metric, product=product, n=n, N=N, value=value, lo=lo, hi=hi, note=note))

    # ---- reproduction checks
    add("repro_luminance_max_abs_diff", "MTF_GLP_HPM", int((dg.repro_max_abs == 0).sum()), len(dg),
        dg.repro_max_abs.max(), note="recomputed vs data/processed/fused; n = chips with identical luminance")
    orig = pd.read_csv(O / "controlled_long_v2.csv")
    orig = orig[orig["product"] == "MTF_GLP_HPM"].set_index(["job", "marker"])
    mine = c[c["product"] == "MTF_GLP_HPM"].set_index(["job", "marker"]).reindex(orig.index)
    same_ok = (mine.ok.astype(bool).values == orig.ok.astype(bool).values)
    dh = np.abs(mine.H.values - orig.H.values)
    add("repro_measurement_ok_agree", "MTF_GLP_HPM", int(same_ok.sum()), len(orig), same_ok.mean())
    add("repro_measurement_max_abs_dH", "MTF_GLP_HPM", int(np.isfinite(dh).sum()), len(orig), np.nanmax(dh))

    # ---- acceptance
    w = c.pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max").astype(bool).reset_index()
    for p in PRODS:
        est, lo, hi = cboot(w, lambda x, p=p: x[p].mean() * 100)
        add("acceptance_pct", p, int(w[p].sum()), len(w), est, lo, hi)
    est, lo, hi = cboot(w, lambda x: (x[PRODS[1]].mean() - x[PRODS[0]].mean()) * 100)
    add("acceptance_diff_pp_stab_minus_hpm", "", len(w), len(w), est, lo, hi)
    for a_, b_ in ((True, True), (True, False), (False, True), (False, False)):
        add("acceptance_2x2", f"HPM={'ok' if a_ else 'fail'},HPM*={'ok' if b_ else 'fail'}",
            int(((w[PRODS[0]] == a_) & (w[PRODS[1]] == b_)).sum()), len(w), np.nan)
    for p in PRODS:
        for f, n in c[c["product"] == p].failure.value_counts().items():
            add("failure_reason", p, int(n), int((c["product"] == p).sum()), np.nan, note=f)

    # ---- accuracy on attempts accepted by both, k > 0
    ae = c.pivot_table(index=["unit", "k"], columns="product", values="ae").reset_index()
    both = w.merge(ae, on=["unit", "k"], suffixes=("_ok", ""))
    both = both[both[PRODS[0] + "_ok"] & both[PRODS[1] + "_ok"]]
    bk = both[both.k > 0]
    for p in PRODS:
        add("median_abs_err_k>0_jointly_accepted", p, len(bk), int((w.k > 0).sum()),
            *cboot(bk, lambda x, p=p: x[p].median()))
    add("median_abs_err_diff_stab_minus_hpm_jointly", "", len(bk), int((w.k > 0).sum()),
        *cboot(bk, lambda x: (x[PRODS[1]] - x[PRODS[0]]).median()))

    # ---- baseline-corrected error (own measured set, as in 26/38) and joint set
    for p in PRODS:
        s = c[(c["product"] == p) & (c.k > 0) & c.ok].dropna(subset=["inc_ae"])
        add("median_inc_abs_err_k>0", p, len(s), int(((c["product"] == p) & (c.k > 0)).sum()),
            *cboot(s, lambda x: x.inc_ae.median()))
        add("share_inc_dev_gt_0.5m_pct", p, int((s.inc_ae > 0.5).sum()), len(s),
            *cboot(s, lambda x: (x.inc_ae > 0.5).mean() * 100))
    ia = c[c.k > 0].pivot_table(index=["unit", "k"], columns="product", values="inc_ae").dropna().reset_index()
    for p in PRODS:
        add("median_inc_abs_err_k>0_joint", p, len(ia), int((w.k > 0).sum()), *cboot(ia, lambda x, p=p: x[p].median()))
        add("share_inc_dev_gt_0.5m_pct_joint", p, int((ia[p] > 0.5).sum()), len(ia),
            *cboot(ia, lambda x, p=p: (x[p] > 0.5).mean() * 100))
    add("share_inc_dev_gt_0.5m_diff_pp_joint_stab_minus_hpm", "", len(ia), int((w.k > 0).sum()),
        *cboot(ia, lambda x: ((x[PRODS[1]] > 0.5).mean() - (x[PRODS[0]] > 0.5).mean()) * 100))
    add("median_inc_abs_err_diff_joint_stab_minus_hpm", "", len(ia), int((w.k > 0).sum()),
        *cboot(ia, lambda x: (x[PRODS[1]] - x[PRODS[0]]).median()))

    # ---- paired change of the estimate on jointly accepted attempts
    hh = c.pivot_table(index=["unit", "k"], columns="product", values="H").reset_index()
    hj = hh.merge(w, on=["unit", "k"], suffixes=("", "_ok"))
    hj = hj[hj[PRODS[0] + "_ok"] & hj[PRODS[1] + "_ok"]].copy()
    hj["dH"] = (hj[PRODS[1]] - hj[PRODS[0]]).abs()
    add("paired_median_abs_dH_all_k", "", len(hj), len(w), *cboot(hj, lambda x: x.dH.median()))
    add("paired_share_abs_dH_gt_0.5m_pct", "", int((hj.dH > 0.5).sum()), len(hj),
        *cboot(hj, lambda x: (x.dH > 0.5).mean() * 100))
    add("paired_share_abs_dH_eq_0_pct", "", int((hj.dH < 1e-9).sum()), len(hj), (hj.dH < 1e-9).mean() * 100)
    hk = hj[hj.k > 0]
    add("paired_median_abs_dH_k>0", "", len(hk), int((w.k > 0).sum()), *cboot(hk, lambda x: x.dH.median()))

    # ---- floor activity (share of PAN-grid pixels per chip, averaged over the 228 chip-levels)
    for col in [x for x in dg.columns if x.startswith("floor_")]:
        add(f"{col}_pct_mean", "", len(dg), len(dg), dg[col].mean() * 100, dg[col].quantile(.5) * 100,
            dg[col].max() * 100, note="lo = median over chips, hi = max over chips")
    add("luminance_median_abs_diff_stab_vs_hpm", "", len(dg), len(dg), dg.lum_med_absdiff.median(),
        dg.lum_med_absdiff.min(), dg.lum_med_absdiff.max(), note="per-chip median; lo/hi = min/max over chips")
    add("luminance_max_hpm", "", len(dg), len(dg), dg.lum_max_hpm.median(), dg.lum_max_hpm.min(),
        dg.lum_max_hpm.max(), note="per-chip maximum; value = median over chips")
    add("luminance_max_stab", "", len(dg), len(dg), dg.lum_max_stab.median(), dg.lum_max_stab.min(),
        dg.lum_max_stab.max(), note="per-chip maximum; value = median over chips")
    sm = pd.DataFrame(out)
    sm.to_csv(O / "hpm_stab_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 200)
    print(sm.round(4).to_string())


if __name__ == "__main__":
    main()
