"""Lambda-PNN as an additional product on the controlled chips.

1. Fusion: every controlled chip (57 crossings-chips x k = 0..3) is fused at full
   resolution with Lambda-PNN (published WV2 / WV3 weights, authors' default
   target adaptation, src/lambda_pnn_run.py). Output <job>_lpnn.mat next to the
   chip with LPNN [4h 4w 8] (reflectance x 1e4) and L_LPNN, the mean of the
   R, G, B bands (5, 3, 2), i.e. the luminance used for every fused product.
   Run time per chip is appended to results/tables/lpnn_log.csv. Chips already
   fused are skipped; CUDA out-of-memory errors are retried after a pause.
2. Measurement: the frozen offset estimator (src/offset.py, PARAMS) is applied to
   every controlled marker on L_LPNN, exactly as scripts/12_measure.py does.
3. Analysis with the definitions of scripts/26_analysis_v2.py, 38, 46 and 47:
   results/tables/v2/lpnn_long.csv (one row per Lambda-PNN attempt) and
   results/tables/v2/lpnn_summary.csv (one row per statistic).

Usage: python scripts/50_lambda_pnn.py [fuse|measure|analyse|all]   (default all)
"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offset import PARAMS, Raster, measure_offset  # noqa: E402

CH = ROOT / "data/processed/chips"
T = ROOT / "results/tables"
V = T / "v2"
LOG = T / "lpnn_log.csv"
MEAS = T / "measurements_lpnn.csv"
RGB = [4, 2, 1]                       # bands 5, 3, 2 (1-based), as in matlab/fuse_and_eval.m
P8 = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
NAME = "LPNN"
B = 2000
RNG = np.random.default_rng(50)


# ================================================================== 1 fusion
def fuse_all(max_retry=20, wait_s=60):
    """Fuse every controlled chip, one at a time, retrying on CUDA out-of-memory."""
    import torch
    from lambda_pnn_run import lpnn_fuse
    jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
    jobs = jobs[jobs.kind == "controlled"]
    for i, j in enumerate(jobs.itertuples()):
        out = CH / f"{j.job}_lpnn.mat"
        if out.exists():
            continue
        d = sio.loadmat(CH / f"{j.job}.mat", variable_names=["MS", "PAN", "SENSOR"])
        sensor = str(np.atleast_1d(d["SENSOR"])[0]).strip()
        for attempt in range(max_retry):
            try:
                t = time.time()
                z, hist, (sr, sc) = lpnn_fuse(np.moveaxis(d["MS"], -1, 0), d["PAN"], sensor)
                sec = time.time() - t
                break
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"out of memory on {j.job}, retry {attempt + 1} in {wait_s} s", flush=True)
                time.sleep(wait_s)
        else:
            raise RuntimeError(f"{j.job}: CUDA memory not available after {max_retry} attempts")
        img = np.moveaxis(z, 0, -1).astype(np.float32)
        res = {"LPNN": img, "L_LPNN": img[:, :, RGB].mean(axis=2).astype(np.float32)}
        tmp = CH / f"{j.job}_lpnn_part.mat"
        sio.savemat(tmp, res, do_compression=False)
        tmp.replace(out)
        rec = dict(job=j.job, sensor=sensor, seconds=sec, loss0=hist[0], loss_min=hist.min(),
                   best_iter=int(hist.argmin()) + 1, shift_r=" ".join(map(str, sr)), shift_c=" ".join(map(str, sc)))
        pd.DataFrame([rec]).to_csv(LOG, index=False, mode="a", header=not LOG.exists())
        print(i + 1, len(jobs), j.job, sensor, f"{sec:.1f} s", flush=True)


# ================================================================== 2 measurement
def measure_all():
    """Frozen estimator on L_LPNN for every controlled marker (as scripts/12_measure.py)."""
    jobs = pd.read_csv(ROOT / "data/processed/jobs.csv").set_index("job")
    mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
    mk = mk[mk.kind == "controlled"]
    rows = []
    for job, g in mk.groupby("job"):
        f = CH / f"{job}_lpnn.mat"
        if not f.exists():
            raise FileNotFoundError(f)
        jm = jobs.loc[job]
        img = sio.loadmat(f, variable_names=["L_LPNN"])["L_LPNN"].astype(np.float64)
        ras = Raster(img, jm["x0"], jm["y0"], jm["pan_res"])
        for m in g.to_dict("records"):
            o = measure_offset(ras, np.array([m["E"], m["N"]]), m["strike"], m["az_plus"], m["az_minus"],
                               aperture=m["aperture"], mode=m["mode"], prm=PARAMS)
            rows.append(dict(job=job, marker=m["marker"], ID=m["ID"], product=NAME, **o.flat()))
    out = pd.DataFrame(rows)
    out.to_csv(MEAS, index=False)
    print("measured", len(out), "attempts, accepted", int(out.ok.sum()), flush=True)


# ================================================================== 3 analysis
def classify(row):
    """Failure class, as in scripts/26_analysis_v2.py."""
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


def boot(df, fun, unit="unit", nb=B):
    """Estimate and 95% percentile interval of fun(df), resampling crossings with replacement."""
    df = df.reset_index(drop=True)
    if df.empty:
        return np.nan, np.nan, np.nan
    idx = list(df.groupby(unit).indices.values())
    bs = np.empty(nb)
    for b in range(nb):
        pick = RNG.integers(len(idx), size=len(idx))
        bs[b] = fun(df.iloc[np.concatenate([idx[i] for i in pick])])
    return float(fun(df)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def build_long():
    """Lambda-PNN attempts with the derived columns of controlled_long_v2.csv."""
    old = pd.read_csv(V / "controlled_long_v2.csv")
    jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
    mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
    ms = pd.read_csv(MEAS)
    acq = old.drop_duplicates("catalog_id").set_index("catalog_id").acq
    d = ms.merge(mk, on=["job", "marker", "ID"]).merge(
        jobs[["job", "k", "chip", "off_nadir", "pan_res", "ms_res", "catalog_id"]].rename(
            columns={"catalog_id": "cid"}), on="job")
    d["acq"] = d.cid.map(acq)
    c = d[d.kind == "controlled"].drop(columns="cid").copy()
    c["unit"] = c.chip + "/" + c.marker
    c["ok"] = c.ok.astype(bool)
    c["err"] = c.H - c.H_true
    c["ae"] = c.err.abs()
    c["failure"] = c.apply(classify, axis=1)
    base = c[c.k == 0].set_index("unit")
    c["H0"] = [base.H.get(u, np.nan) if base.ok.get(u, False) else np.nan for u in c.unit]
    c["inc_err"] = (c.H - c.H0) - c.H_true
    c["inc_ae"] = c.inc_err.abs()
    c = c[[x for x in old.columns if x in c.columns]]
    if LOG.exists():
        lg = pd.read_csv(LOG).drop_duplicates("job", keep="last").set_index("job")
        c["seconds"] = c.job.map(lg.seconds)
    old["ok"] = old.ok.astype(bool)
    return c, pd.concat([old, c], ignore_index=True)


def analyse():
    c, a = build_long()
    c.to_csv(V / "lpnn_long.csv", index=False)
    rows = []

    def add(metric, product, est, lo=np.nan, hi=np.nan, n=np.nan, N=np.nan, n_crossings=np.nan, p=np.nan,
            note=""):
        rows.append(dict(metric=metric, product=product, n=n, N=N, n_crossings=n_crossings, estimate=est,
                         lo=lo, hi=hi, p=p, note=note))

    # ---- acceptance (all k), crossing bootstrap
    for p in ["PAN", "ZPNN", NAME]:
        g = a[a["product"] == p]
        e = boot(g, lambda x: x.ok.mean() * 100)
        add("acceptance_pct", p, *e, n=int(g.ok.sum()), N=len(g), n_crossings=g.unit.nunique(),
            note="all k; 95% CI crossing bootstrap")
    # ---- acceptance contrast (scripts/46_success_bootstrap.py definition)
    gs = a.groupby(["unit", "product"]).ok.agg(["sum", "count"]).unstack("product")
    s, n = gs["sum"].fillna(0), gs["count"].fillna(0)
    nu = len(s)
    bidx = RNG.integers(0, nu, size=(B, nu))
    for ref in ("PAN", "ZPNN"):
        def diff(i, ref=ref):
            m = n.iloc[i][NAME] > 0
            return (s.iloc[i][NAME][m].sum() / n.iloc[i][NAME][m].sum()
                    - s.iloc[i][ref][m].sum() / n.iloc[i][ref][m].sum())
        est = diff(np.arange(nu))
        bs = np.array([diff(i) for i in bidx])
        pv = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
        add(f"acceptance_diff_pp_vs_{ref}", NAME, 100 * est, 100 * np.quantile(bs, .025),
            100 * np.quantile(bs, .975), n_crossings=nu, p=pv, note="percentage points; bootstrap p")
    # ---- failure reasons
    for p in ["PAN", "ZPNN", NAME]:
        fr = a[a["product"] == p].failure.value_counts()
        for k_, v in fr.items():
            add("failure_count", p, int(v), n=int(v), N=int(fr.sum()), note=k_)
    # ---- |e| for k > 0 on accepted attempts
    for p in ["PAN", "ZPNN", NAME]:
        g = a[(a["product"] == p) & (a.k > 0) & a.ok]
        e = boot(g, lambda x: x.ae.median())
        add("median_abs_err_k_gt0_m", p, *e, n=len(g), n_crossings=g.unit.nunique(), note="accepted attempts")
    # ---- common set: accepted by all eight existing products and Lambda-PNN
    okk = a[a.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ok", aggfunc="max")
    okk = okk.fillna(False).astype(bool)
    comp8 = set(okk.index[okk[P8].all(axis=1)])
    comp9 = set(okk.index[okk[P8 + [NAME]].all(axis=1)])
    for p in P8 + [NAME]:
        g = a[(a["product"] == p) & (a.k > 0) & a.ok]
        g = g[[(u, k) in comp9 for u, k in zip(g.unit, g.k)]]
        e = boot(g, lambda x: x.ae.median())
        add("median_abs_err_common9_m", p, *e, n=len(g), n_crossings=g.unit.nunique(),
            note=f"attempts accepted by all nine products (eight-product set n={len(comp8)})")
    # ---- crossing-level paired difference of |e| (scripts/26_analysis_v2.py, section 3)
    ae = a[a.k > 0].pivot_table(index=["unit", "k"], columns="product", values="ae")
    for ref in ("PAN", "ZPNN"):
        m = (okk[ref] & okk[NAME]).reindex(ae.index).fillna(False)
        dd = (ae[NAME] - ae[ref])[m.values].dropna()
        per = dd.groupby(level="unit").mean().reset_index(name="v")
        e = boot(per, lambda x: x.v.median())
        w = stats.wilcoxon(per.v.values) if len(per) > 5 else None
        add(f"paired_abs_err_diff_vs_{ref}_m", NAME, *e, n=len(dd), n_crossings=len(per),
            p=w.pvalue if w else np.nan,
            note=f"median over crossings of mean(|e|LPNN-|e|{ref}); Wilcoxon p; {(per.v < 0).mean() * 100:.1f}% "
                 "of crossings better")
    # ---- baseline-corrected error (scripts/47 section 7 and scripts/38 deviation threshold)
    for p in ["PAN", "ZPNN", NAME]:
        g = a[(a["product"] == p) & (a.k > 0) & a.ok].dropna(subset=["inc_ae"])
        e = boot(g, lambda x: x.inc_ae.median())
        add("median_baseline_corrected_err_m", p, *e, n=len(g), n_crossings=g.unit.nunique(),
            note="k>0 accepted and k=0 accepted, same product")
        e = boot(g, lambda x: (x.inc_ae > 0.5).mean() * 100)
        add("pct_baseline_corrected_gt_0p5m", p, *e, n=int((g.inc_ae > 0.5).sum()), N=len(g),
            n_crossings=g.unit.nunique(), note="share of paired attempts with deviation > 0.5 m")
    # ---- run time
    if LOG.exists():
        lg = pd.read_csv(LOG).drop_duplicates("job", keep="last")
        add("runtime_s_per_chip", NAME, lg.seconds.median(), lg.seconds.quantile(.25), lg.seconds.quantile(.75),
            n=len(lg), note="median and IQR; 1024 x 1024 PAN chip, RTX 2060 6 GB, 100 iterations")
        for snr, g in lg.groupby("sensor"):
            add("runtime_s_per_chip", f"{NAME} {snr}", g.seconds.median(), g.seconds.quantile(.25),
                g.seconds.quantile(.75), n=len(g), note="median and IQR")
    rt = pd.read_csv(T / "runtime.csv", index_col=0).iloc[:, 0]
    add("runtime_s_per_chip", "ZPNN", float(rt["ZPNN"]), note="results/tables/runtime.csv (mean)")
    out = pd.DataFrame(rows)
    out.to_csv(V / "lpnn_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("fuse", "all"):
        fuse_all()
    if stage in ("measure", "all"):
        measure_all()
    if stage in ("analyse", "all"):
        analyse()
