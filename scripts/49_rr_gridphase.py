"""Reduced-scale grid-phase experiment: MS grid phase varied at a fixed dislocation.

In the reduced-scale (RR) experiment of 23_rr_fraction.py a dislocation of k latent
MS pixels becomes k/4 reduced-MS pixels, so MS grid phase and dislocation size vary
together. Here the Wald degradation window is offset by s = 0..3 latent MS pixels
(4s PAN pixels) before degradation (matlab/rr_gridphase.m), which moves the reduced
MS grid by s/4 reduced-MS pixel relative to the fault and the road while k stays
fixed. No misregistration is applied (m = 0). Two offset directions are run:
'a' along strike (the direction of the dislocation, the axis on which k sets the
phase) and 'c' across strike (moves the fault line inside a reduced MS pixel).
s = 0 is the same window for both directions and is stored once as '_a0'.

Window: 248 x 248 latent MS pixels starting at latent pixel 4 (+ s on the shifted
axis), i.e. always inside the stored 256 x 256 chip (itself cut from a read with a
4-pixel margin); at s = 0 the fault lies on a reduced-MS pixel boundary as in the
original RR design. The marker geometry is unchanged; the raster origin of each
product is moved by the window origin, so all offsets are in the same map frame.

Products, fusion, Z-PNN (adapted, 100 iterations; pretrained ZPNN0) and the offset
estimator are those of 23_rr_fraction.py; definitions of delta (vs REF) and of the
baseline-corrected error (k = 0 baseline of the same product and window) follow
27_rr_analysis.py.

stages
  list     write data/processed/rr_gridphase_list.txt for matlab/rr_gridphase.m
  zpnn     Z-PNN on each reduced-scale input, one at a time; waits while the shared GPU
           has less than MIN_FREE_MIB free and retries on CUDA out-of-memory
  measure  offsets -> results/tables/v2/rr_gridphase_long.csv (rows already measured are
           reused, so the stage can be re-run after Z-PNN completes)
  analyse  results/tables/v2/
             rr_gridphase_summary.csv     acceptance, baseline-corrected error and |delta|
                                          per direction, product, s and k (crossing bootstrap)
             rr_gridphase_test.csv        Friedman test of s at fixed k, Holm over products
             rr_gridphase_change.csv      |H(s) - H(s')| at fixed k on jointly accepted cells
             rr_gridphase_windowcheck.csv s = 0 window (248 px) vs the original 256-px RR run
             rr_gridphase_inc_by_phase.csv baseline-corrected |error| by the MS phase of the
                                          displaced side, (k + s) mod 4 / 4, along strike
"""
import itertools
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
OUT = ROOT / "data/processed/rr_gridphase"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results/tables/v2"
LIST = ROOT / "data/processed/rr_gridphase_list.txt"
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
ctrl = jobs[jobs.kind == "controlled"]
K = (0, 1, 2, 3)
S = (0, 1, 2, 3)
DIRS = ("a", "c")
ORDER = ["REF", "PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]
NB = 2000


def conditions():
    """(job row, dir, s, file stem suffix) for every reduced-scale input."""
    for j in ctrl.itertuples():
        for d in DIRS:
            for s in S:
                if d == "c" and s == 0:
                    continue  # identical window to a0
                yield j, d, s, f"{j.job}_{d}{s}"


def stage_list():
    # along-strike inputs first, so a partial run already covers the core design
    cond = sorted(conditions(), key=lambda c: (c[1] != "a", c[0].k == 3, c[3]))
    lines = [f"../data/processed/chips/{j.job}.mat;{j.orientation};{s};{d}" for j, d, s, _ in cond]
    LIST.write_text("\n".join(lines), encoding="ascii")
    print(len(lines))


MIN_FREE_MIB = 1500


def gpu_free_mib():
    """Free device memory reported by nvidia-smi (MiB); counts other processes' use."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return MIN_FREE_MIB


def stage_zpnn(pattern="*", device_name="cuda", reverse=False):
    """device_name 'cpu' runs the same network on the CPU (4 threads) when the shared GPU
    is occupied; outputs agree with the GPU run to within ~0.05 % of the signal."""
    import torch
    from zpnn_run import zpnn_fuse
    cpu = device_name == "cpu"
    if cpu:
        torch.set_num_threads(4)
    dev = torch.device("cpu") if cpu else None
    # along-strike first, then low k first, so a partial run already covers the core design
    files = sorted(OUT.glob(f"{pattern}.mat"),
                   key=lambda f: (f.stem[-2] != "a", int(f.stem.split("_k")[1][0]), f.stem), reverse=reverse)
    t0, n = time.time(), 0
    for f in files:
        z = f.with_name(f.stem + "_zpnn.npz")
        if z.exists() or time.time() - f.stat().st_mtime < 30:  # skip files still being written
            continue
        d = sio.loadmat(f, variable_names=["MS_LR", "PAN_LR", "sensor"])
        sensor = str(np.atleast_1d(d["sensor"])[0]).strip()
        ms, pan = np.moveaxis(d["MS_LR"], -1, 0), d["PAN_LR"]
        # the GPU is shared: start a chip only when enough device memory is free, since on
        # WDDM an oversubscribed device pages to host memory instead of raising an error.
        # When busy, exit (releasing this process's CUDA context); the caller re-runs the stage.
        if not cpu and gpu_free_mib() < MIN_FREE_MIB:
            print(f"{time.strftime('%H:%M')} GPU busy ({gpu_free_mib()} MiB free), stopping after {n}", flush=True)
            return
        for attempt in range(200):
            try:
                ad, _ = zpnn_fuse(ms, pan, sensor, epochs=100, device=dev)
                pre, _ = zpnn_fuse(ms, pan, sensor, epochs=0, device=dev)
                break
            except RuntimeError as e:  # includes torch.cuda.OutOfMemoryError
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"CUDA OOM on {f.stem}, waiting (attempt {attempt + 1})", flush=True)
                time.sleep(60)
        else:
            raise RuntimeError(f"GPU memory not available for {f.stem}")
        if z.exists():  # done meanwhile by another worker
            continue
        tmp = z.with_name(z.stem + "_part.npz")
        np.savez_compressed(tmp, ZPNN=ad[[4, 2, 1]].mean(0), ZPNN0=pre[[4, 2, 1]].mean(0),
                            device=np.array("cpu" if cpu else "cuda"))
        tmp.replace(z)
        del ad, pre
        if not cpu:
            torch.cuda.empty_cache()
        n += 1
        if n % 25 == 0:
            print(f"{n} done, {(time.time() - t0) / n:.1f} s each", flush=True)


def stage_measure():
    from offset import Raster, measure_offset
    mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
    derived = ["unit", "phase_disp", "err", "H_ref_img", "delta", "H0", "inc", "inc_ae"]
    prev, done = pd.DataFrame(), set()
    if (TAB / "rr_gridphase_long.csv").exists():  # reuse earlier measurements (e.g. before Z-PNN finished)
        prev = pd.read_csv(TAB / "rr_gridphase_long.csv")
        prev = prev[~((prev.dir == "c") & (prev.s == 0))].drop(columns=derived)
        done = set(zip(prev.job, prev.dir, prev.s, prev["product"]))
    rows = []
    for j, d, s, stem in conditions():
        f, z = OUT / f"{stem}.mat", OUT / f"{stem}_zpnn.npz"
        if not f.exists():
            continue
        r = sio.loadmat(f, squeeze_me=True, variable_names=["L", "NAMES", "R0", "C0"])
        prods = {str(n): r["L"][:, :, i].astype(np.float64) for i, n in enumerate(r["NAMES"])}
        if z.exists():  # Z-PNN rows are absent until the GPU stage has run for this input
            zz = np.load(z)
            prods["ZPNN"], prods["ZPNN0"] = zz["ZPNN"].astype(np.float64), zz["ZPNN0"].astype(np.float64)
        x0 = j.x0 + float(r["C0"]) * j.ms_res
        y0 = j.y0 - float(r["R0"]) * j.ms_res
        for mm in mk[mk.job == j.job].itertuples():
            for p, a in prods.items():
                if (j.job, d, s, p) in done:
                    continue
                o = measure_offset(Raster(a, x0, y0, j.ms_res), np.array([mm.E, mm.N]), mm.strike,
                                   mm.az_plus, mm.az_minus, aperture=0.0, mode="band")
                rows.append(dict(job=j.job, chip=j.chip, orientation=j.orientation, k=j.k, dir=d, s=s,
                                 marker=mm.marker, product=p, H_true=mm.H_true, ms_res=j.ms_res, **o.flat()))
        if s == 3 and d == "c":
            print(j.job, flush=True)
    r = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    # s = 0 window is shared by both directions
    r = pd.concat([r, r[(r.dir == "a") & (r.s == 0)].assign(dir="c")], ignore_index=True)
    r["unit"] = r.chip + "/" + r.marker
    r["ok"] = r.ok.astype(bool)
    r["phase_disp"] = np.where(r.dir == "a", ((r.k + r.s) % 4) / 4.0, (r.k % 4) / 4.0)
    r["err"] = r.H - r.H_true
    key = ["unit", "dir", "s", "k"]
    ref = r[(r["product"] == "REF") & r.ok].set_index(key).H
    r["H_ref_img"] = ref.reindex(pd.MultiIndex.from_frame(r[key])).values
    r.loc[~r.ok, "H_ref_img"] = np.nan
    r["delta"] = r.H - r.H_ref_img
    b0 = r[(r.k == 0) & r.ok].set_index(["unit", "dir", "s", "product"]).H
    r["H0"] = b0.reindex(pd.MultiIndex.from_frame(r[["unit", "dir", "s", "product"]])).values
    r.loc[~r.ok, "H0"] = np.nan
    r["inc"] = (r.H - r.H0) - r.H_true          # baseline-corrected error (signed)
    r["inc_ae"] = r.inc.abs()
    r.to_csv(TAB / "rr_gridphase_long.csv", index=False)
    print(len(r), "rows")


RNG = np.random.default_rng(49)


def cmed(df, col):
    """Median pooled over crossings with a crossing-level bootstrap 95% CI."""
    df = df[df[col].notna()]
    g = {u: x[col].values for u, x in df.groupby("unit")}
    keys = list(g)
    if not keys:
        return np.nan, np.nan, np.nan, 0
    est = np.median(np.concatenate(list(g.values())))
    idx = RNG.integers(0, len(keys), (NB, len(keys)))
    arrs = [g[k] for k in keys]
    bs = [np.median(np.concatenate([arrs[i] for i in row])) for row in idx]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(df)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def stage_analyse():
    r = pd.read_csv(TAB / "rr_gridphase_long.csv")
    r["ok"] = r.ok.astype(bool)
    rows = []
    for d in DIRS:
        for p in ORDER:
            for s in S:
                gs = r[(r.dir == d) & (r["product"] == p) & (r.s == s)]
                for k in K + ("k>0",):
                    g = gs[gs.k > 0] if k == "k>0" else gs[gs.k == k]
                    ok = g[g.ok]
                    a = cmed(ok[ok.k > 0], "inc_ae") if k != 0 else (np.nan,) * 4
                    si = cmed(ok[ok.k > 0], "inc") if k != 0 else (np.nan,) * 4
                    dl = cmed(ok.assign(ad=ok.delta.abs()), "ad") if p != "REF" else (np.nan,) * 4
                    rows.append(dict(dir=d, product=p, s=s, k=k, n_total=len(g), n_ok=int(g.ok.sum()),
                                     acc_pct=100 * g.ok.mean() if len(g) else np.nan,
                                     med_H=ok.H.median(), med_err=ok.err.median(),
                                     med_inc_ae=a[0], inc_ae_lo=a[1], inc_ae_hi=a[2],
                                     med_inc=si[0], inc_lo=si[1], inc_hi=si[2], n_inc=a[3] if k != 0 else 0,
                                     med_abs_delta=dl[0], abs_delta_lo=dl[1], abs_delta_hi=dl[2], n_delta=dl[3]))
    summ = pd.DataFrame(rows)
    summ["k"] = summ.k.astype(str)
    summ.to_csv(TAB / "rr_gridphase_summary.csv", index=False)

    # Friedman test of the s effect at fixed k (crossings with all four s accepted), Holm over products
    tests = []
    for d in DIRS:
        for resp in ("H", "inc", "delta"):
            for k in K:
                if resp == "inc" and k == 0:
                    continue
                block = []
                for p in ORDER[1:]:
                    g = r[(r.dir == d) & (r["product"] == p) & (r.k == k) & r.ok & r[resp].notna()]
                    w = g.pivot_table(index="unit", columns="s", values=resp, aggfunc="mean").dropna()
                    if len(w) > 4 and w.shape[1] == 4:
                        st = stats.friedmanchisquare(*[w[c] for c in w.columns])
                        rng = (w.max(1) - w.min(1))
                        block.append(dict(dir=d, response=resp, k=k, product=p, n=len(w), chi2=st.statistic,
                                          p=st.pvalue, kendall_W=st.statistic / (len(w) * 3),
                                          med_range_over_s=rng.median()))
                if block:
                    adj = holm([b["p"] for b in block])
                    for b, a in zip(block, adj):
                        b["p_holm"] = a
                    tests += block
    pd.DataFrame(tests).to_csv(TAB / "rr_gridphase_test.csv", index=False)

    # absolute change of the offset between s levels at fixed k on jointly accepted cells
    ch = []
    for d in DIRS:
        for p in ORDER:
            for k in K:
                g = r[(r.dir == d) & (r["product"] == p) & (r.k == k) & r.ok]
                w = g.pivot_table(index="unit", columns="s", values="H", aggfunc="mean")
                pairs = []
                for s1, s2 in itertools.combinations(S, 2):
                    if s1 in w and s2 in w:
                        x = (w[s1] - w[s2]).abs().dropna()
                        pairs.append(pd.DataFrame(dict(unit=x.index, v=x.values, pair=f"{s1}-{s2}")))
                if not pairs:
                    continue
                pp = pd.concat(pairs)
                e = cmed(pp, "v")
                adj = pp[pp.pair.isin(["0-1", "1-2", "2-3"])]
                e1 = cmed(adj, "v")
                full = w.dropna()
                ch.append(dict(dir=d, product=p, k=k, med_abs_change=e[0], lo=e[1], hi=e[2], n_pairs=e[3],
                               n_units=pp.unit.nunique(), med_abs_change_adjacent=e1[0], adj_lo=e1[1],
                               adj_hi=e1[2], n_adj=e1[3], med_range_all4=(full.max(1) - full.min(1)).median(),
                               n_all4=len(full)))
    pd.DataFrame(ch).to_csv(TAB / "rr_gridphase_change.csv", index=False)

    # window check: s = 0 (248-pixel window) against the original RR run (256-pixel window, m = 0).
    # Global, scene-adaptive methods (GSA, BDSD-PC, Z-PNN) depend on window content, so this gives
    # the size of an offset change caused by window content alone, without any grid-phase change.
    old = pd.read_csv(TAB / "rr_long.csv")
    old = old[(old.m == 0) & old.ok.astype(bool)].set_index(["unit", "k", "product"]).H
    a0 = r[(r.dir == "a") & (r.s == 0) & r.ok].set_index(["unit", "k", "product"]).H
    wc = (a0 - old.reindex(a0.index)).abs().dropna().rename("v").reset_index()
    win = []
    for p in ORDER:
        for k in K:
            e = cmed(wc[(wc["product"] == p) & (wc.k == k)], "v")
            win.append(dict(product=p, k=k, med_abs_change_vs_256=e[0], lo=e[1], hi=e[2], n=e[3]))
    pd.DataFrame(win).to_csv(TAB / "rr_gridphase_windowcheck.csv", index=False)

    # baseline-corrected |error| by the MS phase of the displaced side, decoupled from k
    q = r[(r.dir == "a") & (r.k > 0) & r.ok & r.inc_ae.notna()]
    q.groupby(["product", "phase_disp"]).inc_ae.median().unstack().to_csv(TAB / "rr_gridphase_inc_by_phase.csv")

    pd.set_option("display.width", 250)
    cols = ["dir", "product", "s", "k", "n_ok", "n_total", "acc_pct", "med_inc_ae", "inc_ae_lo", "inc_ae_hi",
            "med_inc", "med_abs_delta", "n_delta"]
    print(summ[summ.k.isin(["1", "2"])][cols].round(3).to_string())
    print(pd.DataFrame(tests).round(4).to_string())
    print(pd.DataFrame(ch).round(3).to_string())


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "list":
        stage_list()
    elif st == "zpnn":
        stage_zpnn(sys.argv[2] if len(sys.argv) > 2 else "*",
                   sys.argv[3] if len(sys.argv) > 3 else "cuda", "reverse" in sys.argv[4:])
    elif st == "measure":
        stage_measure()
    elif st == "analyse":
        stage_analyse()
