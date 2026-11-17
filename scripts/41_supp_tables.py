"""LaTeX table bodies for the supplementary material (manuscript/supp_tables/*.tex)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
V = T / "v2"
O = ROOT / "manuscript/supp_tables"
O.mkdir(exist_ok=True)
P = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
LAB = {"PAN": "PAN", "EXP": "EXP", "GSA": "GSA", "BDSD_PC": "BDSD-PC", "MTF_GLP_HPM": "MTF-GLP-HPM", "AWLP": "AWLP",
       "ZPNN": "Z-PNN", "VIS": "Vendor RGB", "ZPNN0": "Z-PNN, pretrained", "REF": "Reference (original MS)"}
ACQ = {"10300500D9F8D200": "8 Feb", "10300100E291D100": "11 Feb", "10300100E3154100": "28 Feb, WV-2",
       "1040010082B6D600": "28 Feb, WV-3", "10300100E383D900": "11 Mar"}


def f(x, d=2):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci(e, lo, hi, d=2):
    return f"{f(e, d)} [{f(lo, d)}, {f(hi, d)}]"


def w(name, rows):
    (O / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


# S1 reduced-resolution summary (already built for the main text earlier)
src = ROOT / "manuscript/tables/tab8_rr.tex"
if src.exists():
    (O / "s_rr.tex").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
# S2 epsilon sensitivity
lm = pd.read_csv(V / "lmm_eps.csv")
rows = []
for p in P[1:]:
    cells = []
    for eps in (0.001, 0.01, 0.05, 0.1):
        r = lm[(lm.model == "primary") & np.isclose(lm.eps, eps) & (lm["product"] == p)].iloc[0]
        cells.append(f"{f(r.ratio)} [{f(r.lo)}, {f(r.hi)}]")
    a = lm[(lm.model == "acquisition fixed effect") & (lm["product"] == p)].iloc[0]
    rows.append(f"{LAB[p]} & " + " & ".join(cells) + f" & {f(a.ratio)} [{f(a.lo)}, {f(a.hi)}] \\\\")
w("s_eps.tex", rows)
# S3 deviation classes
dc = pd.read_csv(V / "deviation_classes.csv").set_index("product").reindex(P)
w("s_dev.tex", [f"{LAB[p]} & {int(r.measured_pairs)} & {int(r.events)} & {int(r['different edge pair'])} & "
                f"{int(r.events - r['different edge pair'])} \\\\" for p, r in dc.iterrows()])
# S4 cross-correlation estimator
xs = pd.read_csv(V / "xcorr_summary.csv").set_index("product").reindex(P)
w("s_xcorr.tex", [f"{LAB[p]} & {int(r.n_success)}/{int(r.n_total)} & {f(r.success_pct, 1)} & {f(r.main_success_pct, 1)} & "
                  f"{ci(r.med, r.lo, r.hi)} & {ci(r.cc_med, r.cc_lo, r.cc_hi)} & {f(r.inc_med, 3)} & {f(r.inc_gt05, 1)} \\\\"
                  for p, r in xs.iterrows()])
# S5 intensity definitions
if (V / "intensity_summary.csv").exists():
    it = pd.read_csv(V / "intensity_summary.csv")
    rows = []
    for p in ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]:
        g = it[it["product"] == p].set_index("intensity")
        cells = [f"{f(g.loc[i].success_pct, 1)} / {f(g.loc[i].med)} / {f(g.loc[i].inc_gt05, 1)}" for i in ("RGB", "PANW", "NIR1", "PC1")]
        rows.append(f"{LAB[p]} & " + " & ".join(cells) + " \\\\")
    w("s_intens.tex", rows)
# S6 field aperture and support length for M01
if (V / "field_aperture.csv").exists():
    fa = pd.read_csv(V / "field_aperture.csv")
    fa = fa[fa.marker == "M01"]
    rows = []
    for (kind, ID, cid), g in fa.groupby(["kind", "ID", "catalog_id"]):
        g = g.set_index("value")
        vals = sorted(g.index)
        cells = [f"{int(v)}: ${f(g.loc[v]['min'])}$ to ${f(g.loc[v]['max'])}$" for v in vals]
        rows.append(f"{'$A$' if kind == 'aperture' else '$L$'} & {ID} & {ACQ.get(cid, cid)} & " + ", ".join(cells) + " \\\\")
    w("s_field.tex", rows)
# S7 Z-PNN ablation
za = pd.read_csv(V / "zpnn_ablation_summary.csv", header=[0, 1], index_col=0)
rows = []
for cfg in ["0/s0", "25/s0", "100/s0", "100/s1", "200/s0"]:
    r = za.loc[cfg]
    it_, sd = cfg.split("/")
    rows.append(f"{it_} & {sd[1:]} & {int(r[('dH', 'count')])} & {f(r[('dH', 'median')])} & {f(r[('dH', 'max')])} & "
                f"{f(r[('du', 'median')])} & {f(r[('du', 'max')])} \\\\")
w("s_ablation.tex", rows)
# S8 registration per acquisition
reg = pd.read_csv(T / "registration.csv")
u = reg.drop_duplicates("job")
rows = []
for cid, g in u.groupby("catalog_id"):
    loc = reg[reg.catalog_id == cid]
    rows.append(f"{ACQ[cid]} & {len(g)} & {f(g.dE_chip_m.median())} & {f(g.dN_chip_m.median())} & {f(g.mag_chip_m.median())} & "
                f"{f(loc.mag_local_m.median())} & {f(loc.mag_local_m.quantile(.9))} \\\\")
w("s_reg.tex", rows)
# S9 side-dependent misregistration
if (V / "side_summary.csv").exists():
    ss = pd.read_csv(V / "side_summary.csv")
    w("s_side.tex", [f"{LAB.get(r['product'], r['product'])} & {int(r.n)} & {ci(r.med_frac, r.lo, r.hi)} & {f(r.share_follow_ms, 1)} & "
                     f"{f(r.success_pct, 1)} \\\\" for _, r in ss.iterrows()])
# main-text tables moved to the supplement
import shutil
for src, dst in (("tab2_field.tex", "s_fieldobs.tex"), ("tab4_quality.tex", "s_quality.tex"), ("tab6_models.tex", "s_models.tex")):
    shutil.copy(ROOT / "manuscript/tables" / src, O / dst)
print("supp tables written")
