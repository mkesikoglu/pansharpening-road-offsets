"""LaTeX tables and a key-number file generated directly from the result tables,
so that numbers quoted in the manuscript match the analysis outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
OUT = ROOT / "manuscript/tables"
OUT.mkdir(parents=True, exist_ok=True)
LAB = {"PAN": "PAN", "EXP": "EXP", "GSA": "GSA", "BDSD_PC": "BDSD-PC", "MTF_GLP_HPM": "MTF-GLP-HPM",
       "AWLP": "AWLP", "ZPNN": "Z-PNN", "VIS": "Vendor RGB"}
P = list(LAB)
ACQ = {"10300500D9F8D200": "WV-2, 8 Feb", "10300100E291D100": "WV-2, 11 Feb", "10300100E3154100": "WV-2, 28 Feb",
       "1040010082B6D600": "WV-3, 28 Feb", "10300100E383D900": "WV-2, 11 Mar"}
K = {}


def f(x, d=2):
    if pd.isna(x):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


# ------------------------------------------------------------- Table 1 acquisitions
jobs = pd.read_csv(ROOT / "data/processed/jobs.csv")
mk = pd.read_csv(ROOT / "data/processed/markers_measured.csv")
inv = pd.read_csv(ROOT / "data/processed/tile_inventory.csv").drop_duplicates("catalog_id").set_index("catalog_id")
rows = []
for cid, lab in ACQ.items():
    j = jobs[jobs.catalog_id == cid]
    if j.empty:
        continue
    fobs = mk[(mk.kind == "field") & (mk.catalog_id == cid)].ID.nunique()
    cchips = j[(j.kind == "controlled") & (j.k == 0)].job.nunique()
    ccross = mk[(mk.kind == "controlled") & (mk.catalog_id == cid) & mk.job.str.endswith("_k0")].shape[0]
    r = j.iloc[0]
    rows.append(f"{lab} & {cid} & {r.pan_res:.2f} & {r.ms_res:.2f} & {inv.loc[cid, 'off_nadir']:.1f} & "
                f"{inv.loc[cid, 'sun_el']:.1f} & {fobs} & {cchips} & {ccross} \\\\")
(OUT / "tab1_acquisitions.tex").write_text("\n".join(rows), encoding="utf-8")
K["n_controlled_chips"] = int(jobs[(jobs.kind == "controlled") & (jobs.k == 0)].job.nunique())
K["n_crossings"] = int(mk[(mk.kind == "controlled") & mk.job.str.endswith("_k0")].shape[0])
K["n_field_jobs"] = int((jobs.kind == "field").sum())
K["n_jobs"] = int(len(jobs))

# ------------------------------------------------------------- Table 2 field observations
sel = pd.read_csv(ROOT / "data/processed/sites_selected.csv")
scr = pd.read_csv(ROOT / "data/processed/field_marker_screening.csv")
x = sel.merge(scr, on="ID", how="left")
cov = pd.read_csv(ROOT / "data/processed/site_coverage.csv")
nacq = (cov.valid >= 0.95).groupby(cov.ID).sum()
rows = []
for r in x.sort_values("ID").itertuples():
    rng = f"{f(r.H_min_m)}--{f(r.H_max_m)}" if np.isfinite(r.H_min_m) else "--"
    note = "used" if r.include == 1 else ("periodic rows" if "rows" in str(r.screen_note) else "not resolved")
    rows.append(f"{r.ID} & {r.marker} & {r.Feature} & {f(r.H_pref_m)} & {rng} & {r.MsmtMethod} & {r.MsmtType} & "
                f"{int(nacq.get(r.ID, 0))} & {note} \\\\")
(OUT / "tab2_field.tex").write_text("\n".join(rows), encoding="utf-8")
K["n_field_candidates"] = int(len(x))
K["n_field_used"] = int((x.include == 1).sum())
K["n_field_markers_used"] = int(x[x.include == 1].marker.nunique())

# ------------------------------------------------------------- Table 4 quality indices
q = pd.read_csv(T / "quality_long.csv")
FU = ["EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN"]
rows = []
for p in FU:
    g = q[q["product"] == p]
    cell = []
    for ind, d in (("Q2n", 3), ("SAM", 2), ("ERGAS", 2), ("SCC", 3), ("SAM_road", 2), ("D_lambda", 3), ("D_s", 3),
                   ("HQNR", 3)):
        v = g[ind].dropna()
        cell.append(f"{v.median():.{d}f}")
    rows.append(f"{LAB[p]} & " + " & ".join(cell) + " \\\\")
(OUT / "tab4_quality.tex").write_text("\n".join(rows), encoding="utf-8")
K["n_quality_chips"] = int(q.job.nunique())
K["hpm_ergas_gt100"] = int((q[q["product"] == "MTF_GLP_HPM"].ERGAS > 100).sum())
K["quality_median"] = q.groupby("product")[["Q2n", "SAM", "ERGAS", "SCC", "SAM_road", "SAM_bg", "D_lambda", "D_s",
                                            "HQNR"]].median().round(3).to_dict()

# ------------------------------------------------------------- Table 5 controlled results
s = pd.read_csv(T / "controlled_summary.csv")
pr = pd.read_csv(T / "controlled_paired_vs_pan.csv").set_index("product")
sh = pd.read_csv(T / "controlled_shift_vs_pan.csv").set_index("product")
lmm = pd.read_csv(T / "controlled_lmm.csv").set_index("term")
c = pd.read_csv(T / "controlled_long.csv")
rows = []
for p in P:
    r = s[(s["product"] == p) & (s.set == "k>0")].iloc[0]
    r0 = s[(s["product"] == p) & (s.set == "k=0")].iloc[0]
    nunit = c[(c["product"] == p) & c.ok.astype(bool) & (c.k > 0)].unit.nunique()
    base = (f"{LAB[p]} & {r.measurable:.0f} & {nunit} & {f(r.mae_median)} ({f(r.mae_lo)}, {f(r.mae_hi)}) & "
            f"{f(r.rmse)} & {r.p_gt_1:.0f} & {f(r0.mae_median)}")
    if p == "PAN":
        rows.append(base + " & -- & -- & -- \\\\")
    else:
        a, b, l_ = pr.loc[p], sh.loc[p], lmm.loc[p]
        rows.append(base + f" & {f(a.d_median)} ({f(a.d_lo)}, {f(a.d_hi)}) & {a.p_holm:.3f} & "
                    f"{f(b.med_abs_shift)} & {f(l_.ratio)} ({f(l_.lo)}, {f(l_.hi)}) \\\\")
(OUT / "tab5_controlled.tex").write_text("\n".join(rows), encoding="utf-8")
K["controlled"] = {p: dict(measurable=round(float(s[(s["product"] == p) & (s.set == "k>0")].measurable.iloc[0]), 1),
                           mae=round(float(s[(s["product"] == p) & (s.set == "k>0")].mae_median.iloc[0]), 2),
                           rmse=round(float(s[(s["product"] == p) & (s.set == "k>0")].rmse.iloc[0]), 2),
                           p_gt1=round(float(s[(s["product"] == p) & (s.set == "k>0")].p_gt_1.iloc[0]), 1),
                           mae_k0=round(float(s[(s["product"] == p) & (s.set == "k=0")].mae_median.iloc[0]), 2))
                   for p in P}
K["paired"] = pr[["n", "d_median", "d_lo", "d_hi", "frac_better", "p_wilcoxon", "p_holm"]].round(4).to_dict("index")
K["shift"] = sh[["n", "med_abs_shift", "lo", "hi", "p90_abs_shift", "frac_gt_05"]].round(3).to_dict("index")
K["lmm"] = lmm.round(3).to_dict("index")
K["rho"] = pd.read_csv(T / "quality_vs_error_rho.csv").set_index("index").round(3).to_dict("index")
K["by_acq"] = pd.read_csv(T / "controlled_by_acq.csv").set_index("product").round(2).to_dict("index")
K["units_by_acq"] = pd.read_csv(T / "controlled_units_by_acq.csv").set_index("acq").unit.to_dict()
K["runtime"] = pd.read_csv(T / "runtime.csv", index_col=0).iloc[:, 0].round(2).to_dict()
K["sensitivity"] = pd.read_csv(T / "sensitivity_cnr.csv").round(2).to_dict("records")

# ------------------------------------------------------------- field numbers
fl = pd.read_csv(T / "field_long.csv")
fl["ok"] = fl.ok.astype(bool)
K["field_measurable"] = fl.groupby("product").ok.mean().mul(100).round(1).to_dict()
K["field_n_ok"] = fl[fl.ok].groupby("product").size().to_dict()
K["field_obs_any"] = sorted(fl[fl.ok].ID.unique().tolist())
m01 = fl[fl.ok & fl.ID.isin([1, 2])]
K["M01_range"] = [round(float(m01.H.min()), 2), round(float(m01.H.max()), 2)]
K["M01_by_acq_PAN"] = m01[m01["product"] == "PAN"].groupby(["ID", "catalog_id"]).H.first().round(2).rename(lambda i: str(i)).to_dict()
K["M01_spread_products"] = m01.groupby(["ID", "catalog_id"]).H.agg(lambda v: v.max() - v.min()).round(2).rename(lambda i: str(i)).to_dict()
K["parallax"] = pd.read_csv(T / "field_parallax.csv").dropna(subset=["dH_parallax"]).round(2).to_dict("records")
def _keys(o):
    if isinstance(o, dict):
        return {(" / ".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_keys(v) for v in o]
    return o


K = _keys(K)
(T / "key_numbers.json").write_text(json.dumps(K, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: K[k] for k in ("n_controlled_chips", "n_crossings", "n_field_used", "n_field_markers_used",
                                    "hpm_ergas_gt100", "field_obs_any", "M01_range")}, default=str))
