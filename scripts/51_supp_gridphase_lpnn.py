"""LaTeX rows for Tables S16 (grid-phase experiment), S17 (Lambda-PNN) and S18
(paired uniform misregistration).

Inputs:  results/tables/v2/rr_gridphase_change.csv (scripts/49_rr_gridphase.py)
         results/tables/v2/lpnn_summary.csv       (scripts/50_lambda_pnn.py)
         results/tables/v2/review5_rr_misreg.csv  (scripts/47_review5_checks.py)
Outputs: manuscript/supp_tables/s_gridphase.tex, s_lpnn.tex, s_misreg.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results/tables/v2"
OUT = ROOT / "manuscript/supp_tables"
NAMES = {"PAN": "PAN", "EXP": "EXP", "GSA": "GSA", "BDSD_PC": "BDSD-PC",
         "MTF_GLP_HPM": "MTF-GLP-HPM", "AWLP": "AWLP", "ZPNN": "Z-PNN",
         "ZPNN0": "Z-PNN (pretrained)", "VIS": "Vendor", "LPNN": r"$\lambda$-PNN"}


def num(x, nd):
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s  # no negative zero


def cell(est, lo, hi, nd=3):
    return f"{num(est, nd)} [{num(lo, nd)}, {num(hi, nd)}]"


def gridphase():
    d = pd.read_csv(V / "rr_gridphase_change.csv")
    d = d[d["product"] != "REF"]
    rows = []
    for direction, label in (("a", "Along strike"), ("c", "Across strike")):
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\")
        for p in ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]:
            s = d[(d.dir == direction) & (d["product"] == p)].set_index("k")
            cells = [cell(s.loc[k, "med_abs_change"], s.loc[k, "lo"], s.loc[k, "hi"]) for k in range(4)]
            counts = "/".join(str(int(s.loc[k, "n_pairs"])) for k in range(4))
            rows.append(f"{NAMES[p]} & " + " & ".join(cells) + f" & {counts}" + r" \\")
    (OUT / "s_gridphase.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def misreg():
    """Table S18: uniform misregistration, paired on cells accepted with and without it."""
    d = pd.read_csv(V / "review5_rr_misreg.csv")
    rows = []
    for p in ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "ZPNN0"]:
        for m, lab in ((1, "1/4"), (2, "1/2")):
            r = d[(d["product"] == p) & (d.m == m)].iloc[0]
            rows.append(" & ".join([
                NAMES[p] if m == 1 else "", lab, f"{int(r.n_cells)} ({int(r.n_crossings)})",
                cell(r.med_signed, r.signed_lo, r.signed_hi, 2),
                cell(r.med_abs, r.abs_lo, r.abs_hi, 2),
                cell(r.pct_abs_gt05, r.pct_lo, r.pct_hi, 0),
            ]) + r" \\")
    (OUT / "s_misreg.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def lpnn():
    s = pd.read_csv(V / "lpnn_summary.csv")

    def get(metric, product):
        r = s[(s.metric == metric) & (s["product"] == product)]
        return r.iloc[0] if len(r) else None

    rows = []
    for p in ["PAN", "ZPNN", "LPNN"]:
        a = get("acceptance_pct", p)
        e = get("median_abs_err_k_gt0_m", p)
        c = get("median_abs_err_common9_m", p)
        g = get("pct_baseline_corrected_gt_0p5m", p)
        rows.append(" & ".join([
            NAMES[p],
            f"{int(a.n)}/{int(a.N)}",
            cell(a.estimate, a.lo, a.hi, 1),
            f"{int(e.n)}", cell(e.estimate, e.lo, e.hi, 2),
            cell(c.estimate, c.lo, c.hi, 2),
            f"{int(g.N)}", cell(g.estimate, g.lo, g.hi, 1),
        ]) + r" \\")
    (OUT / "s_lpnn.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gridphase()
    lpnn()
    misreg()
    for name in ("s_gridphase.tex", "s_lpnn.tex", "s_misreg.tex"):
        print((OUT / name).read_text(encoding="utf-8"))
