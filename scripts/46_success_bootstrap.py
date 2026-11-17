"""Crossing-cluster bootstrap of the measurement-success contrasts against PAN, as a
sensitivity check of the logistic GEE (which uses the default robust covariance).
Output results/tables/v2/success_bootstrap.csv"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables/v2"
B = 2000
rng = np.random.default_rng(11)

c = pd.read_csv(T / "controlled_long_v2.csv")
c["ok"] = c.ok.astype(bool)
# successes and attempts per crossing and product; the vendor product lacks 11 March
g = c.groupby(["unit", "product"]).ok.agg(["sum", "count"]).unstack("product")
s, n = g["sum"].fillna(0), g["count"].fillna(0)
units = s.index.values
prods = [p for p in s.columns if p != "PAN"]


def diff(idx, p):
    m = n.iloc[idx][p] > 0  # crossings where the product was attempted
    a = s.iloc[idx][p][m].sum() / n.iloc[idx][p][m].sum()
    b = s.iloc[idx]["PAN"][m].sum() / n.iloc[idx]["PAN"][m].sum()
    return a - b


rows = []
allidx = np.arange(len(units))
boot = rng.integers(0, len(units), size=(B, len(units)))
for p in prods:
    est = diff(allidx, p)
    bs = np.array([diff(i, p) for i in boot])
    pval = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
    rows.append(dict(product=p, diff_pp=100 * est, lo=100 * np.quantile(bs, .025), hi=100 * np.quantile(bs, .975),
                     p_boot=pval))
r = pd.DataFrame(rows)
r["p_holm"] = multipletests(r.p_boot, method="holm")[1]
r.to_csv(T / "success_bootstrap.csv", index=False)
print(r.round(3).to_string(index=False))
