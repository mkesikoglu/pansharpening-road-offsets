"""Agreement and proxy-validity statistics for the blinded manual adjudication.

Inputs (exported from validation/blind_gallery/index.html)
  validation/ratings/adjudication_R1*.csv, adjudication_R2*.csv   (one file per rater)
  validation/ratings/adjudication_consensus*.csv                  (optional, disagreements only)
  validation/gallery_blind_key.csv
Outputs
  results/tables/v2/gallery_index.csv        (rater, agreement and consensus columns added)
  results/tables/v2/adjudication_summary.csv
  results/tables/v2/adjudication_confusion.csv, adjudication_proxy.csv, adjudication_by_product.csv
Confidence intervals resample the 20 crossing units (cluster bootstrap), never single events.
Usage: python scripts/45_adjudication_stats.py [ratings_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables/v2"
RD = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "validation/ratings"
CLASSES = ["same_physical_edge", "different_physical_edge", "invalid_tracking", "ambiguous"]
B = 2000
rng = np.random.default_rng(7)


def latest(pattern):
    f = sorted(RD.glob(pattern))
    return pd.read_csv(f[-1], dtype=str).fillna("") if f else None


key = pd.read_csv(ROOT / "validation/gallery_blind_key.csv")
r1, r2, cons = latest("adjudication_R1*.csv"), latest("adjudication_R2*.csv"), latest("adjudication_consensus*.csv")
if r1 is None or r2 is None:
    sys.exit(f"Need adjudication_R1*.csv and adjudication_R2*.csv in {RD}")
d = key.copy()
for tag, r in (("rater1", r1), ("rater2", r2)):
    r = r.set_index("code")
    d[f"adjudication_{tag}"] = d.code.map(r["class"])
    d[f"adjudication_{tag}_confidence"] = d.code.map(r["confidence"])
    d[f"adjudication_{tag}_note"] = d.code.map(r["note"])
bad = d[~d.adjudication_rater1.isin(CLASSES) | ~d.adjudication_rater2.isin(CLASSES)]
if len(bad):
    sys.exit(f"Missing or invalid classes for {len(bad)} events: {', '.join(bad.code)}")
d["adjudication_agreement"] = d.adjudication_rater1 == d.adjudication_rater2
d["adjudication_consensus"] = np.where(d.adjudication_agreement, d.adjudication_rater1, "")
d["adjudication_consensus_note"] = ""
if cons is not None:
    cs = cons.set_index("code")
    m = ~d.adjudication_agreement & d.code.isin(cs.index)
    d.loc[m, "adjudication_consensus"] = d.loc[m, "code"].map(cs["class"])
    d.loc[m, "adjudication_consensus_note"] = d.loc[m, "code"].map(cs["note"])
open_ = (d.adjudication_consensus == "").sum()


def kappa(a, b):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in CLASSES)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def proxy(df, col):
    """Proxy (automated different-edge-pair label) against manual same/different."""
    x = df[df[col].isin(CLASSES[:2])]
    pos, man = x.cls == "different edge pair", x[col] == "different_physical_edge"
    tp, fp = int((pos & man).sum()), int((pos & ~man).sum())
    fn, tn = int((~pos & man).sum()), int((~pos & ~man).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return dict(n=len(x), tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=div(tp, tp + fn),
                specificity=div(tn, tn + fp), ppv=div(tp, tp + fp), npv=div(tn, tn + fn))


def stats(df):
    po, k = kappa(df.adjudication_rater1.values, df.adjudication_rater2.values)
    out = dict(agreement=po, kappa=k)
    if open_ == 0:
        out.update({f"consensus_{c}": np.mean(df.adjudication_consensus == c) for c in CLASSES})
        out.update({f"proxy_{a}": b for a, b in proxy(df, "adjudication_consensus").items()
                    if a in ("sensitivity", "specificity", "ppv", "npv")})
    return out


units = d.unit.unique()
groups = {u: d[d.unit == u] for u in units}
point = stats(d)
boot = pd.DataFrame([stats(pd.concat([groups[u] for u in rng.choice(units, len(units))])) for _ in range(B)])
summ = pd.DataFrame({"estimate": pd.Series(point), "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)})
summ.loc["n_events"] = [len(d), np.nan, np.nan]
summ.loc["n_units"] = [len(units), np.nan, np.nan]
summ.loc["n_disagreements"] = [int((~d.adjudication_agreement).sum()), np.nan, np.nan]
summ.loc["n_consensus_open"] = [open_, np.nan, np.nan]
summ.index.name = "statistic"
summ.to_csv(T / "adjudication_summary.csv")

conf = pd.crosstab(pd.Categorical(d.adjudication_rater1, CLASSES), pd.Categorical(d.adjudication_rater2, CLASSES),
                   rownames=["rater1"], colnames=["rater2"], dropna=False)
conf.to_csv(T / "adjudication_confusion.csv")
counts = pd.DataFrame({c: d[c].value_counts().reindex(CLASSES, fill_value=0)
                       for c in ("adjudication_rater1", "adjudication_rater2", "adjudication_consensus")})
counts.to_csv(T / "adjudication_counts.csv")
prox = pd.DataFrame({c: proxy(d, c) for c in ("adjudication_rater1", "adjudication_rater2", "adjudication_consensus")})
prox.to_csv(T / "adjudication_proxy.csv")
if open_ == 0:
    pd.crosstab([d["product"]], pd.Categorical(d.adjudication_consensus, CLASSES), dropna=False) \
        .to_csv(T / "adjudication_by_product.csv")

# LaTeX macros and Table S15 body for the manuscript (only once consensus is complete)
if open_ == 0:
    MS = ROOT / "manuscript/supp_tables"
    ci_ = lambda s, k=1, d=2: f"{k * summ.loc[s, 'ci_low']:.{d}f}--{k * summ.loc[s, 'ci_high']:.{d}f}"  # noqa: E731
    pc, pos = prox["adjudication_consensus"], d.cls == "different edge pair"
    cc = counts["adjudication_consensus"]
    mac = {"AdjAgree": f"{100 * point['agreement']:.0f}", "AdjAgreeCI": ci_("agreement", 100, 0),
           "AdjKappa": f"{point['kappa']:.2f}", "AdjKappaCI": ci_("kappa"),
           "AdjSame": cc[CLASSES[0]], "AdjDiff": cc[CLASSES[1]], "AdjInvalid": cc[CLASSES[2]], "AdjAmb": cc[CLASSES[3]],
           "AdjProxyPos": int(pos.sum()),
           "AdjProxyPosDiff": int((pos & (d.adjudication_consensus == CLASSES[1])).sum()),
           "AdjProxyPosSame": int((pos & (d.adjudication_consensus == CLASSES[0])).sum()),
           "AdjPPV": f"{100 * pc.ppv:.0f}", "AdjPPVCI": ci_("proxy_ppv", 100, 0),
           "AdjSens": f"{100 * pc.sensitivity:.0f}", "AdjSensCI": ci_("proxy_sensitivity", 100, 0), "AdjSpec": f"{100 * pc.specificity:.0f}",
           "AdjProxyNeg": int((~pos).sum()),
           "AdjProxyNegDiff": int((~pos & (d.adjudication_consensus == CLASSES[1])).sum()),
           "AdjProxyNegSame": int((~pos & (d.adjudication_consensus == CLASSES[0])).sum()),
           "AdjDisagree": int((~d.adjudication_agreement).sum())}
    (MS / "adjudication_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n", encoding="utf-8")
    lab = {"adjudication_rater1": "Rater 1", "adjudication_rater2": "Rater 2", "adjudication_consensus": "Consensus"}
    body = [f"{lab[k]} & " + " & ".join(str(counts.loc[c_, k]) for c_ in CLASSES) + " \\\\" for k in lab]
    for name, m in (("Consensus, indicator: edge-pair change", pos), ("Consensus, indicator: not flagged", ~pos)):
        body.append(f"{name} & " + " & ".join(str(int((m & (d.adjudication_consensus == c_)).sum())) for c_ in CLASSES)
                    + " \\\\")
    (MS / "s_adjud.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

gi = pd.read_csv(T / "gallery_index.csv")
gi = gi.drop(columns=[c for c in gi.columns if c.startswith("adjudication") or c == "code"])
cols = [c for c in d.columns if c.startswith("adjudication")]
gi = gi.merge(d[["event", "code"] + cols], on="event", how="left")
gi.to_csv(T / "gallery_index.csv", index=False)

print(summ.round(3).to_string())
print("\nRater 1 x rater 2\n", conf.to_string())
print("\nClass counts\n", counts.to_string())
print("\nProxy vs manual (same/different only)\n", prox.round(3).to_string())
if open_:
    print(f"\n{open_} uyuşmazlık için uzlaşı kararı bekleniyor (index.html, Uzlaşı modu).")
