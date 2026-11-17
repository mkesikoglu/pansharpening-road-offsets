"""Quick-look PAN images of each field-site cluster (for marker identification)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chips  # noqa: E402
from offset import unit  # noqa: E402

sel = pd.read_csv(ROOT / "data/processed/sites_selected.csv")
cov = pd.read_csv(ROOT / "data/processed/site_coverage.csv")
out = ROOT / "results/qc/quicklooks"
out.mkdir(parents=True, exist_ok=True)
HALF = 60.0
for cl, g in sel.groupby("cluster"):
    c = cov[cov.ID.isin(g.ID) & (cov.valid >= 0.95)]
    if c.empty:
        continue
    # least off-nadir acquisition available for the cluster
    inv = chips.INV.drop_duplicates("catalog_id").set_index("catalog_id")
    cid = min(c.catalog_id.unique(), key=lambda k: inv.loc[k, "off_nadir"])
    E, N = g.E.mean(), g.N.mean()
    ch = chips.make_chip(cid, E, N, size_pan=512)
    pan = ch["PAN"]
    n = pan.shape[0]
    ext = [ch["x0"], ch["x0"] + n * ch["pan_res"], ch["y0"] - n * ch["pan_res"], ch["y0"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    lo, hi = np.percentile(pan[pan > 0], [1, 99])
    ax.imshow(pan, cmap="gray", vmin=lo, vmax=hi, extent=ext)
    for _, s in g.iterrows():
        ax.plot(s.E, s.N, "r+", ms=12)
        ax.text(s.E + 2, s.N + 2, f"{s.ID} {s.Feature} H={s.H_pref_m}", color="yellow", fontsize=8)
        if np.isfinite(s.FaultStrike_deg):
            v = unit(s.FaultStrike_deg) * 50
            ax.plot([s.E - v[0], s.E + v[0]], [s.N - v[1], s.N + v[1]], "c--", lw=0.8)
        for col, sty in (("FeatAzim_N_deg", "g-"), ("FeatAzim_S_deg", "m-")):
            if np.isfinite(s[col]):
                v = unit(s[col]) * 25
                ax.plot([s.E - v[0], s.E + v[0]], [s.N - v[1], s.N + v[1]], sty, lw=0.8)
    ax.set_xlim(E - HALF * 1.6, E + HALF * 1.6)
    ax.set_ylim(N - HALF * 1.6, N + HALF * 1.6)
    ax.set_title(f"cluster {cl} | {ch['date']} {ch['platform']} off-nadir {ch['off_nadir']}")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.savefig(out / f"cluster_{cl:02d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(cl, cid, list(g.ID))
