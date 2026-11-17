"""Sanity test of the offset estimator on a synthetic road crossing a fault."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offset import Raster, measure_offset, unit  # noqa: E402


def synth(H, phi, az, res=0.5, width=7.0, noise=0.05, seed=0, blur=1.0):
    rng = np.random.default_rng(seed)
    N = int(200 / res)
    x0, y0 = 0.0, N * res
    ss = 8  # supersampling factor (area-averaged rendering, no aliasing)
    yy, xx = np.mgrid[0:N * ss, 0:N * ss]
    X = x0 + (xx + 0.5) * res / ss
    Y = y0 - (yy + 0.5) * res / ss
    p0 = np.array([100.0, 100.0])
    s, n = unit(phi), unit(phi + 90)
    D = np.stack([X - p0[0], Y - p0[1]], -1)
    u, v = D @ s, D @ n
    d = unit(az)
    e = np.array([d[1], -d[0]])
    # + side shifted by +H/2 along s, - side by -H/2
    shift = np.where(v > 0, H / 2, -H / 2)
    Dm = D - shift[..., None] * s
    dist = Dm @ e
    img = 0.3 + 0.4 * (np.abs(dist) < width / 2)
    img = img.reshape(N, ss, N, ss).mean((1, 3))
    from scipy import ndimage
    img = ndimage.gaussian_filter(img, blur) + noise * rng.standard_normal(img.shape)
    return Raster(img, x0, y0, res), p0


if __name__ == "__main__":
    for H, phi, az in [(4.0, 60, 150), (2.5, 60, 120), (-3.0, 45, 0), (6.0, 60, 100)]:
        img, p0 = synth(H, phi, az)
        r = measure_offset(img, p0, phi, az + 4, az - 3, aperture=4)
        print(f"H={H:5.2f} phi={phi} az={az} -> {r.H:6.3f} +/- {r.se_H:.3f} ok={r.ok} {r.reason}")
