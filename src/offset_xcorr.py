"""Second, edge-free offset estimator used as a robustness check.

On each side of the fault, intensity profiles perpendicular to the marker are
sampled at the same stations as in offset.py, with the initial (map) azimuth and
without refinement, and their median gives one mean profile per side. The offset
is the lag that maximises the normalised cross-correlation of the two mean
profiles, refined by a parabola and converted from the profile axis to the strike
direction. No edge is detected, paired or tracked.
"""
import numpy as np

from offset import unit

XPARAMS = dict(step_t=1.0, length=40.0, t_min_floor=5.0, dw=0.25, strip=1.0, half_window=20.0,
               max_offset=15.0, min_corr=0.8)


def _mean_profile(img, p0, phi, az, sign, aperture, prm, w):
    s, n = unit(phi), unit(phi + 90.0)
    d = unit(az)
    if np.dot(d, n) * sign < 0:
        d = -d
    e = np.array([d[1], -d[0]])
    if np.dot(e, s) < 0:
        e = -e
    dv = abs(np.dot(d, n))
    t_min = max(aperture / 2.0, prm["t_min_floor"])
    t = np.arange(t_min, t_min + prm["length"] + 1e-9, prm["step_t"])
    P = 0.0
    offs = np.linspace(-prm["strip"] / 2, prm["strip"] / 2, 3)
    for o in offs:
        pts = (p0[None, None, :] + ((t / dv + o)[:, None, None] * d[None, None, :]) + w[None, :, None] * e[None, None, :])
        P = P + img.sample(pts)
    return np.median(P / len(offs), axis=0), abs(np.dot(e, s))


def measure_offset_xcorr(img, p0, phi, az_plus, az_minus, aperture=0.0, prm=XPARAMS):
    p0 = np.asarray(p0, float)
    out = dict(H=np.nan, r=np.nan, ok=False)
    if not np.all(np.isfinite([p0[0], p0[1], phi, az_plus, az_minus])):
        return out
    hw, dw = prm["half_window"], prm["dw"]
    W = hw + prm["max_offset"] + 1.0
    w = np.arange(-W, W + 1e-9, dw)
    gp, esp = _mean_profile(img, p0, phi, az_plus, +1, aperture, prm, w)
    gm, esm = _mean_profile(img, p0, phi, az_minus, -1, aperture, prm, w)
    es = 0.5 * (esp + esm)
    core = np.abs(w) <= hw
    a = gp[core] - gp[core].mean()
    na = np.linalg.norm(a) + 1e-12
    lags = np.arange(-int(round(prm["max_offset"] * es / dw)), int(round(prm["max_offset"] * es / dw)) + 1)
    idx = np.nonzero(core)[0]
    r = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        j = idx - L
        if j.min() < 0 or j.max() >= len(w):
            continue
        b = gm[j] - gm[j].mean()
        r[i] = float(a @ b / (na * (np.linalg.norm(b) + 1e-12)))
    if not np.isfinite(r).any():
        return out
    k = int(np.nanargmax(r))
    sub = 0.0
    if 0 < k < len(r) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den != 0:
            sub = float(np.clip(0.5 * (r[k - 1] - r[k + 1]) / den, -0.5, 0.5))
    # gm(w - lag) matches gp(w): the + side lies 'lag' further along +e than the - side
    H = (lags[k] + sub) * dw / es
    out.update(H=float(H), r=float(r[k]), ok=bool(r[k] >= prm["min_corr"] and abs(H) <= prm["max_offset"]))
    return out
