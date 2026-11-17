"""Deterministic estimation of the strike-slip offset of a linear marker from a
single image channel.

Local frame: origin p0 on the fault, unit vector s along strike (azimuth phi),
unit vector n = s rotated 90 deg clockwise; u = (x - p0).s, v = (x - p0).n.
The marker is modelled on each side of the fault as a straight line; the
offset is the difference of the u-coordinates where the two lines meet v = 0.
H = u(+) - u(-) is positive for left-lateral slip.

Markers are either 'band' (road, ditch, canal: two opposite edges, the centre
line is used) or 'edge' (field boundary: one step edge).
All lengths are in metres. Parameters are fixed a priori (PARAMS).
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

PARAMS = dict(
    step_t=1.0,        # station spacing along the marker (m)
    length=40.0,       # marker length used on each side (m)
    t_min_floor=5.0,   # minimum distance from the fault (m)
    dw=0.25,           # profile sampling interval (m)
    strip=1.0,         # along-marker averaging length per station (m)
    sigma_d=0.5,       # Gaussian derivative scale (m)
    search_u=15.0,     # search half-range along strike around p0 (m)
    max_offset=15.0,   # largest admissible |H| (m)
    width_min=1.5,     # admissible band width (m)
    width_max=25.0,
    width_tol=0.3,     # relative width tolerance between the two sides
    az_search=10.0,    # half-range of azimuth refinement (deg)
    az_step=1.0,
    rel_cnr=0.5,       # candidates retained if CNR >= rel_cnr * best CNR
    edge_tol=1.5,      # per-station search tolerance around the template edge (m)
    min_cnr=3.0,       # minimum template contrast-to-noise ratio
    min_valid=0.5,     # minimum fraction of stations with a valid detection
    max_rmse=1.5,      # maximum line-fit RMSE (m)
    huber_c=1.345,
    n_iter=2,          # re-centring iterations after the initial detection
)


def unit(az_deg):
    a = np.deg2rad(az_deg)
    return np.array([np.sin(a), np.cos(a)])


@dataclass
class Raster:
    """Single-band image with north-up geometry (UTM): x0, y0 = upper-left corner."""
    data: np.ndarray
    x0: float
    y0: float
    res: float

    def __post_init__(self):
        # cubic B-spline coefficients computed once (identical to prefilter=True per call)
        self._coef = ndimage.spline_filter(np.asarray(self.data, dtype=np.float64), order=3, mode="nearest")

    def sample(self, xy):
        col = (xy[..., 0] - self.x0) / self.res - 0.5
        row = (self.y0 - xy[..., 1]) / self.res - 0.5
        return ndimage.map_coordinates(self._coef, [row, col], order=3, mode="nearest", prefilter=False)


@dataclass
class SideResult:
    ok: bool = False
    reason: str = ""
    u0: float = np.nan        # intersection with the fault (m along strike)
    se_u0: float = np.nan
    az: float = np.nan        # refined marker azimuth (deg)
    width: float = np.nan
    cnr: float = np.nan
    rmse: float = np.nan
    frac_valid: float = np.nan
    polarity: int = 0


@dataclass
class OffsetResult:
    H: float = np.nan
    se_H: float = np.nan
    ok: bool = False
    reason: str = ""
    plus: SideResult = field(default_factory=SideResult)
    minus: SideResult = field(default_factory=SideResult)

    def flat(self):
        d = dict(H=self.H, se_H=self.se_H, ok=self.ok, reason=self.reason)
        for k, s in (("p", self.plus), ("m", self.minus)):
            for kk, vv in asdict(s).items():
                d[f"{k}_{kk}"] = vv
        return d


class _Side:
    """Profiles, template and candidates for one side of the fault."""

    def __init__(self, img, p0, phi, az0, sign, aperture, prm):
        self.img, self.p0, self.sign, self.prm = img, np.asarray(p0, float), sign, prm
        self.s, self.n = unit(phi), unit(phi + 90.0)
        t_min = max(aperture / 2.0, prm["t_min_floor"])
        self.t = np.arange(t_min, t_min + prm["length"] + 1e-9, prm["step_t"])
        W = prm["search_u"] + prm["width_max"] / 2 + 3.0
        self.w = np.arange(-W, W + 1e-9, prm["dw"])
        best = (-1.0, az0)
        for daz in np.arange(-prm["az_search"], prm["az_search"] + 1e-9, prm["az_step"]):
            G = self._derivs(az0 + daz, 0.0)[0]
            sh = float(np.sum(np.median(G, axis=0) ** 2))
            if sh > best[0]:
                best = (sh, az0 + daz)
        self.az, self.base_u = best[1], 0.0
        self._update()

    def _geom(self, az):
        d = unit(az)
        if np.dot(d, self.n) * self.sign < 0:
            d = -d
        e = np.array([d[1], -d[0]])
        if np.dot(e, self.s) < 0:  # profile axis oriented along +strike on both sides
            e = -e
        return d, e

    def _derivs(self, az, base_u):
        d, e = self._geom(az)
        dv = abs(np.dot(d, self.n))
        c = self.p0 + base_u * self.s
        offs = np.linspace(-self.prm["strip"] / 2, self.prm["strip"] / 2, 3)
        P = 0.0
        for o in offs:
            pts = (c[None, None, :] + ((self.t / dv + o)[:, None, None] * d[None, None, :])
                   + self.w[None, :, None] * e[None, None, :])
            P = P + self.img.sample(pts)
        P = P / len(offs)
        G = ndimage.gaussian_filter1d(P, self.prm["sigma_d"] / self.prm["dw"], order=1, axis=-1,
                                      mode="nearest")
        return G, d, e, dv

    def _update(self):
        self.G, self.d, self.e, self.dv = self._derivs(self.az, self.base_u)
        self.tmpl = np.median(self.G, axis=0)
        self.noise = 1.4826 * np.median(np.abs(self.G - self.tmpl[None, :])) + 1e-12
        self.es = abs(np.dot(self.e, self.s))

    def candidates(self, mode, wc_lim):
        """(cnr, ia, ib, polarity, width, w_centre) ; for 'edge' ia == ib."""
        g, w, prm = self.tmpl, self.w, self.prm
        pk, tr = find_peaks(g)[0], find_peaks(-g)[0]
        out = []
        if mode == "edge":
            for i in pk:
                out.append((abs(g[i]) / self.noise, i, i, +1, 0.0, w[i]))
            for i in tr:
                out.append((abs(g[i]) / self.noise, i, i, -1, 0.0, w[i]))
        else:
            for i in pk:
                for j in tr:
                    a, b, pol = (i, j, +1) if i < j else (j, i, -1)
                    width = w[b] - w[a]
                    if prm["width_min"] <= width <= prm["width_max"]:
                        c = min(abs(g[a]), abs(g[b])) / self.noise
                        out.append((c, a, b, pol, width, 0.5 * (w[a] + w[b])))
        return [c for c in out if abs(c[5]) <= wc_lim]

    def u_of(self, wc):
        """Along-strike coordinate where a marker line centred at profile position wc meets v = 0."""
        p_on = self.p0 + self.base_u * self.s + wc * self.e
        lam = -np.dot(p_on - self.p0, self.n) / np.dot(self.d, self.n)
        return float(np.dot(p_on + lam * self.d - self.p0, self.s))

    def track(self, cand, mode):
        prm, w = self.prm, self.w
        cnr, a, b, pol, width, wc = cand
        res = SideResult(polarity=int(pol), width=float(width), cnr=float(cnr))
        for it in range(prm["n_iter"] + 1):
            tol = int(round(prm["edge_tol"] / prm["dw"]))
            pos, valid = np.full(len(self.t), np.nan), np.zeros(len(self.t), bool)
            for k in range(len(self.t)):
                g = self.G[k]
                ia = slice(max(a - tol, 1), min(a + tol + 1, len(w) - 1))
                ka = ia.start + int(np.argmax(pol * g[ia]))
                if abs(g[ka]) / self.noise < prm["min_cnr"] / 2:
                    continue
                wa = w[ka] + _subpix(g, ka, prm["dw"])
                if mode == "edge":
                    pos[k], valid[k] = wa, True
                    continue
                ib = slice(max(b - tol, 1), min(b + tol + 1, len(w) - 1))
                kb = ib.start + int(np.argmax(-pol * g[ib]))
                if abs(g[kb]) / self.noise < prm["min_cnr"] / 2 or kb <= ka:
                    continue
                wb = w[kb] + _subpix(g, kb, prm["dw"])
                pos[k], valid[k] = 0.5 * (wa + wb), True
            res.frac_valid = float(valid.mean())
            if valid.sum() < 6:
                res.reason = "too_few_stations"
                return res
            a0, b1, se_a, rmse = _huber_line(self.t[valid] / self.dv, pos[valid], prm["huber_c"])
            dref = self.d + b1 * self.e
            dref /= np.linalg.norm(dref)
            p_on = self.p0 + self.base_u * self.s + a0 * self.e
            lam = -np.dot(p_on - self.p0, self.n) / np.dot(dref, self.n)
            u0 = float(np.dot(p_on + lam * dref - self.p0, self.s))
            se_u = se_a / max(self.es, 1e-3)
            if it < prm["n_iter"]:
                # re-centre profiles on the fitted line and re-detect the template features
                self.az = float(np.rad2deg(np.arctan2(dref[0], dref[1])))
                self.base_u = u0
                self._update()
                cand2 = [c for c in self.candidates(mode, 3.0) if c[3] == pol and
                         (mode == "edge" or abs(c[4] - width) <= max(1.5, prm["width_tol"] * width))]
                if not cand2:
                    res.reason = "lost_marker"
                    return res
                cnr, a, b, pol, width, wc = max(cand2, key=lambda c: c[0])
        res.u0, res.se_u0, res.az, res.rmse = u0, float(se_u), self.az, float(rmse)
        res.cnr, res.width = float(cnr), float(width)
        res.ok = True
        if cnr < prm["min_cnr"]:
            res.ok, res.reason = False, "low_cnr"
        elif res.frac_valid < prm["min_valid"]:
            res.ok, res.reason = False, "low_valid_fraction"
        elif rmse > prm["max_rmse"]:
            res.ok, res.reason = False, "high_rmse"
        return res


def _subpix(g, k, dw):
    if 0 < k < len(g) - 1:
        y0, y1, y2 = g[k - 1], g[k], g[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) * dw)
    return 0.0


def _huber_line(t, y, c):
    """IRLS Huber fit y = a + b t. Returns a, b, se(a), RMSE of inliers."""
    X = np.column_stack([np.ones_like(t), t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    wts = np.ones_like(t)
    for _ in range(50):
        r = y - X @ beta
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * sc)
        wts = np.where(u <= 1, 1.0, 1.0 / u)
        W = np.sqrt(wts)
        nb = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
        done = np.allclose(nb, beta, atol=1e-6)
        beta = nb
        if done:
            break
    r = y - X @ beta
    s2 = np.sum(wts * r ** 2) / max(len(t) - 2, 1)
    cov = s2 * np.linalg.inv((X * wts[:, None]).T @ X)
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(np.mean(r[wts == 1] ** 2)))


def measure_offset(img, p0, phi, az_plus, az_minus, aperture=0.0, mode="band", prm=PARAMS):
    """Left-lateral offset H (m) of a marker crossing a fault of strike phi at p0 = (E, N).
    az_plus / az_minus: initial marker azimuths on the +n and -n sides (deg)."""
    res = OffsetResult()
    if not np.all(np.isfinite([p0[0], p0[1], phi, az_plus, az_minus, aperture])):
        res.reason = "invalid_input"
        return res
    P = _Side(img, p0, phi, az_plus, +1, aperture, prm)
    M = _Side(img, p0, phi, az_minus, -1, aperture, prm)
    cp = sorted(P.candidates(mode, prm["search_u"] * P.es), key=lambda c: -c[0])[:40]
    cm = sorted(M.candidates(mode, prm["search_u"] * M.es), key=lambda c: -c[0])[:40]
    pairs = []
    for a in cp:
        for b in cm:
            if a[3] != b[3]:
                continue
            if mode == "band" and abs(a[4] - b[4]) > max(1.5, prm["width_tol"] * max(a[4], b[4])):
                continue
            ua, ub = P.u_of(a[5]), M.u_of(b[5])
            if abs(ua - ub) > prm["max_offset"]:
                continue
            pairs.append((min(a[0], b[0]), abs(ua) + abs(ub), a, b))
    if not pairs:
        res.reason = "no_candidate_pair"
        return res
    best = max(p[0] for p in pairs)
    strong = [p for p in pairs if p[0] >= prm["rel_cnr"] * best]
    _, _, a, b = min(strong, key=lambda p: p[1])  # nearest strong pair to the field point
    res.plus, res.minus = P.track(a, mode), M.track(b, mode)
    if np.isfinite(res.plus.u0) and np.isfinite(res.minus.u0):
        res.H = res.plus.u0 - res.minus.u0
        res.se_H = float(np.hypot(res.plus.se_u0, res.minus.se_u0))
    res.ok = bool(res.plus.ok and res.minus.ok and np.isfinite(res.H) and abs(res.H) <= prm["max_offset"])
    res.reason = ";".join(r for r in (res.plus.reason, res.minus.reason) if r)
    return res
