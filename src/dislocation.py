"""Controlled (synthetic) strike-slip dislocation of co-registered PAN/MS chips.

A vertical fault through the chip centre is simulated by translating the image
content on one side of the fault by k MS pixels (= 4k PAN pixels) parallel to
the fault. Integer MS-pixel shifts are exact on both grids, so no resampling is
introduced and all bands remain mutually consistent.

orientation 'NS': strike 0 deg, the east side moves north (left-lateral, H = +k*ms_res)
orientation 'EW': strike 90 deg, the south side moves east (left-lateral, H = +k*ms_res)
"""
import numpy as np

RATIO = 4


def apply(ms, pan, vis, k, orientation, margin_ms):
    """ms (B,H,W), pan (4H,4W), vis (3,4H,4W) or None, read with a margin of
    margin_ms MS pixels on every side. Returns cropped arrays."""
    B, H, W = ms.shape
    ms2, pan2 = ms.copy(), pan.copy()
    vis2 = None if vis is None else vis.copy()
    if k:
        if orientation == "NS":
            c = W // 2
            ms2[:, :H - k, c:] = ms[:, k:, c:]
            pan2[:RATIO * (H - k), RATIO * c:] = pan[RATIO * k:, RATIO * c:]
            if vis is not None:
                vis2[:, :RATIO * (H - k), RATIO * c:] = vis[:, RATIO * k:, RATIO * c:]
        else:
            r = H // 2
            ms2[:, r:, k:] = ms[:, r:, :W - k]
            pan2[RATIO * r:, RATIO * k:] = pan[RATIO * r:, :RATIO * (W - k)]
            if vis is not None:
                vis2[:, RATIO * r:, RATIO * k:] = vis[:, RATIO * r:, :RATIO * (W - k)]
    m, M = margin_ms, RATIO * margin_ms
    ms2 = ms2[:, m:H - m, m:W - m]
    pan2 = pan2[M:RATIO * H - M, M:RATIO * W - M]
    if vis2 is not None:
        vis2 = vis2[:, M:RATIO * H - M, M:RATIO * W - M]
    return ms2, pan2, vis2


def fault_geometry(x0, y0, pan_res, n_pan, orientation):
    """Fault strike (deg) and a point on the fault line for the cropped chip.
    The fault passes through the boundary between MS columns/rows W/2-1 and W/2
    of the margin-padded read, i.e. through the centre of the cropped chip."""
    xc = x0 + n_pan * pan_res / 2
    yc = y0 - n_pan * pan_res / 2
    return (0.0 if orientation == "NS" else 90.0), np.array([xc, yc])
