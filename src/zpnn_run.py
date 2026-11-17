"""Z-PNN (Ciotola et al., 2022) applied with its published WV2 weights and the
default target-adaptive fine-tuning (100 iterations, beta=0.36, lr=1e-5).
Wraps the authors' code in external/zpnn without modifying the algorithm."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ZP = Path(__file__).resolve().parents[1] / "external/zpnn/Z-PNN-master"
sys.path.insert(0, str(ZP))
import losses  # noqa: E402
import networks  # noqa: E402
import utils  # noqa: E402
from input_prepocessing import input_preparation  # noqa: E402
from sensor import Sensor  # noqa: E402
from spectral_tools import generate_mtf_variables  # noqa: E402

# ARD reflectance (x10^4) is rescaled to the 11-bit range the weights were trained on
SCALE = 2047.0 / 10000.0


def zpnn_fuse(ms, pan, sensor="WV2", epochs=100, seed=0, device=None):
    """ms: (8,h,w) reflectance x1e4 ; pan: (4h,4w). Returns (8,4h,4w) float32."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = Sensor(sensor)
    I_MS = np.moveaxis(ms, 0, -1).astype("float32") * SCALE
    I_PAN = pan.astype("float32") * SCALE
    net = networks.PNN(s.nbands + 1, s.kernels, s.net_scope)
    I_in = input_preparation(I_MS, I_PAN, s.ratio, s.nbits, s.net_scope)
    I_in = np.expand_dims(np.moveaxis(I_in, -1, 0), 0)
    I_inp = torch.from_numpy(np.copy(I_in)).float()
    I_in = torch.from_numpy(I_in[:, :, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope]).float()
    thr = utils.local_corr_mask(I_in, s.ratio, s.sensor, device, 8).float()
    ns = s.net_scope
    spec_ref = I_in[:, :-1, ns:-ns, ns:-ns]
    struct_ref = torch.unsqueeze(I_in[:, -1, ns:-ns, ns:-ns], dim=1)
    thr = thr[:, :, ns:-ns, ns:-ns]
    net.load_state_dict(torch.load(ZP / f"weights/{sensor}_Z-PNN_model.tar", map_location=device))
    LSpec = losses.SpectralLossNocorr(generate_mtf_variables(s.ratio, sensor, I_PAN, I_MS),
                                      ns, I_PAN.shape, s.ratio, device)
    LStruct = losses.StructuralLoss(s.ratio, device)
    net = net.to(device)
    opt = optim.Adam(net.parameters(), lr=s.learning_rate)
    net.train()
    I_in, spec_ref, struct_ref, thr = (x.to(device) for x in (I_in, spec_ref, struct_ref, thr))
    LSpec = LSpec.to(device)
    LSpec.mask = LSpec.mask.to(device)
    LStruct = LStruct.to(device)
    best, best_state, hist = np.inf, None, []
    for _ in range(epochs):
        opt.zero_grad()
        out = net(I_in)
        ls = LSpec(out, spec_ref)
        lt, _ = LStruct(out, struct_ref, thr)
        loss = ls + s.beta * lt
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:  # epochs == 0: pretrained weights only
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(I_inp.to(device)).cpu().numpy()[0]
    out = np.clip(out * (2 ** s.nbits), 0, None) / SCALE
    torch.cuda.empty_cache()
    return out.astype("float32"), np.array(hist)
