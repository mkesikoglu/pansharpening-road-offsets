"""Lambda-PNN (Ciotola, Poggi and Scarpa, 2023) applied with its published WV2 / WV3
weights and the authors' default target-adaptive fine-tuning at full resolution.

The wrapper reproduces the full-resolution path of the authors' test script
(external/lambda_pnn/Lambda-PNN-main/test.py) without modifying the algorithm:
network re-initialised from the published weights for every chip, global MS-PAN
co-registration enabled, 100 Adam iterations (weight decay 1e-4) with the
sensor-specific learning rate and loss weights (alpha * reprojected ERGAS +
beta * reprojected D_lambda(Khan) + gamma * thresholded structural loss), the
lowest-loss weights retained, negative output clipped at zero. A 1024 x 1024
PAN chip does not exceed the 16 x 256^2 pixel limit above which the authors
select salient patches, so the whole chip is used for adaptation, as in their code.
Differences from test.py that do not change the result: during adaptation the
network forward pass is evaluated block by block with activation checkpointing
(identical arithmetic, recomputed in the backward pass) so that a whole 1024 x
1024 chip fits in 6 GB of GPU memory; the reprojected D_lambda(Khan) term is
computed with the authors' q_index applied to all 32 x 32 blocks at once
(instead of a Python loop over blocks) and outside the autograd graph, which is
exact because the authors' Q2n rounds its inputs (torch.round has zero
gradient), so the term enters the loss value and the best-iteration choice but
never the weight update; the best weights are kept in memory
instead of a temporary file; the prediction runs under torch.no_grad(); and the
output is returned as float instead of being cast to uint16 (the same
convention as the Z-PNN wrapper in zpnn_run.py).
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

LP = Path(__file__).resolve().parents[1] / "external/lambda_pnn/Lambda-PNN-main"
sys.path.insert(0, str(LP))
import metrics as mt  # noqa: E402
from coregistration import Coregistration  # noqa: E402
from loss import StructuralLoss  # noqa: E402
from network import LPNN  # noqa: E402
from tools import utils  # noqa: E402
from tools.input_prepocessing import input_preparation  # noqa: E402
from tools.spectral_tools import gen_mtf, gen_mtf_pan  # noqa: E402

# ARD reflectance (x10^4) is rescaled to the 11-bit range the weights were trained on
# (identical to the Z-PNN wrapper; the network input is then divided by 2^11).
SCALE = 2047.0 / 10000.0
RATIO = 4
SEMI_WIDTH = 8


def forward_checkpointed(net, inp):
    """LPNN.forward evaluated block by block with activation checkpointing.

    Same operations, in the same order, as network.LPNN.forward; only the
    intermediate activations inside each block are recomputed during backward
    instead of being stored.
    """
    def ck(f, *a):
        return checkpoint(f, *a, use_reentrant=False)

    x = ck(lambda t: func.relu(net.conv_1(t)), inp)
    x = ck(lambda t: func.relu(net.conv_2(t)), x)
    x = ck(lambda t: net.CBAM_1(t) + t, x)
    x = ck(net.res_block_1, x)
    x = ck(net.res_block_2, x)
    x = ck(lambda t: net.CBAM_2(t) + t, x)
    return ck(lambda t, i: net.conv_3(t) + i[:, :-1, :, :], x, inp)


def q2n_batched(outputs, labels, q_block_size=32, q_shift=32):
    """metrics.Q2n.forward with the block loop replaced by one batched q_index call.

    Padding, rounding and band completion follow the authors' code line by line;
    the non-overlapping blocks (q_shift == q_block_size) are stacked along the
    batch dimension, which q_index treats independently.
    """
    assert q_shift == q_block_size
    nn = torch.nn
    bs, dim3, dim1, dim2 = labels.size()
    if math.ceil(math.log2(dim1)) - math.log2(dim1) != 0:
        difference = 2 ** (math.ceil(math.log2(dim1))) - dim1
        pads_2n = nn.ReplicationPad2d((math.floor(difference / 2), math.ceil(difference / 2), 0, 0))
        labels, outputs = pads_2n(labels), pads_2n(outputs)
    if math.ceil(math.log2(dim2)) - math.log2(dim2) != 0:
        difference = 2 ** (math.ceil(math.log2(dim2))) - dim2
        pads_2n = nn.ReplicationPad2d((0, 0, math.floor(difference / 2), math.ceil(difference / 2)))
        labels, outputs = pads_2n(labels), pads_2n(outputs)
    bs, dim3, dim1, dim2 = labels.size()
    stepx, stepy = math.ceil(dim1 / q_shift), math.ceil(dim2 / q_shift)
    if stepy <= 0:
        stepy = stepx = 1
    est1 = (stepx - 1) * q_shift + q_block_size - dim1
    est2 = (stepy - 1) * q_shift + q_block_size - dim2
    if (est1 != 0) + (est2 != 0) > 0:
        padding = nn.ReflectionPad2d((0, est1, 0, est2))
        labels, outputs = padding(labels), padding(outputs)
    outputs, labels = torch.round(outputs), torch.round(labels)
    bs, dim3, dim1, dim2 = labels.size()
    if math.ceil(math.log2(dim3)) - math.log2(dim3) != 0:
        exp_difference = 2 ** (math.ceil(math.log2(dim3))) - dim3
        diff = torch.zeros((bs, exp_difference, dim1, dim2), device=labels.device)
        labels, outputs = torch.cat((labels, diff), dim=1), torch.cat((outputs, diff), dim=1)
    nb = labels.shape[1]

    def tiles(x):
        x = x[:, :, :stepx * q_shift, :stepy * q_shift]
        x = x.reshape(bs, nb, stepx, q_block_size, stepy, q_block_size).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(bs * stepx * stepy, nb, q_block_size, q_block_size)

    q = mt.q_index(tiles(labels), tiles(outputs), q_block_size, labels.device)
    values = q.reshape(bs, stepx, stepy, nb).permute(0, 3, 1, 2)
    return torch.mean(torch.sqrt(torch.sum(values ** 2, dim=1)), dim=(-2, -1))


def d_lambda_khan(shifted_downgraded_outputs, ms):
    """metrics.ReproDLambdaKhan.forward using q2n_batched."""
    return 1.0 - torch.mean(q2n_batched(shifted_downgraded_outputs, ms))


def lpnn_fuse(ms, pan, sensor="WV2", epochs=-1, seed=0, device=None):
    """Fuse one chip with Lambda-PNN.

    Parameters
    ----------
    ms : ndarray (8, h, w), reflectance x 1e4.
    pan : ndarray (4h, 4w), reflectance x 1e4.
    sensor : "WV2" or "WV3"; selects weights, MTF and hyper-parameters.
    epochs : number of adaptation iterations; -1 keeps the authors' default (100).
    seed : torch / numpy seed.

    Returns
    -------
    out : ndarray (8, 4h, 4w) float32, reflectance x 1e4, non-negative.
    hist : ndarray of the total loss per iteration.
    shifts : (rows, cols) integer co-registration shifts per band (PAN pixels / 2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    nbands, nbits, lr, alpha, beta, gamma, epochs = utils.parameters_def(sensor, -1.0, epochs)

    net = LPNN(nbands + 1)
    net.load_state_dict(torch.load(LP / f"weights/{sensor}_L-PNN_model.tar", map_location="cpu"))
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    coreg = Coregistration(gen_mtf_pan(RATIO, sensor), device, RATIO).to(device)
    downgrade = mt.DowngradeProtocol(gen_mtf(RATIO, sensor), RATIO, device).to(device)
    loss_ergas = mt.ERGAS(RATIO).to(device)
    loss_struct = StructuralLoss(RATIO).to(device)

    ms_np = np.moveaxis(ms, 0, -1).astype("float32") * SCALE
    pan_np = pan.astype("float32") * SCALE
    inputs, _, ms_exp, pan_t = input_preparation(ms_np, pan_np, RATIO, nbits, cpu)
    with torch.no_grad():
        r, c = coreg(ms_exp.to(device), pan_t.to(device))
    del ms_exp, pan_t

    labels_ms = inputs[:, :-1, 2::4, 2::4].to(device)
    labels_struct = torch.unsqueeze(inputs[:, -1, :, :], dim=1).to(device)
    threshold = utils.local_corr_mask(inputs, RATIO, sensor, device, SEMI_WIDTH)
    inputs = inputs.to(device)

    best, best_state, hist = np.inf, None, []
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = forward_checkpointed(net, inputs)
        down = downgrade(outputs, r, c)
        l_erg = loss_ergas(down, labels_ms)
        with torch.no_grad():  # zero gradient in the authors' code (rounded inputs)
            l_lam = d_lambda_khan(down * 2048.0, labels_ms * 2048.0)
        l_str, _ = loss_struct(outputs, labels_struct, threshold)
        loss = alpha * l_erg + beta * l_lam + gamma * l_str
        loss.backward()
        optimizer.step()
        hist.append(loss.item())
        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        del outputs, down, l_erg, l_lam, l_str, loss
    if best_state is not None:  # epochs == 0: published weights only
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(inputs).cpu().numpy()[0]
    out = np.clip(out * (2 ** nbits), 0, None) / SCALE
    shifts = (r.cpu().numpy().ravel(), c.cpu().numpy().ravel())
    del net, optimizer, inputs, labels_ms, labels_struct, threshold, best_state
    torch.cuda.empty_cache()
    return out.astype("float32"), np.array(hist), shifts
