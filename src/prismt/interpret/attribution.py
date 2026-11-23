"""The PRISMt attribution tensor.

For held-out trial n, layer l, head h and data token i::

    G[n, l, h, i] = rho[n, l, h, i] * msg[n, l, h, i]

* **Routing** rho is attention rollout from the CLS token. Each layer's attention gets a
  residual term, A_hat = (A + gamma I) / (1 + gamma) with gamma = 0.1, and layers compose
  with the later layer on the left, A_N ... A_1 (a layer mixes tokens as x_out = A x_in, so
  after N layers x_N = A_N ... A_1 x_0 and the CLS row of the product is the relevance of
  each input). For the pair (l, h), head h is used at layer l and the head mean elsewhere.
* **Message** msg = u^T W_O^(l,h) v_(l,h,i): the signed contribution token i's value would
  make to the decision through that head. u is the unit-norm mean gradient of the target
  logit with respect to the CLS residual stream (where every head writes); per-trial u is
  available and changes recovery little.

The target logit is zeta = logit[b] - logit[a] for a chosen pair of classes (a, b), so
positive attribution pushes towards class b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prismt.data.tensors import TrialTensors, batches
from prismt.data.tokens import TokenGrid

GAMMA = 0.1


@dataclass
class Attribution:
    G: np.ndarray            # [n, layers, heads, tokens] float32
    rollout: np.ndarray      # [n, tokens] head-mean rollout without messages (baseline)
    grad_input: np.ndarray   # [n, tokens, patch] gradient x input (baseline)
    zeta: np.ndarray         # [n] target logit
    u: np.ndarray            # [d_model] the decision direction used
    rows: np.ndarray         # positions of the trials in the selection
    classes: tuple[int, int]


def residual_attention(A: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """(A + gamma I) / (1 + gamma) on the last two axes."""
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return (A + gamma * eye) / (1.0 + gamma)


def rollout_routing(attentions: list[torch.Tensor], gamma: float = GAMMA) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(layer, head) CLS routing and the head-mean rollout.

    attentions: per layer [B, H, S, S] (row = query). Returns rho [B, L, H, S] (CLS row of
    A_N ... A_hat(l,h) ... A_1) and the plain rollout [B, S] (head mean at every layer).
    """
    L = len(attentions)
    B, H, S, _ = attentions[0].shape
    heads = [residual_attention(A, gamma) for A in attentions]           # [B, H, S, S]
    means = [residual_attention(A.mean(1), gamma) for A in attentions]   # [B, S, S]
    # right[l] = A_{l-1} ... A_1 (identity for l = 0)
    eye = torch.eye(S, dtype=means[0].dtype, device=means[0].device).expand(B, S, S)
    right = [eye]
    for l in range(1, L):
        right.append(means[l - 1] @ right[-1])
    # left[l] = e_cls^T A_N ... A_{l+1} (a row vector)
    e = torch.zeros(B, 1, S, dtype=means[0].dtype, device=means[0].device)
    e[:, 0, 0] = 1.0
    left = [None] * L
    cur = e
    for l in range(L - 1, -1, -1):
        left[l] = cur
        cur = cur @ means[l]
    full = cur[:, 0]                                                       # e^T A_N ... A_1
    rho = torch.empty(B, L, H, S, dtype=means[0].dtype, device=means[0].device)
    for l in range(L):
        through = left[l][:, None] @ heads[l]                              # [B, H, 1, S]
        rho[:, l] = (through @ right[l][:, None])[:, :, 0]
    return rho, full


def _target(logits: torch.Tensor, classes: tuple[int, int]) -> torch.Tensor:
    a, b = classes
    return logits[:, b] - logits[:, a]


@torch.no_grad()
def _no_grad_pass(model, x, v):
    return model(x, v, need_weights=True)


def compute(model, tensors: TrialTensors, rows: np.ndarray, *, classes: tuple[int, int] = (0, 1),
            batch_size: int = 32, per_trial_u: bool = False, gamma: float = GAMMA) -> Attribution:
    """Attribution for ``rows`` of ``tensors`` (normally one fold's test trials)."""
    model.eval()
    rows = np.asarray(rows)
    grads, gx, zetas = [], [], []
    # pass 1: the decision direction (gradient at the CLS residual) and gradient x input
    for b in batches(rows, batch_size, shuffle=False):
        x, v, _ = tensors.batch(b)
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = model(x, v, need_weights=True)
            zeta = _target(out.logits.float(), classes)
            g_cls, g_x = torch.autograd.grad(zeta.sum(), [out.internals["cls_residual"], x])
        grads.append(g_cls.float().cpu().numpy())
        gx.append((g_x * x).detach().float().cpu().numpy())
        zetas.append(zeta.detach().float().cpu().numpy())
    grads = np.concatenate(grads)
    mean = grads.mean(0)
    u_global = mean / max(np.linalg.norm(mean), 1e-12)
    # pass 2: routing and messages
    Gs, rolls = [], []
    W_O = [blk.attn.per_head_out_proj().detach() for blk in model.encoder.blocks]  # [H, d_head, d]
    offset = 0
    for b in batches(rows, batch_size, shuffle=False):
        x, v, _ = tensors.batch(b)
        out = _no_grad_pass(model, x, v)
        att = [w.float() for w in out.internals["attention"]]
        rho, full = rollout_routing(att, gamma)
        if per_trial_u:
            g = grads[offset:offset + len(b)]
            u = torch.as_tensor(g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12), device=x.device)
        else:
            u = torch.as_tensor(u_global, device=x.device).expand(len(b), -1)
        offset += len(b)
        msgs = []
        for l, vals in enumerate(out.internals["values"]):                 # [B, H, S, d_head]
            proj = torch.einsum("bhsk,hkd->bhsd", vals.float(), W_O[l].float().to(x.device))
            msgs.append(torch.einsum("bhsd,bd->bhs", proj, u.float()))
        msg = torch.stack(msgs, 1)                                          # [B, L, H, S]
        Gs.append((rho * msg)[..., 1:].cpu().numpy().astype(np.float32))
        rolls.append(full[:, 1:].cpu().numpy().astype(np.float32))
    return Attribution(np.concatenate(Gs), np.concatenate(rolls), np.concatenate(gx), np.concatenate(zetas),
                       u_global.astype(np.float32), rows, tuple(classes))


def to_rtm(token_values: np.ndarray, grid: TokenGrid, n_channels: int, n_modalities: int) -> np.ndarray:
    """[..., tokens] -> [..., channels, bins, modalities]. A token spanning several time bins
    shares its value equally among them; channel-modality pairs that do not exist are 0."""
    lead = token_values.shape[:-1]
    V = token_values.reshape(-1, grid.n_patches, grid.n_pairs)
    out = np.zeros((V.shape[0], n_channels, grid.n_bins, n_modalities), dtype=np.float32)
    for q, (c, m) in enumerate(grid.pairs):
        out[:, c, :, m] = np.repeat(V[:, :, q], grid.patch, axis=1) / grid.patch
    return out.reshape(*lead, n_channels, grid.n_bins, n_modalities)


def exists_mask(grid: TokenGrid, n_channels: int, n_modalities: int) -> np.ndarray:
    """[channels, modalities] True where the pair is part of the model's input."""
    ok = np.zeros((n_channels, n_modalities), dtype=bool)
    for c, m in grid.pairs:
        ok[c, m] = True
    return ok
