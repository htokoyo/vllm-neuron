# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference of ``gdn_tkg_multi`` (fp32): T one-token gated delta-rule steps.

Same contract and layouts as the kernel (``q`` / ``k`` [B, T, Hv, K] RAW, ``v`` / ``z``
[B, T, Hv, V], ``a`` / ``b`` [B, T, Hv], ``state_in`` [B, Hv, K, V]); returns ``out``
[B, T, Hv, V] (in ``v``'s dtype) and the state after every token [B, T, Hv, K, V] fp32.
Used as the CPU path of the multi-token verify (no NKI) and as the simulator reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gdn_tkg_multi_torch(q, k, v, b, a, A_log, dt_bias, z, norm_weight, state_in, gate_activation="silu"):
    act = torch.sigmoid if gate_activation == "sigmoid" else F.silu
    T, K = q.shape[1], q.shape[-1]
    q = q.float()
    k = k.float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b.float())  # [B, T, Hv]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    s = state_in.float()
    outs, states = [], []
    for t in range(T):
        s = s * g[:, t].exp()[..., None, None]
        kv = torch.einsum("bhk,bhkv->bhv", k[:, t], s)
        delta = (v[:, t].float() - kv) * beta[:, t, :, None]
        s = s + k[:, t, :, :, None] * delta[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], s))
        states.append(s)
    o = torch.stack(outs, dim=1)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_weight.float()
    return (o * act(z.float())).to(v.dtype), torch.stack(states, dim=1)
