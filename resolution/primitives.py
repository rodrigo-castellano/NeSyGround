"""Core unification primitives: pairwise MGU and substitution application.

Leaf-level, dependency-free, vectorized, torch.compile-safe; generic leading dims
``[...]`` (``[L,3]`` facts and ``[B,S,3]`` states alike). Id classification is owned
by ``Encoding``. Two non-obvious invariants:
  1. var boundary at ``constant_no+1``, ``pad`` inert;
  2. ``apply_substitutions`` applies its two slots SEQUENTIALLY (slot 1 sees slot 0's
     result) — the grounder's S=2 path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from grounder.data.encoding import Encoding


@torch.no_grad()
def unify_one_to_one(a: Tensor, b: Tensor, enc: "Encoding") -> Tuple[Tensor, Tensor]:
    """Pairwise MGU of two ``[...,3]`` atoms → (ok ``[...]`` bool, subs ``[...,2,2]``
    per-arg ``(from,to)``, ``pad``=no binding). Fails on predicate mismatch, constant
    clash, or one variable bound to two different targets."""
    pad = enc.pad
    pad_t = torch.tensor(pad, dtype=a.dtype, device=a.device)

    pred_ok = a[..., 0] == b[..., 0]                              # [...]
    qa, ta = a[..., 1:], b[..., 1:]                              # [...,2]

    qc, tc = enc.is_const(qa), enc.is_const(ta)                  # [...,2]
    qv, tv = enc.is_var(qa), enc.is_var(ta)

    const_clash = (qc & tc & (qa != ta)).any(dim=-1)            # [...]
    ok = pred_ok & ~const_clash

    bind_q = qv & tc                                             # a-var ↦ b-const
    bind_t = tv & (qa != pad)         # b-var ↦ any non-pad a-term ((qc|qv)≡a!=pad)
    frm = torch.where(bind_q, qa, torch.where(bind_t, ta, pad_t))   # [...,2]
    to_ = torch.where(bind_q, ta, torch.where(bind_t, qa, pad_t))   # [...,2]
    subs = torch.stack([frm, to_], dim=-1)                       # [...,2,2]

    same_var = (subs[..., 0, 0] == subs[..., 1, 0]) & (subs[..., 0, 0] != pad)
    ok = ok & ~(same_var & (subs[..., 0, 1] != subs[..., 1, 1]))

    subs = torch.where((~ok).unsqueeze(-1).unsqueeze(-1), pad_t, subs)
    return ok, subs


@torch.no_grad()
def apply_substitutions(atoms: Tensor, subs: Tensor, enc: "Encoding") -> Tensor:
    """Apply substitution slots ``(from→to)`` to atom args: ``[N,M,3]``, subs ``[N,S,2]``.

    S=2 hot path: two SEQUENTIAL ``where``s (slot 1 sees slot 0 — needed for chained
    var→var bindings); a ``pad→-1`` sentinel folds the validity guard into the tiny
    operand (one fewer big op/slot). S≠2: vectorized first-match (cold path).
    """
    if atoms.numel() == 0:
        return atoms
    pad = enc.pad
    N, M = atoms.shape[:2]
    S = subs.shape[1]
    preds = atoms[:, :, 0:1]                          # [N, M, 1] (view)
    args = atoms[:, :, 1:]                            # [N, M, 2] (view)

    if S == 2:
        neg = torch.tensor(-1, dtype=args.dtype, device=args.device)   # absent from args
        frm0 = torch.where(subs[:, 0, 0] != pad, subs[:, 0, 0], neg).view(N, 1, 1)
        out = torch.where(args == frm0, subs[:, 0, 1].view(N, 1, 1), args)
        frm1 = torch.where(subs[:, 1, 0] != pad, subs[:, 1, 0], neg).view(N, 1, 1)
        out = torch.where(out == frm1, subs[:, 1, 1].view(N, 1, 1), out)
        return torch.cat([preds, out], dim=2)

    # General fallback: first-match over the S slots (parallel; cold path).
    frm = subs[:, :, 0].view(N, S, 1, 1)
    valid = (subs[:, :, 0] != pad).view(N, S, 1, 1)
    match = (args.view(N, 1, M, 2) == frm) & valid               # [N, S, M, 2]
    any_match = match.any(dim=1)                                 # [N, M, 2]
    idx = match.long().argmax(dim=1).view(N, M * 2)              # [N, M*2]
    to_gathered = subs[:, :, 1].gather(1, idx).view(N, M, 2)
    out = torch.where(any_match, to_gathered, args)
    return torch.cat([preds, out], dim=2)


__all__ = ["unify_one_to_one", "apply_substitutions"]
