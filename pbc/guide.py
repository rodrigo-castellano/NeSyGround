"""The model seam: a scorer of atoms, and the selection PBC makes with it at each step.

    guide = Guide(scorer, k=4)                      # per goal: its all-fact groundings, and the 4 best others
    g = PBC(kb, depth=3, guide=guide)

A step's kept groundings of a goal are ranked by the t-norm (``min`` or ``product``) of their unknown atoms' scores —
the scorer's, in (0, 1]; a fact scores 1 — and only the ``k`` best are kept, with every grounding whose body atoms are
all facts (proof material, never rationed). ``k`` bounds a goal's successors to ``k × width``: a learned grounder's
budget. ``tau`` samples instead (Gumbel top-k on ``log(score) / tau``: a Plackett-Luce draw of the kept sequence);
``capture`` records each selection (the candidates' atoms, goal, scores, the scores ranked by, the exempt and kept
masks), what a policy-gradient trainer needs. Candidates are ranked in a canonical order (goal, rule, body atoms), so
ties break the same way every run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Scorer(Protocol):
    def score(self, atoms: Tensor) -> Tensor:
        """``[N, 3] -> [N]``: each ground atom's score in (0, 1]."""
        ...


@dataclass(frozen=True)
class Guide:
    scorer: Scorer
    k: Union[int, Tensor]                 # per goal; a Tensor [S]: per pool (a learned budget per query)
    tnorm: str = "min"
    tau: Optional[float] = None
    capture: Optional[list] = None

    def __post_init__(self) -> None:
        if self.tnorm not in ("min", "product"):
            raise ValueError(f"tnorm: 'min' or 'product', got {self.tnorm!r}")


def select(guide: Guide, fact: Tensor, rule: Tensor, goal: Tensor, body: Tensor, pool: Tensor, step: int,
           pad: int) -> Tensor:
    """``[T]``: which of a step's kept groundings ``(rule, goal row, body [T, M, 3])`` the guide keeps (``fact
    [T, M]``: which body atoms are facts; ``pool [T]``: each grounding's pool)."""
    T = rule.shape[0]
    if T == 0:
        return torch.ones(0, dtype=torch.bool, device=rule.device)
    unknown = (body[..., 0] != pad) & ~fact                                            # [T, M]
    rows, inv = torch.unique(torch.cat([goal.unsqueeze(1), rule.unsqueeze(1), body.flatten(1)], 1), dim=0,
                             return_inverse=True)                         # canonical: by goal, rule, body atoms
    first = torch.full((rows.shape[0],), T, dtype=torch.long, device=rule.device).scatter_reduce(
        0, inv, torch.arange(T, device=rule.device), "amin")               # each distinct grounding's first row
    u_unknown, u_body, u_goal = unknown[first], body[first], goal[first]
    s = torch.ones(u_unknown.shape, dtype=torch.float32, device=rule.device)
    if u_unknown.any():
        s[u_unknown] = guide.scorer.score(u_body[u_unknown]).float()
    score = s.amin(1) if guide.tnorm == "min" else s.prod(1)
    u_exempt = ~u_unknown.any(1)
    rank_by = score.clamp_min(1e-30).log()
    if guide.tau is not None:
        u = torch.rand_like(score).clamp_min(1e-10)
        rank_by = rank_by / float(guide.tau) - torch.log((-torch.log(u)).clamp_min(1e-10))
    rank_by = torch.where(u_exempt, torch.full_like(rank_by, float("inf")), rank_by)
    order = torch.argsort(-rank_by, stable=True)                     # best first (the exempt ones), ties canonical
    order = order[torch.sort(u_goal[order], stable=True)[1]]         # grouped by goal, best first within a group
    g, ex = u_goal[order], u_exempt[order].long()
    _, gid = torch.unique_consecutive(g, return_inverse=True)
    exempt_in = torch.zeros_like(g).index_add_(0, gid, ex)[gid]      # each row's group's exempt groundings
    rank = torch.arange(g.shape[0], device=g.device) - torch.searchsorted(g, g) - exempt_in   # among the others
    k = guide.k if isinstance(guide.k, int) else guide.k[pool[first][order]]
    kept = torch.zeros_like(u_exempt)
    kept[order] = u_exempt[order] | (rank < k)
    if guide.capture is not None:
        guide.capture.append(dict(step=step, atoms=u_body, goal=u_goal, pool=pool[first], score=score,
                                  rank_by=rank_by, exempt=u_exempt, kept=kept, tau=guide.tau))
    return kept[inv]


__all__ = ["Scorer", "Guide", "select"]
