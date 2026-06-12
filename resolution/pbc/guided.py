"""PBC guided beam — the KGE-prior beam over the flat-pruned join expansion.

A neural/KGE concern layered on the flat enumeration: ``GuidedBeam`` scores the
atoms each free-var binding determines (facts exactly 1.0, ground unknowns the
consumer's ``GuidedScorer`` prior) and keeps only fact-perfect rows + the top-k
speculative rows per state. ``GuidedStats`` is the matching per-depth census.

Selected by ``PBC.guided_topk`` and driven by ``FlatMaterializer`` on the pruned
path (``enumerate.enumerate_flat_pruned(beam=...)``).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from grounder.resolution.pbc.enumerate import arange_cached, cumcount_flat


class GuidedStats:
    """Per-depth search-census counters (attach via ``grounder.guided_stats``).

    ``bindings_*``: rows entering / leaving the per-fv guided select inside the
    join loop; ``states_*``: rows entering / leaving the per-query state beam at
    emit. With ``guided_topk=None`` the join path counts identically while
    keeping everything — the exhaustive census arm."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bindings_in: Dict[int, int] = {}
        self.bindings_out: Dict[int, int] = {}
        self.states_in: Dict[int, int] = {}
        self.states_out: Dict[int, int] = {}

    def _bump(self, table: Dict[int, int], d: int, n: int) -> None:
        table[d] = table.get(d, 0) + int(n)

    def totals(self) -> Dict[str, Dict[int, int]]:
        return {"bindings_in": dict(self.bindings_in),
                "bindings_out": dict(self.bindings_out),
                "states_in": dict(self.states_in),
                "states_out": dict(self.states_out)}


class GuidedBeam:
    """KGE-guided beam state for one resolve step (flat-pruned path only).

    ONE scoring rule at both levels: fact atoms score EXACTLY 1.0 (fact-index
    membership, never the KGE), ground non-fact atoms take the consumer's
    ``GuidedScorer`` prior, variable/padding atoms are neutral 1.0. Rows whose
    every determined atom is a fact are PROOF MATERIAL — exempt from the
    budget; ``k`` rations speculative expansion only. ``k=None`` = census mode
    (count, keep everything — byte-identical to the plain join).

    ``tnorm='min'`` accumulation is order-independent (bitwise-deterministic
    under scatter atomics); ``'product'`` is FP-order-sensitive there — 'min'
    is the paper default and the deterministic choice.
    """

    def __init__(self, k: Optional[int], tnorm: str, scorer, fact_index,
                 constant_no: int, padding_idx: int, stats: Optional[GuidedStats] = None):
        self._k = k
        self._tnorm = tnorm
        self._scorer = scorer
        self._fact_index = fact_index
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        self.stats = stats
        self.d = 0                      # current depth (set per step by the materializer)

    @property
    def active(self) -> bool:
        return self._k is not None

    def atom_scores(self, atoms: Tensor) -> Tuple[Tensor, Tensor]:
        """[N, 3] → (score [N] f32, proofish [N] bool). Facts/padding 1.0 and
        proofish; variables neutral 1.0, NOT proofish; ground unknowns σ-prior."""
        N = atoms.size(0)
        is_fact = self._fact_index.exists(atoms)
        pred, a0, a1 = atoms[:, 0], atoms[:, 1], atoms[:, 2]
        is_pad = pred == self._padding_idx
        unknown = (~is_fact & ~is_pad
                   & (a0 <= self._constant_no) & (a1 <= self._constant_no))
        score = torch.ones(N, dtype=torch.float32, device=atoms.device)
        todo = torch.nonzero(unknown, as_tuple=False).squeeze(1)
        if todo.numel():
            score.scatter_(0, todo, self._scorer.score_atoms(atoms[todo]).float())
        return score, is_fact | is_pad

    def update_bindings(self, src: Tensor, n_idx: Tensor, r_idx: Tensor, fv: int,
                        run_score: Tensor, run_fact: Tensor,
                        arg_source_dep_q: Tensor, body_preds_dep_q: Tensor,
                        num_body_q: Tensor, ready_after_q: Tensor, M: int
                        ) -> Tuple[Tensor, Tensor]:
        """Fold the atoms NEWLY determined by binding ``fv`` (each atom scored
        exactly once across the loop) into the running t-norm / all-fact state."""
        dev = src.device
        if src.size(0) == 0:
            return run_score, run_fact
        newly = ((ready_after_q[n_idx, r_idx] == fv)
                 & (arange_cached(M, dev).unsqueeze(0)
                    < num_body_q[n_idx, r_idx].unsqueeze(1)))          # [T, M]
        sel = torch.nonzero(newly, as_tuple=False)
        if sel.size(0) == 0:
            return run_score, run_fact
        rows, cols = sel[:, 0], sel[:, 1]
        W = src.size(1)
        asd = arg_source_dep_q[n_idx[rows], r_idx[rows], cols]          # [P, 2]
        args = src[rows].gather(1, asd.clamp(max=W - 1))                # [P, 2]
        preds = body_preds_dep_q[n_idx[rows], r_idx[rows], cols]
        score, proofish = self.atom_scores(
            torch.cat([preds.unsqueeze(1), args], dim=1))
        run_score.scatter_reduce_(
            0, rows, score, reduce=("amin" if self._tnorm == "min" else "prod"))
        bad = rows[~proofish]
        if bad.numel():
            run_fact.scatter_(0, bad, torch.zeros_like(bad, dtype=torch.bool))
        return run_score, run_fact

    def topk_keep(self, groups: Tensor, score: Tensor, exempt: Tensor) -> Tensor:
        """Keep indices: ALL exempt rows + top-k speculative rows per group
        (stable score-desc rank within group — deterministic under ties)."""
        spec = ~exempt
        rank_score = torch.where(spec, score, torch.full_like(score, -1.0))
        perm = torch.argsort(rank_score, descending=True, stable=True)
        rank = cumcount_flat(groups[perm])
        in_beam = torch.zeros_like(spec)
        in_beam.scatter_(0, perm, rank < self._k)
        return torch.nonzero((in_beam & spec) | exempt, as_tuple=False).squeeze(1)

    def state_beam(self, body: Tensor, mask: Tensor, cand, work, inp, M: int) -> Tensor:
        """Per-query state beam before emit: t-norm over the state's goal atoms
        (new body + inherited remaining tail); fact-perfect states are exempt."""
        n_in = int(mask.sum())
        if self.stats is not None:
            self.stats._bump(self.stats.states_in, self.d, n_in)
        if not self.active or n_in == 0:
            if self.stats is not None:
                self.stats._bump(self.stats.states_out, self.d, n_in)
            return mask
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        S = work.S
        surv_n = work.active_pos[cand.n_idx_eff[idx]]
        b_idx, s_idx = surv_n // S, surv_n % S
        G = inp.remaining.shape[2]
        n_rem = min(G - M, G - 1)
        atoms = body[idx]
        if n_rem > 0:
            atoms = torch.cat(
                [atoms, inp.remaining[b_idx, s_idx, 1:1 + n_rem, :]], dim=1)
        R, L = atoms.shape[0], atoms.shape[1]
        score, proofish = self.atom_scores(atoms.reshape(-1, 3))
        score, proofish = score.view(R, L), proofish.view(R, L)
        state_score = score.amin(dim=-1) if self._tnorm == "min" else score.prod(dim=-1)
        keep = self.topk_keep(b_idx, state_score, proofish.all(dim=-1))
        new_mask = torch.zeros_like(mask)
        new_mask.scatter_(0, idx[keep], torch.ones_like(keep, dtype=torch.bool))
        if self.stats is not None:
            self.stats._bump(self.stats.states_out, self.d, keep.numel())
        return new_mask


__all__ = ["GuidedStats", "GuidedBeam"]
