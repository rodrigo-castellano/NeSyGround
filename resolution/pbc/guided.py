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

    ``query_topk`` ([B] long, chunk-local) overrides ``k`` PER QUERY — the
    learned-budget seam: a policy may spend k=5 on one query and k=0 (keep only
    fact-perfect rows) on another. ``capture`` (a plain list) is the training
    hook: ``state_beam`` appends one record per selection decision (the state
    atoms, query index lifted by ``query_offset`` to the pre-chunk batch, the
    t-norm scores, the exempt + kept masks) so a consumer can recompute row
    scores WITH grad and run policy-gradient updates outside the grounder.
    """

    def __init__(self, k: Optional[int], tnorm: str, scorer, fact_index,
                 constant_no: int, padding_idx: int, stats: Optional[GuidedStats] = None,
                 query_topk: Optional[Tensor] = None, capture: Optional[list] = None,
                 query_offset: int = 0, sample_tau: Optional[float] = None,
                 query_depth: Optional[Tensor] = None):
        self._k = k
        self._tnorm = tnorm
        self._scorer = scorer
        self._fact_index = fact_index
        self._constant_no = constant_no
        self._padding_idx = padding_idx
        self.stats = stats
        self._query_topk = query_topk   # [B] per-query width; overrides scalar k
        self._capture = capture         # list; state_beam appends training records
        self._q_offset = int(query_offset)  # chunk offset: local b_idx → global query
        self._state_q = None            # [n_states] state row → local query (bind_states)
        # Per-query DEPTH gate ([B] long): query b expands only at steps
        # d < query_depth[b] — beyond it EVERY row dies, fact-perfect
        # included (the exemption rations speculation; the depth gate ends
        # the query's search outright: depth 0 = no expansion, KGE-only).
        # The one lever that bounds evidence VOLUME, which budgets cannot
        # touch (proof-material rows ride outside every budget).
        self._query_depth = query_depth
        # Stochastic STATE-level selection (training only): perturb row
        # log-scores with Gumbel noise at temperature tau before the top-k.
        # Top-k of Gumbel-perturbed logits == Plackett-Luce sampling of an
        # ordered k-sequence with logits log(score)/tau — the exact density a
        # policy-gradient consumer needs. Binding-level selection stays
        # deterministic (part of the environment). None = deterministic.
        self._sample_tau = sample_tau
        if self.budget_active and scorer is None:
            raise ValueError("guided selection requires a GuidedScorer")
        self.d = 0                      # current depth (set per step by the materializer)

    @property
    def budget_active(self) -> bool:
        """A scored top-k selection happens (needs the scorer); the depth
        gate alone does not select — it terminates."""
        return self._k is not None or self._query_topk is not None

    @property
    def active(self) -> bool:
        return self.budget_active or self._query_depth is not None

    def bind_states(self, active_pos: Tensor, S: int, state_to_goal=None) -> None:
        """Map flat state rows → local query index (per-query budgets, the
        per-query depth gate, and binding-level capture all need it)."""
        if (self._query_topk is None and self._capture is None
                and self._query_depth is None):
            return
        if state_to_goal is not None:
            raise ValueError("guided_query_topk/guided_capture/guided_query_"
                             "depth require dedup_goals=False (state rows "
                             "map 1:1 to queries)")
        self._state_q = active_pos // S

    def depth_alive(self, b_idx: Tensor) -> Optional[Tensor]:
        """[rows] bool: this query still expands at the current step (the
        per-query depth gate; ``None`` = no gate). Applied BEFORE selection,
        to EVERY row — fact-perfect rows die past the gate too."""
        if self._query_depth is None:
            return None
        return self._query_depth[b_idx] > self.d

    def binding_k(self, n_idx: Tensor) -> Optional[Tensor]:
        """Per-row budget for the binding-level select (rows grouped by state)."""
        if self._query_topk is None:
            return None
        return self._query_topk[self._state_q[n_idx]]

    def query_k(self, b_idx: Tensor) -> Optional[Tensor]:
        """Per-row budget for the state-level beam (rows grouped by query)."""
        if self._query_topk is None:
            return None
        return self._query_topk[b_idx]

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

    def topk_keep(self, groups: Tensor, score: Tensor, exempt: Tensor,
                  k_rows: Optional[Tensor] = None,
                  sample_tau: Optional[float] = None) -> Tensor:
        """Keep indices: ALL exempt rows + top-k speculative rows per group
        (stable score-desc rank within group — deterministic under ties).
        ``k_rows`` ([T] long) gives each row its own budget (per-query
        ``query_topk``); ``None`` = the scalar ``k``. ``k=0`` keeps nothing
        speculative — only fact-perfect rows survive for that group.
        ``sample_tau`` switches to STOCHASTIC selection: rows are ranked by
        ``log(score)/tau + Gumbel`` — exact Plackett-Luce sampling of the
        kept ordered sequence (training mode; the caller captures the
        perturbed scores so a consumer can recompute the sample's exact
        log-probability)."""
        spec = ~exempt
        if sample_tau is not None:
            u = torch.rand_like(score).clamp_min(1e-10)
            g = -torch.log((-torch.log(u)).clamp_min(1e-10))   # Gumbel(0,1)
            rank_score = score.clamp_min(1e-30).log() / float(sample_tau) + g
            rank_score = torch.where(spec, rank_score,
                                     torch.full_like(score, float("-inf")))
        else:
            rank_score = torch.where(spec, score, torch.full_like(score, -1.0))
        perm = torch.argsort(rank_score, descending=True, stable=True)
        rank = cumcount_flat(groups[perm])
        kmax = k_rows[perm] if k_rows is not None else self._k
        in_beam = torch.zeros_like(spec)
        in_beam.scatter_(0, perm, rank < kmax)
        keep = torch.nonzero((in_beam & spec) | exempt, as_tuple=False).squeeze(1)
        self._last_rank_score = rank_score      # for capture (selection order)
        return keep

    def capture_binding(self, src: Tensor, n_idx: Tensor, r_idx: Tensor,
                        fv: int, run_score: Tensor, run_fact: Tensor,
                        keep: Tensor, arg_source_dep_q: Tensor,
                        body_preds_dep_q: Tensor, num_body_q: Tensor,
                        ready_after_q: Tensor, M: int) -> None:
        """Record one BINDING-level selection (sampled mode only — the PL
        trainer needs every selection with pressure, and binding-level is
        where nearly all pruning happens). Each record carries the rows'
        determined atoms (padded [T, M, 3] + ``det`` mask) so a consumer can
        recompute row t-norm logits WITH grad, the group key (``groups`` =
        state row; selection is per state), the query lift ``b_idx``, and the
        Gumbel-perturbed ``order_score`` whose descending order over kept rows
        IS the sampled PL sequence."""
        if self._capture is None or self._sample_tau is None or n_idx.numel() == 0:
            return
        dev = src.device
        T, W = n_idx.size(0), src.size(1)
        det = ((ready_after_q[n_idx, r_idx] <= fv)
               & (arange_cached(M, dev).unsqueeze(0)
                  < num_body_q[n_idx, r_idx].unsqueeze(1)))            # [T, M]
        asd = arg_source_dep_q[n_idx, r_idx].clamp(max=W - 1)          # [T, M, 2]
        args = src.unsqueeze(1).expand(T, M, W).gather(2, asd)         # [T, M, 2]
        preds = body_preds_dep_q[n_idx, r_idx]                         # [T, M]
        kept_mask = torch.zeros(T, dtype=torch.bool, device=dev)
        kept_mask[keep] = True
        self._capture.append(dict(
            level="binding", d=self.d, fv=fv,
            atoms=torch.cat([preds.unsqueeze(-1), args], dim=-1).detach(),
            det=det, groups=n_idx.detach(),
            b_idx=self._state_q[n_idx] + self._q_offset,
            score=run_score.detach(),
            order_score=self._last_rank_score.detach(),
            sample_tau=self._sample_tau,
            exempt=run_fact.detach(), kept=kept_mask))

    def state_beam(self, body: Tensor, mask: Tensor, cand, work, inp, M: int) -> Tensor:
        """Per-query state beam before emit: t-norm over the state's goal atoms
        (new body + inherited remaining tail); fact-perfect states are exempt.
        The per-query depth gate applies FIRST — a gated query's rows never
        become candidates (and never reach capture)."""
        n_in = int(mask.sum())
        if self.stats is not None:
            self.stats._bump(self.stats.states_in, self.d, n_in)
        if not self.active or n_in == 0:
            if self.stats is not None:
                self.stats._bump(self.stats.states_out, self.d, n_in)
            return mask
        if self._query_depth is not None:
            S_ = work.S
            all_b = work.active_pos[cand.n_idx_eff] // S_
            mask = mask & self.depth_alive(all_b)
            if not self.budget_active or not bool(mask.any()):
                # gate-only mode (or nothing left): no scored selection.
                if self.stats is not None:
                    self.stats._bump(self.stats.states_out, self.d,
                                     int(mask.sum()))
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
        state_exempt = proofish.all(dim=-1)
        keep = self.topk_keep(b_idx, state_score, state_exempt,
                              k_rows=self.query_k(b_idx),
                              sample_tau=self._sample_tau)
        if self._capture is not None:
            kept_mask = torch.zeros(R, dtype=torch.bool, device=atoms.device)
            kept_mask[keep] = True
            self._capture.append(dict(
                level="state", d=self.d, atoms=atoms.detach(),
                groups=b_idx.detach(),
                b_idx=b_idx + self._q_offset, score=state_score.detach(),
                # the (possibly Gumbel-perturbed) ranking used for selection —
                # sorting kept rows by it recovers the sampled PL order.
                order_score=self._last_rank_score.detach(),
                sample_tau=self._sample_tau,
                exempt=state_exempt, kept=kept_mask))
        new_mask = torch.zeros_like(mask)
        new_mask.scatter_(0, idx[keep], torch.ones_like(keep, dtype=torch.bool))
        if self.stats is not None:
            self.stats._bump(self.stats.states_out, self.d, keep.numel())
        return new_mask


__all__ = ["GuidedStats", "GuidedBeam"]
