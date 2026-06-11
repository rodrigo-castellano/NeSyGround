"""PBC candidate generation: enumerate free-var bindings + fill body atoms.

No MGU — bindings are pre-compiled (PbcRuleIndex). Two materializations:
  ``*_dense`` — padded fixed-shape candidates for compile/CUDA-graph (the dense
  body fill itself lives in ``resolve.DenseMaterializer`` — per-M-slice, no
  [B,K_r,Y_r,M,3] tensor)
  ``*_flat``  — compact [T,...] via nonzero, eager zero-waste (compacts early
  and never materializes the [B,K_r,G] grid)
``_gather_body_atoms`` is the flat path's inner op.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor


# ── stage 1: cluster (query predicate → candidate rules + per-rule view) ──

class ClusteredRules(NamedTuple):
    active_idx: Tensor      # [N, K_r] candidate rule rows per query
    active_mask: Tensor     # [N, K_r] rule exists AND query active
    K_r: int
    has_free_q: Tensor      # [N, K_r]


def cluster(query_preds: Tensor, query_valid: Tensor, tables) -> ClusteredRules:
    """Map each query predicate to its candidate rules (shared dense/flat)."""
    active_idx = tables.pred_rule_indices[query_preds]               # [N, K_r]
    active_mask = tables.pred_rule_mask[query_preds] & query_valid.unsqueeze(1)
    return ClusteredRules(active_idx, active_mask, active_idx.size(1),
                          tables.has_free[active_idx])


# Cached read-only arange (eager); fresh under compile (CUDA-graph aliasing).
_ARANGE_CACHE: Dict[Tuple[int, str], Tensor] = {}


def arange_cached(n: int, device) -> Tensor:
    """Cached read-only ``arange(n)`` (eager); fresh under torch.compile."""
    if torch.compiler.is_compiling():
        return torch.arange(n, device=device)
    key = (int(n), str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        _ARANGE_CACHE[key] = t
    return t


def cumcount_flat(keys: Tensor, assume_sorted: bool = False) -> Tensor:
    """0-based position within each group of equal keys (sort + cummax).

    [A,A,B,A,B,C] → [0,1,0,2,1,0]. Used by the flat pack for per-batch slots.
    ``assume_sorted=True`` skips the stable argsort + scatter (the caller
    guarantees ``keys`` is non-decreasing, so the sort is the identity) —
    the pbc flat emit preserves the enumerate order ``(b, r, g)``, making
    ``flat_batch_idx`` sorted by construction. SLD/RTF flat emits are NOT
    sorted (facts-then-rules concat) and must use the default.
    """
    T = keys.size(0)
    if T == 0:
        return keys.new_empty(0, dtype=torch.long)
    dev = keys.device
    running_idx = torch.arange(T, device=dev)
    if assume_sorted:
        ne = keys[1:] != keys[:-1]
        group_change = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=dev), ne], dim=0)
        group_starts = (running_idx * group_change).cummax(0).values
        return running_idx - group_starts
    sort_perm = torch.argsort(keys, stable=True)
    sorted_keys = keys[sort_perm]
    ne = sorted_keys[1:] != sorted_keys[:-1]
    group_change = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), ne], dim=0)
    group_starts = (running_idx * group_change).cummax(0).values
    result = torch.zeros(T, dtype=torch.long, device=dev)
    result[sort_perm] = running_idx - group_starts
    return result


# ── body-atom fill ──

def _gather_body_atoms(source_m: Tensor, check_arg_m: Tensor,
                       body_preds_m: Tensor) -> Tensor:
    """Shared gather (dense+flat): ``source_m [...,M,W]``, ``check_arg_m [...,M,2]``
    (indexes W, clamped), ``body_preds_m [...,M]`` → ``[...,M,3]`` (pred,a0,a1).

    Both args come from ONE 2-wide gather (not two single-column gathers)."""
    W = source_m.size(-1)
    args = source_m.gather(-1, check_arg_m.clamp(max=W - 1))         # [...,M,2]
    return torch.cat([body_preds_m.unsqueeze(-1), args], dim=-1)


def fill_body_flat(flat_source: Tensor, check_arg_source_flat: Tensor,
                   body_preds_flat: Tensor) -> Tensor:
    """Flat fill: ``flat_source [T,W]`` → ``[T,M,3]``."""
    M = body_preds_flat.size(1)
    source_m = flat_source.unsqueeze(1).expand(-1, M, -1)
    return _gather_body_atoms(source_m, check_arg_source_flat, body_preds_flat)


# ── enumeration ──

def enumerate_single_dense(B: int, K_r: int, Y_r: int, query_subjs: Tensor,
                    query_objs: Tensor, enum_pred_q: Tensor, enum_bound_q: Tensor,
                    enum_dir_q: Tensor, fact_index, cartesian_product: bool = False,
                    E: int = 0) -> Tuple[Tensor, Tensor]:
    """Dense enumerate for ≤1 free var (one bound→free fact-index lookup).

    ``cartesian_product`` → all E entities. Returns
    ``(candidates[B,K_r,G_actual], cand_mask[B,K_r,G_actual])``.
    """
    if cartesian_product:
        dev = query_subjs.device
        candidates = torch.arange(E, device=dev).view(1, 1, E).expand(B * K_r, 1, -1).reshape(B, K_r, E)
        return candidates, torch.ones(B, K_r, E, dtype=torch.bool, device=dev)

    source = torch.stack([query_subjs, query_objs], dim=1)   # [B, 2]
    enum_bound_vals = source.gather(1, enum_bound_q)          # [B, K_r]
    candidates, cand_mask = fact_index.enumerate(
        enum_pred_q.reshape(-1), enum_bound_vals.reshape(-1), enum_dir_q.reshape(-1))
    G_actual = min(Y_r, candidates.size(1))
    return (candidates[:, :G_actual].reshape(B, K_r, G_actual),
            cand_mask[:, :G_actual].reshape(B, K_r, G_actual))


def enumerate_cartesian_dense(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_q,
                              fv_dir_q, fv_valid_q, has_free_q, active_mask, fact_index,
                              K_v, V, G_cap=0, fv_any_valid=None,
                              check_arg_source_q=None, body_preds_q=None,
                              num_body_q=None, M=0) -> Tuple[Tensor, Tensor, int]:
    """Dense Cartesian enumerate of ≥2 free vars; topk-caps G to G_cap each step
    (keeps a static shape). Returns ``(source[B,K_r,G,2+V], mask[B,K_r,G], G)``.

    Memory shape (vs the former ``_cartesian_expand_one_fv`` list-of-columns
    formulation): the per-fv mask list is folded into ONE running mask; the
    bound column is selected virtually (no per-step ``[B,K_r,G,W]`` stack); and
    at a capping step prior columns are gathered straight to G_cap via
    interleave index arithmetic (``j = k*G_cur + g``) instead of first
    expanding every column to ``G_cur*K_use``. Selection and survivors stay
    byte-identical: the topk input mask equals the old per-column fold, and
    ``expanded[j] == col[j % G_cur]``.
    """
    dev = query_subjs.device
    if G_cap <= 0:
        G_cap = K_v
    cols: list = []                  # one entry per fv; None = skipped fv (all-zero col)
    folded: Optional[Tensor] = None  # running AND of (mask_fi | ~valid_fi) at G_current
    G_current = 1
    zero = torch.zeros((), dtype=torch.long, device=dev)

    for fv_idx in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            cols.append(None)        # mask term (ones | ~valid) is a no-op
            continue
        ep = fv_pred_q[:, :, fv_idx]
        eb = fv_bound_q[:, :, fv_idx]
        ed = fv_dir_q[:, :, fv_idx]
        ev = fv_valid_q[:, :, fv_idx]

        # bound value per cell: virtual select over [qs, qo, fv cols] (== the old
        # stack+gather; eb selects exactly one column after the same clamp).
        ebx = eb.clamp(max=2 + len(cols) - 1).unsqueeze(-1)     # [B,K_r,1]
        bound = torch.where(ebx == 0, query_subjs.view(B, 1, 1),
                            query_objs.view(B, 1, 1))
        for i, c in enumerate(cols):
            bound = torch.where(ebx == 2 + i, zero if c is None else c, bound)
        bound = bound.expand(B, K_r, G_current)

        flat_pred = ep.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)
        flat_dir = ed.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)
        new_cands, new_mask = fact_index.enumerate(flat_pred, bound.reshape(-1), flat_dir)
        K_use = min(new_cands.size(1), K_v)
        new_cands = new_cands[:, :K_use].reshape(B, K_r, G_current, K_use)
        new_mask = (new_mask[:, :K_use].reshape(B, K_r, G_current, K_use)
                    & ev.view(B, K_r, 1, 1))
        term = new_mask | ~ev.view(B, K_r, 1, 1)                # (mask_t | ~valid_t)
        folded4 = term if folded is None else (folded.unsqueeze(3) & term)
        G_new = G_current * K_use

        if G_new > G_cap:
            # interleaved position j = k*G_current + g (new candidate k SLOWEST).
            combined = folded4.transpose(2, 3).reshape(B, K_r, G_new)
            cmb8 = combined.to(torch.int8)
            _, top_idx = cmb8.topk(G_cap, dim=2, largest=True, sorted=False)
            g_idx = top_idx % G_current
            cols = [None if c is None else c.gather(2, g_idx) for c in cols]
            cols.append(new_cands.reshape(B, K_r, G_current * K_use)
                        .gather(2, g_idx * K_use + top_idx // G_current))
            folded = cmb8.gather(2, top_idx).bool()
            G_current = G_cap
        else:
            cols = [None if c is None else
                    c.unsqueeze(3).expand(B, K_r, G_current, K_use)
                     .transpose(2, 3).reshape(B, K_r, G_new) for c in cols]
            cols.append(new_cands.transpose(2, 3).reshape(B, K_r, G_new))
            folded = folded4.transpose(2, 3).reshape(B, K_r, G_new)
            G_current = G_new

    G_final = G_current
    if folded is None:
        combined_mask = torch.ones(B, K_r, G_final, dtype=torch.bool, device=dev) \
            & has_free_q.unsqueeze(2)
    else:
        combined_mask = folded & has_free_q.unsqueeze(2)
    combined_mask[:, :, 0] = combined_mask[:, :, 0] | (~has_free_q & active_mask)

    source = torch.empty(B, K_r, G_final, 2 + V, dtype=torch.long, device=dev)
    source[..., 0] = query_subjs.view(B, 1, 1)
    source[..., 1] = query_objs.view(B, 1, 1)
    for i, c in enumerate(cols):
        source[..., 2 + i] = zero if c is None else c
    return source, combined_mask, G_final


def enumerate_cartesian_flat(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_q,
                             fv_dir_q, fv_valid_q, has_free_q, active_mask, fact_index,
                             V, fv_any_valid=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Flat Cartesian enumerate, compact-early: only live rows hit the fact index.

    Row contract is IDENTICAL (same rows, same order) to the dense formulation
    — build [B,K_r,G] cands/masks via the interleaved Cartesian expansion
    (see ``enumerate_cartesian_dense``) then ``nonzero(combined_mask)`` — but
    rows are compacted BEFORE every
    ``fact_index.enumerate`` call instead of once at the end:

      * seed = nonzero(active_mask): inactive cells can never survive
        (``combined &= active_mask``);
      * per fv, keep only (row, slot) pairs that can still survive the final
        mask: ``cmask`` slots when the rule uses the fv (term ``cmask | ~ev``),
        ALL slots when it doesn't (dense term is all-ones there), and slot 0
        only for no-free-var rules (the dense ``|=`` keeps exactly their g=0
        cell). Each fv's mask term is fixed once its digit is chosen, so
        dropping early never removes a final survivor.

    The dense g index decomposes into digits ``g = Σ k_i · G_{i-1}`` (the
    ``.transpose(2,3)`` interleave makes each new fv the MOST significant
    digit); ``g_pos`` accumulates it per live row and a final argsort on the
    unique composite (b, r, g) key restores exact ``nonzero`` row-major order.

    Returns ``(flat_source[T,2+V], b_idx[T], r_idx[T])`` for surviving cells.
    """
    dev = query_subjs.device

    def _empty() -> Tuple[Tensor, Tensor, Tensor]:
        return (torch.empty(0, 2 + V, dtype=query_subjs.dtype, device=dev),
                torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    seed = torch.nonzero(active_mask, as_tuple=False)         # [T0, 2] (b, r)
    b_idx, r_idx = seed[:, 0], seed[:, 1]
    if b_idx.numel() == 0:
        return _empty()

    # single [T, W] source (one row-gather per fv, not one gather per column)
    src = torch.stack([query_subjs[b_idx], query_objs[b_idx]], dim=1)
    keep0 = ~has_free_q[b_idx, r_idx]   # no-free rows: only the g=0 chain survives
    g_pos = torch.zeros(b_idx.size(0), dtype=torch.long, device=dev)
    stride = 1                                                # = G before this fv
    n_expanded = 0

    for fv_idx in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            src = torch.cat([src, src.new_zeros(src.size(0), 1)], dim=1)
            continue
        ep = fv_pred_q[b_idx, r_idx, fv_idx]
        eb = fv_bound_q[b_idx, r_idx, fv_idx].clamp(max=src.size(1) - 1)
        ed = fv_dir_q[b_idx, r_idx, fv_idx]
        ev = fv_valid_q[b_idx, r_idx, fv_idx]
        bound = src.gather(1, eb.unsqueeze(1)).squeeze(1)
        cands, cmask = fact_index.enumerate(ep, bound, ed)    # [T, K_f]
        K_f = cands.size(1)
        live = cmask | ~ev.unsqueeze(1)                       # dense per-fv mask term
        slot0 = arange_cached(K_f, dev) == 0
        keep = torch.where(keep0.unsqueeze(1), slot0, live)
        sel = torch.nonzero(keep, as_tuple=False)             # [T', 2] (row, slot)
        row, slot = sel[:, 0], sel[:, 1]
        if row.numel() == 0:
            return _empty()
        b_idx, r_idx, keep0 = b_idx[row], r_idx[row], keep0[row]
        src = torch.cat([src[row], cands[row, slot].unsqueeze(1)], dim=1)
        g_pos = g_pos[row] + slot * stride
        stride *= K_f
        n_expanded += 1

    if n_expanded <= 1:
        # Already in nonzero row-major order: each step's nonzero(keep) is
        # lexicographic in (previous order, slot), so rows sit in
        # (b, r, slot_1, …, slot_k) order; with k ≤ 1 expanded digits that
        # EQUALS the (b, r, g) target — the composite keys are unique, so the
        # argsort would be the identity. Skip the sort + 3 gathers.
        return src, b_idx, r_idx
    order = torch.argsort((b_idx * K_r + r_idx) * stride + g_pos)
    return src[order], b_idx[order], r_idx[order]


# ── guided beam (KGE prior over the join expansion) ──

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


class GuidedSelect:
    """KGE-guided beam state for one resolve step (join path only).

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


# ── L3 join enumeration (semantics-preserving width branch-pruning) ──

def _determined_unknown_count(flat_source, ready_now, arg_src_dep, bpreds_dep,
                              nbody, fact_index, M):
    """Count UNKNOWN (non-fact) body atoms that are FULLY DETERMINED so far.

    ``ready_now [T,M]`` marks atoms whose every free-var ref is already bound.
    Padded/undetermined atoms are excluded → a sound LOWER BOUND on the final
    ``num_unknown`` (monotone as more fv bind), so a row exceeding ``width`` here
    can never survive the final width filter. Returns ``[T]`` counts.
    """
    T = flat_source.size(0)
    if T == 0:
        return flat_source.new_zeros(0, dtype=torch.long)
    body = _gather_body_atoms(
        flat_source.unsqueeze(1).expand(-1, M, -1), arg_src_dep, bpreds_dep)  # [T,M,3]
    exists = fact_index.exists(body.reshape(-1, 3)).reshape(T, M)
    return (ready_now & ~exists).sum(dim=-1)


def enumerate_join_flat(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_src_q,
                        fv_dir_q, fv_valid_q, active_mask, fact_index,
                        V, fv_any_valid, arg_source_dep_q, body_preds_dep_q,
                        num_body_q, ready_after_q, width, w_is_capped,
                        guided: Optional[GuidedSelect] = None):
    """L3 join: incremental free-var expansion with a width BRANCH PRUNER.

    Same row contract as ``enumerate_cartesian_flat`` (``flat_source[T,2+V]``,
    ``b_idx[T]``, ``r_idx[T]``) but expands compactly (nonzero per step, no dense
    padded cartesian) and — when width is bounded — drops partial rows whose
    DETERMINED unknown count already exceeds ``width``. The prune only removes
    rows the final ``apply_filters_flat`` would also reject, so the survivor SET
    is identical to the flat cartesian path.

    ``guided`` (GuidedSelect, default None = byte-identical exhaustive): after
    each fv binds, the atoms it determines are scored (facts exactly 1.0, ground
    unknowns the KGE prior) and only fact-perfect rows + the top-k speculative
    rows per state survive — the prune-BEFORE-the-next-multiply that bounds
    generation at ×k per free variable.
    """
    dev = query_subjs.device
    # seed: every (active query, valid rule slot) — flat row indices into [B*K_r].
    seed = torch.nonzero(active_mask, as_tuple=False)      # [T0, 2] (b, r)
    n_idx, r_idx = seed[:, 0], seed[:, 1]
    if n_idx.numel() == 0:
        return (torch.empty(0, 2, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    # flat_source columns: [subj, obj, fv0, fv1, ...] (free-var cols filled per step).
    cols = [query_subjs[n_idx], query_objs[n_idx]]
    for _ in range(V):
        cols.append(torch.zeros(n_idx.size(0), dtype=torch.long, device=dev))
    src = torch.stack(cols, dim=1)                          # [T, 2+V]
    M = arg_source_dep_q.size(2)
    if guided is not None and guided.active:
        run_score = torch.ones(n_idx.size(0), dtype=torch.float32, device=dev)
        run_fact = torch.ones(n_idx.size(0), dtype=torch.bool, device=dev)

    for fv in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv]:
            continue
        # rows whose rule actually enumerates this fv (else pass through unchanged).
        does_fv = fv_valid_q[n_idx, r_idx, fv]
        if bool(does_fv.any()):
            act = torch.nonzero(does_fv, as_tuple=False).squeeze(1)
            ep = fv_pred_q[n_idx[act], r_idx[act], fv]
            bsrc = fv_bound_src_q[n_idx[act], r_idx[act], fv].clamp(max=src.size(1) - 1)
            bound = src[act].gather(1, bsrc.unsqueeze(1)).squeeze(1)
            ed = fv_dir_q[n_idx[act], r_idx[act], fv]
            cands, cmask = fact_index.enumerate(ep, bound, ed)   # [Ta, Kf]
            Kf = cands.size(1)
            rep = torch.repeat_interleave(act, Kf)
            keep = cmask.reshape(-1)
            new_src = src[rep].clone()
            new_src[:, 2 + fv] = cands.reshape(-1)
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            exp_n = n_idx[rep][keep_idx]; exp_r = r_idx[rep][keep_idx]
            exp_src = new_src[keep_idx]
            # pass-through rows (rule doesn't enumerate this fv at all).
            pas = torch.nonzero(~does_fv, as_tuple=False).squeeze(1)
            if guided is not None and guided.active:
                exp_rows = rep[keep_idx]
                run_score = torch.cat([run_score[exp_rows], run_score[pas]])
                run_fact = torch.cat([run_fact[exp_rows], run_fact[pas]])
            n_idx = torch.cat([exp_n, n_idx[pas]])
            r_idx = torch.cat([exp_r, r_idx[pas]])
            src = torch.cat([exp_src, src[pas]])
        # BRANCH PRUNE: drop rows whose determined-unknown count already > width.
        if w_is_capped and width is not None:
            asd = arg_source_dep_q[n_idx, r_idx]; bpd = body_preds_dep_q[n_idx, r_idx]
            nb = num_body_q[n_idx, r_idx]
            atom_idx = arange_cached(M, dev).unsqueeze(0)
            ready = (ready_after_q[n_idx, r_idx] <= fv) & (atom_idx < nb.unsqueeze(1))  # [T,M]
            du = _determined_unknown_count(src, ready, asd, bpd, nb, fact_index, M)
            surv = torch.nonzero(du <= width, as_tuple=False).squeeze(1)
            n_idx, r_idx, src = n_idx[surv], r_idx[surv], src[surv]
            if guided is not None and guided.active:
                run_score, run_fact = run_score[surv], run_fact[surv]
        # GUIDED SELECT: fold newly-determined atom scores, beam per state.
        if guided is not None:
            if guided.stats is not None:
                guided.stats._bump(guided.stats.bindings_in, guided.d, n_idx.size(0))
            if guided.active:
                run_score, run_fact = guided.update_bindings(
                    src, n_idx, r_idx, fv, run_score, run_fact,
                    arg_source_dep_q, body_preds_dep_q, num_body_q,
                    ready_after_q, M)
                keep_g = guided.topk_keep(n_idx, run_score, run_fact)
                n_idx, r_idx, src = n_idx[keep_g], r_idx[keep_g], src[keep_g]
                run_score, run_fact = run_score[keep_g], run_fact[keep_g]
            if guided.stats is not None:
                guided.stats._bump(guided.stats.bindings_out, guided.d, n_idx.size(0))
        if n_idx.numel() == 0:
            return (torch.empty(0, 2 + V, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev))

    return src, n_idx, r_idx


__all__ = [
    "cluster", "ClusteredRules", "arange_cached", "cumcount_flat",
    "fill_body_flat",
    "enumerate_single_dense", "enumerate_cartesian_dense", "enumerate_cartesian_flat",
    "enumerate_join_flat", "GuidedSelect", "GuidedStats",
]
