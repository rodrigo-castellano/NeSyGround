"""Iteration strategies — *how* a rule is fired per step.

Each strategy emits a list of ``FireSpec`` records per ``(rule, step)``.
A ``FireSpec`` holds the per-slot mat dict, the per-slot transpose
cache, and the body1 existence-check tensors (``fact_hashes``,
``num_facts``, ``provable_hashes``) appropriate for that fire.

Built-in strategies:

* ``SeminaiveStrategy`` — Bancilhon–Ramakrishnan, ``2^n_slots − 1``
  non-overlapping mask fires per rule per step. At step k each mask m
  routes body slots whose bit is 1 to ``delta_mats`` and the rest to
  ``old_mats``; the union of all masks is exactly the new derivations
  with no ``Δ × Δ`` overlap. Wins big when ``Δ ≪ accum`` (later steps,
  near fixpoint).

* ``TrueIStarStrategy`` — single ``accum @ accum`` fire per rule per
  step, where ``accum_mats[p] = old_mats[p] ∪ delta_mats[p]`` is built
  lazily at step boundaries. One sparse-matmul per rule (libtorch fuses
  all four ``old/delta × old/delta`` terms inside the single matmul).
  Wins when ``Δ ≈ accum`` (early growth) where the algorithmic savings
  of skipping ``old × old`` are smaller than the per-fire dispatch
  overhead semi-naive pays.

* ``HybridStrategy`` (default) — picks ``TrueIStar`` or ``Seminaive``
  per step based on ``nnz(delta) / (nnz(delta) + nnz(old))``. Threshold
  ``0.5`` means: when delta is at least half the total, use ``TrueIStar``;
  otherwise use ``Seminaive``. Choice is cached for the duration of the
  step.

Adding a new strategy: subclass ``IterationStrategy`` and override
``step_fires``. Wire it into ``runner._make_strategy``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from grounder.forward.spmm.matrices import (
    csr_local_hashes, build_csr_from_local_hashes, build_csr_and_transpose,
)
from grounder.forward.spmm.classify import (
    SpMMOp, SpMMRuleDesc, slot_pred, slots_for_op,
)
from grounder.forward.spmm.state import ClosureState


__all__ = [
    "FireSpec",
    "IterationStrategy",
    "SeminaiveStrategy",
    "TrueIStarStrategy",
    "HybridStrategy",
]


@dataclass
class FireSpec:
    """One sparse-matmul fire of a rule at a single step."""
    slot_mats: List[Dict[int, Tensor]]
    slot_mats_T: List[Optional[Dict[int, Tensor]]]
    fact_hashes: Tensor
    num_facts: int
    provable_hashes: Tensor


class IterationStrategy(ABC):
    """Abstract: emit per-(rule, step) fire specs."""

    @abstractmethod
    def step_fires(self, desc: SpMMRuleDesc,
                   state: ClosureState) -> List[FireSpec]: ...


# ── Bancilhon–Ramakrishnan semi-naive ────────────────────────────────


class SeminaiveStrategy(IterationStrategy):
    def step_fires(self, desc, state):
        n_slots = slots_for_op(desc.op)
        if n_slots == 0:
            return []

        if state.step == 0:
            # Prime: all slots read base (= delta at step 0); body1
            # existence check vs facts only.
            slot_mats = [state.delta_mats] * n_slots
            slot_mats_T = [state.delta_mats_T] * n_slots
            return [FireSpec(
                slot_mats, slot_mats_T,
                state.fact_hashes, state.num_facts,
                state.empty_fact_hashes)]

        fires: List[FireSpec] = []
        for mask in range(1, 1 << n_slots):
            sm: List[Dict[int, Tensor]] = []
            smT: List[Optional[Dict[int, Tensor]]] = []
            ok = True
            for slot in range(n_slots):
                sp = slot_pred(desc, slot)
                if (mask >> slot) & 1:  # delta in this slot
                    if sp not in state.delta_mats:
                        ok = False
                        break
                    sm.append(state.delta_mats)
                    smT.append(state.delta_mats_T)
                else:  # old in this slot
                    if sp not in state.old_mats:
                        ok = False
                        break
                    sm.append(state.old_mats)
                    smT.append(state.old_mats_T)
            if not ok:
                continue

            if desc.op in (SpMMOp.CASE_A, SpMMOp.EXIST_AND):
                if (mask >> 1) & 1:
                    # body1 is delta — lookup only against the delta
                    # provable slice (facts are never "delta in").
                    fh, nf, ph = (state.empty_fact_hashes, 0,
                                  state.delta_hashes)
                else:
                    # body1 is old — lookup against facts ∪ old derivations.
                    fh, nf, ph = (state.fact_hashes, state.num_facts,
                                  state.old_hashes)
            else:
                fh, nf, ph = (state.fact_hashes, state.num_facts,
                              state.provable_hashes)
            fires.append(FireSpec(sm, smT, fh, nf, ph))
        return fires


# ── True I* (full re-evaluation) ─────────────────────────────────────


class TrueIStarStrategy(IterationStrategy):
    """Fires ``accum @ accum`` once per rule per step.

    Synthesises ``accum_mats`` lazily by merging ``state.old_mats`` and
    ``state.delta_mats`` per pred at step boundaries. The merge result
    is cached for the duration of the step.
    """

    def __init__(self):
        self._cache_step = -1
        self._accum_mats: Dict[int, Tensor] = {}
        self._accum_mats_T: Dict[int, Tensor] = {}

    def _ensure_accum(self, state: ClosureState) -> None:
        if self._cache_step == state.step:
            return
        self._accum_mats = {}
        self._accum_mats_T = {}
        all_preds = set(state.old_mats) | set(state.delta_mats)
        for p in all_preds:
            o = state.old_mats.get(p)
            d = state.delta_mats.get(p)
            if d is None and o is not None:
                self._accum_mats[p] = o
                if p in state.old_mats_T:
                    self._accum_mats_T[p] = state.old_mats_T[p]
            elif o is None and d is not None:
                self._accum_mats[p] = d
                if p in state.delta_mats_T:
                    self._accum_mats_T[p] = state.delta_mats_T[p]
            else:
                o_local = csr_local_hashes(o, state.E)
                d_local = csr_local_hashes(d, state.E)
                merged = torch.cat([o_local, d_local]).sort().values
                if p in state.needs_T:
                    csr, t = build_csr_and_transpose(merged, state.E)
                    self._accum_mats[p] = csr
                    self._accum_mats_T[p] = t
                else:
                    self._accum_mats[p] = build_csr_from_local_hashes(
                        merged, state.E)
        self._cache_step = state.step

    def step_fires(self, desc, state):
        n_slots = slots_for_op(desc.op)
        if n_slots == 0:
            return []
        self._ensure_accum(state)
        slot_mats = [self._accum_mats] * n_slots
        slot_mats_T = [self._accum_mats_T] * n_slots
        return [FireSpec(
            slot_mats, slot_mats_T,
            state.fact_hashes, state.num_facts, state.provable_hashes)]


# ── Hybrid (per-step auto) ───────────────────────────────────────────


class HybridStrategy(IterationStrategy):
    """Pick TrueIStar or Seminaive per step based on Δ/accum ratio.

    Default threshold = 0.5: if delta is at least half the total nnz,
    use TrueIStar (fewer dispatches when Δ ≈ accum); otherwise use
    Seminaive (algorithmic win when Δ ≪ accum).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._semi = SeminaiveStrategy()
        self._tistar = TrueIStarStrategy()
        self._cached_step = -1
        self._chosen: Optional[IterationStrategy] = None

    def _pick(self, state: ClosureState) -> IterationStrategy:
        if state.step == self._cached_step and self._chosen is not None:
            return self._chosen
        if state.step == 0:
            chosen: IterationStrategy = self._semi
        else:
            old_n = state.old_total_nnz()
            delta_n = state.delta_total_nnz()
            total = old_n + delta_n
            if total == 0:
                chosen = self._semi
            else:
                ratio = delta_n / total
                chosen = self._tistar if ratio >= self.threshold else self._semi
        self._cached_step = state.step
        self._chosen = chosen
        return chosen

    def step_fires(self, desc, state):
        return self._pick(state).step_fires(desc, state)
