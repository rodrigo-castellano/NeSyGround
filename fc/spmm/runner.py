"""Main entrypoint: ``run_forward_chaining_spmm``.

This is the orchestration layer — it wires together
``classify_rule`` (ops), ``build_pred_sparse_matrices`` (matrices),
``ClosureState`` (state), the chosen ``IterationStrategy`` (strategies),
and the per-rule ``apply_spmm_rule_slots`` (kernels) into the
fixpoint-iteration loop.

Heavy lifting lives in the dedicated modules; the runner stays small.
"""
from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

import torch
from torch import Tensor

from grounder.fc.spmm.kernels import apply_spmm_rule_slots
from grounder.fc.spmm.matrices import (
    build_pred_sparse_matrices, csr_to_global_hashes, sorted_merge,
    transpose_csr, unique_sorted,
)
from grounder.fc.spmm.ops import SpMMOp, classify_rule
from grounder.fc.spmm.state import ClosureState
from grounder.fc.spmm.strategies import (
    HybridStrategy, IterationStrategy, SeminaiveStrategy, TrueIStarStrategy,
)


__all__ = ["run_forward_chaining_spmm"]


def _make_strategy(mode: str) -> IterationStrategy:
    if mode == "seminaive":
        return SeminaiveStrategy()
    if mode == "true_istar":
        return TrueIStarStrategy()
    if mode == "hybrid":
        return HybridStrategy()
    raise ValueError(
        f"unknown SpMM mode {mode!r}; expected one of "
        "'hybrid' (default), 'seminaive', 'true_istar'")


def _scan_tracked_preds(rule_descs):
    """Walk classified rules to determine which preds need to be tracked
    in the closure state and which need a transpose cache."""
    tracked: Set[int] = set()
    needs_T: Set[int] = set()
    for desc in rule_descs:
        if desc.op == SpMMOp.UNSUPPORTED:
            continue
        tracked.add(desc.pred_A)
        if desc.op in (SpMMOp.MATMUL, SpMMOp.ELEM_AND,
                       SpMMOp.CASE_A, SpMMOp.EXIST_AND):
            tracked.add(desc.pred_B)
        if desc.op == SpMMOp.MATMUL3:
            tracked.add(desc.pred_B)
            tracked.add(desc.pred_C)
        if desc.op == SpMMOp.TRANSPOSE:
            needs_T.add(desc.pred_A)
        if desc.op == SpMMOp.MATMUL and desc.transpose_A:
            needs_T.add(desc.pred_A)
        if desc.op == SpMMOp.MATMUL and desc.transpose_B:
            needs_T.add(desc.pred_B)
    return tracked, needs_T


def run_forward_chaining_spmm(
    compiled_rules: List,
    facts_idx: Tensor,
    num_entities: int,
    num_predicates: int,
    depth: int = 10,
    device: str = "cpu",
    *,
    verbose: bool = True,
    mode: str = "hybrid",
) -> Tuple[Tensor, int]:
    """Compute the FC closure via SpMM.

    Args:
        compiled_rules: list of ``RulePattern`` from the rule index.
        facts_idx: ``[F, 3]`` fact triples ``(pred, subj, obj)``.
        num_entities: total entity count (= ``E``).
        num_predicates: total predicate count (= ``P``).
        depth: hard cap on FC iterations (loop exits early at fixpoint).
        device: target device — ``'cpu'`` or ``'cuda'`` / ``'cuda:N'``.
            All sparse matrices, hash tensors, and intermediate state
            live on this device.
        mode: iteration strategy. One of:
            * ``'hybrid'`` (default) — pick True I* / semi-naive per
              step from the Δ/accum ratio. Best general-purpose mode.
            * ``'seminaive'`` — Bancilhon–Ramakrishnan, ``2^n_slots − 1``
              non-overlapping mask fires per rule per step.
            * ``'true_istar'`` — single ``accum @ accum`` fire per rule
              per step.
        verbose: per-step progress printing.

    Returns:
        ``(sorted_hashes, n_provable)`` — same format as
        :func:`grounder.fc.fc.run_forward_chaining`. Hashes are the
        global ``pred * E² + row * E + col`` encoding sorted ascending.

    Notes:
        * Rules with ``num_body >= 4`` are ``UNSUPPORTED`` and skipped;
          combine with FCDynamic to cover those.
        * 1-, 2-, and 3-body rules whose binding shape doesn't fit the
          op table are also ``UNSUPPORTED`` and skipped.
    """
    four_plus = [
        i for i, cr in enumerate(compiled_rules)
        if getattr(cr, "num_body", 0) >= 4
    ]
    if four_plus and verbose:
        print(f"    [SpMM] {len(four_plus)} rules with num_body >= 4 "
              f"will be skipped (UNSUPPORTED)")

    t0 = time.time()
    dev = torch.device(device)
    E = num_entities
    P = num_predicates
    E2 = E * E

    facts = facts_idx.to(dev)
    fact_preds = facts[:, 0]
    fact_subjs = facts[:, 1]
    fact_objs = facts[:, 2]
    num_facts = int(facts.shape[0])
    if num_facts > 0:
        fact_hashes = (fact_preds * E2 + fact_subjs * E
                       + fact_objs).sort().values
    else:
        fact_hashes = torch.zeros(0, dtype=torch.long, device=dev)

    base_mats = build_pred_sparse_matrices(
        fact_preds, fact_subjs, fact_objs, E, P, device=dev)

    rule_descs = [classify_rule(cr) for cr in compiled_rules]
    n_unsupported = sum(1 for d in rule_descs if d.op == SpMMOp.UNSUPPORTED)
    if n_unsupported > 0 and verbose:
        print(f"    [SpMM] {n_unsupported}/{len(rule_descs)} rules "
              f"UNSUPPORTED (skipped — fall back to FCDynamic if needed)")

    tracked_preds, needs_T = _scan_tracked_preds(rule_descs)

    base_mats_T: Dict[int, Tensor] = {}
    for p in needs_T:
        if p in base_mats:
            base_mats_T[p] = transpose_csr(base_mats[p], E)

    state = ClosureState(
        base_mats=base_mats, base_mats_T=base_mats_T,
        tracked_preds=tracked_preds, needs_T=needs_T,
        fact_hashes=fact_hashes, num_facts=num_facts,
        E=E, P=P, device=dev)

    strategy = _make_strategy(mode)

    for step in range(depth):
        # Per-fire dedup against the closure, then a single end-of-step
        # cat-sort-unique to dedup across rules. Uses ``sort + unique_sorted``
        # rather than ``torch.unique`` because the latter's hash-table
        # workspace blows past 5 GB on GPU step 2 of fb15k237; sorting is
        # ~2× memory but predictable and OOM-resistant.
        new_hashes_list: List[Tensor] = []
        n_ph = state.provable_hashes.shape[0]

        for desc in rule_descs:
            if desc.op == SpMMOp.UNSUPPORTED:
                continue
            for fire in strategy.step_fires(desc, state):
                result = apply_spmm_rule_slots(
                    desc, fire.slot_mats, fire.slot_mats_T, E,
                    fire.fact_hashes, fire.num_facts, fire.provable_hashes)
                if result is None:
                    continue
                h = csr_to_global_hashes(result, desc.head_pred, E, E2)
                if h is None:
                    continue
                if n_ph > 0:
                    pos = torch.searchsorted(state.provable_hashes, h)
                    valid = pos < n_ph
                    clamped = torch.clamp(pos, 0, max(n_ph - 1, 0))
                    already = valid & (state.provable_hashes[clamped] == h)
                    h = h[~already]
                    if h.numel() == 0:
                        continue
                new_hashes_list.append(h)

        if not new_hashes_list:
            break

        # Cross-rule dedup. On CUDA we run this on CPU because at large
        # depths the cat/sort temporaries (~3× input nnz) can blow past
        # the GPU's free workspace even though the final ``added`` set
        # is small. On CPU we stay in place.
        if dev.type == "cuda":
            cpu_list = [h.cpu() for h in new_hashes_list]
            del new_hashes_list
            torch.cuda.empty_cache()
            all_new = unique_sorted(
                torch.cat(cpu_list).sort().values)
            ph = (state.provable_hashes.cpu()
                  if state.provable_hashes.numel() > 0
                  else state.provable_hashes)
        else:
            all_new = unique_sorted(
                torch.cat(new_hashes_list).sort().values)
            ph = state.provable_hashes

        if ph.numel() > 0:
            n_ph = ph.shape[0]
            pos = torch.searchsorted(ph, all_new)
            valid = pos < n_ph
            clamped = torch.clamp(pos, 0, max(n_ph - 1, 0))
            already = valid & (ph[clamped] == all_new)
            added = all_new[~already]
        else:
            added = all_new

        if added.numel() == 0:
            break

        if dev.type == "cuda":
            added = added.to(dev)

        state.advance(added)

        if verbose:
            print(f"    [SpMM] step {step}: +{added.numel()} atoms "
                  f"(total {state.total_provable()})")

    n_provable = state.total_provable()
    elapsed = time.time() - t0
    if verbose:
        print(f"  [SpMM] FC complete: {n_provable} provable atoms "
              f"({elapsed:.2f}s)")

    if n_provable > 0:
        return state.provable_hashes.to(dev), n_provable
    return torch.zeros(1, dtype=torch.long, device=dev), 0
