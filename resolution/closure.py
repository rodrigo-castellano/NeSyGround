"""Closure resolution — single-shot lookup grounder over the FC closure.

This module implements a third resolution mode (in addition to
``sld`` / ``rtf`` / ``enum``) that fully bypasses BC's per-depth state
machine. Each query becomes a hash-table lookup in
``fp_global_hashes`` plus a witness-table gather; the per-closure-atom
``(rule_id, body_grounding)`` witness produced by FC at init *is* the
proof. No MGU, no per-depth iteration, no per-state allocation.

Used as a drop-in oracle whenever a model only needs the proof that
the FC closure already certifies — e.g. SBR training when the closure
is at fixpoint and ``sld.fp_global.d1`` would only re-derive what FC
has already provided.

Performance: the tensor body lives in :func:`_resolve_closure_tensors`,
which is ``torch.compile(mode='reduce-overhead')``-wrapped on first
call. The wrapper :func:`resolve_closure` keeps the Python-level
ProofEvidence / RuleGroundings construction out of the compiled
graph (dataclass construction is not dynamo-friendly).

Decoupling: imports only public-looking primitives
(``fc.fp_global``, ``fc.witnesses``, ``filters.soundness``,
``data.fact_index``, ``data.rule_index``, ``groundings``, ``types``).
It deliberately avoids the SLD / RTF / Enum module trees so the
closure path stays independent.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor


def init_closure(grounder, **kwargs) -> None:
    """One-shot init for closure resolution.

    Reuses the existing ``fc.fp_global.build_fp_global_set`` builder to
    produce ``fp_global_hashes`` + the witness table that
    ``fc.witnesses.build_witness_table`` registers on the grounder.
    After this returns the grounder carries enough state to answer any
    query via a single batched lookup in :func:`resolve_closure`.

    ``compile_closure=True`` (default) opts the resolver into
    ``torch.compile(mode='reduce-overhead', dynamic=False)`` on first
    call. Pass ``compile_closure=False`` to keep the eager path (useful
    for tests + tiny KBs where compile overhead would dominate).
    """
    from grounder.data.rule_index import RuleIndexEnum
    from grounder.fc.fp_global import build_fp_global_set
    from grounder.data.fact_index import ArgKeyFactIndex

    # 1. Build the FC closure + witness table. We feed an enum
    #    RuleIndex transiently because ``build_fp_global_set``'s
    #    default expects ``grounder._enum_ri`` (only populated for
    #    enum resolutions). All buffers (``fp_global_hashes``,
    #    ``num_fp_global``, ``fp_global_witness_{rule,body,bcount}``,
    #    flags ``_has_fp_global*``, ``_E_fp_global``, ``_P_fp_global``,
    #    ``_fp_global_last_idx``) are registered on the grounder.
    P = int(grounder.kb.predicate_no) + 1
    E = int(grounder.kb.constant_no) + 1
    enum_ri = RuleIndexEnum(
        grounder.kb.rule_index.rules_heads_sorted,
        grounder.kb.rule_index.rules_bodies_sorted,
        grounder.kb.rule_index.rule_lens_sorted,
        constant_no=grounder.kb.constant_no,
        num_predicates=P,
        padding_idx=grounder.kb.padding_idx,
        device=grounder.kb.device_,
    )
    build_fp_global_set(
        grounder, grounder.kb.device_,
        compiled_rules=enum_ri.patterns, P=P, E=E,
    )

    # 2. Cache an arg-key fact index so open-var queries
    #    ``(pred, h, ?)`` / ``(pred, ?, t)`` can be resolved with a
    #    single targeted lookup. Reuse the KB's index if it already
    #    is an ArgKeyFactIndex (set by ``fact_index_type='arg_key'``),
    #    otherwise build one over the closure-augmented facts.
    fi = grounder.kb.fact_index
    if hasattr(fi, "_a0_offsets"):
        grounder._closure_lookup_index = fi
    else:
        grounder._closure_lookup_index = ArgKeyFactIndex(
            fi.facts_idx,
            constant_no=grounder.kb.constant_no,
            padding_idx=grounder.kb.padding_idx,
            device=grounder.kb.device_,
            pack_base=fi.pack_base,
        )

    # 3. Compile config. Default OFF: empirically the eager closure
    #    path is ~equal to ``torch.compile(mode="reduce-overhead")``
    #    once the host sync is gone, and eager keeps fp_global truly
    #    dynamic-shape across KBs (no recompile when ``n_fp`` changes,
    #    no per-graph CUDA-graph memory pool). Opt in via
    #    ``compile_closure=True`` for static-shape SBR training where
    #    every batch is identically shaped.
    grounder._closure_compile_enabled = bool(
        kwargs.get("compile_closure", False))
    grounder._closure_compiled_fn = None

    # ``closure_fast_rg``: opt in to the dedup-free, compile-friendly
    # ``closure_to_rule_groundings_fast`` builder instead of the legacy
    # ``evidence_to_rule_groundings -> populate_query_pool_idx`` pair.
    # The fast builder uses no ``torch.unique`` / ``nonzero`` so the
    # whole ``run_bc -> rule_tail`` pipeline can be wrapped in a single
    # ``torch.compile(mode='reduce-overhead')`` region without graph
    # breaks. Semantic note: K_iter > 1 ``_rule_loop`` no longer
    # propagates updates across shared atoms (non-dedup pool), so the
    # default is OFF; opt in only when the consumer is K_iter=1 SBR.
    grounder._closure_fast_rg = bool(kwargs.get("closure_fast_rg", False))

    # 4. Nominal BCGrounder attrs other helpers may read defensively.
    grounder.K = 0
    grounder.max_vars_per_rule = 3
    if not hasattr(grounder, "C") or grounder.C is None:
        grounder.C = int(kwargs.get("max_total_groundings", 64) or 64)
    grounder._compiled = False
    grounder._clone_between_steps = False
    grounder._uses_outer_compile = False
    grounder._compiled_inner = None
    grounder._fn_steps_by_depth = {}


@torch.no_grad()
def _resolve_closure_tensors(
    grounder,
    queries: Tensor,
    query_mask: Tensor,
    K: int,
    include_open_var: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure-tensor body of :func:`resolve_closure` — compilable.

    Returns 5 K-slot tensors:
      ``witness_rule_K`` ``[B, K]`` long  — rule index per slot (-1 = invalid)
      ``witness_body_K`` ``[B, K, M, 3]`` long  — body atoms per slot
      ``witness_count_K`` ``[B, K]`` long  — valid body atom count per slot
      ``heads_K``       ``[B, K, 3]`` long  — head atom per slot
      ``valid_rule``    ``[B, K]`` bool  — slot fired & rule valid

    Branchless on tensor scalars (no ``bool(...).any()`` host sync).
    The two top-level guards (``n_fp > 0``, ``include_open_var``) are
    Python ints / bools — dynamo specializes on them so each
    grounder instance gets one compiled graph per (B, K, M, flag)
    tuple. For SBR training (always concrete, K=1, include_open_var
    can be False) the graph is the minimal hash + gather + scrub
    sequence with no host sync — exactly what
    ``torch.compile(mode='reduce-overhead')`` needs to capture into
    a single CUDA graph.
    """
    from grounder.filters.soundness import check_in_fp_global

    pad = int(grounder.kb.padding_idx)
    E = int(grounder._E_fp_global)
    E2 = E * E
    dev = queries.device
    B = int(queries.size(0))
    M = int(grounder.kb.M)
    fp_hashes = grounder.fp_global_hashes
    last_idx = int(getattr(grounder, "_fp_global_last_idx", 0))
    n_fp = int(fp_hashes.numel())

    # ── A. Per-slot buffers (C=K). All start in the "no firing" state
    #    so unprovable rows are clean by construction.
    witness_rule_K = torch.full(
        (B, K), -1, dtype=torch.long, device=dev)
    witness_body_K = torch.full(
        (B, K, M, 3), pad, dtype=torch.long, device=dev)
    witness_count_K = torch.zeros(
        (B, K), dtype=torch.long, device=dev)
    heads_K = torch.full(
        (B, K, 3), pad, dtype=torch.long, device=dev)
    valid_K = torch.zeros((B, K), dtype=torch.bool, device=dev)

    if n_fp > 0:
        q_pred = queries[:, 0].long()
        q_h = queries[:, 1].long()
        q_t = queries[:, 2].long()
        has_free = (q_h == pad) | (q_t == pad)
        is_concrete = ~has_free

        # ── B. Concrete queries: one witness, slot 0 only. ──
        q_hash = q_pred * E2 + q_h * E + q_t
        found_ground = check_in_fp_global(q_hash, fp_hashes)
        idx = torch.searchsorted(fp_hashes, q_hash).clamp(0, last_idx)
        wr = grounder.fp_global_witness_rule[idx]    # [B]
        wb = grounder.fp_global_witness_body[idx]    # [B, M, 3]
        wc = grounder.fp_global_witness_bcount[idx]  # [B]
        fire_concrete = found_ground & is_concrete & query_mask
        witness_rule_K[:, 0] = torch.where(
            fire_concrete, wr, witness_rule_K[:, 0])
        witness_body_K[:, 0, :, :] = torch.where(
            fire_concrete.view(B, 1, 1), wb, witness_body_K[:, 0, :, :])
        witness_count_K[:, 0] = torch.where(
            fire_concrete, wc, witness_count_K[:, 0])
        heads_K[:, 0, :] = torch.where(
            fire_concrete.view(B, 1), queries.long(), heads_K[:, 0, :])
        valid_K[:, 0] = fire_concrete

        # ── C. Open-var queries: up to K matching closure atoms.
        #    Statically gated by ``include_open_var`` — when False,
        #    dynamo elides this whole block, which is what SBR
        #    training (always concrete) wants. Branchless on
        #    tensor scalars — no host sync.
        if include_open_var:
            fact_idx, valid_lookup = (
                grounder._closure_lookup_index.targeted_lookup(
                    queries, max_results=K))
            # ``targeted_lookup`` returns ``[B, K]``: K candidate
            # facts per query, with ``valid_lookup`` flagging real
            # matches. For concrete queries it returns same-pred-
            # same-head facts that may not equal the query; the
            # ``in_closure`` + ``sel_K`` masks below gate them out.
            fi_flat = fact_idx.clamp(min=0)                       # [B, K]
            all_facts = grounder.kb.fact_index.facts_idx
            picked = all_facts[fi_flat]                           # [B, K, 3]
            picked_hash = (
                picked[..., 0].long() * E2
                + picked[..., 1].long() * E
                + picked[..., 2].long()
            )                                                     # [B, K]
            idx2 = torch.searchsorted(
                fp_hashes, picked_hash).clamp(0, last_idx)        # [B, K]
            in_closure = fp_hashes[idx2] == picked_hash           # [B, K]
            ok_k = (
                in_closure
                & valid_lookup
                & has_free.unsqueeze(-1)
                & query_mask.unsqueeze(-1)
            )                                                     # [B, K]

            wr_k = grounder.fp_global_witness_rule[idx2]          # [B, K]
            wb_k = grounder.fp_global_witness_body[idx2]          # [B, K, M, 3]
            wc_k = grounder.fp_global_witness_bcount[idx2]        # [B, K]

            # Splice open-var rows into the K-slot buffers. Concrete
            # rows are gated out by the ``sel_K = has_free.unsqueeze``
            # mask so they keep slot-0 from the concrete pass above.
            sel_K = has_free.unsqueeze(-1).expand(B, K)            # [B, K]
            witness_rule_K = torch.where(sel_K, wr_k, witness_rule_K)
            sel_Kbody = sel_K.view(B, K, 1, 1).expand(B, K, M, 3)
            witness_body_K = torch.where(sel_Kbody, wb_k, witness_body_K)
            witness_count_K = torch.where(sel_K, wc_k, witness_count_K)
            sel_Khead = sel_K.view(B, K, 1).expand(B, K, 3)
            heads_K = torch.where(sel_Khead, picked, heads_K)
            valid_K = torch.where(sel_K, ok_k, valid_K)

    valid_rule = valid_K & (witness_rule_K >= 0)               # [B, K]

    # ── D. Scrub unfired rows to padding sentinels.
    pad_long = torch.full((), pad, dtype=torch.long, device=dev)
    zero_long = torch.zeros((), dtype=torch.long, device=dev)
    neg_long = torch.full((), -1, dtype=torch.long, device=dev)
    witness_body_K = torch.where(
        valid_rule.view(B, K, 1, 1), witness_body_K, pad_long)
    heads_K = torch.where(
        valid_rule.view(B, K, 1), heads_K, pad_long)
    witness_count_K = torch.where(valid_rule, witness_count_K, zero_long)
    witness_rule_K = torch.where(valid_rule, witness_rule_K, neg_long)

    return witness_rule_K, witness_body_K, witness_count_K, heads_K, valid_rule


@torch.no_grad()
def resolve_closure(
    grounder,
    queries: Tensor,
    query_mask: Tensor,
    *,
    max_open_var_witnesses: int = 1,
    closure_skip_open_var: Optional[bool] = None,
    closure_skip_inner_pool_idx: bool = True,
    **init_kwargs: Any,
):
    """Resolve all queries via a single closure lookup.

    Returns a :class:`grounder.types.GrounderOutput` whose ``state``,
    ``evidence`` and ``rule_groundings`` mirror what an
    ``sld.fp_global.d1`` grounder would emit on the same KB *for
    queries actually in the FC closure*. Queries outside the closure
    are dropped (``state_valid=False``, no ``A_in`` / ``A_out`` entry);
    this is the semantic-correctness improvement over the SLD-d1
    baseline, which over-fires by emitting phantom groundings whose
    body atoms aren't real facts.

    Concrete queries hash directly into ``fp_global_hashes``. Open-var
    queries (``(pred, h, ?)`` or ``(pred, ?, t)``) consult the
    closure-side ``ArgKeyFactIndex`` for up to
    ``max_open_var_witnesses`` matching closure atoms; each completion
    becomes a separate ``C``-slot in the returned evidence. Default
    ``max_open_var_witnesses=1`` preserves the single-witness shape
    used by SBR training.

    ``closure_skip_open_var`` controls whether the open-var lookup
    runs at all. When ``True`` the open-var branch is statically
    elided from the compiled graph, which lets
    ``mode='reduce-overhead'`` capture the whole resolver into a
    single CUDA graph (no host sync). When ``False`` the open-var
    lookup always runs (branchless, ``torch.where``-gated). Default
    ``None`` → auto: ``True`` when ``max_open_var_witnesses == 1``
    (the SBR-training shape — open-var queries are not expected and
    the lookup would be dead work).

    ``closure_skip_inner_pool_idx=True`` (default) skips the
    redundant ``populate_query_pool_idx`` call inside this resolver
    — ``run_bc`` re-runs it with the original queries anyway. Set
    ``False`` if you call ``.forward()`` directly and need
    ``rule_groundings.query_pool_idx`` populated by the resolver.

    The tensor body is lazily compiled (``torch.compile(mode=
    'reduce-overhead', dynamic=False)``) on first call, gated by the
    ``compile_closure`` init kwarg. ProofEvidence / RuleGroundings /
    ProofState construction happens here outside the graph because
    dataclass init isn't dynamo-friendly.
    """
    from grounder.groundings import populate_query_pool_idx
    from grounder.types import GrounderOutput, ProofEvidence, ProofState

    K = max(1, int(max_open_var_witnesses or 1))
    # Auto-default: skip open-var when K==1 (the SBR-training case
    # never produces open-var queries, so the lookup is dead work).
    if closure_skip_open_var is None:
        skip_open_var = (K == 1)
    else:
        skip_open_var = bool(closure_skip_open_var)
    include_open_var = not skip_open_var

    # Lazy compile on first call, keyed by ``include_open_var`` so a
    # single grounder can be compiled for both a fast (concrete-only)
    # and a general (open-var) path if needed.
    if grounder._closure_compile_enabled:
        cache = grounder._closure_compiled_fn
        if cache is None:
            cache = {}
            grounder._closure_compiled_fn = cache
        key = bool(include_open_var)
        fn = cache.get(key)
        if fn is None:
            fn = torch.compile(
                _resolve_closure_tensors,
                mode="reduce-overhead",
                dynamic=False,
            )
            cache[key] = fn
    else:
        fn = _resolve_closure_tensors

    wr_K, wb_K, wc_K, h_K, valid_rule = fn(
        grounder, queries, query_mask, K, include_open_var)

    pad = int(grounder.kb.padding_idx)
    M = int(grounder.kb.M)
    B = int(queries.size(0))
    dev = queries.device

    body = wb_K.view(B, K, 1, M, 3)
    head = h_K.view(B, K, 1, 3).long()
    rule_idx_t = wr_K.view(B, K, 1)
    body_count = wc_K.view(B, K, 1)
    mask = valid_rule                                          # [B, K]
    count = mask.sum(dim=1)

    evidence = ProofEvidence(
        body=body, mask=mask, count=count,
        rule_idx=rule_idx_t, body_count=body_count,
        D=1, M=M, head=head,
    )

    rule_groundings: Optional[Any] = None
    if grounder._collect_rule_groundings:
        if getattr(grounder, "_closure_fast_rg", False):
            # Static-shape, dedup-free, fully compile-friendly builder.
            # Skips ``evidence_to_rule_groundings``'s ``torch.unique`` /
            # ``nonzero`` calls and the follow-up
            # ``populate_query_pool_idx`` (queries are slots 0..B-1 of
            # the fresh atom_table). Closure-resolver only — see
            # ``closure_to_rule_groundings_fast`` docstring for the
            # K_iter > 1 semantic note.
            from grounder.groundings import closure_to_rule_groundings_fast
            rule_groundings = closure_to_rule_groundings_fast(
                witness_rule=wr_K, witness_body=wb_K,
                witness_count=wc_K, valid_rule=valid_rule,
                queries=queries,
                num_rules=int(grounder.kb.num_rules),
                M_max=M, padding_idx=pad,
            )
        else:
            rule_groundings = evidence.to_rule_groundings(
                pad, num_rules=int(grounder.kb.num_rules))
            if not closure_skip_inner_pool_idx:
                rule_groundings = populate_query_pool_idx(
                    rule_groundings, queries, pad)

    state = ProofState(
        proof_goals=torch.full(
            (B, K, 1, 3), pad, dtype=torch.long, device=dev),
        state_valid=valid_rule,                                # [B, K]
        top_ridx=wr_K,                                         # [B, K]
        next_var_indices=None,
    )
    return GrounderOutput(
        state=state, evidence=evidence,
        rule_groundings=rule_groundings,
    )


__all__ = ["init_closure", "resolve_closure"]
