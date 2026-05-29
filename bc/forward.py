"""Forward orchestration for ``BCGrounder``.

The class's :meth:`BCGrounder.forward` dispatches into one of three
paths depending on resolution / compile / batch_size:

* :func:`forward_chunked` — required when
  ``_uses_outer_compile=True`` (sld/rtf with ``compile_mode`` set), and
  also used when ``batch_size`` is set on a large input. Pads each
  chunk to ``batch_size`` so the captured CUDA graph sees a stable
  shape.
* :func:`forward_one_batch_inner` — single chunk, no r2g build. Used
  inside :func:`forward_chunked`'s inner loop.
* :func:`forward_one_batch` — single batch with r2g finalisation.
  Used when no chunking is needed.

After all chunks finish, :func:`merge_chunk_outputs` concatenates the
per-chunk outputs and (when ``_collect_rule_groundings=True``) builds
the unified :class:`grounder.types.RuleGroundings` from the merged
:class:`grounder.types.ProofEvidence`.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grounder.bc.state import init_states
from grounder.bc.step import step
from grounder.bc.terminal import filter_terminal
from grounder.types import (
    GrounderOutput,
    ProofEvidence,
    ProofState,
    RuleGroundings,
)


def forward(
    grounder, queries: Tensor, query_mask: Tensor,
    *, batch_size: Optional[int] = None,
    **init_kwargs,
) -> GrounderOutput:
    """Public dispatch — picks chunked vs single-batch path.

    Outer-compile resolutions (sld/rtf) always go through the chunked
    path so the captured CUDA graph is replayed against a stable
    per-chunk shape; caller must supply ``batch_size``. For other
    resolutions, an explicit ``batch_size`` (or compile_mode auto)
    triggers chunking when ``batch_size < N``.
    """
    # Reset the per-step "considered" accumulator once at the top of
    # the public forward call. The chunked path's per-chunk
    # ``forward_one_batch_inner`` MUST NOT reset (each chunk's
    # captures need to accumulate; resetting per-chunk would only
    # keep the final chunk's firings). The single-batch path uses
    # this same top-level reset and skips the per-call reset inside
    # ``forward_one_batch``.
    if grounder._collect_rule_groundings:
        from grounder.bc.considered import reset_accumulator
        reset_accumulator(grounder)
    N = queries.size(0)
    if grounder._uses_outer_compile:
        if batch_size is None or batch_size <= 0:
            raise ValueError(
                f"resolution={grounder.resolution!r} with "
                f"compile_mode={grounder.compile_mode!r} requires an "
                "explicit positive ``batch_size`` so the captured "
                "CUDA graph sees a stable per-chunk shape.")
        return forward_chunked(
            grounder, queries, query_mask, batch_size, **init_kwargs)
    if batch_size is None and grounder.compile_mode is not None:
        batch_size = auto_batch_size(grounder, N)
    if batch_size is not None and 0 < batch_size < N:
        return forward_chunked(
            grounder, queries, query_mask, batch_size, **init_kwargs)
    return forward_one_batch(grounder, queries, query_mask, **init_kwargs)


def auto_batch_size(grounder, N: int) -> int:
    """Pick a chunk size for compile mode based on per-chunk memory.

    Heuristic: ``[B*S, K_r, G_r, M, 3]`` body tensor (the dominant
    intermediate at d=0 with ``init_state_shape='full'``) should
    stay below ~1 GB to leave room for the rest of the graph and
    for CUDA-graph private pools. ``B = 1e9 / (S*K_r*G_r*M*3*8)``.
    """
    if grounder.resolution != "enum":
        return N      # only the enum dense path needs chunking
    S = grounder.S
    K_r = getattr(grounder, "K_r", 1) or 1
    G_r = getattr(grounder, "G_r", 1) or 1
    M = max(grounder.kb.M, 1)
    # bytes per body element = 8 (long); 3 args per atom
    per_query_bytes = S * K_r * G_r * M * 3 * 8
    budget = 1_000_000_000   # ~1 GB
    B_max = max(1, budget // max(per_query_bytes, 1))
    return min(N, int(B_max))


def forward_chunked(
    grounder, queries: Tensor, query_mask: Tensor,
    batch_size: int, **init_kwargs,
) -> GrounderOutput:
    """Pad each chunk to ``batch_size`` so the compiled graph sees stable
    shapes, then concat outputs.

    The rule-groundings build happens once at the end across the
    merged evidence (see :func:`merge_chunk_outputs`).
    """
    N = queries.size(0)
    dev = queries.device
    pad_q = queries.new_zeros(batch_size, queries.shape[1])
    pad_m = torch.zeros(batch_size, dtype=torch.bool, device=dev)

    # Choose the inner step. When outer compile is enabled (sld/
    # rtf with compile_mode set), wrap forward_one_batch_inner
    # with torch.compile once and reuse for every chunk; the
    # padded chunk shape stays constant so the captured CUDA
    # graph replays. Otherwise run the eager inner.
    if grounder._uses_outer_compile:
        if grounder._compiled_inner is None:
            # Compile the bound method so dynamo's per-instance
            # specialization keys on ``grounder`` identity (the same
            # behavior the pre-extraction bc.py path had).
            from grounder.bc.compile import Compiler
            grounder._compiled_inner = Compiler.wrap(
                grounder._forward_one_batch_inner,
                mode=grounder.compile_mode, fullgraph=True)
        inner_fn = grounder._compiled_inner
    else:
        inner_fn = grounder._forward_one_batch_inner

    chunk_outputs: List[GrounderOutput] = []
    chunk_sizes: List[int] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        q = queries[start:end]
        m = query_mask[start:end]
        n_real = q.size(0)
        if n_real < batch_size:
            # Pad to static shape; padded rows have mask=False so
            # init_states / step / collection all skip them.
            q_padded = pad_q.clone()
            m_padded = pad_m.clone()
            q_padded[:n_real] = q
            m_padded[:n_real] = m
            q, m = q_padded, m_padded
        if grounder._uses_outer_compile and not torch.compiler.is_compiling():
            # Tell CUDA-graph trees a new iteration is starting so
            # static-address output buffers are safe to reuse.
            # Mirrors dprl's per-step pattern (PPO compilation.py
            # calls this before every rollout_step).
            # Skipped when called from inside torch.compile tracing —
            # the outer compile owns step boundaries, and
            # mark_step_begin is a forbidden tracing op anyway.
            torch.compiler.cudagraph_mark_step_begin()
        out = inner_fn(q, m, **init_kwargs)
        if grounder._uses_outer_compile:
            # Captured graph outputs share static memory across
            # calls — clone every tensor field so subsequent
            # chunks don't overwrite earlier per-chunk results
            # (``merge_chunk_outputs`` reads them all at the end).
            out = clone_grounder_output(out)
        chunk_outputs.append(out)
        chunk_sizes.append(n_real)

    return merge_chunk_outputs(grounder, chunk_outputs, chunk_sizes, queries)


def clone_grounder_output(out: GrounderOutput) -> GrounderOutput:
    """Deep-clone every tensor field of a GrounderOutput.

    Used after compiled (CUDA-graph) chunked calls so the per-chunk
    output survives subsequent chunks overwriting the static-address
    buffers the captured graph re-uses.
    """
    def _c(t):
        return t.clone() if isinstance(t, Tensor) else t

    st = out.state
    new_state = ProofState(
        proof_goals=_c(st.proof_goals),
        state_valid=_c(st.state_valid),
        top_ridx=_c(st.top_ridx),
        next_var_indices=_c(st.next_var_indices),
    )
    new_evidence = None
    if out.evidence is not None:
        ev = out.evidence
        new_evidence = ProofEvidence(
            body=_c(ev.body), mask=_c(ev.mask),
            count=_c(ev.count), rule_idx=_c(ev.rule_idx),
            body_count=_c(ev.body_count),
            D=ev.D, M=ev.M, head=_c(ev.head),
        )
    new_rg = None
    if out.rule_groundings is not None:
        rg = out.rule_groundings
        new_rg = RuleGroundings(
            atom_table=_c(rg.atom_table),
            body_pool_idx=_c(rg.body_pool_idx),
            body_atom_valid=_c(rg.body_atom_valid),
            head_pool_idx=_c(rg.head_pool_idx),
            rule_idx=_c(rg.rule_idx),
            rule_offsets=_c(rg.rule_offsets),
            firing_valid=_c(rg.firing_valid),
            num_atoms=rg.num_atoms,
            num_rules=rg.num_rules,
            M_max=rg.M_max,
            query_pool_idx=_c(rg.query_pool_idx) if rg.query_pool_idx is not None else None,
            _has_real_firings_valid=rg._has_real_firings_valid,
        )
    return GrounderOutput(
        state=new_state,
        evidence=new_evidence,
        rule_groundings=new_rg,
    )


def merge_chunk_outputs(
    grounder,
    chunk_outputs: List[GrounderOutput],
    chunk_sizes: List[int],
    queries: Tensor,
) -> GrounderOutput:
    """Concat per-chunk evidence/state along B; finalise r2g once."""
    evidences = [o.evidence for o in chunk_outputs
                 if o.evidence is not None]
    if evidences:
        # Trim padding rows by chunk_sizes, then cat.
        def _trim_cat(attr):
            parts = []
            for ev, n in zip(evidences, chunk_sizes):
                t = getattr(ev, attr)
                if t is None:
                    return None
                parts.append(t[:n])
            return torch.cat(parts, dim=0)
        body = _trim_cat("body")
        mask = _trim_cat("mask")
        count = mask.sum(dim=1) if mask is not None else None
        rule_idx = _trim_cat("rule_idx")
        body_count = _trim_cat("body_count")
        head = _trim_cat("head")
        evidence = ProofEvidence(
            body=body, mask=mask, count=count, rule_idx=rule_idx,
            body_count=body_count,
            D=evidences[0].D, M=evidences[0].M, head=head,
        )
    else:
        evidence = None

    # State: concat trimmed proof_goals / state_valid / top_ridx.
    # The flat path produces dynamic S_out per chunk, so chunks may
    # have different inner shape on dim=1 (state) and beyond. Pad
    # each chunk to the max-inner-shape before cat. Padding is at
    # ``padding_idx`` for atoms / ``-1`` for ridx / ``False`` for
    # masks / ``0`` for indices — same conventions as ``init_states``
    # / ``pack_states_flat``'s placeholders.
    states = [o.state for o in chunk_outputs]
    pad = grounder.kb.padding_idx
    def _pad_to(t: Tensor, target_inner: List[int], pad_val) -> Tensor:
        cur = list(t.shape[1:])
        if cur == target_inner:
            return t
        out = torch.full(
            (t.size(0), *target_inner), pad_val,
            dtype=t.dtype, device=t.device)
        slices = (slice(None),) + tuple(slice(0, c) for c in cur)
        out[slices] = t
        return out
    _attr_pad = {
        "proof_goals": pad,
        "state_valid": False,
        "top_ridx": -1,
        "next_var_indices": 0,
    }
    def _state_cat(attr):
        parts_raw = []
        for s, n in zip(states, chunk_sizes):
            t = getattr(s, attr)
            if t is None:
                return None
            parts_raw.append(t[:n])
        max_inner: List[int] = []
        for d in range(1, parts_raw[0].dim()):
            max_inner.append(max(p.size(d) for p in parts_raw))
        pad_val = _attr_pad.get(attr, 0)
        parts = [_pad_to(p, max_inner, pad_val) for p in parts_raw]
        return torch.cat(parts, dim=0)
    state = ProofState(
        proof_goals=_state_cat("proof_goals"),
        state_valid=_state_cat("state_valid"),
        top_ridx=_state_cat("top_ridx"),
        next_var_indices=_state_cat("next_var_indices"),
    )

    # Build RuleGroundings once across ALL chunks (per-chunk
    # ``forward_one_batch_inner`` returns evidence already spanning
    # every chunk via the merged proof_goals / accumulated_body).
    # ``sync_accumulated`` propagates winning_subs across all depth
    # slots in evidence.body, so the substituted body atoms come
    # for free here (matching keras-ns).
    rule_groundings = None
    if grounder._collect_rule_groundings:
        # "Considered" semantics: every rule application the BFS
        # proposed (including ones whose proof tree doesn't complete
        # within depth) — captured per-step in ``bc.considered``.
        # Matches keras-ns ``ApproximateBackwardChainingGrounder``'s
        # ``rule2groundings`` accumulator. The fp_batch fixpoint then
        # drops apps whose body atoms aren't transitively groundable.
        #
        # FALLBACK: when ``capture_step`` was skipped (compiled trace,
        # see ``bc.step.step``), the accumulator is empty. Build
        # rule_groundings from ``evidence`` instead so callers still
        # get a well-defined output — the cost is the in-proof-only
        # set rather than the full considered set.
        from grounder.bc.considered import finalize as considered_finalize
        rule_groundings = considered_finalize(grounder)
        if rule_groundings is None and evidence is not None:
            from grounder.groundings import evidence_to_rule_groundings
            rule_groundings = evidence_to_rule_groundings(
                evidence, grounder.kb.padding_idx,
                num_rules=grounder.kb.num_rules)
        if rule_groundings is not None and grounder.filter_mode == "fp_batch":
            from grounder.bc.pruning import prune_rule_groundings
            rule_groundings = prune_rule_groundings(
                rule_groundings,
                facts_idx=grounder.kb.fact_index.facts_idx,
                depth=grounder.depth,
                padding_idx=grounder.kb.padding_idx)

    return GrounderOutput(state=state, evidence=evidence,
                          rule_groundings=rule_groundings)


def forward_one_batch_inner(
    grounder, queries: Tensor, query_mask: Tensor, **init_kwargs,
) -> GrounderOutput:
    """Single-batch grounding without finalising r2g.

    Used by :func:`forward_chunked`'s inner loop; the chunked path
    builds RuleGroundings once at the end across all chunks rather
    than per chunk.

    NOTE: does NOT reset the considered accumulator — the chunked
    path needs captures to accumulate across all chunks. The public
    ``forward`` resets it once at the top.
    """
    if grounder.resolution == "closure":
        from grounder.resolution.closure import resolve_closure
        return resolve_closure(grounder, queries, query_mask, **init_kwargs)
    states = init_states(grounder, queries, query_mask, **init_kwargs)
    for d in range(grounder.depth):
        states = step(grounder, states, d)
        if grounder.step_hook is not None:
            cb, cm, cr = grounder.step_hook.on_step(
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"], d)
            states["collected_body"] = cb
            states["collected_mask"] = cm
            states["collected_ridx"] = cr
    evidence = filter_terminal(grounder, states)
    if isinstance(evidence, dict):
        if grounder.collect_evidence:
            evidence = ProofEvidence(
                body=evidence["collected_body"],
                mask=evidence["collected_mask"],
                count=evidence["collected_mask"].sum(dim=1),
                rule_idx=evidence["collected_ridx"],
                body_count=evidence["collected_bcount"],
                D=grounder.depth,
                M=grounder.kb.M,
                head=evidence.get("collected_head"),
            )
        else:
            evidence = None
    if evidence is not None:
        for hook in grounder.hooks:
            body, mask, ridx = hook.apply(
                evidence.body_flat, evidence.mask, evidence.rule_idx_top)
            evidence = ProofEvidence(
                body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx,
                body_count=evidence.body_count)
    state = ProofState(
        proof_goals=states["proof_goals"],
        state_valid=states["state_valid"],
        top_ridx=states["top_ridx"],
        next_var_indices=(
            states["next_var_indices"]
            if grounder._standardize_fn is not None else None),
    )
    return GrounderOutput(state=state, evidence=evidence,
                          rule_groundings=None)


def forward_one_batch(
    grounder, queries: Tensor, query_mask: Tensor, **init_kwargs,
) -> GrounderOutput:
    """Single-batch forward + r2g finalisation.

    NOTE: does NOT reset the considered accumulator — public
    ``forward`` already does that at the top of the call. Skipping
    the per-batch reset keeps semantics consistent between the
    single-batch and chunked entry paths.
    """
    if grounder.resolution == "closure":
        from grounder.resolution.closure import resolve_closure
        return resolve_closure(grounder, queries, query_mask, **init_kwargs)
    states = init_states(grounder, queries, query_mask, **init_kwargs)
    for d in range(grounder.depth):
        states = step(grounder, states, d)
        if grounder.step_hook is not None:
            cb, cm, cr = grounder.step_hook.on_step(
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"], d)
            states["collected_body"] = cb
            states["collected_mask"] = cm
            states["collected_ridx"] = cr
    evidence = filter_terminal(grounder, states)
    if isinstance(evidence, dict):
        if grounder.collect_evidence:
            evidence = ProofEvidence(
                body=evidence["collected_body"],
                mask=evidence["collected_mask"],
                count=evidence["collected_mask"].sum(dim=1),
                rule_idx=evidence["collected_ridx"],
                body_count=evidence["collected_bcount"],
                D=grounder.depth,
                M=grounder.kb.M,
                head=evidence.get("collected_head"),
            )
        else:
            evidence = None
    if evidence is not None:
        for hook in grounder.hooks:
            body, mask, ridx = hook.apply(
                evidence.body_flat, evidence.mask, evidence.rule_idx_top)
            evidence = ProofEvidence(
                body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx,
                body_count=evidence.body_count)
    # Build RuleGroundings from the per-step "considered" accumulator
    # captured by ``bc.considered.capture_step`` during BFS. Matches
    # keras-ns ``rule2groundings`` semantics (every rule application
    # the BFS proposed, regardless of proof-tree completion). The
    # fp_batch fixpoint pruning then drops apps whose body atoms
    # aren't transitively groundable.
    rule_groundings = None
    if grounder._collect_rule_groundings:
        from grounder.bc.considered import finalize as considered_finalize
        rule_groundings = considered_finalize(grounder)
        if rule_groundings is not None and grounder.filter_mode == "fp_batch":
            from grounder.bc.pruning import prune_rule_groundings
            rule_groundings = prune_rule_groundings(
                rule_groundings,
                facts_idx=grounder.kb.fact_index.facts_idx,
                depth=grounder.depth,
                padding_idx=grounder.kb.padding_idx)

    state = ProofState(
        proof_goals=states["proof_goals"],
        state_valid=states["state_valid"],
        top_ridx=states["top_ridx"],
        next_var_indices=(
            states["next_var_indices"]
            if grounder._standardize_fn is not None else None),
    )
    return GrounderOutput(state=state, evidence=evidence,
                          rule_groundings=rule_groundings)


def run_bc(
    grounder, queries: Tensor, query_mask: Tensor,
    *, batch_size: Optional[int] = None,
    pad_outputs: bool = False,
    **init_kwargs,
) -> "RuleGroundings":
    """Rule-evidence entry point — for SBR/DCR/R2N pool-iter consumers.

    Wraps :func:`forward` and post-processes the output:
      * forces ``collect_rule_groundings=True`` for the duration of
        the call so callers don't need to know about that
        ``__init__`` flag;
      * extends ``rule_groundings.atom_table`` to include every
        query atom and populates ``rule_groundings.query_pool_idx``
        so the pool-iter loop has a well-defined readout slot per
        query (even when no firing produced that atom as a head);
      * (opt-in via ``pad_outputs=True``) pads ``A_in[r]``,
        ``A_out[r]`` and ``atom_table`` to fixed sizes (per-rule
        ``grounder.G_r`` for firings; ``populate_query_pool_idx``
        output + a small headroom for atom_table) so the downstream
        compiled reasoner sees a single tensor shape across batches.
        Trades a constant per-batch atom-table cost for the ability
        to keep ``torch.compile(mode="reduce-overhead")`` on cells
        whose flat-path output would otherwise oscillate per batch
        (countries_s3 + BC13, family + BC{12,13}).

    Returns the augmented :class:`grounder.types.RuleGroundings`
    directly; callers that also need the per-tree ``ProofEvidence``
    or the search ``ProofState`` should still call :func:`forward`.

    ``init_kwargs`` are forwarded to :func:`forward` (and onward to
    ``_init_resolution`` for any per-call resolution overrides).
    """
    # Lazy import to avoid a cycle: groundings.py imports from types.py.
    from grounder.groundings import populate_query_pool_idx, pad_rule_groundings
    from grounder.types import RuleGroundings as _RG

    prev_collect = grounder._collect_rule_groundings
    grounder._collect_rule_groundings = True
    # Only forward ``batch_size`` when caller explicitly set it.
    # Passing ``batch_size=None`` triggers the SLD+reduce-overhead
    # validation in ``forward`` (which requires an explicit positive
    # batch_size for stable CUDA-graph capture); omitting the arg
    # entirely lets ``forward`` apply its own ``auto_batch_size``
    # heuristic, matching the regular ``__call__`` adapter path.
    fwd_kwargs = dict(init_kwargs)
    if batch_size is not None:
        fwd_kwargs["batch_size"] = batch_size
    try:
        out = forward(grounder, queries, query_mask, **fwd_kwargs)
    finally:
        grounder._collect_rule_groundings = prev_collect

    rg = out.rule_groundings
    if rg is None:
        # Build an empty-firings RuleGroundings whose atom_table
        # still covers the queries — pool-iter consumers can then
        # just gather KGE-init at every slot and produce a "no
        # proof" score (constant per atom) for every query.
        rg = _RG.empty(
            num_rules=int(getattr(grounder.kb, "num_rules", 0) or 0),
            device=queries.device,
        )
    # Idempotent: ``closure_to_rule_groundings_fast`` already populated
    # ``query_pool_idx`` (queries occupy the first B rows of its
    # purpose-built atom_table). Skip the legacy
    # ``populate_query_pool_idx`` pass in that case to avoid a redundant
    # ``torch.unique`` host sync.
    if rg.query_pool_idx is None:
        rg = populate_query_pool_idx(rg, queries, grounder.kb.padding_idx)
    if pad_outputs and not getattr(grounder, "_closure_fast_rg", False):
        # ``closure_fast_rg`` already produces a fully static-shape rg
        # (``body_pool_idx`` is always ``[B*K, M_max]``, ``atom_table``
        # is always ``[B + B*K*M_max + 1, 3]``). Forcing pow2 pad on
        # top of that buys nothing — and is actively harmful, because
        # ``G_pad = next_pow2(max_K_r)`` varies per batch (different
        # rules fire on different slot counts), giving the downstream
        # compiled reasoner a new shape variant per pow2 bucket. In
        # benchmarking this took the MetaQA bs=8 closure SBR step
        # from ~1 ms with 1 unique graph to ~3 ms with ~39 unique
        # graphs (every transition between pow2 buckets recompiles).
        # When ``_closure_fast_rg`` is on, the rg is already
        # compile-friendly, so skip the pad path entirely.
        #
        # Per-rule firings count after dedup can far exceed grounder.G_r
        # (the per-query-per-rule budget): batch aggregation + multi-step
        # state expansion drives the post-dedup K_r into the
        # ``O(B * S * G_r)`` range. Pad to a power-of-two ceiling of the
        # actual observed max K_r so the compiled downstream sees a
        # small fixed set of graph variants (~log2(K_r_max)) instead of
        # one per batch. The atom_table pad uses the same power-of-two
        # rule on its own range so atom-side compile captures are
        # bounded too.
        if rg.rule_offsets.numel() > 1:
            # Per-rule sizes are diff of offsets — one tensor reduce
            # to find the max instead of iterating the dict shim.
            sizes = rg.rule_offsets[1:] - rg.rule_offsets[:-1]
            max_K_r = int(sizes.max().item()) if sizes.numel() else 0
            if max_K_r > 0:
                G_pad = _next_pow2(max(max_K_r, 1))
                cur_n = int(rg.atom_table.size(0))
                atom_cap = _next_pow2(max(cur_n + 1, 16))
                rg = pad_rule_groundings(
                    rg,
                    pad_per_rule_to=G_pad,
                    pad_atom_table_to=atom_cap,
                    pad_idx_for_atoms=0,
                )
    return rg


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n. Returns 1 for n <= 1."""
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


__all__ = [
    "forward", "run_bc", "auto_batch_size",
    "forward_chunked", "clone_grounder_output", "merge_chunk_outputs",
    "forward_one_batch_inner", "forward_one_batch",
]
