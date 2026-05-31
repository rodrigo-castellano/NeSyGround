"""Terminal-state soundness filter for ``BCGrounder``.

After the per-depth proof loop finishes, the collected groundings need
a soundness check (the per-step accumulator records candidates without
proving body atoms). :func:`filter_terminal` dispatches to one of:

* ``filter_mode='none'`` — no check; returns the raw states dict.
* ``filter_mode='fp_batch'`` — Kleene fixed-point pruning over the
  collected proof tree (``filters/soundness/fp_batch.apply_fp_batch``).
* ``filter_mode='fp_global'`` — membership check against the
  precomputed global closure set
  (``filters/soundness/fp_global.apply_fp_global``), then witness
  injection for empty-body fact-match slots so SBR/DCR have body atoms
  to score.

Returns a :class:`ProofEvidence` (or the raw states dict for
``'none'``).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from grounder.types import ProofEvidence


def filter_terminal(grounder, states: Dict[str, Tensor]):
    """Apply soundness filter on collected groundings -> ProofEvidence.

    When ``filter='none'``, returns the raw states dict (no collection).
    """
    if grounder.filter_mode == "none":
        return states

    B = states["collected_body"].size(0)
    C = grounder.C
    dev = states["collected_body"].device

    body = states["collected_body"]     # [B, C, D, M, 3]
    mask = states["collected_mask"]     # [B, C]
    ridx = states["collected_ridx"]     # [B, C, D]

    if grounder.kb.num_rules == 0:
        D = body.shape[2]
        M = body.shape[3]
        G_body = D * M
        return _empty_result(grounder, B, C, G_body, dev)

    head = states.get("collected_head")  # [B, C, D, 3] or None

    # Inner body dim is ``D * M``; compute it explicitly rather than via
    # ``-1`` so the reshape is well-defined even when ``B == 0`` (the
    # all-cache-hit case under the per-query tabling cache, where the
    # depth loop runs on an empty miss subset).
    G_body = body.shape[2] * body.shape[3]

    if grounder.filter_mode == "fp_batch":
        from grounder.filters.soundness.fp_batch import apply_fp_batch
        body_flat = body.reshape(B, C, G_body, 3)
        # Use per-grounding heads if available (grounded collection mode)
        grounding_heads = None
        if head is not None:
            grounding_heads = head  # [B, C, D, 3]
        mask = apply_fp_batch(
            body_flat, mask, states["queries"], grounder.kb.fact_index,
            grounder.kb.fact_index.pack_base, grounder.kb.padding_idx,
            grounder.depth, grounding_heads=grounding_heads)

    elif grounder.filter_mode == "fp_global":
        from grounder.filters.soundness.fp_global import apply_fp_global
        body_flat = body.reshape(B, C, G_body, 3)
        # NOTE: fp_global_hashes is built by run_forward_chaining with
        # E = num_entities (= constant_no + 1), so body atoms must be
        # hashed with that same base — NOT fact_index.pack_base, which
        # is max(constant_no, padding_idx) + 2 and generally differs.
        # Using pack_base here silently drops every derived atom from
        # the fp_global set, so 2+-hop provable queries emit no grounding.
        mask = apply_fp_global(
            body_flat, mask, grounder.kb.fact_index,
            grounder._E_fp_global, grounder.kb.padding_idx,
            grounder.fp_global_hashes)

    bcount = states["collected_bcount"]   # [B, C, D]

    # Witness injection for fact-match groundings.
    #
    # When the query atom IS itself in the augmented KB (i.e. it's
    # a closure member), the pack_states initial-depth fact-match
    # path emits a grounding with an empty body. That's a valid
    # Boolean proof but collapses to reasoning_score = 1.0 under
    # fuzzy-AND identity, which kills ranking among multi-valid
    # closure members at the SBR head.
    #
    # Fix: look up each query atom's precomputed witness body
    # (see witnesses.build_witness_table) and splice it into the
    # empty fact-match slot. The caller's reasoning head then
    # computes min(KGE(witness body atoms)) just as for any rule
    # match, restoring principled per-atom scoring.
    if (grounder.filter_mode == "fp_global"
            and getattr(grounder, "_has_fp_global_witnesses", False)):
        from grounder.fc.witnesses import inject_witnesses_into_evidence
        body, ridx, bcount = inject_witnesses_into_evidence(
            grounder, body, mask, ridx, bcount, states["queries"])

    count = mask.sum(dim=1)
    D_val = grounder.depth if grounder.collect_evidence else 0
    M_val = grounder.kb.M if grounder.collect_evidence else 0
    return ProofEvidence(
        body=body, mask=mask,
        count=count, rule_idx=ridx,
        body_count=bcount,
        D=D_val, M=M_val,
        head=head,
    )


def _empty_result(
    grounder, B: int, C: int, G_body: int, dev: torch.device,
) -> ProofEvidence:
    """Empty proof evidence for KBs with no rules."""
    D = grounder.depth if grounder.collect_evidence else 1
    M = grounder.kb.M if grounder.collect_evidence else 1
    return ProofEvidence(
        body=torch.zeros(
            B, C, D, M, 3, dtype=torch.long, device=dev),
        mask=torch.zeros(
            B, C, dtype=torch.bool, device=dev),
        count=torch.zeros(
            B, dtype=torch.long, device=dev),
        rule_idx=torch.full(
            (B, C, D), -1, dtype=torch.long, device=dev),
        body_count=torch.zeros(
            B, C, D, dtype=torch.long, device=dev),
        D=D if grounder.collect_evidence else 0,
        M=M if grounder.collect_evidence else 0,
    )


__all__ = ["filter_terminal"]
