"""BC finalization helpers — assemble ProofEvidence / ProofState /
RuleGroundings from a completed depth-loop ``states`` dict.

Shared by bc/forward.py's single-batch, chunk-inner, and chunk-merge
paths. The inner (per-chunk) path deliberately returns
``rule_groundings=None`` and finalises once at merge time (the
"no-finalize contract"); ``finalize_rule_groundings`` therefore only runs
on the single-batch and merge paths.
"""
from __future__ import annotations

from typing import Optional

from grounder.bc.terminal import filter_terminal
from grounder.types import ProofEvidence, ProofState, RuleGroundings


def finalize_evidence(grounder, states) -> Optional[ProofEvidence]:
    """``filter_terminal`` → ``ProofEvidence`` (when ``collect_evidence``),
    then apply the registered evidence hooks. Byte-identical to the inline
    block previously duplicated in forward_one_batch / *_inner.
    """
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
    return evidence


def build_proof_state(grounder, states) -> ProofState:
    """Assemble the output ``ProofState`` from the final ``states`` dict."""
    return ProofState(
        proof_goals=states["proof_goals"],
        state_valid=states["state_valid"],
        top_ridx=states["top_ridx"],
        next_var_indices=(
            states["next_var_indices"]
            if grounder._standardize_fn is not None else None),
    )


def finalize_rule_groundings(
    grounder, evidence: Optional[ProofEvidence] = None, *,
    evidence_fallback: bool = False,
) -> Optional[RuleGroundings]:
    """Build the "considered" RuleGroundings and apply fp_batch pruning.

    ``evidence_fallback`` (chunk-merge path only): when the considered
    accumulator is empty — e.g. a compiled trace skipped ``capture_step`` —
    rebuild from ``evidence`` so callers still get a well-defined output.
    The single-batch path passes ``False`` (no fallback), reproducing its
    original behaviour exactly.
    """
    if not grounder._collect_rule_groundings:
        return None
    from grounder.bc.considered import finalize as considered_finalize
    rule_groundings = considered_finalize(grounder)
    if (rule_groundings is None and evidence_fallback
            and evidence is not None):
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
    return rule_groundings
