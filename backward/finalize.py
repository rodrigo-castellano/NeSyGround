"""Finalization — assemble CompletedTreeFirings / GoalState / RuleGroundings.

Produces the output types:
  * ``finalize_evidence``        — terminal fp_batch filter → ``CompletedTreeFirings``.
  * ``build_proof_state``        — final goals → ``GoalState``.
  * ``finalize_rule_groundings`` — firing accumulator → fp_batch-pruned
                                   ``RuleGroundings``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from grounder.vocab.shapes import Shapes
from grounder.core import BackwardResult
from grounder.base.types import CompletedTreeFirings, GoalState, RuleGroundings


def _make_shapes(plan, B: int, Y_q: int) -> Shapes:
    """Minimal valid Shapes for the completed-tree-firings view (consumers preserve D/M).
    Shapes.G=plan.S (states/goals per step), Shapes.L=plan.max_goals (atoms per goal)."""
    kb, sh = plan.kb, plan.shapes
    return Shapes(
        B=B, G=plan.S, L=plan.max_goals, M=kb.M, D=plan.depth,
        A=plan.depth * kb.M, Y_q=Y_q,
        K_f=kb.K_f, K_r=sh.K_r, Y_r=sh.Y_r, K_v=sh.K_v, V=sh.V,
        E=kb.constant_no + 1, N=B * plan.S, pad=kb.padding_idx,
    )


def finalize_evidence(plan, trees) -> Optional[CompletedTreeFirings]:
    """Build ``CompletedTreeFirings`` from run-scoped ``ProofTrees`` (chunk pieces).

    Concat the per-chunk pieces on B, then ``fp_batch`` applies ONE terminal
    Kleene filter (never per-chunk). The pre-filter validity mask rides in the
    ProofTrees ``count`` slot. ``filter='none'`` keeps the raw mask."""
    if trees is None or not trees.body:
        return None

    body = torch.cat(trees.body, 0)          # [B, Y_q, D, M, 3]
    rule_idx = torch.cat(trees.rule_idx, 0)      # [B, Y_q, D]
    head = torch.cat(trees.head, 0)          # [B, Y_q, D, 3]
    body_count = torch.cat(trees.body_count, 0)  # [B, Y_q, D]
    mask = torch.cat(trees.count, 0)         # [B, Y_q]  (pre-filter validity)

    B = body.size(0)
    Y_q = plan.Y_q

    G_body = body.shape[2] * body.shape[3]

    if plan.filter_mode == "fp_batch":
        from grounder.filters.fp_batch import apply_fp_batch
        body_flat = body.reshape(B, Y_q, G_body, 3)
        mask = apply_fp_batch(
            body_flat, mask, plan.kb.fact_index,
            plan.kb.fact_index.pack_base, plan.kb.padding_idx,
            plan.depth, grounding_heads=head)

    count = mask.sum(dim=1)
    shapes = _make_shapes(plan, B, Y_q)
    return CompletedTreeFirings(
        body=body, grounding_valid=mask, count=count, rule_idx=rule_idx,
        body_count=body_count, head=head, shapes=shapes,
    )


def build_proof_state(plan, fr) -> GoalState:
    """Assemble the output ``GoalState`` from the final ``Frontier``.

    When a standardization fn is configured (and not collecting evidence), the
    output goal_atoms' variables are renamed into the runtime range relative to
    the passed-in next_var/goals so downstream consumers (DpRL env embeddings) see
    bounded var ids."""
    goal_atoms = fr.goal_atoms
    next_var = fr.next_var if plan.standardize else None
    if plan.standardize_fn is not None and not plan.collect_evidence:
        counts = fr.goal_valid.long().sum(dim=1)
        nv = fr.initial_next_var if fr.initial_next_var is not None else fr.next_var
        inp = (fr.initial_goals if fr.initial_goals is not None
               else goal_atoms.new_zeros(goal_atoms.shape[0], 0, 3))
        std, std_nv = plan.standardize_fn(goal_atoms, counts, nv, inp)
        # Clone to detach from any compiled/CUDA-graph output buffers.
        goal_atoms = std.clone()
        next_var = std_nv.clone()
    return GoalState(
        goal_atoms=goal_atoms,
        goal_valid=fr.goal_valid,
        top_rule_idx=fr.top_rule_idx,
        next_var=next_var,
    )


def finalize_rule_groundings(plan, firings) -> Optional[RuleGroundings]:
    """Build the firing-set RuleGroundings and apply fp_batch pruning.

    ``firings`` is the run-scoped ``FiringSet`` (global concat; query_idx IGNORED)."""
    if not plan.collect_rule_groundings:
        return None
    from grounder.backward.considered import finalize as considered_finalize
    rule_groundings = considered_finalize(plan, firings)
    if rule_groundings is not None and plan.filter_mode == "fp_batch":
        from grounder.filters.fp_batch import prune_rule_groundings
        rule_groundings = prune_rule_groundings(
            rule_groundings,
            facts_idx=plan.kb.fact_index.facts_idx,
            depth=plan.depth,
            padding_idx=plan.kb.padding_idx)
    return rule_groundings


def _merge_proof_states(parts: Sequence[GoalState]) -> GoalState:
    """Concat per-chunk GoalStates on B (dim 0). Single chunk = identity; for
    >1 chunk the dynamic G (flat path) is right-padded to the max across parts."""
    if len(parts) == 1:
        return parts[0]
    S = max(p.goal_atoms.shape[1] for p in parts)
    G = max(p.goal_atoms.shape[2] for p in parts)

    def _pad_s(t):  # right-pad dim-1 (G) to S; padded goals stay inactive
        ps = S - t.shape[1]
        return torch.nn.functional.pad(t, (0, ps)) if ps else t

    goal_atoms = torch.cat(
        [torch.nn.functional.pad(p.goal_atoms, (0, 0, 0, G - p.goal_atoms.shape[2],
                                                0, S - p.goal_atoms.shape[1]))
         for p in parts], dim=0)
    goal_valid = torch.cat([_pad_s(p.goal_valid) for p in parts], dim=0)
    top_rule_idx = torch.cat([_pad_s(p.top_rule_idx) for p in parts], dim=0)
    nvs = [p.next_var for p in parts]
    next_var = torch.cat(nvs, dim=0) if all(n is not None for n in nvs) else None
    return GoalState(goal_atoms=goal_atoms, goal_valid=goal_valid,
                     top_rule_idx=top_rule_idx, next_var=next_var)


def merge_finalize(plan) -> Callable[[Sequence[GoalState], object], BackwardResult]:
    """Return the ``finalize_fn`` handed to ``ExecStrategy.merge``: stitch the
    per-chunk goal-state ``parts`` + threaded ``RunState`` into a BackwardResult.
    State concats on B; completed-tree-firings/rule_groundings come from the
    run-scoped, already globally-concatenated trees/firings (ONE terminal fp_batch,
    never per-chunk)."""
    def _fn(parts: Sequence[GoalState], run) -> BackwardResult:
        goal_state = _merge_proof_states(parts)
        completed_tree_firings = finalize_evidence(plan, run.trees)
        rule_groundings = finalize_rule_groundings(plan, run.firings)
        return BackwardResult(goal_state=goal_state,
                              completed_tree_firings=completed_tree_firings,
                              rule_groundings=rule_groundings)
    return _fn


__all__ = ["finalize_evidence", "build_proof_state", "finalize_rule_groundings",
           "merge_finalize"]
