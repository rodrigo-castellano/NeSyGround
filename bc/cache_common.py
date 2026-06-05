"""Shared guardrails for the BC grounding caches — the per-query tabling
cache (:mod:`grounder.bc.tabling`) and the per-subgoal memo
(:mod:`grounder.bc.subgoal`).

Both caches are OFF by default. When enabled, these preconditions hard-fail
(assert) rather than silently degrading: the caches are only valid on the
eager enum FLAT considered path with a stable KB.
"""
from __future__ import annotations

import torch


def verify_cache_preconditions(grounder, cache_name: str) -> bool:
    """Assert the mandatory guardrails for a grounding cache, returning True
    when all hold. ``cache_name`` is interpolated into the messages.

    The cache only applies to the eager enum FLAT considered path (never the
    compiled / dense path, which derives rule_groundings from evidence rather
    than the considered accumulator), requires ``collect_rule_groundings``,
    and is only valid while the KB (fact index + rule count) is unchanged.
    """
    assert grounder.resolution == "enum", (
        f"{cache_name} requires resolution=='enum', "
        f"got {grounder.resolution!r}")
    assert getattr(grounder, "_flat_intermediate", False) \
        or getattr(grounder, "_flat_intermediate_flag", False), (
        f"{cache_name} requires the enum FLAT path (flat_intermediate=True)")
    assert not torch.compiler.is_compiling(), (
        f"{cache_name} must not run inside torch.compile tracing")
    assert not getattr(grounder, "_compiled", False), (
        f"{cache_name} is incompatible with the compiled/dense step path "
        "(it uses the considered accumulator, not evidence-derived "
        "rule_groundings)")
    assert grounder._collect_rule_groundings, (
        f"{cache_name} requires collect_rule_groundings=True (it caches the "
        "considered accumulator)")
    # KB-immutability: the cache is only valid while facts + rules are
    # unchanged for the grounder's lifetime.
    stamp = (id(grounder.kb.fact_index), int(grounder.kb.num_rules))
    assert stamp == grounder._tabling_kb_stamp, (
        f"{cache_name} KB-immutability stamp changed "
        f"(was {grounder._tabling_kb_stamp}, now {stamp}); the fact index or "
        "rule count was mutated, invalidating every cached entry")
    return True
