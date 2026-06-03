"""fp_batch / Kleene fixed-point pruning over RuleGroundings.

Used by ``BCGrounder`` when ``filter_mode='fp_batch'`` (the
keras-ns ``prune_incomplete_proofs=True`` equivalent). Operates on a
``RuleGroundings`` produced by ``evidence_to_rule_groundings``: drops
rule applications whose body atoms aren't all in the proved-set
(facts ∪ heads of proved apps) after ``depth`` snapshot iterations.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def prune_rule_groundings(rg, *, facts_idx: Tensor, depth: int,
                          padding_idx: int = None):
    """Snapshot-based ``num_steps``-iteration pruning.

    Mirrors ``common.prune_rule_groundings`` semantics in tensor form.

    Args:
        rg: input ``RuleGroundings``.
        facts_idx: ``[F, 3]`` ground KB facts.
        depth: number of snapshot iterations.

    Returns:
        New ``RuleGroundings`` with ``A_in`` / ``A_out`` filtered.
    """
    from grounder.types import RuleGroundings

    atom_table = rg.atom_table                              # [num_atoms, 3]
    num_atoms = atom_table.size(0)
    device = atom_table.device

    # Build fact-set membership: which atom_table rows are facts.
    # ``atom_hash`` projects each ``(p, a0, a1)`` row to int64, then
    # ``torch.isin`` does the set-membership check in O((N+F) log F)
    # on GPU — replaces the old ``unique_dim`` + ``.item()``-sized
    # fact_mask buffer, which was slow per-row-sort plus a host sync.
    if facts_idx.numel() > 0:
        import os as _os
        fi_dev = facts_idx.to(device)
        if _os.environ.get("GROUNDER_EXACT_FACT_MEMBERSHIP") == "1":
            # COLLISION-FREE fact membership (eval-only, env-gated default OFF).
            # The default ``atom_hash`` packs (p,a0,a1) with three NEAR-EQUAL
            # primes; on large eval atom_tables (exhaustive candidate pools)
            # this yields FALSE-POSITIVE "facts", seeding spurious proofs that
            # boost distractor scores and suppress exhaustive MRR. Pack
            # injectively into int64 with a base > every component so
            # membership is exact. Default (hash) path kept for byte-identity.
            base = int(torch.maximum(atom_table.max(), fi_dev.max()).item()) + 1
            bb = base * base
            atom_h = (atom_table[:, 0].long() * bb
                      + atom_table[:, 1].long() * base + atom_table[:, 2].long())
            fact_h = (fi_dev[:, 0].long() * bb
                      + fi_dev[:, 1].long() * base + fi_dev[:, 2].long())
        else:
            from grounder.groundings import atom_hash
            atom_h = atom_hash(atom_table)                  # [num_atoms]
            fact_h = atom_hash(fi_dev)                       # [F]
        is_fact = torch.isin(atom_h, fact_h)                # [num_atoms]
    else:
        is_fact = torch.zeros(num_atoms, dtype=torch.bool, device=device)

    # Padding atoms in shorter-rule body slots count as "always proved"
    # — a body of [real_atom, pad] for an M=2 rule (when global M=3) is
    # complete iff the real atom is proved; the pad shouldn't block
    # the firing. Mark every atom_table row whose predicate equals
    # ``padding_idx`` as proved. Without this, every rule with body
    # shorter than the global ``kb.M`` is silently filtered out.
    if padding_idx is not None and num_atoms > 0:
        is_pad = atom_table[:, 0] == padding_idx
        is_fact = is_fact | is_pad

    # Fully-tensorized fixpoint + final filter over the flat firings.
    # ``rg.body_pool_idx`` is already ``[N, M_max]``; we apply the
    # fixpoint directly on it and use ``rg.body_atom_valid`` to mask
    # padded slots as proved (they're padding sentinels in atom_table
    # via ``is_pad`` above).
    A_in_all = rg.body_pool_idx                                  # [N, M_max]
    A_out_all = rg.head_pool_idx                                 # [N]
    body_atom_valid = rg.body_atom_valid                         # [N, M_max]
    rule_idx_all = rg.rule_idx                                   # [N]

    if A_in_all.size(0) == 0:
        # All rules empty — nothing to filter; return rg unchanged.
        return rg

    # Fixpoint over proved-set, fully GPU-resident:
    #   - ``proved[A_in_all].all(-1)``: gather + reduce, fixed shapes.
    #   - scatter_reduce(reduce="amax", include_self=True) into a
    #     fresh ``[num_atoms]`` byte tensor at the head indices, ORed
    #     into ``proved``. Replaces the previous form
    #     ``new_proved[A_out_all[body_proved]] = True`` which had two
    #     data-dependent ops (boolean-mask gather producing variable
    #     output + fancy-index scatter), each launching slower kernels
    #     and forcing a CUDA stream sync.
    #   - No ``torch.equal`` (host sync). Run ``depth`` iterations
    #     unconditionally; the fixpoint converges in <= depth steps
    #     by construction (each new fact adds at most one indirection).
    # scatter_reduce(amax) doesn't support bool on CUDA — use int8.
    proved = is_fact.to(torch.int8)
    for _ in range(max(1, depth)):
        body_proved = (proved[A_in_all] == 1).all(dim=-1).to(torch.int8)
        new_heads = torch.zeros_like(proved)
        new_heads.scatter_reduce_(
            0, A_out_all, body_proved, reduce="amax", include_self=False,
        )
        proved = torch.maximum(proved, new_heads)

    keep_all = (proved[A_in_all] == 1).all(dim=-1)              # [N]

    # Filter firings to the kept set. ``keep_all`` is data-dependent,
    # so this is the one sync-inducing op (masked_select). The flat
    # layout means we get the per-rule split for free via
    # bincount + cumsum, no per-rule Python loop.
    body_pool_idx_kept = A_in_all[keep_all]
    head_pool_idx_kept = A_out_all[keep_all]
    body_atom_valid_kept = body_atom_valid[keep_all]
    rule_idx_kept = rule_idx_all[keep_all].long()

    sizes = torch.bincount(rule_idx_kept, minlength=rg.num_rules)
    rule_offsets = torch.zeros(
        rg.num_rules + 1, dtype=torch.long, device=device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)
    firing_valid_kept = torch.ones(
        rule_idx_kept.size(0), dtype=torch.bool, device=device)

    return RuleGroundings(
        atom_table=rg.atom_table,
        body_pool_idx=body_pool_idx_kept,
        body_atom_valid=body_atom_valid_kept,
        head_pool_idx=head_pool_idx_kept,
        rule_idx=rule_idx_kept,
        rule_offsets=rule_offsets,
        firing_valid=firing_valid_kept,
        num_atoms=rg.num_atoms,
        num_rules=rg.num_rules,
        M_max=rg.M_max,
        query_pool_idx=rg.query_pool_idx,
    )


__all__ = ["prune_rule_groundings"]
