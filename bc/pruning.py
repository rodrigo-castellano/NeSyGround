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
        from grounder.groundings import atom_hash
        fi_dev = facts_idx.to(device)
        atom_h = atom_hash(atom_table)                      # [num_atoms]
        fact_h = atom_hash(fi_dev)                          # [F]
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

    # Fully-tensorized fixpoint + final filter. Key insight: every
    # ``rg.A_in[r]`` already shares the same M dim (the global max
    # body length set by ``considered.finalize`` — shorter rules have
    # padding atoms in unused positions, and ``is_pad`` above marks
    # those as proved). So we can cat all rules into one [total_K, M]
    # via a single torch.cat and run the fixpoint without per-rule
    # padding logic. After the fixpoint, the per-rule split uses an
    # argsort + bincount + view-slicing pattern that fires O(1)
    # kernels in the rule count.
    rule_keys = sorted(rg.A_in.keys())
    a_in_chunks = [rg.A_in[r] for r in rule_keys]
    a_out_chunks = [rg.A_out[r][:, 0] for r in rule_keys]
    K_per_rule = torch.tensor(
        [t.size(0) for t in a_in_chunks], dtype=torch.long, device=device)

    if not a_in_chunks or all(t.size(0) == 0 for t in a_in_chunks):
        # All rules empty — nothing to filter.
        return RuleGroundings(
            atom_table=rg.atom_table,
            A_in=dict(rg.A_in), A_out=dict(rg.A_out),
            num_atoms=rg.num_atoms, num_rules=rg.num_rules,
        )

    A_in_all = torch.cat(a_in_chunks, dim=0)                    # [total_K, M]
    A_out_all = torch.cat(a_out_chunks, dim=0)                  # [total_K]

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

    keep_all = (proved[A_in_all] == 1).all(dim=-1)              # [total_K]

    # Per-rule split via cumsum of K_per_rule — view-slicing only,
    # 0 CUDA kernels regardless of rule count. The slices are views
    # into ``A_in_all`` / ``A_out_all`` (already filtered by keep_all
    # via a single gather above).
    A_in_kept = A_in_all[keep_all]                              # 1 kernel
    A_out_kept = A_out_all[keep_all]                            # 1 kernel
    # Recompute per-rule sizes after filtering.
    # rule_idx_per_row: which rule each surviving row came from.
    # Built by gathering from a flat rule_idx vector via the same
    # keep_all mask. One torch.repeat_interleave + one gather.
    rule_idx_all = torch.repeat_interleave(
        torch.arange(len(rule_keys), device=device, dtype=torch.long),
        K_per_rule)                                              # [total_K]
    rule_idx_kept = rule_idx_all[keep_all]                       # [K_kept]
    sizes_kept = torch.bincount(
        rule_idx_kept, minlength=len(rule_keys))                 # [num_rule_keys]
    sizes_cpu = sizes_kept.tolist()                              # 1 sync

    new_A_in: Dict[int, Tensor] = {}
    new_A_out: Dict[int, Tensor] = {}
    offset = 0
    for i, r in enumerate(rule_keys):
        K_r_kept = sizes_cpu[i]
        end = offset + K_r_kept
        new_A_in[r] = A_in_kept[offset:end]                      # view, 0 kernels
        new_A_out[r] = A_out_kept[offset:end].unsqueeze(-1)       # view, 0 kernels
        offset = end

    return RuleGroundings(
        atom_table=rg.atom_table,
        A_in=new_A_in,
        A_out=new_A_out,
        num_atoms=rg.num_atoms,
        num_rules=rg.num_rules,
    )


__all__ = ["prune_rule_groundings"]
