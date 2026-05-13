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

    # proved-set: starts as facts, grows with proved heads. Snapshot
    # iteration: each pass uses a frozen view from the previous pass.
    proved = is_fact.clone()
    for _ in range(max(1, depth)):
        new_proved = proved.clone()
        for r, a_in in rg.A_in.items():
            if a_in.numel() == 0:
                continue
            a_out = rg.A_out[r]
            head_idx = a_out[:, 0]
            if a_in.size(1) == 0:
                body_proved = torch.ones(
                    a_in.size(0), dtype=torch.bool, device=device)
            else:
                body_proved = proved[a_in].all(dim=-1)
            new_proved[head_idx[body_proved]] = True
        if torch.equal(new_proved, proved):
            break
        proved = new_proved

    # Final filter: keep apps whose body atoms are all in proved.
    new_A_in: Dict[int, Tensor] = {}
    new_A_out: Dict[int, Tensor] = {}
    for r, a_in in rg.A_in.items():
        if a_in.numel() == 0:
            new_A_in[r] = a_in
            new_A_out[r] = rg.A_out[r]
            continue
        if a_in.size(1) == 0:
            keep = torch.ones(
                a_in.size(0), dtype=torch.bool, device=device)
        else:
            keep = proved[a_in].all(dim=-1)
        new_A_in[r] = a_in[keep].contiguous()
        new_A_out[r] = rg.A_out[r][keep].contiguous()

    return RuleGroundings(
        atom_table=rg.atom_table,
        A_in=new_A_in,
        A_out=new_A_out,
        num_atoms=rg.num_atoms,
        num_rules=rg.num_rules,
    )


__all__ = ["prune_rule_groundings"]
