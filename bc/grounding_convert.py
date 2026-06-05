"""BC RuleGroundings conversion: python-dict ``prune_rule_groundings`` and
the tensor ``build_rule_grounding_tensors`` builder.

Split out of ``bc/common.py``. ``RuleGroundings`` is imported lazily inside
``build_rule_grounding_tensors`` (as in the original) to avoid a circular
import with ``grounder.types``.
"""

from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor


def prune_rule_groundings(
    rule2groundings: Dict[int, Set[Tuple]],
    fact_set: Set[Tuple[int, int, int]],
    max_iterations: int = 10,
) -> Dict[int, Set[Tuple]]:
    """Iterative fixed-point pruning of rule groundings (Kleene T_P).

    Mirrors keras-ns ``PruneIncompleteProofs`` exactly: each iteration uses
    a snapshot of the proved set from the previous iteration so each pass
    extends the chain by exactly one step. After ``max_iterations`` passes,
    keep groundings whose body atoms are all either facts or proved heads.

    A previous in-place version converged to a strictly larger proved set
    than keras at the same iteration count (because heads added earlier in
    a pass were immediately usable later in the same pass), causing torch
    to keep groundings keras dropped.

    Args:
        rule2groundings: rule_idx → set of (head, body) tuples
        fact_set: set of (pred, subj, obj) known facts
        max_iterations: number of snapshot passes (= keras ``num_steps``)

    Returns:
        Pruned dict with same structure.
    """
    proved: Set[Tuple[int, int, int]] = set()

    for _ in range(max_iterations):
        # Snapshot from the previous pass — heads added here are NOT
        # visible to other groundings until the next iteration.
        snapshot = proved | fact_set
        new_proved = set(proved)
        for r, groundings in rule2groundings.items():
            for head, body in groundings:
                if head in new_proved:
                    continue
                if all(atom in snapshot for atom in body):
                    new_proved.add(head)
        if new_proved == proved:
            break
        proved = new_proved

    proved_or_fact = proved | fact_set
    pruned: Dict[int, Set[Tuple]] = {}
    for r, groundings in rule2groundings.items():
        kept = set()
        for head, body in groundings:
            if all(atom in proved_or_fact for atom in body):
                kept.add((head, body))
        if kept:
            pruned[r] = kept
    return pruned


def build_rule_grounding_tensors(
    rule2groundings: Dict[int, Set[Tuple]],
    num_rules: int,
    device: torch.device,
) -> "RuleGroundings":
    """Convert Python rule2groundings to (A_in, A_out) tensors.

    Each entry in rule2groundings[r] is (head_tuple, body_tuple) where:
    - head_tuple = (pred, subj, obj)
    - body_tuple = ((p,s,o), (p,s,o), ...)

    Builds a global atom table and per-rule index tensors.

    Returns:
        RuleGroundings with atom_table [num_atoms, 3] and per-rule A_in/A_out.
    """
    from grounder.types import RuleGroundings

    # 1. Collect all unique atoms
    all_atoms: Dict[Tuple[int, int, int], int] = {}

    def get_idx(atom: Tuple[int, int, int]) -> int:
        if atom not in all_atoms:
            all_atoms[atom] = len(all_atoms)
        return all_atoms[atom]

    # Pre-scan to build atom table
    for r, groundings in rule2groundings.items():
        for head, body in groundings:
            get_idx(head)
            for atom in body:
                get_idx(atom)

    num_atoms = len(all_atoms)
    atom_table = torch.zeros(num_atoms, 3, dtype=torch.long, device=device)
    for atom, idx in all_atoms.items():
        atom_table[idx, 0] = atom[0]
        atom_table[idx, 1] = atom[1]
        atom_table[idx, 2] = atom[2]

    # 2. Build flat per-firing tensors. ``M_max`` is the max body
    # length across all firings. Body atoms shorter than ``M_max`` get
    # padded with atom_table slot 0 (the consumer-side sentinel) and
    # marked invalid in ``body_atom_valid``.
    flat_rows: list = []  # (rule_idx, head_idx, [body_idx_padded], [body_valid])
    M_max = 0
    for r in range(num_rules):
        groundings = rule2groundings.get(r, set())
        if not groundings:
            continue
        for head, body in sorted(groundings):
            M_max = max(M_max, len(body))

    if M_max == 0:
        # Every rule empty.
        return RuleGroundings.empty(num_rules=num_rules, device=device)

    body_rows: list = []
    valid_rows: list = []
    head_idxs: list = []
    rule_idxs: list = []
    for r in range(num_rules):
        groundings = rule2groundings.get(r, set())
        if not groundings:
            continue
        for head, body in sorted(groundings):
            row = [all_atoms[a] for a in body] + [0] * (M_max - len(body))
            vrow = [True] * len(body) + [False] * (M_max - len(body))
            body_rows.append(row)
            valid_rows.append(vrow)
            head_idxs.append(all_atoms[head])
            rule_idxs.append(r)

    if not body_rows:
        return RuleGroundings.empty(num_rules=num_rules, device=device)

    body_pool_idx = torch.tensor(body_rows, dtype=torch.long, device=device)
    body_atom_valid = torch.tensor(valid_rows, dtype=torch.bool, device=device)
    head_pool_idx = torch.tensor(head_idxs, dtype=torch.long, device=device)
    rule_idx = torch.tensor(rule_idxs, dtype=torch.long, device=device)
    sizes = torch.bincount(rule_idx, minlength=num_rules)
    rule_offsets = torch.zeros(
        num_rules + 1, dtype=torch.long, device=device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)
    firing_valid = torch.ones(
        rule_idx.size(0), dtype=torch.bool, device=device)

    return RuleGroundings(
        atom_table=atom_table,
        body_pool_idx=body_pool_idx,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_pool_idx,
        rule_idx=rule_idx,
        rule_offsets=rule_offsets,
        firing_valid=firing_valid,
        num_atoms=num_atoms,
        num_rules=num_rules,
        M_max=M_max,
    )
