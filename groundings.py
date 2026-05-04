"""Conversions between grounder output forms.

A grounder fundamentally produces ``ProofEvidence`` — per-query proof
trees with fully-ground head and body atoms. From that single source
two scalar metrics summarise the result, both of which are derivable
*from* evidence (not from independent accumulators):

  * **Rule groundings** — the number of distinct ``(rule, head,
    sorted_body)`` tuples appearing across all proof trees. This is
    the keras-comparable ``rule2groundings`` metric. Compute via
    :func:`evidence_unique_app_count` (count) or
    :func:`evidence_to_rule_groundings` (full ``RuleGroundings``
    dataclass for downstream consumers like SBR / R2N / DCR).

  * **Proof groundings** — the number of distinct proof trees rooted
    at the queries. This is :attr:`ProofEvidence.count` aggregated
    across queries (already produced by the grounder), and can also
    be **recomputed** from rule_groundings + facts + queries via
    :func:`count_proof_trees` (AND-OR graph fixpoint), useful for
    cross-checking and for grounders that don't populate evidence
    counts directly.

The two transforms are duals:

  P → R is straightforward: walk every proof tree, collect each
  rule application, deduplicate.

  R → P needs facts + queries: build an AND-OR graph where every
  ``(rule, head, body)`` is a hyperedge ``head ← AND(body)``. Each
  atom can have multiple incoming hyperedges (multiple ways to
  derive it). The proof count of an atom is the sum over its
  hyperedges of the product of body-atom proof counts, with
  ``proofs(fact) = 1``. Iterated to fixpoint.

For convenience, ``ProofEvidence`` exposes
:meth:`ProofEvidence.unique_app_count` and
:meth:`ProofEvidence.to_rule_groundings` as bound methods (see
:mod:`grounder.types`).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from grounder.types import ProofEvidence, RuleGroundings


__all__ = [
    "atom_hash",
    "evidence_unique_app_count",
    "evidence_unique_app_keys",
    "evidence_to_rule_groundings",
    "count_proof_trees",
]


# Three large coprime primes chosen to scatter ``(p, a0, a1)`` far
# apart so all three components contribute to the hash without
# overflowing 64-bit signed. Collisions are vanishingly rare for the
# entity / predicate ranges we work with (≤ 10^5 each).
_HASH_P0 = 1_000_003
_HASH_P1 = 999_983
_HASH_P2 = 999_979


def atom_hash(atoms: Tensor) -> Tensor:
    """Hash atom triples ``(p, a0, a1)`` to int64 ids.

    ``atoms`` may be any shape ending in ``(..., 3)``. Output drops
    the last dim and is suitable as a dictionary key for proof
    counting and unique-set deduplication.
    """
    a = atoms.long()
    return (a[..., 0] * _HASH_P0
            + a[..., 1] * _HASH_P1
            + a[..., 2] * _HASH_P2)


def _flatten_evidence(
    evidence: ProofEvidence,
    padding_idx: int,
):
    """Pull active (rule_idx, head, body) rows out of a ProofEvidence.

    Handles both the structured layout (rule_idx is ``[B, C, D]``,
    body is ``[B, C, D, M, 3]``) and the legacy flat layout
    (``[B, C]`` and ``[B, C, G_body, 3]``). Returns ``(ridx, head,
    body, M)`` long tensors restricted to active rows, or ``(None,
    None, None, M)`` if there are no active rows.
    """
    if (evidence is None
            or evidence.rule_idx is None
            or evidence.body is None):
        return None, None, None, 0

    if evidence.rule_idx.dim() == 3:
        # Structured layout — one row per (B, C, D).
        M = evidence.M if evidence.M else evidence.body.shape[-2]
        mask = evidence.mask.unsqueeze(-1) & (evidence.rule_idx >= 0)
        flat_mask = mask.reshape(-1)
        if not bool(flat_mask.any()):
            return None, None, None, M
        flat_ridx = evidence.rule_idx.reshape(-1)
        flat_body = evidence.body.reshape(-1, M, 3)
        if evidence.head is not None:
            flat_head = evidence.head.reshape(-1, 3)
        else:
            flat_head = torch.full(
                (flat_ridx.size(0), 3), padding_idx,
                dtype=torch.long, device=flat_ridx.device)
    else:
        # Legacy flat layout — one row per (B, C); body is [B, C, G, 3].
        # Take the first M atoms as the rule body slice.
        M = evidence.M if evidence.M else evidence.body.shape[-2]
        flat_mask = evidence.mask.reshape(-1)
        flat_mask = flat_mask & (evidence.rule_idx.reshape(-1) >= 0)
        if not bool(flat_mask.any()):
            return None, None, None, M
        flat_ridx = evidence.rule_idx.reshape(-1)
        body_full = evidence.body.reshape(-1, evidence.body.shape[-2], 3)
        flat_body = body_full[:, :M, :]
        if evidence.head is not None:
            flat_head = evidence.head.reshape(-1, 3)
        else:
            flat_head = torch.full(
                (flat_ridx.size(0), 3), padding_idx,
                dtype=torch.long, device=flat_ridx.device)

    keep = flat_mask
    return (flat_ridx[keep].long(),
            flat_head[keep].long(),
            flat_body[keep].long(),
            M)


def evidence_unique_app_keys(
    evidence: ProofEvidence,
    padding_idx: int,
) -> Tensor:
    """Unique ``(rule_idx, head, sorted_body)`` rows from evidence.

    Returns a ``[N_unique, 1 + 3 + M*3]`` long tensor:

      * column 0:        ``rule_idx``
      * columns 1..3:    head atom (pred, arg0, arg1)
      * columns 4..3+M*3: M body atoms, each as (pred, arg0, arg1),
                          sorted within the row by atom-hash so
                          different anchor variants of the same
                          logical app share a key.

    Inactive entries (``mask == False`` or ``rule_idx < 0``) are
    dropped before deduplication.
    """
    ridx, head, body, M = _flatten_evidence(evidence, padding_idx)
    if ridx is None:
        device = (evidence.body.device if evidence is not None
                  and evidence.body is not None else torch.device("cpu"))
        return torch.zeros(0, 1 + 3 + M * 3, dtype=torch.long, device=device)

    # Sort body atoms within each entry by hash so different anchor
    # variants of the same logical app share a canonical key.
    body_h = atom_hash(body)               # [N, M]
    active = body[..., 0] != padding_idx
    sentinel = torch.full_like(body_h, (2 ** 62) - 1)
    sort_keys = torch.where(active, body_h, sentinel)
    sort_idx = sort_keys.argsort(dim=-1)
    body_sorted = body.gather(
        1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))

    keys = torch.cat([
        ridx.unsqueeze(-1),
        head,
        body_sorted.reshape(-1, M * 3),
    ], dim=-1)
    return torch.unique(keys, dim=0)


def evidence_unique_app_count(
    evidence: ProofEvidence,
    padding_idx: int,
) -> int:
    """Count distinct ``(rule, head, sorted_body)`` tuples in evidence.

    This is keras's ``rule2groundings`` metric. Equivalent to
    ``evidence_unique_app_keys(evidence, padding_idx).shape[0]``.
    """
    if (evidence is None
            or evidence.rule_idx is None
            or evidence.body is None):
        return 0
    return int(evidence_unique_app_keys(evidence, padding_idx).shape[0])


def evidence_to_rule_groundings(
    evidence: ProofEvidence,
    padding_idx: int,
    num_rules: Optional[int] = None,
) -> RuleGroundings:
    """Build a ``RuleGroundings`` dataclass from ``ProofEvidence``.

    Equivalent to keras's ``rule2groundings``: each rule application is
    a separate entry, body atoms point into a global atom table.

    Args:
        evidence: per-query proof evidence from a grounder.
        padding_idx: predicate index used to mark padded atoms.
        num_rules: total number of rules in the KB. If omitted, taken
            as ``max(rule_idx) + 1`` over the evidence; callers that
            need a stable ``A_in`` keyspace across runs should pass
            this explicitly.

    Returns:
        ``RuleGroundings(atom_table, A_in, A_out, num_atoms, num_rules)``
        with ``A_in[r]`` of shape ``[G_r, M_r]`` (body-atom indices into
        ``atom_table``) and ``A_out[r]`` of shape ``[G_r, 1]`` (head
        atom indices).
    """
    keys = evidence_unique_app_keys(evidence, padding_idx)
    if keys.shape[0] == 0:
        empty = torch.zeros(0, 3, dtype=torch.long)
        return RuleGroundings(
            atom_table=empty,
            A_in={}, A_out={},
            num_atoms=0,
            num_rules=int(num_rules or 0))

    M = (keys.shape[1] - 4) // 3   # 1 ridx + 3 head + 3*M body
    N = keys.shape[0]

    ridx = keys[:, 0]                            # [N]
    head = keys[:, 1:4]                          # [N, 3]
    body = keys[:, 4:].reshape(N, M, 3)          # [N, M, 3]

    # Build a global atom table by deduplicating heads + active body
    # atoms. Padded body slots map to a dedicated padding entry.
    body_active = body[..., 0] != padding_idx     # [N, M]
    head_h = atom_hash(head)
    body_h = atom_hash(body)
    flat_active_body_h = body_h[body_active]
    flat_active_body = body[body_active]
    pad_atom = torch.tensor(
        [padding_idx, padding_idx, padding_idx],
        dtype=torch.long, device=keys.device).unsqueeze(0)
    pad_h = atom_hash(pad_atom)

    all_h = torch.cat([head_h, flat_active_body_h, pad_h])
    uniq_h, inv = torch.unique(all_h, return_inverse=True)

    # Reconstruct one representative atom per unique hash.
    n_head = head_h.shape[0]
    n_body = flat_active_body_h.shape[0]
    repr_atoms = torch.zeros(uniq_h.shape[0], 3,
                             dtype=torch.long, device=keys.device)
    repr_atoms[inv[:n_head]] = head
    repr_atoms[inv[n_head:n_head + n_body]] = flat_active_body
    repr_atoms[inv[n_head + n_body]] = pad_atom.squeeze(0)
    pad_id = int(inv[n_head + n_body].item())

    head_id = inv[:n_head]                       # [N]
    body_id = torch.full((N, M), pad_id,
                         dtype=torch.long, device=keys.device)
    body_id[body_active] = inv[n_head:n_head + n_body]

    if num_rules is None:
        num_rules_ = int(ridx.max().item()) + 1 if ridx.numel() else 0
    else:
        num_rules_ = int(num_rules)

    A_in: dict = {}
    A_out: dict = {}
    for r in range(num_rules_):
        sel = ridx == r
        if not bool(sel.any()):
            continue
        A_in[r] = body_id[sel]                   # [G_r, M]
        A_out[r] = head_id[sel].unsqueeze(-1)    # [G_r, 1]

    return RuleGroundings(
        atom_table=repr_atoms,
        A_in=A_in,
        A_out=A_out,
        num_atoms=int(repr_atoms.shape[0]),
        num_rules=num_rules_,
    )


def count_proof_trees(
    rule_apps_atoms: Tensor,
    rule_apps_lens: Tensor,
    facts_atoms: Tensor,
    queries: Tensor,
    *,
    max_iters: int = 64,
) -> int:
    """Count distinct proof trees rooted at the supplied queries.

    A proof tree of query ``q`` is a tree whose root is ``q`` and whose
    leaves are facts; every internal node is the head of a rule
    application whose body atoms are children. Recurrence::

        proofs[atom] = 1                              if atom ∈ facts
        proofs[atom] = Σ_R  ∏_{b ∈ body(R)} proofs[b] otherwise

    Iterated to fixpoint. The total returned is
    ``Σ_{q ∈ queries} proofs[q]``.

    Args:
        rule_apps_atoms: ``[N, 1+M, 3]`` head plus M body atoms per
            rule application; body slots beyond ``rule_apps_lens[i]``
            may be anything (treated as padding).
        rule_apps_lens: ``[N]`` valid body length per app.
        facts_atoms: ``[F, 3]`` facts (proof base case).
        queries: ``[Q, 3]`` queries (proof root).
        max_iters: bailout for non-convergent rule sets.

    Returns:
        Sum of proof tree counts across the queries.
    """
    if rule_apps_atoms.numel() == 0:
        q_h = atom_hash(queries)
        f_h = atom_hash(facts_atoms)
        return int(torch.isin(q_h, f_h).sum().item())

    N, MM1, _ = rule_apps_atoms.shape
    M = MM1 - 1
    head = rule_apps_atoms[:, 0, :]
    body = rule_apps_atoms[:, 1:, :]
    head_h = atom_hash(head)
    body_h = atom_hash(body)
    fact_h = atom_hash(facts_atoms)
    q_h = atom_hash(queries)
    body_pos = torch.arange(M, device=body_h.device)
    body_active = body_pos.unsqueeze(0) < rule_apps_lens.unsqueeze(1)

    all_atoms_h = torch.cat([fact_h, head_h, body_h[body_active]])
    uniq_h, _ = torch.unique(all_atoms_h, return_inverse=True)

    def _id_of(h: Tensor) -> Tensor:
        idx = torch.searchsorted(uniq_h, h)
        return idx.clamp(max=uniq_h.shape[0] - 1)

    fact_id = _id_of(fact_h)
    head_id = _id_of(head_h)
    body_id = _id_of(body_h.reshape(-1)).reshape(N, M)
    q_id = _id_of(q_h)

    U = uniq_h.shape[0]
    proofs = torch.zeros(U, dtype=torch.float64, device=uniq_h.device)
    proofs.index_fill_(0, fact_id, 1.0)

    sentinel_id = (fact_id[0] if fact_id.numel()
                   else torch.tensor(0, device=body_id.device))
    body_id_safe = torch.where(body_active, body_id, sentinel_id)

    for _ in range(max_iters):
        body_proofs = proofs[body_id_safe]
        body_proofs = torch.where(
            body_active, body_proofs, torch.ones_like(body_proofs))
        per_app = body_proofs.prod(dim=1)
        new_head_proofs = torch.zeros_like(proofs)
        new_head_proofs.scatter_add_(0, head_id, per_app)
        prev = proofs.clone()
        proofs = torch.maximum(proofs, new_head_proofs)
        proofs.index_fill_(0, fact_id, 1.0)
        if torch.equal(proofs, prev):
            break

    return int(proofs[q_id].sum().item())
