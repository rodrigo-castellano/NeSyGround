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

from typing import Dict, Optional

import torch
from torch import Tensor

from grounder.types import ProofEvidence, RuleGroundings


__all__ = [
    "atom_hash",
    "evidence_unique_app_count",
    "evidence_unique_app_keys",
    "evidence_to_rule_groundings",
    "closure_to_rule_groundings_fast",
    "count_proof_trees",
    "populate_query_pool_idx",
    "pad_rule_groundings",
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

    # Branchless layout dispatch — no data-dependent ``bool(mask.any())``
    # early returns. When every row is inactive, the ``keep`` mask is
    # all-False and the indexed tensors come out empty; the caller's
    # downstream ``torch.unique`` / ``representatives[inv] = keys``
    # handles the empty case naturally.
    if evidence.rule_idx.dim() == 3:
        # Structured layout — one row per (B, C, D).
        M = evidence.M if evidence.M else evidence.body.shape[-2]
        mask = evidence.mask.unsqueeze(-1) & (evidence.rule_idx >= 0)
        flat_mask = mask.reshape(-1)
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
    body_sorted_h = body_h.gather(1, sort_idx)  # sorted body hashes

    keys = torch.cat([
        ridx.unsqueeze(-1),
        head,
        body_sorted.reshape(-1, M * 3),
    ], dim=-1)
    # Hash-based dedup instead of ``torch.unique(keys, dim=0)``: the
    # latter calls ``unique_dim`` which does a per-row sort on a
    # (1 + 3 + 3M)-wide int64 row. Polynomial hash + 1D ``unique``
    # is faster (the path the rest of this module already takes).
    # Collisions share the vanishingly-rare risk that ``atom_hash``
    # already accepts elsewhere.
    head_h = atom_hash(head)                                   # [N]
    P = _HASH_P0
    row_hash = ridx * P + head_h
    for m in range(M):
        row_hash = row_hash * P + body_sorted_h[:, m]
    uniq_h, inv = torch.unique(row_hash, return_inverse=True)
    n_uniq = uniq_h.size(0)
    representatives = torch.empty(
        n_uniq, keys.size(1), dtype=keys.dtype, device=keys.device)
    representatives[inv] = keys
    return representatives


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
        ``RuleGroundings`` with flat firings concatenated in sorted
        ``rule_idx`` order. Use ``rg.rule_offsets[r]:rg.rule_offsets[r+1]``
        to slice per-rule firings; the legacy ``rg.A_in[r]`` /
        ``rg.A_out[r]`` dict views are still available for
        backward-compatible callers.

    Compile note (``torch.compile(dynamic=True)``):
        This path uses ``torch.unique`` twice (once over body-atom
        hashes, once via the follow-up ``populate_query_pool_idx``).
        Under ``dynamic=True`` the resulting ``aten._unique2``
        produces unbacked SymInts that dynamo's codegen does not
        always bind into the generated wrapper — observed as
        ``NameError: name 's<N>' is not defined`` at trace finalize
        for the full closure-resolution graph. The bug is
        torch-upstream (pytorch/pytorch#108014, #108877). The
        production path is ``closure_to_rule_groundings_fast``
        (see below), which is dedup-free and fully static-shape;
        wrap that in ``torch.compile(dynamic=False)`` to get the
        one-graph fast path. ``dynamic=True`` over the legacy
        builder is not supported and is not a regression for any
        production cell — call the fast builder instead.
    """
    keys = evidence_unique_app_keys(evidence, padding_idx)
    if keys.shape[0] == 0:
        return RuleGroundings.empty(num_rules=int(num_rules or 0))

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
    pad_id_t = inv[n_head + n_body]                          # 0-dim long

    head_id = inv[:n_head]                       # [N]
    # ``Tensor.fill_`` accepts a 0-dim scalar tensor and keeps it on
    # device, avoiding the ``.item()`` host sync that ``torch.full``'s
    # ``fill_value`` would force.
    body_id = torch.empty((N, M), dtype=torch.long, device=keys.device)
    body_id.fill_(pad_id_t)
    body_id[body_active] = inv[n_head:n_head + n_body]

    if num_rules is None:
        num_rules_ = int(ridx.max().item()) + 1 if ridx.numel() else 0
    else:
        num_rules_ = int(num_rules)

    # Sort firings by rule_idx (stable) so each rule's slice is a
    # contiguous range in the flat tensors. ``bincount`` then gives
    # per-rule sizes, and the cumulative sum is ``rule_offsets``.
    sort_idx = torch.argsort(ridx, stable=True)               # [N]
    rule_idx_sorted = ridx[sort_idx].long()                   # [N]
    body_pool_idx = body_id[sort_idx]                         # [N, M]
    body_atom_valid = body_active[sort_idx]                   # [N, M]
    head_pool_idx = head_id[sort_idx]                         # [N]

    sizes = torch.bincount(rule_idx_sorted, minlength=num_rules_)
    rule_offsets = torch.zeros(
        num_rules_ + 1, dtype=torch.long, device=ridx.device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)             # [num_rules + 1]

    firing_valid = torch.ones(N, dtype=torch.bool, device=ridx.device)

    return RuleGroundings(
        atom_table=repr_atoms,
        body_pool_idx=body_pool_idx,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_pool_idx,
        rule_idx=rule_idx_sorted,
        rule_offsets=rule_offsets,
        firing_valid=firing_valid,
        num_atoms=int(repr_atoms.shape[0]),
        num_rules=num_rules_,
        M_max=int(M),
    )


@torch.no_grad()
def closure_to_rule_groundings_fast(
    witness_rule: Tensor,
    witness_body: Tensor,
    witness_count: Tensor,
    valid_rule: Tensor,
    queries: Tensor,
    *,
    num_rules: int,
    M_max: int,
    padding_idx: int,
) -> RuleGroundings:
    """Fast, compile-friendly RuleGroundings builder for the closure resolver.

    Specialised for the closure resolution path where each query slot
    has at most one (rule, body) witness — see :func:`resolve_closure`.
    Builds a non-deduplicated ``RuleGroundings`` whose shape is a static
    function of ``(B, K, M_max, num_rules)``:

      * ``atom_table`` is ``[B + B*K*M_max + 1, 3]``: queries first,
        then per-firing body slots, then one padding sentinel.
      * ``query_pool_idx[b]`` is simply ``b`` — queries occupy the first
        ``B`` rows of the atom_table.
      * ``head_pool_idx[firing] = b`` (the query that owns the firing's
        slot) when valid, padding-slot index otherwise.
      * ``body_pool_idx[firing, m]`` is the precomputed offset
        ``B + firing*M_max + m`` when ``m < witness_count[firing]``,
        padding-slot index otherwise.
      * Firings are sorted ascending by ``rule_idx`` so ``rule_offsets``
        gives a contiguous slice per rule. Invalid firings are sorted to
        the trailing ``num_rules`` sentinel bucket; ``rule_offsets`` is
        ``[num_rules + 1]`` so the per-rule iteration ``r in range(num_rules)``
        never sees the sentinel firings.

    The function uses **no** ``torch.unique``, ``boolean_mask_index``,
    ``nonzero``, ``bincount``, or host-syncing ``.item()`` calls — every
    op is a static-shape tensor op. Combined with
    ``_resolve_closure_tensors`` it lets the whole
    ``run_bc -> rule_tail`` pipeline be wrapped in one
    ``torch.compile(mode='reduce-overhead')`` region.

    Semantic note: non-deduplicated atom_table means each firing has
    its own body-atom pool slots. For K_iter=1 ``_rule_loop`` this is
    bit-identical to the deduped path (only the per-firing-slot scatter
    matters). For K_iter>1 the inter-firing propagation through shared
    atoms is absent — see the closure resolver docs for when that
    matters.

    Args:
        witness_rule:   ``[B, K]`` long, rule index per slot
                        (``-1`` for non-firing slots).
        witness_body:   ``[B, K, M_max, 3]`` long, body atoms per slot.
        witness_count:  ``[B, K]`` long, valid body atom count per slot.
        valid_rule:     ``[B, K]`` bool, slot fired and rule index valid.
        queries:        ``[B, 3]`` long, query atoms.
        num_rules:      total rules in the KB (Python int).
        M_max:          body width (Python int, the same ``M`` used by
                        the witness table).
        padding_idx:    predicate index used to mark padded atoms
                        (Python int).

    Returns:
        ``RuleGroundings`` with all flat fields populated, including
        ``query_pool_idx``. The trailing slot of ``atom_table`` is a
        ``[padding_idx, padding_idx, padding_idx]`` sentinel used for
        invalid body slots and unfired-firing heads.
    """
    queries = queries.long()
    witness_rule = witness_rule.long()
    witness_count = witness_count.long()
    device = queries.device
    B, K = witness_rule.shape
    N = B * K
    M = int(M_max)
    num_rules_ = int(num_rules)

    # ── 1. atom_table: queries | flattened body slots | pad sentinel.
    pad_atom_row = torch.full(
        (1, 3), int(padding_idx), dtype=torch.long, device=device)
    atom_table = torch.cat(
        [queries, witness_body.view(N * M, 3).long(), pad_atom_row], dim=0,
    )
    N_pool = B + N * M + 1
    pad_pool_idx_scalar = torch.tensor(
        N_pool - 1, dtype=torch.long, device=device)

    # ── 2. Per-firing index tensors (unsorted, firing index = b*K + k).
    valid_firing = valid_rule.reshape(N)                       # [N]
    firing_arange = torch.arange(N, device=device, dtype=torch.long)
    m_arange = torch.arange(M, device=device, dtype=torch.long)

    # head_pool_idx[firing] = b (the query owning the slot).
    head_pool_idx_u = firing_arange // K                       # [N]
    head_pool_idx_u = torch.where(
        valid_firing, head_pool_idx_u, pad_pool_idx_scalar)

    # body_pool_idx[firing, m] = B + firing*M + m  when m < count.
    body_base = B + firing_arange * M                          # [N]
    body_pool_idx_u = (
        body_base.unsqueeze(-1) + m_arange.unsqueeze(0)
    )                                                          # [N, M]
    body_active_u = (
        m_arange.unsqueeze(0) < witness_count.reshape(N).unsqueeze(-1)
    ) & valid_firing.unsqueeze(-1)                             # [N, M]
    body_pool_idx_u = torch.where(
        body_active_u, body_pool_idx_u, pad_pool_idx_scalar)

    # ── 3. Sort by rule_idx so rule_offsets is contiguous-per-rule.
    #     Invalid firings get the sentinel bucket ``num_rules`` so they
    #     land at the end of the sorted layout — never accessed by
    #     ``rule_offsets[r:r+1]`` for ``r in range(num_rules)``.
    sentinel = torch.tensor(
        num_rules_, dtype=torch.long, device=device)
    rule_idx_for_sort = torch.where(
        valid_firing, witness_rule.reshape(N), sentinel,
    )                                                          # [N]
    sort_idx = torch.argsort(rule_idx_for_sort, stable=True)   # [N]
    rule_idx_sorted = rule_idx_for_sort[sort_idx]              # [N]
    body_pool_idx = body_pool_idx_u[sort_idx]
    body_atom_valid = body_active_u[sort_idx]
    head_pool_idx = head_pool_idx_u[sort_idx]
    firing_valid = valid_firing[sort_idx]

    # ── 4. rule_offsets via scatter_add (static shape, no bincount sync).
    #     ``rule_idx_sorted`` values are in ``[0, num_rules]`` (with
    #     ``num_rules`` reserved for the invalid-sentinel bucket).
    sizes_full = torch.zeros(
        num_rules_ + 1, dtype=torch.long, device=device)
    sizes_full.scatter_add_(
        0, rule_idx_sorted, torch.ones_like(rule_idx_sorted))
    # rule_offsets[r] = sum(sizes[:r]) for r in [0, num_rules]. Drop the
    # sentinel-bucket count when building offsets so the per-rule
    # iteration ``r in range(num_rules)`` never crosses into the
    # invalid range.
    rule_offsets = torch.zeros(
        num_rules_ + 1, dtype=torch.long, device=device)
    rule_offsets[1:] = torch.cumsum(sizes_full[:num_rules_], dim=0)

    # query_pool_idx[b] = b (queries are the first B rows of atom_table).
    query_pool_idx = torch.arange(B, device=device, dtype=torch.long)

    return RuleGroundings(
        atom_table=atom_table,
        body_pool_idx=body_pool_idx,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_pool_idx,
        rule_idx=rule_idx_sorted,
        rule_offsets=rule_offsets,
        firing_valid=firing_valid,
        num_atoms=int(N_pool),
        num_rules=num_rules_,
        M_max=M,
        query_pool_idx=query_pool_idx,
    )


def populate_query_pool_idx(
    rg: RuleGroundings,
    queries: Tensor,
    padding_idx: int,
) -> RuleGroundings:
    """Extend ``rg.atom_table`` to include the queries and populate
    ``rg.query_pool_idx``.

    Pool-iter reasoners (SBR / DCR / R2N) need every query atom to
    have a slot in the atom pool, even when no grounding produced
    that atom as a head — so the final ``pool[query_pool_idx]``
    gather is well-defined regardless of provability. Atoms already
    present in ``rg.atom_table`` keep their existing slot; novel
    queries are appended to the end of the table.

    Args:
        rg: input ``RuleGroundings`` (typically straight from the
            grounder's per-rule pipeline).
        queries: ``[B, 3]`` long tensor of query atoms ``(pred, h, t)``.
        padding_idx: predicate index used to mark padded atoms in
            ``atom_table``.

    Returns:
        A new ``RuleGroundings`` whose ``atom_table`` covers every
        query atom and whose ``query_pool_idx`` is shape ``[B]``,
        containing the pool index of each query. ``A_in`` / ``A_out``
        are unchanged (their indices already point into the prefix
        of ``atom_table`` that wasn't extended).
    """
    if queries.dim() != 2 or queries.shape[1] != 3:
        raise ValueError(
            f"queries must be [B, 3]; got shape {tuple(queries.shape)}")
    queries = queries.long()
    device = queries.device

    pool = rg.atom_table.to(device)

    # Branchless union: pool ∪ unique(queries). Works whether pool is
    # empty or non-empty — when empty, ``in_pool_mask`` is all-False
    # (the broadcast comparison reduces over a zero-size dim → False
    # everywhere), ``novel_atoms`` is just ``queries``, and
    # ``new_pool`` becomes the deduped queries. No data-dependent
    # ``if pool.numel() == 0`` branch — the whole function is
    # fullgraph-traceable.
    pool_h = atom_hash(pool)                                 # [N]
    query_h = atom_hash(queries)                             # [B]
    # Equivalent of ``torch.isin(query_h, pool_h)`` but without the
    # decomposition's data-dependent ``len(test) * log2(len(elem))``
    # heuristic that breaks dynamo. O(B*N) broadcast compare; for the
    # sizes seen here (B ~ thousands, N ~ thousands) the GPU swallows
    # this trivially.
    in_pool_mask = (
        query_h.unsqueeze(1) == pool_h.unsqueeze(0)
    ).any(dim=1)                                             # [B]

    # Append novel queries (deduped among themselves) so the existing
    # slots 0..N-1 stay stable and rg.A_in/A_out remain valid as-is.
    # Even when every query is already in the pool, ``queries[~in_pool_mask]``
    # produces an empty [0, 3] tensor and ``torch.cat`` is a no-op.
    novel_atoms = queries[~in_pool_mask]
    novel_h = atom_hash(novel_atoms)
    novel_h_uniq, novel_inv = torch.unique(novel_h, return_inverse=True)
    novel_repr = torch.zeros(
        novel_h_uniq.shape[0], 3, dtype=torch.long, device=device)
    novel_repr[novel_inv] = novel_atoms
    new_pool = torch.cat([pool, novel_repr], dim=0)

    # Final lookup: each query's index in new_pool, via sorted-hash
    # binary search. Stable for tie-broken duplicates because each
    # unique hash maps to exactly one slot in new_pool.
    new_pool_h = atom_hash(new_pool)
    sort_idx = new_pool_h.argsort()
    sorted_h = new_pool_h[sort_idx]
    pos = torch.searchsorted(sorted_h, query_h)
    new_query_pool_idx = sort_idx[pos]

    # All flat firing fields stay valid: the prefix 0..N-1 of the
    # atom table is unchanged (new entries are appended), so the
    # ``body_pool_idx`` / ``head_pool_idx`` indices that point into
    # the prefix still resolve to the same atoms.
    return RuleGroundings(
        atom_table=new_pool,
        body_pool_idx=rg.body_pool_idx,
        body_atom_valid=rg.body_atom_valid,
        head_pool_idx=rg.head_pool_idx,
        rule_idx=rg.rule_idx,
        rule_offsets=rg.rule_offsets,
        firing_valid=rg.firing_valid,
        num_atoms=int(new_pool.shape[0]),
        num_rules=rg.num_rules,
        M_max=rg.M_max,
        query_pool_idx=new_query_pool_idx,
        _has_real_firings_valid=rg._has_real_firings_valid,
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


def pad_rule_groundings(
    rg: "RuleGroundings",
    *,
    pad_per_rule_to: Optional[int] = None,
    pad_atom_table_to: Optional[int] = None,
    pad_idx_for_atoms: int = 0,
) -> "RuleGroundings":
    """Return a copy of ``rg`` whose ``A_in[r]``/``A_out[r]`` are
    padded to ``[pad_per_rule_to, M]``/``[pad_per_rule_to, 1]`` and
    whose ``atom_table`` is padded to ``[pad_atom_table_to, 3]``.

    Padding rows in each ``A_in[r]`` / ``A_out[r]`` point at the
    consumer-side sentinel pool slot (``pad_idx_for_atoms``, default 0
    — the standard ``_rule_loop`` pad slot). The per-rule
    ``firings_valid`` mask is emitted so that
    ``build_firings_from_rule_groundings`` can forward validity into
    ``FiringsTensors.firing_valid`` and the rule loop masks the
    padded rows out of the T-norm composition.

    This is a post-processing step (after any
    ``prune_rule_groundings``) used to give the downstream compiled
    reasoner a fixed-shape input on cells whose flat-path output
    would otherwise oscillate across batches (countries_s3 + BC13,
    family + BC{12,13}). On main, those cells fall back to
    ``compile_mode=None`` instead.

    No-ops when both pad arguments are ``None``.
    """
    from grounder.types import RuleGroundings
    if pad_per_rule_to is None and pad_atom_table_to is None:
        return rg

    atom_table = rg.atom_table
    device = atom_table.device
    M_max = rg.M_max
    num_rules = rg.num_rules
    has_real_firings_valid = rg._has_real_firings_valid

    if pad_per_rule_to is not None:
        G = pad_per_rule_to
        # Per-rule firing counts. We want each rule's slice in the
        # flat tensors to grow to exactly G rows (clamping if K_r > G,
        # padding if K_r < G). The actual firings stay in the front
        # of each rule's slice; trailing rows are pad sentinels.
        #
        # Branchless construction: build *fixed-shape* ``[num_rules * G]``
        # gather indices, then mask invalid (out-of-K_r) slots with
        # ``torch.where``. This avoids the previous ``total_K.item()``
        # host sync and the ``if total_K > 0`` Python branch.
        rule_offsets = rg.rule_offsets                              # [num_rules + 1]
        rule_sizes = (rule_offsets[1:] - rule_offsets[:-1]).long()  # [num_rules]
        K_clamped = rule_sizes.clamp(max=G)                         # [num_rules]

        new_N = num_rules * G
        if new_N == 0:
            # Genuinely-empty case (no rules at all). Allocate the same
            # zero-sized outputs the loop would have produced; skip the
            # gather/mask path entirely without consulting GPU state.
            body_pool_idx = torch.empty(
                (0, M_max), dtype=rg.body_pool_idx.dtype, device=device)
            body_atom_valid = torch.empty(
                (0, M_max), dtype=torch.bool, device=device)
            head_pool_idx = torch.empty(
                (0,), dtype=rg.head_pool_idx.dtype, device=device)
            firing_valid = torch.empty(
                (0,), dtype=torch.bool, device=device)
        else:
            # ``rule_idx_flat[i]`` = which rule does flat slot i belong to,
            # ``local_idx[i]`` = position within that rule's G-slot.
            rule_idx_flat = torch.arange(
                num_rules, device=device, dtype=torch.long
            ).repeat_interleave(G)                                   # [num_rules * G]
            local_idx = torch.arange(
                new_N, device=device, dtype=torch.long
            ) % G                                                    # [num_rules * G]
            # Valid iff local_idx < K_clamped[rule]; broadcast-compare
            # avoids any data-dependent shape.
            K_per_slot = K_clamped.repeat_interleave(G)              # [num_rules * G]
            valid = local_idx < K_per_slot                           # [num_rules * G]
            # Source row in rg.body_pool_idx. For invalid (padding)
            # slots we still need a safe index (0 is always in-range
            # whenever new_N > 0 implies num_rules > 0 — but the source
            # tensors may still be empty if rg had zero real firings,
            # which we handle by clamping the gather index to the
            # available source size). When ``rg`` has no real firings
            # (``rule_offsets`` all zero), ``K_clamped`` is all zero so
            # ``valid`` is all False — the masked gathers don't need
            # any real source rows.
            src_idx_raw = rule_offsets[rule_idx_flat] + local_idx    # [num_rules * G]
            src_size = rg.body_pool_idx.size(0)
            # ``clamp(max=src_size - 1)`` keeps the index in range when
            # ``valid`` is False (the result will be masked out anyway).
            # When ``src_size == 0``, ``valid`` is necessarily all False
            # (rule_sizes all 0 → K_clamped all 0); we replace the
            # gather with a zero/pad value built directly so the index
            # never hits an empty tensor.
            pad_atom_id = torch.tensor(
                pad_idx_for_atoms,
                dtype=rg.body_pool_idx.dtype, device=device)
            if src_size > 0:
                src_idx_safe = src_idx_raw.clamp(max=src_size - 1)
                gathered_body = rg.body_pool_idx[src_idx_safe]
                gathered_body_valid = rg.body_atom_valid[src_idx_safe]
                gathered_head = rg.head_pool_idx[src_idx_safe]
                gathered_firing_valid = rg.firing_valid[src_idx_safe]
            else:
                # Zero-firing case: every slot is padding; build the
                # destination tensors directly so we never index a
                # zero-size source.
                gathered_body = torch.empty(
                    (new_N, M_max), dtype=rg.body_pool_idx.dtype,
                    device=device).fill_(pad_atom_id)
                gathered_body_valid = torch.zeros(
                    (new_N, M_max), dtype=torch.bool, device=device)
                gathered_head = torch.empty(
                    (new_N,), dtype=rg.head_pool_idx.dtype,
                    device=device).fill_(pad_atom_id)
                gathered_firing_valid = torch.zeros(
                    (new_N,), dtype=torch.bool, device=device)
            valid_2d = valid.unsqueeze(-1)
            body_pool_idx = torch.where(
                valid_2d, gathered_body, pad_atom_id)
            body_atom_valid = gathered_body_valid & valid_2d
            head_pool_idx = torch.where(
                valid, gathered_head, pad_atom_id)
            firing_valid = gathered_firing_valid & valid
        rule_idx = torch.repeat_interleave(
            torch.arange(num_rules, device=device, dtype=torch.long), G)
        rule_offsets = torch.arange(
            num_rules + 1, device=device, dtype=torch.long) * G
        has_real_firings_valid = True
    else:
        body_pool_idx = rg.body_pool_idx
        body_atom_valid = rg.body_atom_valid
        head_pool_idx = rg.head_pool_idx
        rule_idx = rg.rule_idx
        rule_offsets = rg.rule_offsets
        firing_valid = rg.firing_valid

    if pad_atom_table_to is not None:
        cur_n = atom_table.size(0)
        if pad_atom_table_to > cur_n:
            extra = pad_atom_table_to - cur_n
            pad_rows = torch.zeros(
                extra, 3, dtype=atom_table.dtype, device=device,
            )
            atom_table = torch.cat([atom_table, pad_rows], dim=0)
        # If pad_atom_table_to <= cur_n we leave atom_table as-is.

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        body_pool_idx=body_pool_idx,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_pool_idx,
        rule_idx=rule_idx,
        rule_offsets=rule_offsets,
        firing_valid=firing_valid,
        num_atoms=int(atom_table.size(0)),
        num_rules=num_rules,
        M_max=M_max,
        query_pool_idx=rg.query_pool_idx,
        _has_real_firings_valid=has_real_firings_valid,
    )
