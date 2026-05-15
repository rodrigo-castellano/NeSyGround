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

    return RuleGroundings(
        atom_table=new_pool,
        A_in=rg.A_in,
        A_out=rg.A_out,
        num_atoms=int(new_pool.shape[0]),
        num_rules=rg.num_rules,
        query_pool_idx=new_query_pool_idx,
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
    A_in = rg.A_in
    A_out = rg.A_out

    new_A_in: Dict[int, Tensor] = {}
    new_A_out: Dict[int, Tensor] = {}
    firings_valid: Optional[Dict[int, Tensor]] = (
        {} if pad_per_rule_to is not None else None
    )
    if pad_per_rule_to is not None:
        G = pad_per_rule_to
        # Fully-vectorized pad: build one [num_rules*G, M] target buffer
        # and scatter each rule's real firings into its slice via a
        # single index assignment. The previous per-rule torch.cat fired
        # ~5 small CUDA kernels per rule (zeros×2, cat×2, ones×1) which
        # on the 48-rule family workload dominated the pad cost. The
        # vectorized form runs ~6 kernels regardless of rule count.
        rule_keys = sorted(A_in.keys())
        # Determine M and dtypes from the first non-empty rule (every
        # rule shares M after considered.finalize / prune).
        sample = next((A_in[r] for r in rule_keys if A_in[r].size(1) > 0),
                      A_in[rule_keys[0]] if rule_keys else None)
        if sample is None or sample.numel() == 0 and all(A_in[r].size(0) == 0 for r in rule_keys):
            # Empty input — nothing to pad; emit per-rule empty tensors.
            for r in rule_keys:
                new_A_in[r] = A_in[r]
                new_A_out[r] = A_out[r]
                firings_valid[r] = torch.zeros(G, dtype=torch.bool, device=device)
        else:
            M = sample.size(1)
            a_in_dtype = sample.dtype
            a_out_dtype = A_out[rule_keys[0]].dtype
            # Per-rule firing counts (clamped to G — truncate if K_r > G).
            K_clamped = torch.tensor(
                [min(A_in[r].size(0), G) for r in rule_keys],
                dtype=torch.long, device=device,
            )                                                       # [num_rules]
            num_rules_local = len(rule_keys)
            # Flatten real firings across rules in one cat call. Truncate
            # per-rule first so torch.cat sees clamped sizes.
            a_in_chunks = []
            a_out_chunks = []
            for i, r in enumerate(rule_keys):
                K_r_clamped = int(K_clamped[i].item())
                if K_r_clamped == 0:
                    continue
                a_in_chunks.append(A_in[r][:K_r_clamped])
                a_out_chunks.append(A_out[r][:K_r_clamped, 0])
            if a_in_chunks:
                a_in_flat = torch.cat(a_in_chunks, dim=0)            # [total_K, M]
                a_out_flat = torch.cat(a_out_chunks, dim=0)          # [total_K]
            else:
                a_in_flat = torch.empty(0, M, dtype=a_in_dtype, device=device)
                a_out_flat = torch.empty(0, dtype=a_out_dtype, device=device)
            # Allocate target buffers shaped [num_rules, G, ...] and
            # initialize to pad_idx_for_atoms.
            A_in_3d = torch.full(
                (num_rules_local, G, M), pad_idx_for_atoms,
                dtype=a_in_dtype, device=device,
            )
            A_out_3d = torch.full(
                (num_rules_local, G, 1), pad_idx_for_atoms,
                dtype=a_out_dtype, device=device,
            )
            valid_2d = torch.zeros(
                (num_rules_local, G), dtype=torch.bool, device=device)
            # Compute target rows for the flat input via:
            #   rule_idx[i] = which rule firing i belongs to
            #   local_idx[i] = position within that rule (0..K_r-1)
            #   target[i] = rule_idx[i] * G + local_idx[i]
            rule_idx_flat = torch.repeat_interleave(
                torch.arange(num_rules_local, device=device, dtype=torch.long),
                K_clamped,
            )                                                       # [total_K]
            total_K = int(K_clamped.sum().item())
            rule_offset = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                           K_clamped[:-1]]), dim=0)                  # [num_rules]
            local_idx = (torch.arange(total_K, device=device, dtype=torch.long)
                         - rule_offset[rule_idx_flat])
            target_idx = rule_idx_flat * G + local_idx                # [total_K]
            # Scatter via .view(-1, ...) — A_in_3d is contiguous so
            # view(-1, M) is a single dispatch (no copy).
            A_in_3d.view(-1, M)[target_idx] = a_in_flat
            A_out_3d.view(-1, 1)[target_idx, 0] = a_out_flat
            valid_2d.view(-1)[target_idx] = True
            # Per-rule slicing: views into the 3D tensor, 0 kernels.
            for i, r in enumerate(rule_keys):
                new_A_in[r] = A_in_3d[i]                              # view [G, M]
                new_A_out[r] = A_out_3d[i]                            # view [G, 1]
                firings_valid[r] = valid_2d[i]                        # view [G]
    else:
        new_A_in = dict(A_in)
        new_A_out = dict(A_out)

    if pad_atom_table_to is not None:
        cur_n = atom_table.size(0)
        if pad_atom_table_to > cur_n:
            extra = pad_atom_table_to - cur_n
            pad_rows = torch.zeros(
                extra, 3, dtype=atom_table.dtype, device=device,
            )
            atom_table = torch.cat([atom_table, pad_rows], dim=0)
        # If pad_atom_table_to <= cur_n we leave atom_table as-is.
        # Truncating would silently invalidate gathers from real
        # firings whose indices point past the truncation.

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        A_in=new_A_in,
        A_out=new_A_out,
        num_atoms=int(atom_table.size(0)),
        num_rules=rg.num_rules,
        query_pool_idx=rg.query_pool_idx,
        firings_valid=firings_valid,
    )
