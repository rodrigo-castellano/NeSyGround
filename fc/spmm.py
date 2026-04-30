"""Sparse-matrix-multiplication forward chaining.

Computes the FC closure (provable atoms) using sparse matrix
representations per predicate. For each predicate ``p``, build a
``[E, E]`` sparse boolean CSR matrix where entry ``(s, o)`` = 1 iff
``p(s, o)`` is a fact. A 2-body chain rule
``p(X,Y), q(Y,Z) → r(X,Z)`` then collapses to a sparse matmul
``M_p @ M_q``.

Compared to the staged ragged-join FC (``fc.fc.FCDynamic``), SpMM
keeps work proportional to the number of NON-ZERO entries (= facts
+ derived atoms) rather than to the per-row Python dispatch cost,
and reuses libtorch's ``torch.sparse.mm`` kernel — which scales
without the Python-loop overhead that limited FCDynamic to
~30M-atom closures on commodity 24 GiB GPU / 128 GiB RAM systems.

This module ports the original ``ns_lib.grounding.provable_set``
SpMM implementation (commit ``7a6a025`` of torch-ns,
Feb 2026) verbatim, with imports adapted to the grounder package
and minor cleanup. The semi-naive incremental delta architecture
is preserved:

  * Step 0   — apply all rules with ``base_mats`` (full SpMM).
  * Step >0  — ``MATMUL`` / ``ELEM_AND`` use ``delta_mats`` for
               body1; ``COPY`` / ``TRANSPOSE`` are constant from
               step 0 and skipped; ``CASE_A`` / ``EXIST_AND``
               re-evaluated against accumulated ``provable_hashes``.

Rule classification (``classify_rule``):

  =========== ====================================================
  ``COPY``       1-body: ``h(X,Y) :- b(X,Y)``  (direct copy)
  ``TRANSPOSE``  1-body: ``h(X,Y) :- b(Y,X)``  (matrix transpose)
  ``MATMUL``     2-body chain: sparse ``A @ B``
  ``ELEM_AND``   2-body, both bodies bind same head vars
  ``CASE_A``     2-body, body1 vars all derivable from body0
  ``EXIST_AND``  2-body, body1 has 1 head var + 1 existential
  ``UNSUPPORTED`` Falls back to the staged FCDynamic (handled by
                  the dispatch in ``fc.fc.run_forward_chaining``).
  =========== ====================================================

The ``num_body=3`` case is *not* supported by SpMM; rules with three
or more body atoms must use the staged FCDynamic.

Public entry point: :func:`run_forward_chaining_spmm`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "SpMMOp",
    "SpMMRuleDesc",
    "classify_rule",
    "build_pred_sparse_matrices",
    "apply_spmm_rule",
    "run_forward_chaining_spmm",
]


# ══════════════════════════════════════════════════════════════════════
# Rule classification
# ══════════════════════════════════════════════════════════════════════


class SpMMOp(Enum):
    """Classification of a rule's binding pattern into a SpMM operation."""
    COPY = auto()          # 1-body: direct copy
    TRANSPOSE = auto()     # 1-body: transpose (swap args)
    MATMUL = auto()        # 2-body: sparse matrix multiplication
    ELEM_AND = auto()      # 2-body: element-wise AND (both args identical)
    CASE_A = auto()        # 2-body: fully resolved (existence check)
    EXIST_AND = auto()     # 2-body: body0 has head vars, body1 has existential
    UNSUPPORTED = auto()   # Falls back to staged FCDynamic.


@dataclass
class SpMMRuleDesc:
    """Describes how to compute a rule's provable atoms via SpMM."""
    op: SpMMOp
    head_pred: int
    pred_A: int = -1
    pred_B: int = -1
    transpose_A: bool = False
    transpose_B: bool = False
    transpose_result: bool = False
    reflexive_body0: bool = False
    reflexive_body1: bool = False
    b10_binding: int = -1
    b11_binding: int = -1
    b00: int = -1
    b01: int = -1
    exist_body1_head_arg: int = -1
    exist_filter_body0_arg: int = -1


def classify_rule(cr) -> SpMMRuleDesc:
    """Classify a ``RulePattern`` into a SpMM operation.

    Uses variable names (not just binding indices) so 2-body rules
    are categorised by their *shared variable* pattern. See module
    docstring for the full op table.
    """
    head_pred = cr.head_pred_idx
    hv0 = cr.head_var0
    hv1 = cr.head_var1

    if cr.num_body == 1:
        bp = cr.body_patterns[0]
        b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
        pred_A = bp["pred_idx"]
        reflexive = (b0 == b1)

        if b0 == 0 and b1 == 1:
            return SpMMRuleDesc(op=SpMMOp.COPY, head_pred=head_pred,
                                pred_A=pred_A, reflexive_body0=reflexive)
        if b0 == 1 and b1 == 0:
            return SpMMRuleDesc(op=SpMMOp.TRANSPOSE, head_pred=head_pred,
                                pred_A=pred_A, reflexive_body0=reflexive)
        return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

    if cr.num_body == 2:
        bp0, bp1 = cr.body_patterns[0], cr.body_patterns[1]
        b00, b01 = bp0["arg0_binding"], bp0["arg1_binding"]
        b10, b11 = bp1["arg0_binding"], bp1["arg1_binding"]
        pred0, pred1 = bp0["pred_idx"], bp1["pred_idx"]

        v00, v01 = bp0["arg0_var"], bp0["arg1_var"]
        v10, v11 = bp1["arg0_var"], bp1["arg1_var"]

        reflexive0 = (v00 == v01)
        reflexive1 = (v10 == v11)

        b0_vars = {v00, v01}
        b1_vars = {v10, v11}
        shared_vars = (b0_vars & b1_vars) - {hv0, hv1}

        b1_from_b0 = b1_vars.issubset(b0_vars)
        if b1_from_b0:
            return SpMMRuleDesc(
                op=SpMMOp.CASE_A, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1,
                b10_binding=b10, b11_binding=b11,
                b00=b00, b01=b01)

        # Element-wise AND: both bodies bind to the same head vars.
        if (b00 in (0, 1) and b01 in (0, 1) and b10 in (0, 1) and b11 in (0, 1)
                and b00 == b10 and b01 == b11):
            return SpMMRuleDesc(
                op=SpMMOp.ELEM_AND, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1)

        # EXIST_AND: body0 has only head vars; body1 has 1 head var
        # + 1 existential.
        b0_head_vars = b0_vars & {hv0, hv1}
        b1_head_vars = b1_vars & {hv0, hv1}
        b1_exist_vars = b1_vars - {hv0, hv1}
        if (b0_head_vars == b0_vars and len(b1_head_vars) == 1
                and len(b1_exist_vars) == 1):
            shared_head_var = next(iter(b1_head_vars))
            exist_b1_arg = 0 if v10 == shared_head_var else 1
            exist_filter_b0 = 0 if v00 == shared_head_var else 1
            if v00 == hv0 and v01 == hv1:
                t_R = False
            elif v00 == hv1 and v01 == hv0:
                t_R = True
            else:
                return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
            return SpMMRuleDesc(
                op=SpMMOp.EXIST_AND, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                transpose_result=t_R,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1,
                exist_body1_head_arg=exist_b1_arg,
                exist_filter_body0_arg=exist_filter_b0)

        # MATMUL: exactly one shared variable that's not a head var.
        if len(shared_vars) != 1:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

        shared = next(iter(shared_vars))
        shared_is_a0 = (v00 == shared)
        shared_is_a1 = (v01 == shared)
        shared_is_c0 = (v10 == shared)
        shared_is_c1 = (v11 == shared)

        if not (shared_is_a0 or shared_is_a1):
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        if not (shared_is_c0 or shared_is_c1):
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

        t_A = shared_is_a0
        t_B = shared_is_c1

        non_shared_a_var = v01 if shared_is_a0 else v00
        non_shared_c_var = v10 if shared_is_c1 else v11

        if non_shared_a_var == hv0 and non_shared_c_var == hv1:
            t_R = False
        elif non_shared_a_var == hv1 and non_shared_c_var == hv0:
            t_R = True
        else:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

        return SpMMRuleDesc(
            op=SpMMOp.MATMUL, head_pred=head_pred,
            pred_A=pred0, pred_B=pred1,
            transpose_A=t_A, transpose_B=t_B,
            transpose_result=t_R,
            reflexive_body0=reflexive0, reflexive_body1=reflexive1)

    # 3+ body atoms not supported by SpMM.
    return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)


# ══════════════════════════════════════════════════════════════════════
# Sparse matrix construction
# ══════════════════════════════════════════════════════════════════════


def build_pred_sparse_matrices(
    fact_preds: Tensor,
    fact_subjs: Tensor,
    fact_objs: Tensor,
    E: int,
    P: int,
) -> Dict[int, Tensor]:
    """Build per-predicate ``[E, E]`` sparse boolean CSR matrices."""
    mats: Dict[int, Tensor] = {}
    for p in range(P):
        mask = fact_preds == p
        if not mask.any():
            continue
        rows = fact_subjs[mask]
        cols = fact_objs[mask]
        hashes = rows * E + cols
        unique_h = torch.unique(hashes)
        rows = unique_h // E
        cols = unique_h % E
        vals = torch.ones(rows.shape[0], dtype=torch.float32)
        coo = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(E, E))
        mats[p] = coo.to_sparse_csr()
    return mats


def _sparse_to_bool_csr(mat: Tensor) -> Tensor:
    """Clamp a sparse CSR matrix's values to 0/1."""
    vals = mat.values()
    new_vals = (vals > 0).float()
    return torch.sparse_csr_tensor(
        mat.crow_indices(), mat.col_indices(), new_vals, size=mat.shape)


def _sparse_nnz(mat: Tensor) -> int:
    return mat.values().shape[0]


def _transpose_csr(mat: Tensor, E: int) -> Tensor:
    coo = mat.to_sparse_coo().coalesce()
    idx = coo.indices()
    if idx.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(E, E)).to_sparse_csr()
    new_idx = torch.stack([idx[1], idx[0]])
    vals = torch.ones(new_idx.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(new_idx, vals, size=(E, E)).to_sparse_csr()


def _get_diagonal_mask(mat: Tensor, E: int) -> Tensor:
    """Filter sparse matrix to only diagonal entries (row == col)."""
    coo = mat.to_sparse_coo().coalesce()
    indices = coo.indices()
    if indices.shape[1] == 0:
        return mat
    diag_mask = indices[0] == indices[1]
    if not diag_mask.any():
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(E, E)).to_sparse_csr()
    new_indices = indices[:, diag_mask]
    new_vals = torch.ones(new_indices.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(
        new_indices, new_vals, size=(E, E)).to_sparse_csr()


def _sorted_merge(a: Tensor, b: Tensor) -> Tensor:
    """Merge two sorted 1-D tensors into a single sorted tensor."""
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    insert_pos = torch.searchsorted(a, b)
    n_a, n_b = a.numel(), b.numel()
    n_total = n_a + n_b
    merged = torch.empty(n_total, dtype=a.dtype)
    b_dest = insert_pos + torch.arange(n_b, dtype=torch.long)
    mask = torch.ones(n_total, dtype=torch.bool)
    mask[b_dest] = False
    merged[b_dest] = b
    merged[mask] = a
    return merged


def _unique_sorted(t: Tensor) -> Tensor:
    """Drop consecutive duplicates from a sorted 1-D tensor.

    O(n) — much faster than ``torch.unique`` on already-sorted input.
    """
    if t.numel() <= 1:
        return t
    mask = torch.empty(t.shape[0], dtype=torch.bool, device=t.device)
    mask[0] = True
    mask[1:] = t[1:] != t[:-1]
    return t[mask]


def _build_csr_from_local_hashes(local_hashes: Tensor, E: int) -> Tensor:
    """Build sparse CSR from sorted local hashes (``row*E + col``)."""
    unique_h = _unique_sorted(local_hashes)
    n = unique_h.shape[0]
    if n == 0:
        crow = torch.zeros(E + 1, dtype=torch.long)
        cols = torch.zeros(0, dtype=torch.long)
        vals = torch.zeros(0, dtype=torch.float32)
        return torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))
    rows = unique_h // E
    cols = unique_h % E
    vals = torch.ones(n, dtype=torch.float32)
    crow = torch.zeros(E + 1, dtype=torch.long)
    crow.scatter_add_(0, rows + 1, torch.ones(n, dtype=torch.long))
    crow = torch.cumsum(crow, 0)
    return torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))


def _build_csr_and_transpose(
    local_hashes: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """Build CSR and its transpose from sorted local hashes."""
    unique_h = _unique_sorted(local_hashes)
    n = unique_h.shape[0]
    if n == 0:
        crow = torch.zeros(E + 1, dtype=torch.long)
        cols = torch.zeros(0, dtype=torch.long)
        vals = torch.zeros(0, dtype=torch.float32)
        empty = torch.sparse_csr_tensor(
            crow.clone(), cols.clone(), vals.clone(), size=(E, E))
        return empty, torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))

    rows = unique_h // E
    cols = unique_h % E
    vals = torch.ones(n, dtype=torch.float32)

    crow = torch.zeros(E + 1, dtype=torch.long)
    crow.scatter_add_(0, rows + 1, torch.ones(n, dtype=torch.long))
    crow = torch.cumsum(crow, 0)
    csr = torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))

    sort_idx = torch.argsort(cols, stable=True)
    t_col_indices = rows[sort_idx]
    t_row_keys = cols[sort_idx]
    t_vals = torch.ones(n, dtype=torch.float32)
    t_crow = torch.zeros(E + 1, dtype=torch.long)
    t_crow.scatter_add_(0, t_row_keys + 1, torch.ones(n, dtype=torch.long))
    t_crow = torch.cumsum(t_crow, 0)
    csr_t = torch.sparse_csr_tensor(
        t_crow, t_col_indices, t_vals, size=(E, E))
    return csr, csr_t


def _csr_to_hashes(
    mat: Tensor, pred: int, E: int, E2: int,
) -> Optional[Tensor]:
    """Extract global hashes (``pred*E² + row*E + col``) from a CSR mat."""
    crow = mat.crow_indices()
    col = mat.col_indices()
    nnz = col.shape[0]
    if nnz == 0:
        return None
    row_counts = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(
        torch.arange(E, dtype=torch.long), row_counts)
    return pred * E2 + rows * E + col


# ══════════════════════════════════════════════════════════════════════
# Per-rule SpMM application
# ══════════════════════════════════════════════════════════════════════


def apply_spmm_rule(
    desc: SpMMRuleDesc,
    all_mats: Dict[int, Tensor],
    base_mats: Dict[int, Tensor],
    E: int,
    fact_hashes: Tensor,
    num_facts: int,
    provable_hashes: Tensor,
    base_mats_T: Optional[Dict[int, Tensor]] = None,
    all_mats_T: Optional[Dict[int, Tensor]] = None,
) -> Optional[Tensor]:
    """Apply one rule via the right sparse-matrix operation.

    Critical semi-naive semantics: ``body0`` always reads from
    ``base_mats`` (base facts only); ``body1`` reads from
    ``all_mats`` which is ``base_mats`` at step 0 and ``delta_mats``
    at later steps.
    """
    if desc.op == SpMMOp.UNSUPPORTED:
        return None

    if desc.op == SpMMOp.COPY:
        mat = base_mats.get(desc.pred_A)
        if mat is None:
            return None
        if desc.reflexive_body0:
            mat = _get_diagonal_mask(mat, E)
        return mat if _sparse_nnz(mat) > 0 else None

    if desc.op == SpMMOp.TRANSPOSE:
        if (not desc.reflexive_body0
                and base_mats_T is not None
                and desc.pred_A in base_mats_T):
            mat = base_mats_T[desc.pred_A]
            return mat if _sparse_nnz(mat) > 0 else None
        mat = base_mats.get(desc.pred_A)
        if mat is None:
            return None
        if desc.reflexive_body0:
            mat = _get_diagonal_mask(mat, E)
        return _transpose_csr(mat, E) if _sparse_nnz(mat) > 0 else None

    if desc.op == SpMMOp.ELEM_AND:
        mat_A = base_mats.get(desc.pred_A)
        mat_B = all_mats.get(desc.pred_B)
        if mat_A is None or mat_B is None:
            return None
        if desc.reflexive_body0:
            mat_A = _get_diagonal_mask(mat_A, E)
        if desc.reflexive_body1:
            mat_B = _get_diagonal_mask(mat_B, E)
        A_coo = mat_A.to_sparse_coo().coalesce()
        B_coo = mat_B.to_sparse_coo().coalesce()
        A_idx = A_coo.indices()
        B_idx = B_coo.indices()
        if A_idx.shape[1] == 0 or B_idx.shape[1] == 0:
            return None
        A_hash = A_idx[0] * E + A_idx[1]
        B_hash = B_idx[0] * E + B_idx[1]
        B_sorted, _ = B_hash.sort()
        pos = torch.searchsorted(B_sorted, A_hash)
        n_B = B_sorted.shape[0]
        valid = pos < n_B
        clamped = torch.clamp(pos, 0, max(n_B - 1, 0))
        in_B = valid & (B_sorted[clamped] == A_hash)
        if not in_B.any():
            return None
        new_indices = A_idx[:, in_B]
        new_vals = torch.ones(new_indices.shape[1], dtype=torch.float32)
        result = torch.sparse_coo_tensor(
            new_indices, new_vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.CASE_A:
        mat_A = base_mats.get(desc.pred_A)
        if mat_A is None:
            return None
        if desc.reflexive_body0:
            mat_A = _get_diagonal_mask(mat_A, E)

        A_coo = mat_A.to_sparse_coo().coalesce()
        A_idx = A_coo.indices()
        if A_idx.shape[1] == 0:
            return None

        s0 = A_idx[0]
        o0 = A_idx[1]
        E2 = E * E

        def _resolve(binding, b0_arg0, b0_arg1):
            if binding == desc.b00:
                return b0_arg0
            if binding == desc.b01:
                return b0_arg1
            return torch.zeros_like(b0_arg0)

        s1_val = _resolve(desc.b10_binding, s0, o0)
        o1_val = _resolve(desc.b11_binding, s0, o0)

        query_h = desc.pred_B * E2 + s1_val * E + o1_val

        pos = torch.searchsorted(fact_hashes, query_h)
        valid = pos < num_facts
        clamped = torch.clamp(pos, 0, max(num_facts - 1, 0))
        in_facts = valid & (fact_hashes[clamped] == query_h)

        if provable_hashes.numel() > 0:
            n_ph = provable_hashes.shape[0]
            pos_p = torch.searchsorted(provable_hashes, query_h)
            valid_p = pos_p < n_ph
            clamped_p = torch.clamp(pos_p, 0, max(n_ph - 1, 0))
            in_prov = valid_p & (provable_hashes[clamped_p] == query_h)
            found = in_facts | in_prov
        else:
            found = in_facts

        if not found.any():
            return None

        def _head_resolve(head_var, b00, b01, arg0, arg1):
            if b00 == head_var:
                return arg0
            if b01 == head_var:
                return arg1
            return torch.zeros_like(arg0)

        hx = _head_resolve(0, desc.b00, desc.b01, s0[found], o0[found])
        hy = _head_resolve(1, desc.b00, desc.b01, s0[found], o0[found])

        hashes = hx * E + hy
        unique_h = torch.unique(hashes)
        rows = unique_h // E
        cols = unique_h % E
        vals = torch.ones(rows.shape[0], dtype=torch.float32)
        result = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.EXIST_AND:
        mat_A = base_mats.get(desc.pred_A)
        if mat_A is None:
            return None
        if desc.reflexive_body0:
            mat_A = _get_diagonal_mask(mat_A, E)

        A_coo = mat_A.to_sparse_coo().coalesce()
        A_idx = A_coo.indices()
        if A_idx.shape[1] == 0:
            return None

        E2 = E * E
        b_start = desc.pred_B * E2
        b_end = b_start + E2

        f_start = torch.searchsorted(fact_hashes, b_start).item()
        f_end = torch.searchsorted(fact_hashes, b_end).item()
        fact_local = fact_hashes[f_start:f_end] - b_start

        if provable_hashes.numel() > 0:
            p_start = torch.searchsorted(provable_hashes, b_start).item()
            p_end = torch.searchsorted(provable_hashes, b_end).item()
            prov_local = provable_hashes[p_start:p_end] - b_start
        else:
            prov_local = torch.zeros(0, dtype=torch.long)

        all_local = (torch.cat([fact_local, prov_local])
                     if prov_local.numel() > 0 else fact_local)
        if all_local.numel() == 0:
            return None

        if desc.exist_body1_head_arg == 0:
            exist_entities = torch.unique(all_local // E)
        else:
            exist_entities = torch.unique(all_local % E)

        if desc.exist_filter_body0_arg == 0:
            filter_vals = A_idx[0]
        else:
            filter_vals = A_idx[1]

        pos = torch.searchsorted(exist_entities, filter_vals)
        n_e = exist_entities.shape[0]
        valid = pos < n_e
        clamped = torch.clamp(pos, 0, max(n_e - 1, 0))
        in_exist = valid & (exist_entities[clamped] == filter_vals)

        if not in_exist.any():
            return None

        new_rows = A_idx[0][in_exist]
        new_cols = A_idx[1][in_exist]

        if desc.transpose_result:
            new_rows, new_cols = new_cols, new_rows

        hashes = new_rows * E + new_cols
        unique_h = torch.unique(hashes)
        rows = unique_h // E
        cols = unique_h % E
        vals = torch.ones(rows.shape[0], dtype=torch.float32)
        result = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.MATMUL:
        mat_A = base_mats.get(desc.pred_A)
        mat_B = all_mats.get(desc.pred_B)
        if mat_A is None or mat_B is None:
            return None
        if desc.reflexive_body0:
            mat_A = _get_diagonal_mask(mat_A, E)
        if desc.reflexive_body1:
            mat_B = _get_diagonal_mask(mat_B, E)
        if _sparse_nnz(mat_A) == 0 or _sparse_nnz(mat_B) == 0:
            return None

        A = mat_A
        B = mat_B
        if desc.transpose_A:
            if (not desc.reflexive_body0
                    and base_mats_T is not None
                    and desc.pred_A in base_mats_T):
                A = base_mats_T[desc.pred_A]
            else:
                A = _transpose_csr(A, E)
        if desc.transpose_B:
            if (not desc.reflexive_body1
                    and all_mats_T is not None
                    and desc.pred_B in all_mats_T):
                B = all_mats_T[desc.pred_B]
            else:
                B = _transpose_csr(B, E)

        result = torch.sparse.mm(A, B)
        result = _sparse_to_bool_csr(result)

        if desc.transpose_result:
            result = _transpose_csr(result, E)

        return result if _sparse_nnz(result) > 0 else None

    return None


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════


def run_forward_chaining_spmm(
    compiled_rules: List,
    facts_idx: Tensor,
    num_entities: int,
    num_predicates: int,
    depth: int = 10,
    device: str = "cpu",
    *,
    verbose: bool = True,
) -> Tuple[Tensor, int]:
    """Compute the FC closure via incremental SpMM with delta tracking.

    Same I/O contract as :func:`grounder.fc.fc.run_forward_chaining`
    so callers can swap implementations with no API change. Falls
    back is the responsibility of the caller — rules whose
    ``classify_rule`` result is ``UNSUPPORTED`` (or any rule with
    ``num_body >= 3``) are skipped here; combine with the staged
    FCDynamic if you need to cover those.

    Args:
        compiled_rules: list of ``RulePattern`` (must expose
            ``head_pred_idx``, ``head_var0``, ``head_var1``,
            ``num_body``, and ``body_patterns[i]['arg0_binding'/...]``).
        facts_idx: ``[F, 3]`` tensor of fact triples (pred, subj, obj).
        num_entities: total entity count.
        num_predicates: total predicate count.
        depth: hard cap on FC iterations (the loop also exits early
            when no new atoms are derived).
        device: target device for the returned hash tensor. Sparse
            ops run on the device of the input matrices; for very
            large closures CPU is usually the right choice.

    Returns:
        ``(sorted_hashes, n_provable)`` — the same format as
        :func:`grounder.fc.fc.run_forward_chaining`.
    """
    three_body = [
        i for i, cr in enumerate(compiled_rules)
        if getattr(cr, "num_body", 0) >= 3
    ]
    if three_body:
        raise ValueError(
            f"SpMM FC does not support num_body >= 3 (rules at "
            f"indices {three_body}). Fall back to the staged "
            f"FCDynamic for these rule sets.")

    t0 = time.time()
    E = num_entities
    P = num_predicates
    E2 = E * E

    facts = facts_idx.to(device)
    fact_preds = facts[:, 0]
    fact_subjs = facts[:, 1]
    fact_objs = facts[:, 2]
    num_facts = int(facts.shape[0])
    if num_facts > 0:
        fact_hashes = (fact_preds * E2 + fact_subjs * E
                       + fact_objs).sort().values
    else:
        fact_hashes = torch.zeros(0, dtype=torch.long, device=device)

    base_mats = build_pred_sparse_matrices(
        fact_preds, fact_subjs, fact_objs, E, P)

    rule_descs = [classify_rule(cr) for cr in compiled_rules]
    n_unsupported = sum(1 for d in rule_descs if d.op == SpMMOp.UNSUPPORTED)
    if n_unsupported > 0 and verbose:
        print(f"    [SpMM] {n_unsupported}/{len(rule_descs)} rules "
              f"UNSUPPORTED (skipped — fall back to FCDynamic if needed)")

    body1_preds: set = set()
    needs_delta_T: set = set()
    for desc in rule_descs:
        if desc.op in (SpMMOp.MATMUL, SpMMOp.ELEM_AND):
            body1_preds.add(desc.pred_B)
        if desc.op == SpMMOp.MATMUL and desc.transpose_B:
            needs_delta_T.add(desc.pred_B)

    base_mats_T: Dict[int, Tensor] = {}
    for desc in rule_descs:
        if desc.op == SpMMOp.TRANSPOSE or (
                desc.op == SpMMOp.MATMUL and desc.transpose_A):
            p = desc.pred_A
            if p not in base_mats_T and p in base_mats:
                base_mats_T[p] = _transpose_csr(base_mats[p], E)
        if desc.op == SpMMOp.MATMUL and desc.transpose_B:
            p = desc.pred_B
            if p not in base_mats_T and p in base_mats:
                base_mats_T[p] = _transpose_csr(base_mats[p], E)

    pred_boundaries = torch.arange(P + 1, dtype=torch.long) * E2

    provable_hashes = torch.zeros(0, dtype=torch.long)
    delta_mats: Dict[int, Tensor] = {}
    delta_mats_T: Dict[int, Tensor] = {}

    for step in range(depth):
        new_hashes_list: List[Tensor] = []
        n_ph = provable_hashes.shape[0]

        for desc in rule_descs:
            if desc.op == SpMMOp.UNSUPPORTED:
                continue

            # COPY / TRANSPOSE produce identical output every step;
            # apply them only at step 0.
            if desc.op in (SpMMOp.COPY, SpMMOp.TRANSPOSE) and step > 0:
                continue

            if step == 0:
                result = apply_spmm_rule(
                    desc, base_mats, base_mats, E,
                    fact_hashes, num_facts, provable_hashes,
                    base_mats_T=base_mats_T, all_mats_T=base_mats_T)
            else:
                if desc.op in (SpMMOp.MATMUL, SpMMOp.ELEM_AND):
                    result = apply_spmm_rule(
                        desc, delta_mats, base_mats, E,
                        fact_hashes, num_facts, provable_hashes,
                        base_mats_T=base_mats_T, all_mats_T=delta_mats_T)
                elif desc.op in (SpMMOp.CASE_A, SpMMOp.EXIST_AND):
                    result = apply_spmm_rule(
                        desc, base_mats, base_mats, E,
                        fact_hashes, num_facts, provable_hashes,
                        base_mats_T=base_mats_T)
                else:
                    continue

            if result is None:
                continue

            h = _csr_to_hashes(result, desc.head_pred, E, E2)
            if h is None:
                continue

            if n_ph > 0:
                pos = torch.searchsorted(provable_hashes, h)
                valid = pos < n_ph
                clamped = torch.clamp(pos, 0, max(n_ph - 1, 0))
                already = valid & (provable_hashes[clamped] == h)
                h = h[~already]
                if h.numel() == 0:
                    continue

            new_hashes_list.append(h)

        if not new_hashes_list:
            break

        all_new = torch.unique(torch.cat(new_hashes_list))

        if provable_hashes.numel() > 0:
            n_ph = provable_hashes.shape[0]
            pos = torch.searchsorted(provable_hashes, all_new)
            valid = pos < n_ph
            clamped = torch.clamp(pos, 0, max(n_ph - 1, 0))
            already = valid & (provable_hashes[clamped] == all_new)
            added = all_new[~already]
        else:
            added = all_new

        if added.numel() == 0:
            break

        provable_hashes = _sorted_merge(provable_hashes, added)

        # Build delta sparse matrices for body1 predicates only (the
        # only ones the next step actually reads from).
        delta_mats.clear()
        delta_mats_T.clear()
        starts = torch.searchsorted(added, pred_boundaries[:-1])
        ends = torch.searchsorted(added, pred_boundaries[1:])

        for p in body1_preds:
            s = int(starts[p].item())
            e = int(ends[p].item())
            if s >= e:
                continue
            new_local = added[s:e] - p * E2
            if p in needs_delta_T:
                delta_mats[p], delta_mats_T[p] = _build_csr_and_transpose(
                    new_local, E)
            else:
                delta_mats[p] = _build_csr_from_local_hashes(new_local, E)

        if verbose:
            print(f"    [SpMM] step {step}: +{added.numel()} atoms "
                  f"(total {provable_hashes.numel()})")

    n_provable = int(provable_hashes.numel())
    elapsed = time.time() - t0
    if verbose:
        print(f"  [SpMM] FC complete: {n_provable} provable atoms "
              f"({elapsed:.2f}s)")

    if n_provable > 0:
        return provable_hashes.to(device), n_provable
    return torch.zeros(1, dtype=torch.long, device=device), 0
