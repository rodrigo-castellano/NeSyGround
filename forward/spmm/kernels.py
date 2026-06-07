"""Per-rule sparse-matmul application — the inner loop kernel.

``apply_spmm_rule_slots`` evaluates one ``SpMMRuleDesc`` against a list
of body-slot mat dicts (one per body atom), returning the resulting
sparse CSR matrix of head atoms or ``None`` if the rule fires empty.

The slot interface is what makes semi-naive evaluation possible: each
body slot can read from a different mat dict (e.g. ``delta_mats`` vs
``old_mats``), so the iteration strategy can pick which combination of
old / delta sources to fire per step. See ``strategies.py``.

A backwards-compatible 2-source ``apply_spmm_rule(all_mats, base_mats)``
wrapper is also provided.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from grounder.forward.spmm.ops import SpMMOp, SpMMRuleDesc, slots_for_op
from grounder.forward.spmm.matrices import (
    sparse_nnz, sparse_to_bool_csr, transpose_csr, get_diagonal_mask,
    safe_sparse_mm,
)


__all__ = [
    "apply_spmm_rule_slots",
    "apply_spmm_rule",
]


def apply_spmm_rule_slots(
    desc: SpMMRuleDesc,
    slot_mats: List[Dict[int, Tensor]],
    slot_mats_T: List[Optional[Dict[int, Tensor]]],
    E: int,
    fact_hashes: Tensor,
    num_facts: int,
    provable_hashes: Tensor,
) -> Optional[Tensor]:
    """Apply one rule with per-body-slot mat sources.

    For 1-body rules only ``slot_mats[0]`` is consulted; for 2-body
    rules slots 0 and 1; for ``MATMUL3`` slots 0, 1, 2.

    For ``CASE_A`` and ``EXIST_AND``, body1 is an existence check
    against ``fact_hashes ∪ provable_hashes`` — the caller controls
    these tensors per fire (e.g. pass ``empty_fact_hashes`` plus
    ``delta_provable_hashes`` for a "body1 grew" semi-naive fire).
    """
    if desc.op == SpMMOp.UNSUPPORTED:
        return None
    device = slot_mats[0].get(desc.pred_A, fact_hashes).device \
        if slot_mats[0] else fact_hashes.device

    if desc.op == SpMMOp.COPY:
        mat = slot_mats[0].get(desc.pred_A)
        if mat is None:
            return None
        if desc.reflexive_body0:
            mat = get_diagonal_mask(mat, E)
        return mat if sparse_nnz(mat) > 0 else None

    if desc.op == SpMMOp.TRANSPOSE:
        if (not desc.reflexive_body0
                and slot_mats_T[0] is not None
                and desc.pred_A in slot_mats_T[0]):
            mat = slot_mats_T[0][desc.pred_A]
            return mat if sparse_nnz(mat) > 0 else None
        mat = slot_mats[0].get(desc.pred_A)
        if mat is None:
            return None
        if desc.reflexive_body0:
            mat = get_diagonal_mask(mat, E)
        return transpose_csr(mat, E) if sparse_nnz(mat) > 0 else None

    if desc.op == SpMMOp.ELEM_AND:
        mat_A = slot_mats[0].get(desc.pred_A)
        mat_B = slot_mats[1].get(desc.pred_B)
        if mat_A is None or mat_B is None:
            return None
        if desc.reflexive_body0:
            mat_A = get_diagonal_mask(mat_A, E)
        if desc.reflexive_body1:
            mat_B = get_diagonal_mask(mat_B, E)
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
        new_vals = torch.ones(
            new_indices.shape[1], dtype=torch.float32, device=device)
        result = torch.sparse_coo_tensor(
            new_indices, new_vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.CASE_A:
        mat_A = slot_mats[0].get(desc.pred_A)
        if mat_A is None:
            return None
        if desc.reflexive_body0:
            mat_A = get_diagonal_mask(mat_A, E)
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

        if num_facts > 0:
            pos = torch.searchsorted(fact_hashes, query_h)
            valid = pos < num_facts
            clamped = torch.clamp(pos, 0, max(num_facts - 1, 0))
            in_facts = valid & (fact_hashes[clamped] == query_h)
        else:
            in_facts = torch.zeros_like(query_h, dtype=torch.bool)

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
        vals = torch.ones(rows.shape[0], dtype=torch.float32, device=device)
        result = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.EXIST_AND:
        mat_A = slot_mats[0].get(desc.pred_A)
        if mat_A is None:
            return None
        if desc.reflexive_body0:
            mat_A = get_diagonal_mask(mat_A, E)
        A_coo = mat_A.to_sparse_coo().coalesce()
        A_idx = A_coo.indices()
        if A_idx.shape[1] == 0:
            return None

        E2 = E * E
        b_start = desc.pred_B * E2
        b_end = b_start + E2

        if num_facts > 0:
            f_start = torch.searchsorted(fact_hashes, b_start).item()
            f_end = torch.searchsorted(fact_hashes, b_end).item()
            fact_local = fact_hashes[f_start:f_end] - b_start
        else:
            fact_local = torch.zeros(0, dtype=torch.long, device=device)

        if provable_hashes.numel() > 0:
            p_start = torch.searchsorted(provable_hashes, b_start).item()
            p_end = torch.searchsorted(provable_hashes, b_end).item()
            prov_local = provable_hashes[p_start:p_end] - b_start
        else:
            prov_local = torch.zeros(0, dtype=torch.long, device=device)

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
        vals = torch.ones(rows.shape[0], dtype=torch.float32, device=device)
        result = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, size=(E, E))
        return result.to_sparse_csr()

    if desc.op == SpMMOp.MATMUL:
        mat_A = slot_mats[0].get(desc.pred_A)
        mat_B = slot_mats[1].get(desc.pred_B)
        if mat_A is None or mat_B is None:
            return None
        if desc.reflexive_body0:
            mat_A = get_diagonal_mask(mat_A, E)
        if desc.reflexive_body1:
            mat_B = get_diagonal_mask(mat_B, E)
        if sparse_nnz(mat_A) == 0 or sparse_nnz(mat_B) == 0:
            return None

        A = mat_A
        B = mat_B
        if desc.transpose_A:
            if (not desc.reflexive_body0
                    and slot_mats_T[0] is not None
                    and desc.pred_A in slot_mats_T[0]):
                A = slot_mats_T[0][desc.pred_A]
            else:
                A = transpose_csr(A, E)
        if desc.transpose_B:
            if (not desc.reflexive_body1
                    and slot_mats_T[1] is not None
                    and desc.pred_B in slot_mats_T[1]):
                B = slot_mats_T[1][desc.pred_B]
            else:
                B = transpose_csr(B, E)

        result = safe_sparse_mm(A, B)
        result = sparse_to_bool_csr(result)
        if desc.transpose_result:
            result = transpose_csr(result, E)
        return result if sparse_nnz(result) > 0 else None

    if desc.op == SpMMOp.MATMUL3:
        mat_A = slot_mats[0].get(desc.pred_A)
        mat_B = slot_mats[1].get(desc.pred_B)
        mat_C = slot_mats[2].get(desc.pred_C)
        if mat_A is None or mat_B is None or mat_C is None:
            return None
        if desc.reflexive_body0:
            mat_A = get_diagonal_mask(mat_A, E)
        if desc.reflexive_body1:
            mat_B = get_diagonal_mask(mat_B, E)
        if desc.reflexive_body2:
            mat_C = get_diagonal_mask(mat_C, E)
        if (sparse_nnz(mat_A) == 0 or sparse_nnz(mat_B) == 0
                or sparse_nnz(mat_C) == 0):
            return None

        A, B, C = mat_A, mat_B, mat_C
        if desc.transpose_A:
            A = transpose_csr(A, E)
        if desc.transpose_B:
            B = transpose_csr(B, E)
        if desc.transpose_C:
            C = transpose_csr(C, E)

        AB = safe_sparse_mm(A, B)
        AB = sparse_to_bool_csr(AB)
        if sparse_nnz(AB) == 0:
            return None
        result = safe_sparse_mm(AB, C)
        result = sparse_to_bool_csr(result)
        if desc.transpose_result:
            result = transpose_csr(result, E)
        return result if sparse_nnz(result) > 0 else None

    return None


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
    """Legacy 2-source signature: body0 from ``base_mats``, body1+ from ``all_mats``.

    Provided for backward compatibility with code that pre-dates the
    per-slot interface. New code should call ``apply_spmm_rule_slots``.
    """
    n_slots = slots_for_op(desc.op)
    if n_slots == 0:
        return None
    slot_mats = [base_mats] + [all_mats] * (n_slots - 1)
    slot_mats_T = [base_mats_T] + [all_mats_T] * (n_slots - 1)
    return apply_spmm_rule_slots(
        desc, slot_mats, slot_mats_T, E,
        fact_hashes, num_facts, provable_hashes)
