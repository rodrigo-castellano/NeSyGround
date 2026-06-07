"""Sparse-matrix helpers for SpMM forward chaining.

All functions operate on torch sparse CSR / COO tensors of shape ``(E, E)``.
Every function honours ``device`` — outputs live on the same device as
the input (or the explicit ``device`` arg for build helpers).

Hot-path note: ``csr_local_hashes`` reads CSR ``crow_indices`` /
``col_indices`` directly without round-tripping through COO. This is
substantially faster than ``mat.to_sparse_coo().coalesce().indices()``
on large CSR matrices, because CSR is already coalesced and sorted by
``(row, col)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "build_pred_sparse_matrices",
    "sparse_to_bool_csr",
    "sparse_nnz",
    "transpose_csr",
    "get_diagonal_mask",
    "csr_local_hashes",
    "csr_to_global_hashes",
    "build_csr_from_local_hashes",
    "build_csr_and_transpose",
    "sorted_merge",
    "unique_sorted",
    "empty_csr",
    "safe_sparse_mm",
]


# Default row-chunk size for the OOM-resistant CUDA fallback. Tuned to
# keep cusparseSpGEMM workspace below a few GB on a 24 GB GPU even for
# moderately dense outputs.
_SPMM_DEFAULT_CHUNK = 2048


def safe_sparse_mm(
    A: Tensor, B: Tensor,
    *, chunk_size: int = _SPMM_DEFAULT_CHUNK,
) -> Tensor:
    """Sparse-sparse matmul with CUDA-OOM-resistant fallback.

    First tries the libtorch ``torch.sparse.mm(A, B)`` path directly.
    On CUDA workspace failure (``insufficient resources`` or OOM),
    falls back to row-chunked matmul: split A into chunks of
    ``chunk_size`` rows, compute each ``A_chunk @ B``, and reassemble.
    Bounds workspace per call to ``O(chunk_size × density(B))`` which
    avoids the cusparseSpGEMM workspace blowup on very wide outputs.

    Last resort: if chunked still OOMs, do the matmul on CPU and copy
    the result back to A's device.
    """
    try:
        return torch.sparse.mm(A, B)
    except RuntimeError as e:
        msg = str(e).lower()
        if A.device.type != "cuda":
            raise
        if ("insufficient resources" not in msg
                and "out of memory" not in msg):
            raise
        torch.cuda.empty_cache()

    # ── Try 1: row-chunked matmul on GPU ──
    try:
        return _spmm_row_chunked(A, B, chunk_size=chunk_size)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("insufficient resources" not in msg
                and "out of memory" not in msg):
            raise
        torch.cuda.empty_cache()

    # ── Try 2: do this single matmul on CPU and ship back ──
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    out_cpu = torch.sparse.mm(A_cpu, B_cpu)
    return out_cpu.to(A.device)


def _spmm_row_chunked(A: Tensor, B: Tensor, *, chunk_size: int) -> Tensor:
    """Row-chunked sparse @ sparse: split A's rows into chunks of size
    ``chunk_size``, do each ``A_chunk @ B`` independently, reassemble.

    Output shape is ``(A.shape[0], B.shape[1])`` and the device matches
    ``A``. Empty result returns an empty CSR with the right shape.
    """
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    crow = A.crow_indices()
    col = A.col_indices()
    vals = A.values()

    all_rows: List[Tensor] = []
    all_cols: List[Tensor] = []
    all_vals: List[Tensor] = []

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        cs = int(crow[start].item())
        ce = int(crow[end].item())
        if ce <= cs:
            continue
        sub_crow = crow[start:end + 1] - cs
        sub_col = col[cs:ce]
        sub_vals = vals[cs:ce]
        A_chunk = torch.sparse_csr_tensor(
            sub_crow, sub_col, sub_vals,
            size=(end - start, K))
        out = torch.sparse.mm(A_chunk, B)
        out_coo = out.to_sparse_coo().coalesce()
        idx = out_coo.indices()
        if idx.shape[1] == 0:
            continue
        all_rows.append(idx[0] + start)
        all_cols.append(idx[1])
        all_vals.append(out_coo.values())

    if not all_rows:
        crow0 = torch.zeros(M + 1, dtype=torch.long, device=A.device)
        col0 = torch.zeros(0, dtype=torch.long, device=A.device)
        val0 = torch.zeros(0, dtype=torch.float32, device=A.device)
        return torch.sparse_csr_tensor(crow0, col0, val0, size=(M, N))
    rows = torch.cat(all_rows)
    cols = torch.cat(all_cols)
    vals = torch.cat(all_vals)
    coo = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(M, N))
    return coo.to_sparse_csr()


def empty_csr(E: int, *, device: torch.device) -> Tensor:
    crow = torch.zeros(E + 1, dtype=torch.long, device=device)
    cols = torch.zeros(0, dtype=torch.long, device=device)
    vals = torch.zeros(0, dtype=torch.float32, device=device)
    return torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))


def build_pred_sparse_matrices(
    fact_preds: Tensor,
    fact_subjs: Tensor,
    fact_objs: Tensor,
    E: int,
    P: int,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, Tensor]:
    """Build per-predicate ``[E, E]`` sparse boolean CSR matrices."""
    if device is None:
        device = fact_preds.device
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
        vals = torch.ones(rows.shape[0], dtype=torch.float32, device=device)
        coo = torch.sparse_coo_tensor(
            torch.stack([rows, cols]).to(device), vals,
            size=(E, E), device=device)
        mats[p] = coo.to_sparse_csr()
    return mats


def sparse_to_bool_csr(mat: Tensor) -> Tensor:
    """Clamp a sparse CSR matrix's values to 0/1 (boolean)."""
    vals = mat.values()
    new_vals = (vals > 0).to(vals.dtype)
    return torch.sparse_csr_tensor(
        mat.crow_indices(), mat.col_indices(), new_vals, size=mat.shape)


def sparse_nnz(mat: Tensor) -> int:
    return int(mat.values().shape[0])


def transpose_csr(mat: Tensor, E: int) -> Tensor:
    coo = mat.to_sparse_coo().coalesce()
    idx = coo.indices()
    device = mat.device
    if idx.shape[1] == 0:
        return empty_csr(E, device=device)
    new_idx = torch.stack([idx[1], idx[0]])
    vals = torch.ones(new_idx.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        new_idx, vals, size=(E, E)).to_sparse_csr()


def get_diagonal_mask(mat: Tensor, E: int) -> Tensor:
    """Filter sparse matrix to only diagonal entries (row == col)."""
    coo = mat.to_sparse_coo().coalesce()
    indices = coo.indices()
    device = mat.device
    if indices.shape[1] == 0:
        return mat
    diag_mask = indices[0] == indices[1]
    if not diag_mask.any():
        return empty_csr(E, device=device)
    new_indices = indices[:, diag_mask]
    new_vals = torch.ones(
        new_indices.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        new_indices, new_vals, size=(E, E)).to_sparse_csr()


def csr_local_hashes(mat: Tensor, E: int) -> Tensor:
    """Read sorted local hashes (``row*E + col``) directly from a CSR.

    Equivalent to ``mat.to_sparse_coo().coalesce().indices()`` followed
    by ``rows*E + cols``, but skips the COO round-trip and coalesce.
    Output is in CSR's natural row-major sort order, suitable for
    ``unique_sorted`` / ``sorted_merge``.
    """
    crow = mat.crow_indices()
    col = mat.col_indices()
    nnz = col.shape[0]
    if nnz == 0:
        return torch.zeros(0, dtype=torch.long, device=mat.device)
    row_counts = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(
        torch.arange(E, dtype=torch.long, device=mat.device), row_counts)
    return rows * E + col


def csr_to_global_hashes(
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
        torch.arange(E, dtype=torch.long, device=mat.device), row_counts)
    return pred * E2 + rows * E + col


def sorted_merge(a: Tensor, b: Tensor) -> Tensor:
    """Merge two sorted 1-D tensors into a single sorted tensor.

    Returns the multiset union (does not deduplicate). Output is
    sorted ascending. Implementation is ``cat + sort`` which is
    O(N log N) but predictable.

    On CUDA, falls back to CPU for the sort if the device runs out
    of memory: the sort temporary buffer is ~3× input size and at
    deep iterations of large closures (fb15k237 step 2+) this can
    blow past available VRAM even when the *result* would fit.
    """
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    try:
        return torch.cat([a, b]).sort().values
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if a.device.type != "cuda":
            raise
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        merged_cpu = torch.cat([a_cpu, b_cpu]).sort().values
        return merged_cpu.to(a.device)


def unique_sorted(t: Tensor) -> Tensor:
    """Drop consecutive duplicates from a sorted 1-D tensor."""
    if t.numel() <= 1:
        return t
    mask = torch.empty(t.shape[0], dtype=torch.bool, device=t.device)
    mask[0] = True
    mask[1:] = t[1:] != t[:-1]
    return t[mask]


def build_csr_from_local_hashes(local_hashes: Tensor, E: int) -> Tensor:
    """Build sparse CSR from sorted local hashes (``row*E + col``)."""
    unique_h = unique_sorted(local_hashes)
    n = unique_h.shape[0]
    device = local_hashes.device
    if n == 0:
        return empty_csr(E, device=device)
    rows = unique_h // E
    cols = unique_h % E
    vals = torch.ones(n, dtype=torch.float32, device=device)
    crow = torch.zeros(E + 1, dtype=torch.long, device=device)
    crow.scatter_add_(0, rows + 1,
                      torch.ones(n, dtype=torch.long, device=device))
    crow = torch.cumsum(crow, 0)
    return torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))


def build_csr_and_transpose(
    local_hashes: Tensor, E: int,
) -> Tuple[Tensor, Tensor]:
    """Build CSR and its transpose from sorted local hashes."""
    unique_h = unique_sorted(local_hashes)
    n = unique_h.shape[0]
    device = local_hashes.device
    if n == 0:
        empty = empty_csr(E, device=device)
        return empty, empty_csr(E, device=device)

    rows = unique_h // E
    cols = unique_h % E
    vals = torch.ones(n, dtype=torch.float32, device=device)

    crow = torch.zeros(E + 1, dtype=torch.long, device=device)
    crow.scatter_add_(0, rows + 1,
                      torch.ones(n, dtype=torch.long, device=device))
    crow = torch.cumsum(crow, 0)
    csr = torch.sparse_csr_tensor(crow, cols, vals, size=(E, E))

    sort_idx = torch.argsort(cols, stable=True)
    t_col_indices = rows[sort_idx]
    t_row_keys = cols[sort_idx]
    t_vals = torch.ones(n, dtype=torch.float32, device=device)
    t_crow = torch.zeros(E + 1, dtype=torch.long, device=device)
    t_crow.scatter_add_(0, t_row_keys + 1,
                        torch.ones(n, dtype=torch.long, device=device))
    t_crow = torch.cumsum(t_crow, 0)
    csr_t = torch.sparse_csr_tensor(
        t_crow, t_col_indices, t_vals, size=(E, E))
    return csr, csr_t
