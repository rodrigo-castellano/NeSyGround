"""BlockSparseFactIndex — dense [P,E,K] blocks for compile-friendly pbc.

Materializes the inverted tables as dense ``[P, E, K]`` blocks so enumerate and
exists are pure gathers (no offset arithmetic, no data-dependent slicing) — the
shape the compiled / CUDA-graph dense path needs. Falls back to the parent
``InvertedFactIndex`` offset tables when the dense blocks would exceed
``max_memory_mb``.

BIT-EXACT RECIPES (fingerprint-enforced):
  * K = min(max per-slot fact count, max_facts_per_query)
  * dense fill: stable argsort of composite keys -> within-group position ->
    scatter values at ``key*K + pos`` for positions < K
  * exists: gather block + count, match arg2 against block AND in-range
The ``arange(K)`` constant is cached for eager but re-materialized under
``torch.compile`` (a persistent buffer aliases across CUDA-graph replays).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from grounder._build.data.fact_index.inverted import InvertedFactIndex


class BlockSparseFactIndex(InvertedFactIndex):
    """Dense [P,E,K] enumerate + exists, with an offset-table fallback."""

    def __init__(
        self,
        facts_idx: Tensor,
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
        **_: object,
    ) -> None:
        super().__init__(facts_idx, constant_no=constant_no,
                         predicate_no=predicate_no, padding_idx=padding_idx,
                         device=device, max_facts_per_query=max_facts_per_query)
        self._build_dense(device, max_memory_mb)

    def _build_dense(self, device: torch.device, max_memory_mb: int) -> None:
        facts = self.facts_idx
        P, E, M = self._num_predicates, self._num_entities, self._max_facts_per_query
        preds, subjs, objs = facts[:, 0], facts[:, 1], facts[:, 2]

        _, ps_cnt = torch.unique(preds * E + subjs, return_counts=True)
        _, po_cnt = torch.unique(preds * E + objs, return_counts=True)
        K = min(max(int(ps_cnt.max().item()) if ps_cnt.numel() else 0,
                    int(po_cnt.max().item()) if po_cnt.numel() else 0), M)

        # Two [P,E,K] int64 block tensors -> 2*P*E*K*8 bytes. Fall back if over budget.
        if 2 * P * E * K * 8 / (1024 ** 2) > max_memory_mb:
            self._use_dense = False
            return

        self._use_dense = True
        self._K = K
        # Cached eager arange(K); the compiled path uses a fresh one (see _k_arange).
        self._k_arange_const = torch.arange(K, device=device)

        def _fill(group_a: Tensor, group_b: Tensor,
                  values: Tensor) -> Tuple[Tensor, Tensor]:
            comp = group_a * E + group_b
            order = torch.argsort(comp, stable=True)
            s_keys, s_vals = comp[order], values[order]
            _, _, counts = torch.unique_consecutive(
                s_keys, return_inverse=True, return_counts=True)
            group_starts = torch.zeros(counts.shape[0] + 1, dtype=torch.long, device=facts.device)
            group_starts[1:] = counts.cumsum(0)
            group_ids = torch.arange(counts.shape[0], device=facts.device).repeat_interleave(counts)
            within = (torch.arange(s_keys.shape[0], dtype=torch.long, device=facts.device)
                      - group_starts[group_ids])
            keep = within < K
            blocks = torch.zeros(P * E, K, dtype=torch.long, device=facts.device)
            blocks.view(-1)[s_keys[keep] * K + within[keep]] = s_vals[keep]
            count_arr = torch.zeros(P * E, dtype=torch.long, device=facts.device)
            count_arr[s_keys[group_starts[:-1]]] = counts.clamp(max=K)
            return blocks.view(P, E, K), count_arr.view(P, E)

        ps_blocks, ps_counts = _fill(preds, subjs, objs)
        po_blocks, po_counts = _fill(preds, objs, subjs)
        self.register_buffer("_ps_blocks", ps_blocks.to(device))
        self.register_buffer("_ps_counts", ps_counts.to(device))
        self.register_buffer("_po_blocks", po_blocks.to(device))
        self.register_buffer("_po_counts", po_counts.to(device))

    @property
    def max_fact_pairs(self) -> int:
        return self._K if self._use_dense else self._max_facts_per_query

    def _k_arange(self, device: torch.device) -> Tensor:
        """``arange(K)``: cached eager, fresh under compile (CUDA-graph safety).

        A persistent buffer captured inside a CUDA graph aliases across replays
        and trips "accessing tensor output of CUDAGraphs that has been
        overwritten", so the compiled path must materialize it inside the graph.
        """
        if torch.compiler.is_compiling():
            return torch.arange(self._K, device=device)
        return self._k_arange_const

    def enumerate(self, preds: Tensor, bound_args: Tensor,
                  direction: Tensor) -> Tuple[Tensor, Tensor]:
        if not self._use_dense:
            return super().enumerate(preds, bound_args, direction)
        P, E = self._num_predicates, self._num_entities
        valid_input = (preds < P) & (bound_args < E)
        sp = preds.clamp(max=P - 1)
        sa = bound_args.clamp(max=E - 1)
        is_obj = (direction == 0)
        cands = torch.where(is_obj.unsqueeze(1),
                            self._ps_blocks[sp, sa], self._po_blocks[sp, sa])
        counts = torch.where(is_obj, self._ps_counts[sp, sa], self._po_counts[sp, sa])
        counts = torch.where(valid_input, counts, torch.zeros_like(counts))
        valid = self._k_arange(preds.device).unsqueeze(0) < counts.unsqueeze(1)
        return cands, valid

    def exists(self, atoms: Tensor) -> Tensor:
        """``[N,3] -> [N]`` bool. Chunked in eager (T*K can exceed 16 GiB);
        single inline gather under compile (Python loops break graph capture)."""
        if not self._use_dense:
            return super().exists(atoms)
        K, P, E = self._K, self._num_predicates, self._num_entities
        if torch.compiler.is_compiling():
            return self._exists_chunk(atoms, P, E)
        N = atoms.shape[0]
        chunk = max(1, 256_000_000 // max(K, 1))   # 256 M booleans / chunk
        if N <= chunk:
            return self._exists_chunk(atoms, P, E)
        out = torch.empty(N, dtype=torch.bool, device=atoms.device)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            out[start:end] = self._exists_chunk(atoms[start:end], P, E)
        return out

    def _exists_chunk(self, atoms: Tensor, P: int, E: int) -> Tensor:
        preds, subjs = atoms[:, 0], atoms[:, 1]
        in_range = (preds < P) & (subjs < E)
        sp, ss = preds.clamp(max=P - 1), subjs.clamp(max=E - 1)
        block = self._ps_blocks[sp, ss]
        counts = self._ps_counts[sp, ss]
        pos = self._k_arange(atoms.device).unsqueeze(0)
        matched = ((block == atoms[:, 2].unsqueeze(1)) & (pos < counts.unsqueeze(1))).any(1)
        return matched & in_range

    def __repr__(self) -> str:
        d = f"dense=True, K={self._K}" if self._use_dense else "dense=False"
        return (f"BlockSparseFactIndex(F={self.num_facts}, "
                f"P={self._num_predicates}, E={self._num_entities}, {d})")


__all__ = ["BlockSparseFactIndex"]
