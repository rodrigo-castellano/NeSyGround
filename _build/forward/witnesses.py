"""Per-closure-atom witness table for the fp_global filter.

When ``filter_mode='fp_global'`` and the FC closure is non-empty,
``BCGrounder`` precomputes one ``(rule_id, body_grounding)`` witness
per closure atom so query-time fact matches can be expanded into full
rule-body groundings that SBR can score.

Two free functions:

* :func:`build_witness_table` — one-time CPU-Python pass at init that
  enumerates body-atom bindings via indexed dicts over the augmented
  facts. For industrial-scale closures (~3k atoms, 16 rules, bodies
  up to 6 atoms with a few existentials) the total cost is under a
  second. Registers three dense buffers keyed by position in
  ``fp_global_hashes`` plus a Python-int clamp bound on ``grounder``.
* :func:`inject_witnesses_into_evidence` — query-time tensor op that
  splices the witness body into empty-body fact-match slots
  (``ridx[:, :, 0] == -1``). Pure ``torch.compile(fullgraph=True)``-
  safe ops: searchsorted / indexing / cat / where / expand. No
  in-place writes, no host-device sync.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def build_witness_table(
    grounder, device: torch.device, E: int, compiled_rules=None,
) -> None:
    """Precompute per-closure-atom ``(rule_id, body_grounding)`` witnesses.

    For each atom in the FC closure, find ONE rule whose head unifies
    with the atom and whose body grounds in the (augmented) KB; record
    that rule's index and the specific body-atom instantiation. Stored
    as 3 dense buffers on ``grounder``, keyed by position in
    ``fp_global_hashes``:

      ``fp_global_witness_rule``: ``[N]`` long       — rule index (-1 = base fact)
      ``fp_global_witness_body``: ``[N, M, 3]`` long — body atoms, padded to M
      ``fp_global_witness_bcount``: ``[N]`` long     — per-atom valid count

    At query time, when a fact-match grounding fires (rule_idx == -1,
    see the ``top_ridx == -1`` branch in ``pack_states``), the caller
    binary-searches the query's hash in ``fp_global_hashes`` and
    replaces the empty fact-match body with the stored witness body.
    SBR then computes ``reasoning_score = min(KGE(body atoms))`` just
    as it would for any rule-match grounding — preserving principled
    ranking among multi-valid closure members.
    """
    if compiled_rules is None:
        compiled_rules = getattr(
            grounder._enum_ri, '_original_patterns',
            grounder._enum_ri.patterns)

    N = int(grounder.fp_global_hashes.numel())
    M = int(grounder.kb.M)
    pad = int(grounder.kb.padding_idx)
    c_no = int(grounder.kb.constant_no)
    E2 = int(E) * int(E)

    # Decode sorted closure hashes back to (pred, subj, obj) triples.
    h_cpu = grounder.fp_global_hashes.cpu()
    preds = (h_cpu // E2).tolist()
    rem = (h_cpu % E2)
    subjs = (rem // int(E)).tolist()
    objs = (rem % int(E)).tolist()

    # Build indexed fact lookups over the augmented KB.
    facts_idx_cpu = grounder.kb.fact_index.facts_idx.cpu().tolist()
    by_pred_s: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    by_pred_o: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for row in facts_idx_cpu:
        p, s, o = int(row[0]), int(row[1]), int(row[2])
        by_pred_s[(p, s)].append(o)
        by_pred_o[(p, o)].append(s)

    # Index rules by head predicate.
    rules_by_head: Dict[int, List[Tuple[int, object]]] = defaultdict(list)
    for ri, rp in enumerate(compiled_rules):
        rules_by_head[int(rp.head_pred_idx)].append((ri, rp))

    def find_body_grounding(
        body_patterns: list, bindings: Dict[int, int],
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Return a list of concrete body atoms or None."""
        if not body_patterns:
            return []
        bp = body_patterns[0]
        rest = body_patterns[1:]
        bp_pred = int(bp["pred_idx"])
        a0v = int(bp["arg0_var"])
        a1v = int(bp["arg1_var"])

        def resolve(v: int) -> Optional[int]:
            if v <= c_no:
                return v
            return bindings.get(v)

        a0 = resolve(a0v)
        a1 = resolve(a1v)

        def try_pair(cs: int, co: int):
            new_b = dict(bindings)
            if a0v > c_no:
                new_b[a0v] = cs
            if a1v > c_no:
                new_b[a1v] = co
            tail = find_body_grounding(rest, new_b)
            if tail is None:
                return None
            return [(bp_pred, cs, co)] + tail

        if a0 is not None and a1 is not None:
            if a1 in by_pred_s.get((bp_pred, a0), []):
                r = try_pair(a0, a1)
                if r is not None:
                    return r
            return None
        if a0 is not None:
            for co in by_pred_s.get((bp_pred, a0), []):
                r = try_pair(a0, co)
                if r is not None:
                    return r
            return None
        if a1 is not None:
            for cs in by_pred_o.get((bp_pred, a1), []):
                r = try_pair(cs, a1)
                if r is not None:
                    return r
            return None
        # Both args free: iterate over every fact of this pred.
        for (ps, po_list) in by_pred_s.items():
            if ps[0] != bp_pred:
                continue
            for co in po_list:
                r = try_pair(ps[1], co)
                if r is not None:
                    return r
        return None

    # Allocate outputs on CPU, move to device at the end.
    witness_rule = [-1] * N
    witness_body: List[List[Tuple[int, int, int]]] = [[] for _ in range(N)]

    for i in range(N):
        p, s, o = preds[i], subjs[i], objs[i]
        rules = rules_by_head.get(p, [])
        done = False
        for rule_id, rp in rules:
            bindings: Dict[int, int] = {}
            hv0 = int(rp.head_var0)
            hv1 = int(rp.head_var1)
            # Unify head with the closure atom.
            if hv0 > c_no:
                bindings[hv0] = s
            elif hv0 != s:
                continue
            if hv1 > c_no:
                bindings[hv1] = o
            elif hv1 != o:
                continue
            bg = find_body_grounding(
                list(rp.body_patterns), bindings)
            if bg is not None:
                witness_rule[i] = rule_id
                witness_body[i] = bg
                done = True
                break
        if not done:
            # Pure base fact (no matching rule): witness body is the
            # atom itself, as a 1-atom degenerate grounding. SBR will
            # score it via KGE(atom).
            witness_body[i] = [(p, s, o)]

    # Pack into [N, M, 3] tensors with padding, plus [N] body-atom
    # counts (needed by SBR so body_atom_valid reflects only the
    # real witness atoms, not the padding slots).
    rule_t = torch.tensor(witness_rule, dtype=torch.long)
    body_t = torch.full((N, M, 3), pad, dtype=torch.long)
    bcount_t = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        atoms = witness_body[i][:M]
        bcount_t[i] = len(atoms)
        for j, (bp_pred, bs, bo) in enumerate(atoms):
            body_t[i, j, 0] = bp_pred
            body_t[i, j, 1] = bs
            body_t[i, j, 2] = bo

    grounder.register_buffer(
        "fp_global_witness_rule", rule_t.to(device))
    grounder.register_buffer(
        "fp_global_witness_body", body_t.to(device))
    grounder.register_buffer(
        "fp_global_witness_bcount", bcount_t.to(device))
    grounder._has_fp_global_witnesses = True
    # Static Python-int upper bound for clamp() at query time — keeps
    # the clamp bound a compile-time constant so the injection step
    # stays torch.compile(fullgraph=True)-friendly (no tensor .numel()
    # sync needed in the hot path).
    grounder._fp_global_last_idx = max(N - 1, 0)


def inject_witnesses_into_evidence(
    grounder,
    body: Tensor,      # [B, C, D, M, 3]
    mask: Tensor,      # [B, C]
    ridx: Tensor,      # [B, C, D]
    bcount: Tensor,    # [B, C, D]
    queries: Tensor,   # [B, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Replace empty-body fact-match slots with the stored witness.

    A fact-match slot is identified by ``ridx[:, :, 0] == -1`` (the
    initial-depth fact match emitted by pack_states when the query
    unifies with a fact in the KB). For each such slot we binary-
    search the query's hash in ``fp_global_hashes`` and, if found,
    overwrite the depth-0 body slice with the precomputed witness
    body and tag ``ridx[..., 0]`` with the witness rule index.

    torch.compile(fullgraph=True) safety:
      - All ops are pure tensor ops (searchsorted / indexing / cat /
        where / expand). No in-place writes to sliced views, no
        ``.numel()`` / ``.item()`` CPU sync — the clamp upper bound
        is the pre-computed Python int ``_fp_global_last_idx``.
      - Static shapes: ``body`` / ``ridx`` shapes flow through
        unchanged; witness buffers have fixed shape from init.
    """
    B, C, D, M, _ = body.shape
    E = grounder._E_fp_global  # Python int captured at init
    E2 = E * E

    q_hash = (queries[..., 0].long() * E2
              + queries[..., 1].long() * E
              + queries[..., 2].long())  # [B]

    idx = torch.searchsorted(grounder.fp_global_hashes, q_hash)  # [B]
    idx_clamped = idx.clamp(0, grounder._fp_global_last_idx)
    found = grounder.fp_global_hashes[idx_clamped] == q_hash  # [B]

    wb = grounder.fp_global_witness_body[idx_clamped]   # [B, M, 3]
    wr = grounder.fp_global_witness_rule[idx_clamped]   # [B]
    wc = grounder.fp_global_witness_bcount[idx_clamped]  # [B]

    is_fact_match = (ridx[..., 0] == -1) & mask        # [B, C]
    inject = is_fact_match & found.unsqueeze(-1)       # [B, C]

    # --- Replace body[:, :, 0, :, :] without in-place writes ---
    wb_exp = wb.unsqueeze(1).expand(B, C, M, 3)                 # [B,C,M,3]
    inj_bcm3 = inject.unsqueeze(-1).unsqueeze(-1).expand(
        B, C, M, 3)                                             # [B,C,M,3]
    new_d0 = torch.where(inj_bcm3, wb_exp, body[:, :, 0, :, :])  # [B,C,M,3]
    # Reassemble body via cat along the D axis (avoids in-place slice
    # assignment, which isn't traceable under fullgraph compile).
    if D == 1:
        new_body = new_d0.unsqueeze(2)                          # [B,C,1,M,3]
    else:
        new_body = torch.cat(
            [new_d0.unsqueeze(2), body[:, :, 1:, :, :]], dim=2)  # [B,C,D,M,3]

    # --- Same trick for ridx[:, :, 0] ---
    wr_exp = wr.unsqueeze(1).expand(B, C)                       # [B,C]
    new_r0 = torch.where(inject, wr_exp, ridx[..., 0])          # [B,C]
    if D == 1:
        new_ridx = new_r0.unsqueeze(-1)                         # [B,C,1]
    else:
        new_ridx = torch.cat(
            [new_r0.unsqueeze(-1), ridx[..., 1:]], dim=-1)       # [B,C,D]

    # --- Same trick for bcount[:, :, 0]: inject witness atom count ---
    wc_exp = wc.unsqueeze(1).expand(B, C)                       # [B,C]
    new_c0 = torch.where(inject, wc_exp, bcount[..., 0])        # [B,C]
    if D == 1:
        new_bcount = new_c0.unsqueeze(-1)                       # [B,C,1]
    else:
        new_bcount = torch.cat(
            [new_c0.unsqueeze(-1), bcount[..., 1:]], dim=-1)     # [B,C,D]

    return new_body, new_ridx, new_bcount


__all__ = ["build_witness_table", "inject_witnesses_into_evidence"]
