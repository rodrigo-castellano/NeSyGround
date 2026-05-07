"""Forward-chaining global fixed-point closure for ``BCGrounder``.

Used when ``filter_mode='fp_global'`` (the keras-ns global-closure
equivalent). Two free functions:

* :func:`build_fp_global_set` — runs ``fc.run_forward_chaining`` to
  derive the closure set of provable atoms, registers buffers on the
  grounder, and augments the KB so SLD body-matching can hit derived
  atoms as if they were base facts.
* :func:`augment_kb_with_closure` — decodes the closure hash buffer
  into ``[N, 3]`` triples and merges them into ``kb.fact_index``,
  rebuilding the index in place.

Lives under ``fc/`` (not ``bc/``) because the closure itself is
forward chaining. ``BCGrounder`` invokes it once at init when the
filter mode is ``'fp_global'``.
"""
from __future__ import annotations

import torch


def build_fp_global_set(
    grounder, device: torch.device,
    *, compiled_rules=None, P: int = 0, E: int = 0,
) -> None:
    """Compute fp_global closure, register buffers, and augment KB.

    Args:
        grounder: ``BCGrounder`` instance — buffers ``fp_global_hashes``
            and ``num_fp_global`` are registered on it; flags
            ``_has_fp_global`` / ``_P_fp_global`` / ``_E_fp_global`` are
            set as well. When the closure is non-empty, ``grounder.kb``
            is mutated in place by :func:`augment_kb_with_closure`, and
            the per-closure-atom witness table is built via
            ``grounder._build_witness_table``.
        device: target torch device.
        compiled_rules: list of ``RulePattern`` to use for FC. Defaults
            to ``grounder._enum_ri._original_patterns`` (or
            ``patterns`` if missing).
        P: number of predicates. Defaults to ``grounder._P``.
        E: number of entities. Defaults to ``grounder._E``.

    Returns:
        ``None`` — all outputs are wired onto ``grounder``.
    """
    from grounder.fc.fc import run_forward_chaining

    if compiled_rules is None:
        # Use original patterns (not expanded all_anchors variants)
        compiled_rules = getattr(
            grounder._enum_ri, '_original_patterns',
            grounder._enum_ri.patterns)
    if P == 0:
        P = grounder._P
    if E == 0:
        E = grounder._E
    # ``fc_method`` selects the FC engine: ``'spmm'`` (default —
    # sparse-matrix-multiplication closure with hybrid mask/single-fire
    # semi-naive iteration; falls back to staged for any rule with
    # ``num_body >= 4``) or ``'staged'`` (the original ragged-join
    # FCDynamic). The legacy alias ``'join'`` is silently mapped to
    # ``'staged'`` for back-compat.
    method = getattr(grounder, 'fc_method', 'spmm')
    if method == "join":
        method = "staged"
    fc_depth = getattr(grounder, 'fc_depth', 10)
    fp_global_tensor, n_fp_global = run_forward_chaining(
        compiled_rules=compiled_rules,
        facts_idx=grounder.kb.fact_index.facts_idx,
        num_entities=E,
        num_predicates=P,
        depth=fc_depth,
        device=str(device),
        method=method,
    )
    grounder.register_buffer("fp_global_hashes", fp_global_tensor)
    grounder.register_buffer(
        "num_fp_global",
        torch.tensor(n_fp_global, dtype=torch.long, device=device))
    grounder._has_fp_global = n_fp_global > 0
    grounder._P_fp_global = P
    grounder._E_fp_global = E

    # Augment the KB's fact index with the closure atoms.
    #
    # Without this augmentation, the SLD body-matching step
    # enumerates over BASE facts only. Rules whose body atoms are
    # derived (e.g. activate_failover <- is_down, can_failover_to)
    # can never ground their body at query time because is_down /
    # can_failover_to don't appear in facts.txt. FC at init IS
    # able to derive such atoms, so the closure already contains
    # the proof — but without augmenting the fact index, SLD can't
    # find it. Concretely: merge the decoded closure triples into
    # facts_idx and rebuild the fact index so body-atom enumeration
    # naturally hits derived atoms as if they were base.
    #
    # Same fix removes the two structural failure modes for rules
    # with wide base-fact bodies and many existential variables
    # (e.g. shared_defect / coincident_failures with 6 atoms and
    # 4 existentials): SLD enumeration blows the grounding budget
    # before finding the right (X1, X2, F1, F2) assignment, but
    # the augmented KB already contains the rule head as a fact,
    # so a single 1-atom match is sufficient to prove the query
    # via the identity rule pattern the grounder injects.
    if n_fp_global > 0:
        augment_kb_with_closure(
            grounder.kb, device=device, E=E,
            fp_global_hashes=fp_global_tensor)
        # After KB augmentation, precompute one (rule, body-grounding)
        # witness per closure atom so query-time fact matches can be
        # expanded into full rule-body groundings that SBR can score.
        from grounder.fc.witnesses import build_witness_table
        build_witness_table(grounder, device, E, compiled_rules)
    else:
        grounder._has_fp_global_witnesses = False


def augment_kb_with_closure(
    kb, *, device: torch.device, E: int, fp_global_hashes,
) -> None:
    """Decode closure hashes to triples and merge into ``kb.fact_index``.

    Mutates ``kb.fact_index`` in place (replaces the submodule). Filters
    out triples that are already base facts so we don't duplicate rows.

    Args:
        kb: ``KnowledgeBase`` whose ``fact_index`` is augmented in place.
        device: target torch device.
        E: number of entities (used for hash decoding).
        fp_global_hashes: 1-D long tensor of closure-set hashes
            (typically ``grounder.fp_global_hashes``).
    """
    from grounder.data.fact_index import FactIndex

    fi = kb.fact_index
    E2 = int(E) * int(E)

    # Decode sorted hashes -> (pred, subj, obj) rows.
    h = fp_global_hashes
    preds = (h // E2).long()
    rem = h % E2
    subjs = (rem // int(E)).long()
    objs = (rem % int(E)).long()
    closure_triples = torch.stack([preds, subjs, objs], dim=1)  # [N, 3]

    # Filter out triples that are already base facts (membership via
    # the existing fact-index hash table) so we don't duplicate rows.
    already_in_kb = fi.exists(closure_triples)
    new_triples = closure_triples[~already_in_kb]
    if new_triples.numel() == 0:
        return

    # Build augmented facts_idx and rebuild the fact index.
    augmented = torch.cat(
        [fi.facts_idx, new_triples.to(fi.facts_idx.device)],
        dim=0).contiguous()

    # Pick the same subclass type the KB originally used.
    fact_index_type = (
        "block_sparse" if fi.__class__.__name__ == "BlockSparseFactIndex"
        else "inverted" if fi.__class__.__name__ == "InvertedFactIndex"
        else "arg_key"
    )
    kb.fact_index = FactIndex.create(
        augmented,
        type=fact_index_type,
        constant_no=kb.constant_no,
        predicate_no=kb.predicate_no,
        padding_idx=kb.padding_idx,
        device=device,
        pack_base=fi.pack_base,
        max_facts_per_query=fi.max_fact_pairs,
    )
    # K_f depends on per-pattern fact counts; refresh the cached value.
    kb.K_f = kb.fact_index.max_fact_pairs


__all__ = ["build_fp_global_set", "augment_kb_with_closure"]
