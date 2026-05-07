"""Per-resolution init pass for ``BCGrounder``.

Picks one of three resolution strategies and computes the
resolution-specific shapes (``K``, ``S``, ``C``, vars-per-rule, …)
plus the per-step search-filter buffers and compile flags. Mutates
the passed ``grounder`` in place.

Three resolutions:

* ``sld`` / ``rtf`` — most-general-unifier through
  :func:`grounder.resolution.mgu.init_mgu`. Sets ``K``, ``K_f`` cap,
  ``max_vars_per_rule``, ``C``, ``_max_fact_pairs_body``.
* ``enum`` — dense static-shape enumeration through
  :func:`grounder.resolution.enum.init_enum`. Registers buffers,
  caches ``_enum_ri`` / ``_P`` / ``_E`` / ``K_r`` / ``G_r``, plus
  optional all-anchors variant→original mapping.

Then applies common compile-mode logic (per-step inner vs outer
per-batch compile) and (when ``filter_mode='fp_global'``) builds the
forward-chaining global closure.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def init_resolution(grounder, **kwargs) -> None:
    """Call resolution module's init, apply results, set up compilation.

    Shared layout (G, S, C) is already set by ``BCGrounder.__init__``.
    This function computes resolution-specific params:
      SLD/RTF: K (= K_f + K_r or K_f * K_r), K_f capping, vars_per_rule
      Enum: K_enum, G_use, K_per_fv, enum buffers
    """
    if grounder.resolution in ("sld", "rtf"):
        from grounder.resolution.mgu import init_mgu
        cfg = init_mgu(
            resolution=grounder.resolution,
            K_f=grounder.kb.K_f, K_r=grounder.kb.K_r,
            rule_index=grounder.kb.rule_index,
            max_total_groundings=kwargs["max_total_groundings"],
            K_MAX=kwargs["K_MAX"],
            max_derived_per_state=kwargs["max_derived_per_state"],
            max_states=kwargs["max_states"],
            max_groundings_per_rule=kwargs["max_groundings_per_rule"],
        )
        grounder.K = cfg["K"]
        # init_mgu may cap K_f and recompute S/C — override shared values.
        grounder.S = cfg["S"]
        grounder.kb.K_f = cfg["K_f"]
        grounder.max_vars_per_rule = cfg["max_vars_per_rule"]
        grounder.C = cfg["C"]
        grounder._max_fact_pairs_body = cfg["max_fact_pairs_body"]

    elif grounder.resolution == "enum":
        from grounder.resolution.enum import init_enum
        meta = init_enum(
            rule_index=grounder.kb.rule_index,
            fact_index=grounder.kb.fact_index,
            facts_idx=grounder.kb.fact_index.facts_idx,
            constant_no=grounder.kb.constant_no,
            num_rules=grounder.kb.num_rules, M=grounder.kb.M,
            width=grounder.width,
            max_groundings_per_query=kwargs["max_groundings_per_query"],
            max_total_groundings=kwargs["max_total_groundings"],
            max_states=kwargs["max_states"],
            device=grounder.kb.device_,
            cartesian_product=grounder._cartesian_product,
            all_anchors=grounder._all_anchors,
            flat_intermediate=grounder._flat_intermediate_flag,
        )
        for name, tensor in meta["buffers"].items():
            grounder.register_buffer(name, tensor)
        grounder._enum_ri = meta["enum_rule_index"]
        grounder.max_body_atoms = grounder._enum_ri.max_body
        grounder._P, grounder._E = meta["P"], meta["E"]
        grounder.K_r = meta["K_r"]
        grounder.K = meta["K"]
        # init_enum may recompute S/C — override shared values.
        grounder.S = meta["S"]
        # Optional bump to ensure intermediate-step packing keeps every
        # valid child. The previous ``S = max(S, K)`` used the
        # *budget* ``K = min(K_r * max_groundings_per_query,
        # max_total_groundings)`` — which on small KBs is much
        # larger than the actual per-state children count
        # ``K_r * min(K_f, max_groundings_per_query)``. That overshoot
        # exploded the dense ``[B*S, K_r, G_r, M, 3]`` allocation
        # to GB-scale on ablation/countries even though only ~80
        # children per state are actually possible.
        # Bound by the realistic per-state max instead.
        if (grounder._bump_s_to_k and grounder.depth > 1
                and grounder.width is not None and grounder.width > 0):
            K_v = meta.get("K_v", 0) or 0
            K_actual = max(grounder.K_r * K_v, 1)
            grounder.S = max(grounder.S, min(grounder.K, K_actual))
        grounder.C = meta["C"]
        grounder.any_dual = meta["any_dual"]
        grounder.G_r = meta["G_r"]
        grounder._enum_cartesian = meta.get("cartesian_product", False)
        grounder.V = meta.get("V", 1)
        grounder.K_v = meta.get("K_v", 64)
        grounder._fv_any_valid = meta.get("fv_any_valid", None)
        grounder._flat_intermediate = meta.get("flat_intermediate", False)
        # Variant→original rule mapping (for all_anchors). Stored as
        # both a Python list (consumed by per-rule collection) and a
        # long Tensor buffer (consumed by tensor dedup).
        if grounder._all_anchors:
            v2o = []
            for orig_r, blen in enumerate(
                    grounder.kb.rule_index.rule_lens_sorted.tolist()):
                for _ in range(blen):
                    v2o.append(orig_r)
            grounder._variant_to_orig = v2o
            grounder.register_buffer(
                "_variant_to_orig_t",
                torch.tensor(v2o, dtype=torch.long,
                             device=grounder.kb.device_),
            )
        grounder.fc_method = kwargs["fc_method"]
        grounder.fc_depth = kwargs["fc_depth"]
        # unused for enum, but keeps state uniform
        grounder.max_vars_per_rule = 3

    else:
        raise ValueError(f"Unknown resolution: {grounder.resolution}")

    # Compilation. Two modes are supported:
    #
    # * **Per-step inner compile** (existing path, used by ``enum``):
    #   ``step_compiled`` calls ``torch.compile(step_impl,
    #   fullgraph=True, mode=compile_mode)`` once per ``(depth, kind)``
    #   variant. Pairs with the dense static-shape enum path.
    #
    # * **Outer per-batch compile** (new path, used by ``sld`` /
    #   ``rtf``): wraps ``forward_one_batch_inner`` once with
    #   ``torch.compile(..., fullgraph=True, mode=compile_mode)``.
    #   Dynamo traces the entire single-batch forward (init →
    #   step×depth → finalize) as one graph; the chunked-forward
    #   path replays the same captured graph per padded chunk.
    #   Pairs with resolutions whose per-step ``step_impl`` has
    #   path-sensitive control flow that breaks dynamo at the
    #   per-step granularity (notably SLD).
    #
    # Both paths require ``compile_mode`` to be set and run on
    # CUDA; reduce-overhead specifically wants CUDA graphs. The
    # outer path also requires the caller to pass an explicit
    # ``batch_size`` so the captured graph sees a stable shape.
    grounder._compiled = False
    grounder._clone_between_steps = False
    grounder._fn_steps_by_depth: Dict[int, Any] = {}
    grounder._uses_outer_compile = (
        grounder.compile_mode is not None
        and grounder.resolution != "enum"
        and grounder.kb.device_.type == "cuda"
    )
    grounder._compiled_inner: Optional[Any] = None  # lazy on first call
    if (grounder.compile_mode
            and grounder.depth > 1
            and grounder.kb.device_.type == "cuda"
            and not grounder._uses_outer_compile):
        grounder._clone_between_steps = (
            grounder.compile_mode == "reduce-overhead")
        grounder._compiled = True
        grounder._multi_step = True

    # Per-step search filter buffers
    if grounder._step_prune_dead:
        P = grounder.kb.predicate_no + 1
        head_pred_mask = torch.zeros(
            P, dtype=torch.bool, device=grounder.kb.device_)
        head_preds = grounder.kb.rule_index.rules_heads_sorted[:, 0]
        head_pred_mask.scatter_(0, head_preds, True)
        grounder.register_buffer("_step_head_pred_mask", head_pred_mask)

        fi = grounder.kb.fact_index
        if hasattr(fi, '_a0_offsets'):
            a0_lens = fi._a0_offsets[1:] - fi._a0_offsets[:-1]
            a1_lens = fi._a1_offsets[1:] - fi._a1_offsets[:-1]
            grounder.register_buffer("_step_a0_lens", a0_lens)
            grounder.register_buffer("_step_a1_lens", a1_lens)
            grounder._step_key_scale = fi._key_scale
            grounder._step_has_csr = True
        else:
            grounder._step_has_csr = False
        if hasattr(fi, '_p_offsets'):
            p_lens = fi._p_offsets[1:] - fi._p_offsets[:-1]
            grounder.register_buffer("_step_p_lens", p_lens)

    # fp_global set I_D (for fp_global filter)
    grounder._has_fp_global = False
    if grounder.filter_mode == "fp_global":
        from grounder.fc.fp_global import build_fp_global_set
        if grounder.resolution == "enum":
            build_fp_global_set(grounder, grounder.kb.device_)
        else:
            # SLD/RTF: build a temporary RuleIndexEnum for FC patterns
            from grounder.data.rule_index import RuleIndexEnum
            P = grounder.kb.predicate_no + 1
            E = grounder.kb.constant_no + 1
            enum_ri = RuleIndexEnum(
                grounder.kb.rule_index.rules_heads_sorted,
                grounder.kb.rule_index.rules_bodies_sorted,
                grounder.kb.rule_index.rule_lens_sorted,
                constant_no=grounder.kb.constant_no,
                num_predicates=P,
                padding_idx=grounder.kb.padding_idx,
                device=grounder.kb.device_,
            )
            build_fp_global_set(
                grounder, grounder.kb.device_,
                compiled_rules=enum_ri.patterns, P=P, E=E,
            )


__all__ = ["init_resolution"]
