"""BCGrounder — unified backward chaining with configurable resolution.

Configuration replaces classes:
  resolution: 'sld' | 'rtf' | 'enum'
  filter:     'fp_batch' | 'fp_global' | 'none'
  depth, width, hooks
  standardization: None | StandardizationConfig

Canonical loop (same code path for all resolutions):
  states = init_states(queries, query_mask)
  for d in range(D):
      states = step(states, d)   # SELECT → RESOLVE → PACK → POSTPROCESS
  return GrounderOutput(state, evidence)

Resolution is the only pluggable phase — _select, _pack, _postprocess are shared.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.kb import KB
from grounder.resolution.primitives import apply_substitutions
from grounder.resolution.standardization import StandardizationConfig

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook
from grounder.bc.common import (
    compact_atoms,
    collect_groundings,
    pack_states,
    prune_ground_facts,
)
from grounder.types import (
    FlatResolvedChildren, GrounderOutput, ProofEvidence, ProofState,
    ResolvedChildren, RuleGroundings, SyncParams,
)
from grounder.filters import check_in_fp_global
from grounder.filters.search import filter_width, filter_prune_dead
from grounder.resolution.sld import resolve_sld
from grounder.resolution.rtf import resolve_rtf
from grounder.resolution.enum import resolve_enum_step


class BCGrounder(nn.Module):
    """Unified backward-chaining grounder BC_{w,d}.

    Configurable with orthogonal choices:
      depth (d):    number of proof steps
      width (w):    max unknown body atoms per grounding (enum only; None=∞)
      resolution:   'sld' | 'rtf' | 'enum'
      filter:       'fp_batch' | 'fp_global' | 'none'
      hooks:        GroundingHook list (post-grounding scoring/filtering)
      fact_hook:    ResolutionFactHook (filters fact candidates during resolution)
      rule_hook:    ResolutionRuleHook (filters rule candidates during resolution)
    """

    def __init__(
        self,
        kb: KB,
        *,
        depth: int = 2,
        width: Optional[int] = 1,
        resolution: str = "enum",
        filter: Optional[str] = None,
        max_total_groundings: int = 64,
        compile_mode: Optional[str] = None,
        hooks: Optional[List] = None,
        fact_hook: Optional[ResolutionFactHook] = None,
        rule_hook: Optional[ResolutionRuleHook] = None,
        # MGU params
        max_goals: Optional[int] = None,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        collect_evidence: bool = True,
        step_prune_dead: bool = False,
        max_groundings_per_rule: Optional[int] = None,
        # Enum params
        max_groundings_per_query: int = 32,
        fc_method: str = "spmm",
        fc_depth: int = 10,
        # Testing/validation enum params (not compile-compatible)
        cartesian_product: bool = False,
        all_anchors: bool = False,
        flat_intermediate: bool = False,
        pack_dedup: bool = True,
        collect_rule_groundings: bool = False,
        w_last_depth: Optional[int] = None,
        collect_mode: str = "terminal",
        # Output variable standardization (for consumers of ungrounded states)
        standardization: Optional[StandardizationConfig] = None,
        # Per-step ground-fact pruning: remove known facts from proof goals
        # between resolution steps. Disabled by default (standard SLD semantics
        # where every resolution step costs 1 depth). Enable for "compressed"
        # depth semantics where ground-fact goals are free.
        prune_facts: bool = False,
        # Bump ``S = max(S, K)`` for ``depth>1, width>0`` to keep every
        # intermediate child. Disabled defaults the per-step allocation
        # to ``[B*S_max, K_r, G_r, M, 3]`` — fits high-fan-out KBs at
        # the cost of truncating at packing. Default True (no-loss).
        bump_s_to_k: bool = True,
        # ``init_state_shape``:
        #   "minimal" (default) — depth-0 state has shape ``[B, 1, ...]``;
        #     only one slot per query is allocated. DpRL-friendly: lets
        #     callers stop after step 0 with the smallest possible
        #     buffer.
        #   "full" — depth-0 state has shape ``[B, S, ...]`` from the
        #     start, with only slot 0 valid. NS-friendly: every step
        #     sees the same shape, so a single compiled graph covers
        #     all depths (no d=0 specialization). Recommended when
        #     ``compile_mode`` is set.
        init_state_shape: str = "minimal",
    ) -> None:
        super().__init__()
        self.kb = kb

        self.depth = depth
        self.width = width
        self.resolution = resolution
        # Default filter is the paper BC_{w,d,u=0} convention for enum:
        # 'fp_batch' (keras ``prune_incomplete_proofs=True`` equivalent).
        # Callers that want u>0 semantics (admit unknown leaves) should
        # pass ``filter='none'`` explicitly along with ``w_last_depth>0``;
        # ``make_bcwd`` does this via its ``u`` parameter.
        # SLD/RTF have no parity story; default 'none'.
        if filter is None:
            filter = "fp_batch" if resolution == "enum" else "none"
        self.filter_mode = filter
        # Compile mode is opt-in (default None = eager). The dense
        # ``enum`` path is the right pairing for ``'reduce-overhead'``
        # (static shapes, CUDA-graph capture); the flat path stays
        # eager because its ``torch.nonzero`` produces dynamic shapes
        # incompatible with reduce-overhead. Multi-grounder sweeps
        # should NOT enable compile globally — torch's CUDA-graph
        # weakref bookkeeping accumulates across grounders.
        self.compile_mode = compile_mode
        self.hooks = hooks or []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self.step_hook = None  # Optional StepHook (nn.Module), set externally
        self.collect_evidence = collect_evidence
        self.prune_facts = prune_facts
        # Enum defaults:
        #   all_anchors=True              — try every body atom as anchor
        #                                    (matches keras's per-i loop;
        #                                    forced for correctness because
        #                                    anchoring only on the first body
        #                                    atom misses bindings keras finds)
        #   cartesian_product=False       — fact-anchored enumeration: candidates
        #                                    come from ``fact_index.enumerate``
        #                                    (the partial atom lookup), not from
        #                                    the full entity domain. This
        #                                    matches keras's
        #                                    ``fact_index._index.get(partial_atom)``
        #                                    and keeps the d=1 body tensor
        #                                    bounded by ``G_r`` (typically
        #                                    ``min(K_f, max_groundings_per_query)``)
        #                                    instead of ``E`` (entity count).
        # Callers may opt back into ``cartesian_product=True`` for full-domain
        # exploration; the count is identical (after filtering) because both
        # admit the same set of valid groundings.
        #
        # ``collect_rule_groundings`` is NOT forced for enum: speed-only
        # callers (test_speed.py) pass ``False`` so the per-step
        # ``_step_compiled`` accumulators stay empty. Forcing True caused
        # the chunked path to leak ~100 MB / chunk on wn18rr (step_body /
        # step_head / step_ridx clones from every chunk's depth=1 step
        # piling up across all 293 chunks until ~24 GB OOM).
        if resolution == "enum":
            if not all_anchors:
                all_anchors = True
        self._cartesian_product = cartesian_product
        self._all_anchors = all_anchors
        self._flat_intermediate_flag = flat_intermediate
        self._pack_dedup = pack_dedup
        self._collect_rule_groundings = collect_rule_groundings
        self._bump_s_to_k = bump_s_to_k
        if init_state_shape not in ("minimal", "full"):
            raise ValueError(
                f"init_state_shape must be 'minimal' or 'full', "
                f"got {init_state_shape!r}")
        self._init_state_shape = init_state_shape
        # Paper BC_{w,d,u} convention: u (= w_last_depth) defaults to 0.
        # All body atoms at the last (= terminal) step must be facts;
        # any rule application with leftover unknown leaves is dropped
        # by terminal collection. Callers that want u>0 (admit unknown
        # leaves; e.g. depth=1 with width>0 to surface single-rule
        # applications as in keras-ns ``prune_incomplete_proofs=False``
        # tests) pass ``w_last_depth=u`` explicitly. ``make_bcwd``
        # exposes this as the ``u`` parameter.
        if w_last_depth is None:
            w_last_depth = 0
        self._w_last_depth = w_last_depth
        self._collect_mode = collect_mode

        # Per-step search filters
        self._step_width = width if resolution in ("sld", "rtf") and width is not None else None

        # prune_dead: only for SLD/RTF
        if step_prune_dead and resolution == "enum":
            import warnings
            warnings.warn(
                "step_prune_dead has no effect with enum resolution "
                "(all body atoms are ground). Ignoring.",
                stacklevel=2,
            )
        self._step_prune_dead = step_prune_dead and resolution in ("sld", "rtf")

        self.standardization_mode = standardization.mode if standardization else None

        # ── Shared layout: G, A, S, C ──
        # Standard symbols (see grounder/CLAUDE.md "Naming Convention").
        M = self.kb.M
        D = depth

        # G (max goals per state): M + (M-1)*D.
        if max_goals is None:
            max_goals = M + (M - 1) * D
        self.max_goals = max(max_goals, M)

        # A (accumulated body capacity: D * M).
        self.A = D * M

        # S (max states per depth step).  Default 256.
        if max_states is None:
            max_states = 256
        self.S = max_states

        # C (collected groundings budget).
        self.C = max_total_groundings

        # Init resolution-specific params + compilation
        self._init_resolution(
            max_states=max_states, K_MAX=K_MAX,
            max_derived_per_state=max_derived_per_state,
            max_total_groundings=max_total_groundings,
            max_groundings_per_rule=max_groundings_per_rule,
            max_groundings_per_query=max_groundings_per_query,
            fc_method=fc_method, fc_depth=fc_depth,
        )

        # Output variable standardization
        self._standardize_fn: Optional[Callable] = None
        if standardization is not None:
            from grounder.resolution.standardization import build_standardize_fn
            self._standardize_fn = build_standardize_fn(standardization, self.kb.device_)

    # ==================================================================
    # Resolution init
    # ==================================================================

    def _init_resolution(self, **kwargs) -> None:
        """Call resolution module's init, apply results, set up compilation.

        Shared layout (G, S, C) is already set by __init__.
        This method computes resolution-specific params:
          SLD/RTF: K (= K_f + K_r or K_f * K_r), K_f capping, vars_per_rule
          Enum: K_enum, G_use, K_per_fv, enum buffers
        """
        if self.resolution in ("sld", "rtf"):
            from grounder.resolution.mgu import init_mgu
            cfg = init_mgu(
                resolution=self.resolution,
                K_f=self.kb.K_f, K_r=self.kb.K_r,
                rule_index=self.kb.rule_index,
                max_total_groundings=kwargs["max_total_groundings"],
                K_MAX=kwargs["K_MAX"],
                max_derived_per_state=kwargs["max_derived_per_state"],
                max_states=kwargs["max_states"],
                max_groundings_per_rule=kwargs["max_groundings_per_rule"],
            )
            self.K = cfg["K"]
            # init_mgu may cap K_f and recompute S/C — override shared values.
            self.S = cfg["S"]
            self.kb.K_f = cfg["K_f"]
            self.max_vars_per_rule = cfg["max_vars_per_rule"]
            self.C = cfg["C"]
            # (C may be overridden by resolution-specific init)
            self._max_fact_pairs_body = cfg["max_fact_pairs_body"]

        elif self.resolution == "enum":
            from grounder.resolution.enum import init_enum
            meta = init_enum(
                rule_index=self.kb.rule_index,
                fact_index=self.kb.fact_index,
                facts_idx=self.kb.fact_index.facts_idx,
                constant_no=self.kb.constant_no,
                num_rules=self.kb.num_rules, M=self.kb.M,
                width=self.width,
                max_groundings_per_query=kwargs["max_groundings_per_query"],
                max_total_groundings=kwargs["max_total_groundings"],
                max_states=kwargs["max_states"],
                device=self.kb.device_,
                cartesian_product=self._cartesian_product,
                all_anchors=self._all_anchors,
                flat_intermediate=self._flat_intermediate_flag,
            )
            for name, tensor in meta["buffers"].items():
                self.register_buffer(name, tensor)
            self._enum_ri = meta["enum_rule_index"]
            self.max_body_atoms = self._enum_ri.max_body
            self._P, self._E = meta["P"], meta["E"]
            self.K_r = meta["K_r"]
            self.K = meta["K"]
            # init_enum may recompute S/C — override shared values.
            self.S = meta["S"]
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
            if (self._bump_s_to_k and self.depth > 1
                    and self.width is not None and self.width > 0):
                K_v = meta.get("K_v", 0) or 0
                K_actual = max(self.K_r * K_v, 1)
                self.S = max(self.S, min(self.K, K_actual))
            self.C = meta["C"]
            # (C may be overridden by resolution-specific init)
            self.any_dual = meta["any_dual"]
            self.G_r = meta["G_r"]
            self._enum_cartesian = meta.get("cartesian_product", False)
            self.V = meta.get("V", 1)
            self.K_v = meta.get("K_v", 64)
            self._fv_any_valid = meta.get("fv_any_valid", None)
            self._flat_intermediate = meta.get("flat_intermediate", False)
            # Variant→original rule mapping (for all_anchors). Stored as
            # both a Python list (consumed by per-rule collection) and a
            # long Tensor buffer (consumed by tensor dedup).
            if self._all_anchors:
                v2o = []
                for orig_r, blen in enumerate(self.kb.rule_index.rule_lens_sorted.tolist()):
                    for _ in range(blen):
                        v2o.append(orig_r)
                self._variant_to_orig = v2o
                self.register_buffer(
                    "_variant_to_orig_t",
                    torch.tensor(v2o, dtype=torch.long, device=self.kb.device_),
                )
            self.fc_method = kwargs["fc_method"]
            self.fc_depth = kwargs["fc_depth"]
            self.max_vars_per_rule = 3  # unused for enum, but keeps state uniform

        else:
            raise ValueError(f"Unknown resolution: {self.resolution}")

        # Compilation. Two modes are supported:
        #
        # * **Per-step inner compile** (existing path, used by ``enum``):
        #   ``_step_compiled`` calls ``torch.compile(self._step_impl,
        #   fullgraph=True, mode=compile_mode)`` once per ``(depth, kind)``
        #   variant. Pairs with the dense static-shape enum path.
        #
        # * **Outer per-batch compile** (new path, used by ``sld`` /
        #   ``rtf``): wraps ``_forward_one_batch_inner`` once with
        #   ``torch.compile(..., fullgraph=True, mode=compile_mode)``.
        #   Dynamo traces the entire single-batch forward (init →
        #   step×depth → finalize) as one graph; the chunked-forward
        #   path replays the same captured graph per padded chunk.
        #   Pairs with resolutions whose per-step ``_step_impl`` has
        #   path-sensitive control flow that breaks dynamo at the
        #   per-step granularity (notably SLD).
        #
        # Both paths require ``compile_mode`` to be set and run on
        # CUDA; reduce-overhead specifically wants CUDA graphs. The
        # outer path also requires the caller to pass an explicit
        # ``batch_size`` so the captured graph sees a stable shape.
        self._compiled = False
        self._clone_between_steps = False
        self._fn_steps_by_depth: Dict[int, Any] = {}
        self._uses_outer_compile = (
            self.compile_mode is not None
            and self.resolution != "enum"
            and self.kb.device_.type == "cuda"
        )
        self._compiled_inner: Optional[Any] = None  # lazy on first call
        if (self.compile_mode
                and self.depth > 1
                and self.kb.device_.type == "cuda"
                and not self._uses_outer_compile):
            self._clone_between_steps = (
                self.compile_mode == "reduce-overhead")
            self._compiled = True
            self._multi_step = True

        # Per-step search filter buffers
        if self._step_prune_dead:
            P = self.kb.predicate_no + 1
            head_pred_mask = torch.zeros(P, dtype=torch.bool, device=self.kb.device_)
            head_preds = self.kb.rule_index.rules_heads_sorted[:, 0]
            head_pred_mask.scatter_(0, head_preds, True)
            self.register_buffer("_step_head_pred_mask", head_pred_mask)

            fi = self.kb.fact_index
            if hasattr(fi, '_a0_offsets'):
                a0_lens = fi._a0_offsets[1:] - fi._a0_offsets[:-1]
                a1_lens = fi._a1_offsets[1:] - fi._a1_offsets[:-1]
                self.register_buffer("_step_a0_lens", a0_lens)
                self.register_buffer("_step_a1_lens", a1_lens)
                self._step_key_scale = fi._key_scale
                self._step_has_csr = True
            else:
                self._step_has_csr = False
            if hasattr(fi, '_p_offsets'):
                p_lens = fi._p_offsets[1:] - fi._p_offsets[:-1]
                self.register_buffer("_step_p_lens", p_lens)

        # fp_global set I_D (for fp_global filter)
        self._has_fp_global = False
        if self.filter_mode == "fp_global":
            if self.resolution == "enum":
                self._build_fp_global_set(self.kb.device_)
            else:
                # SLD/RTF: build a temporary RuleIndexEnum for FC patterns
                from grounder.data.rule_index import RuleIndexEnum
                P = self.kb.predicate_no + 1
                E = self.kb.constant_no + 1
                enum_ri = RuleIndexEnum(
                    self.kb.rule_index.rules_heads_sorted,
                    self.kb.rule_index.rules_bodies_sorted,
                    self.kb.rule_index.rule_lens_sorted,
                    constant_no=self.kb.constant_no,
                    num_predicates=P,
                    padding_idx=self.kb.padding_idx,
                    device=self.kb.device_,
                )
                self._build_fp_global_set(
                    self.kb.device_, compiled_rules=enum_ri.patterns,
                    P=P, E=E,
                )

    def _build_fp_global_set(
        self, device: torch.device,
        compiled_rules=None, P: int = 0, E: int = 0,
    ) -> None:
        from grounder.fc.fc import run_forward_chaining
        if compiled_rules is None:
            # Use original patterns (not expanded all_anchors variants)
            compiled_rules = getattr(
                self._enum_ri, '_original_patterns', self._enum_ri.patterns)
        if P == 0:
            P = self._P
        if E == 0:
            E = self._E
        # ``fc_method`` selects the FC engine: ``'spmm'`` (default —
        # sparse-matrix-multiplication closure with hybrid mask/single-fire
        # semi-naive iteration; falls back to staged for any rule with
        # ``num_body >= 4``) or ``'staged'`` (the original ragged-join
        # FCDynamic). The legacy alias ``'join'`` is silently mapped to
        # ``'staged'`` for back-compat.
        method = getattr(self, 'fc_method', 'spmm')
        if method == "join":
            method = "staged"
        fc_depth = getattr(self, 'fc_depth', 10)
        fp_global_tensor, n_fp_global = run_forward_chaining(
            compiled_rules=compiled_rules,
            facts_idx=self.kb.fact_index.facts_idx,
            num_entities=E,
            num_predicates=P,
            depth=fc_depth,
            device=str(device),
            method=method,
        )
        self.register_buffer("fp_global_hashes", fp_global_tensor)
        self.register_buffer(
            "num_fp_global",
            torch.tensor(n_fp_global, dtype=torch.long, device=device))
        self._has_fp_global = n_fp_global > 0
        self._P_fp_global = P
        self._E_fp_global = E

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
            self._augment_kb_with_closure(device, E)
            # After KB augmentation, precompute one (rule, body-grounding)
            # witness per closure atom so query-time fact matches can be
            # expanded into full rule-body groundings that SBR can score.
            self._build_witness_table(device, E, compiled_rules)
        else:
            self._has_fp_global_witnesses = False

    def _build_witness_table(
        self, device: torch.device, E: int, compiled_rules=None,
    ) -> None:
        """Precompute per-closure-atom (rule_id, body_grounding) witnesses.

        For each atom in the FC closure, find ONE rule whose head unifies
        with the atom and whose body grounds in the (augmented) KB; record
        that rule's index and the specific body-atom instantiation. Stored
        as 2 dense buffers keyed by position in ``fp_global_hashes``:

          fp_global_witness_rule: [N] long      — rule index (-1 = base fact)
          fp_global_witness_body: [N, M, 3] long — body atoms, padded to M

        At query time, when a fact-match grounding fires (rule_idx == -1,
        see the ``top_ridx == -1`` branch in ``pack_states``), the caller
        binary-searches the query's hash in ``fp_global_hashes`` and
        replaces the empty fact-match body with the stored witness body.
        SBR then computes ``reasoning_score = min(KGE(body atoms))`` just
        as it would for any rule-match grounding — preserving principled
        ranking among multi-valid closure members.

        The build itself is a one-time CPU-Python pass at init (not in
        the compiled forward-path). It enumerates body-atom bindings via
        indexed dicts over the augmented facts; for industrial-scale
        closures (~3k atoms, 16 rules, bodies up to 6 atoms with a few
        existentials) the total cost is under a second.
        """
        from collections import defaultdict

        if compiled_rules is None:
            compiled_rules = getattr(
                self._enum_ri, '_original_patterns', self._enum_ri.patterns)

        N = int(self.fp_global_hashes.numel())
        M = int(self.kb.M)
        pad = int(self.kb.padding_idx)
        c_no = int(self.kb.constant_no)
        E2 = int(E) * int(E)

        # Decode sorted closure hashes back to (pred, subj, obj) triples.
        h_cpu = self.fp_global_hashes.cpu()
        preds = (h_cpu // E2).tolist()
        rem = (h_cpu % E2)
        subjs = (rem // int(E)).tolist()
        objs = (rem % int(E)).tolist()

        # Build indexed fact lookups over the augmented KB.
        facts_idx_cpu = self.kb.fact_index.facts_idx.cpu().tolist()
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
            # Build an index lazily on first request.
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

        self.register_buffer(
            "fp_global_witness_rule", rule_t.to(device))
        self.register_buffer(
            "fp_global_witness_body", body_t.to(device))
        self.register_buffer(
            "fp_global_witness_bcount", bcount_t.to(device))
        self._has_fp_global_witnesses = True
        # Static Python-int upper bound for clamp() at query time — keeps
        # the clamp bound a compile-time constant so the injection step
        # stays torch.compile(fullgraph=True)-friendly (no tensor .numel()
        # sync needed in the hot path).
        self._fp_global_last_idx = max(N - 1, 0)

    def _augment_kb_with_closure(
        self, device: torch.device, E: int,
    ) -> None:
        """Decode fp_global_hashes to (pred, subj, obj) triples, filter
        out base-fact duplicates, concat to kb.fact_index.facts_idx,
        and rebuild the fact index. Mutates self.kb.fact_index in place
        (replaces the submodule).
        """
        from grounder.data.fact_index import FactIndex

        fi = self.kb.fact_index
        E2 = int(E) * int(E)

        # Decode sorted hashes -> (pred, subj, obj) rows.
        h = self.fp_global_hashes
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
        augmented = torch.cat([fi.facts_idx, new_triples.to(fi.facts_idx.device)],
                              dim=0).contiguous()

        # Pick the same subclass type the KB originally used.
        fact_index_type = (
            "block_sparse" if fi.__class__.__name__ == "BlockSparseFactIndex"
            else "inverted" if fi.__class__.__name__ == "InvertedFactIndex"
            else "arg_key"
        )
        self.kb.fact_index = FactIndex.create(
            augmented,
            type=fact_index_type,
            constant_no=self.kb.constant_no,
            predicate_no=self.kb.predicate_no,
            padding_idx=self.kb.padding_idx,
            device=device,
            pack_base=fi.pack_base,
            max_facts_per_query=fi.max_fact_pairs,
        )
        # K_f depends on per-pattern fact counts; refresh the cached value.
        self.kb.K_f = self.kb.fact_index.max_fact_pairs

    # ==================================================================
    # Canonical loop
    # ==================================================================

    @torch.no_grad()
    def forward(
        self, queries: Tensor, query_mask: Tensor,
        *, batch_size: Optional[int] = None,
        **init_kwargs,
    ) -> GrounderOutput:
        """Ground all ``queries`` and return a ``GrounderOutput``.

        ``batch_size``: when set, queries are processed in chunks of
        exactly ``batch_size`` (last chunk padded with zero queries +
        ``query_mask=False`` so the compiled graph's static shapes hold
        across all chunks). The same ``reduce-overhead`` CUDA graph
        replays per chunk. Use this when ``len(queries)`` is too large
        for a single forward to fit in VRAM. ``rule_groundings``
        accumulators span chunks and dedup once at the end.

        When ``batch_size`` is ``None``:
          * compile_mode is None      → full batch in one forward.
          * compile_mode is set       → auto-default to a chunk size
            small enough to keep the per-chunk CUDA-graph buffers
            bounded. The dense compiled path with
            ``init_state_shape='full'`` allocates intermediate tensors
            of shape ``[B*S, K_r, G_r, M, 3]`` which scales linearly
            with B; chunking keeps peak memory roughly constant.
        """
        N = queries.size(0)
        # Outer-compile resolutions (sld/rtf) always go through the
        # chunked path so the captured CUDA graph is replayed against
        # a stable per-chunk shape. Caller must supply ``batch_size``
        # so the chunk size is reproducible across calls.
        if self._uses_outer_compile:
            if batch_size is None or batch_size <= 0:
                raise ValueError(
                    f"resolution={self.resolution!r} with "
                    f"compile_mode={self.compile_mode!r} requires an "
                    "explicit positive ``batch_size`` so the captured "
                    "CUDA graph sees a stable per-chunk shape.")
            return self._forward_chunked(
                queries, query_mask, batch_size, **init_kwargs)
        if batch_size is None and self.compile_mode is not None:
            batch_size = self._auto_batch_size(N)
        if batch_size is not None and 0 < batch_size < N:
            return self._forward_chunked(
                queries, query_mask, batch_size, **init_kwargs)
        return self._forward_one_batch(queries, query_mask, **init_kwargs)

    @torch.no_grad()
    def run_bc(
        self, queries: Tensor, query_mask: Tensor,
        *, batch_size: Optional[int] = None,
        **init_kwargs,
    ) -> "RuleGroundings":
        """Rule-evidence entry point — for SBR/DCR/R2N pool-iter consumers.

        Wraps :meth:`forward` and post-processes the output:
          * forces ``collect_rule_groundings=True`` for the duration of
            the call so callers don't need to know about that
            ``__init__`` flag;
          * extends ``rule_groundings.atom_table`` to include every
            query atom and populates ``rule_groundings.query_pool_idx``
            so the pool-iter loop has a well-defined readout slot per
            query (even when no firing produced that atom as a head).

        Returns the augmented :class:`grounder.types.RuleGroundings`
        directly; callers that also need the per-tree ``ProofEvidence``
        or the search ``ProofState`` should still call :meth:`forward`.

        ``init_kwargs`` are forwarded to ``forward`` (and onward to
        ``_init_resolution`` for any per-call resolution overrides).
        """
        # Lazy import to avoid a cycle: groundings.py imports from types.py.
        from grounder.groundings import populate_query_pool_idx
        from grounder.types import RuleGroundings as _RG

        prev_collect = self._collect_rule_groundings
        self._collect_rule_groundings = True
        try:
            out = self.forward(
                queries, query_mask, batch_size=batch_size, **init_kwargs)
        finally:
            self._collect_rule_groundings = prev_collect

        rg = out.rule_groundings
        if rg is None:
            # Build an empty-firings RuleGroundings whose atom_table
            # still covers the queries — pool-iter consumers can then
            # just gather KGE-init at every slot and produce a "no
            # proof" score (constant per atom) for every query.
            rg = _RG(
                atom_table=torch.zeros(
                    0, 3, dtype=torch.long, device=queries.device),
                A_in={}, A_out={},
                num_atoms=0,
                num_rules=int(getattr(self.kb, "num_rules", 0) or 0),
            )
        return populate_query_pool_idx(rg, queries, self.kb.padding_idx)

    def _auto_batch_size(self, N: int) -> int:
        """Pick a chunk size for compile mode based on per-chunk memory.

        Heuristic: ``[B*S, K_r, G_r, M, 3]`` body tensor (the dominant
        intermediate at d=0 with ``init_state_shape='full'``) should
        stay below ~1 GB to leave room for the rest of the graph and
        for CUDA-graph private pools. ``B = 1e9 / (S*K_r*G_r*M*3*8)``.
        """
        if self.resolution != "enum":
            return N      # only the enum dense path needs chunking
        S = self.S
        K_r = getattr(self, "K_r", 1) or 1
        G_r = getattr(self, "G_r", 1) or 1
        M = max(self.kb.M, 1)
        # bytes per body element = 8 (long); 3 args per atom
        per_query_bytes = S * K_r * G_r * M * 3 * 8
        budget = 1_000_000_000   # ~1 GB
        B_max = max(1, budget // max(per_query_bytes, 1))
        return min(N, int(B_max))

    def _forward_chunked(
        self, queries: Tensor, query_mask: Tensor,
        batch_size: int, **init_kwargs,
    ) -> GrounderOutput:
        """Chunked forward: pad each chunk to ``batch_size`` so the
        compiled graph sees stable shapes, then concat outputs.

        The rule-groundings accumulators are reset once at the start
        and finalised once at the end — across-chunk dedup is built
        into the existing ``_finalize_r2g_tensor`` torch.unique pass.
        """
        N = queries.size(0)
        dev = queries.device
        pad_q = queries.new_zeros(batch_size, queries.shape[1])
        pad_m = torch.zeros(batch_size, dtype=torch.bool, device=dev)

        # Reset r2g accumulators once for the entire chunked call.
        self._reset_r2g_state()

        # Choose the inner step. When outer compile is enabled (sld/
        # rtf with compile_mode set), wrap _forward_one_batch_inner
        # with torch.compile once and reuse for every chunk; the
        # padded chunk shape stays constant so the captured CUDA
        # graph replays. Otherwise run the eager inner.
        if self._uses_outer_compile:
            if self._compiled_inner is None:
                self._compiled_inner = torch.compile(
                    self._forward_one_batch_inner,
                    mode=self.compile_mode, fullgraph=True)
            inner_fn = self._compiled_inner
        else:
            inner_fn = self._forward_one_batch_inner

        # Run each chunk through the inner forward (which does NOT
        # reset / finalise r2g — those are once-per-call here).
        chunk_outputs: List[GrounderOutput] = []
        chunk_sizes: List[int] = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            q = queries[start:end]
            m = query_mask[start:end]
            n_real = q.size(0)
            if n_real < batch_size:
                # Pad to static shape; padded rows have mask=False so
                # init_states / step / collection all skip them.
                q_padded = pad_q.clone()
                m_padded = pad_m.clone()
                q_padded[:n_real] = q
                m_padded[:n_real] = m
                q, m = q_padded, m_padded
            if self._uses_outer_compile:
                # Tell CUDA-graph trees a new iteration is starting so
                # static-address output buffers are safe to reuse.
                # Mirrors dprl's per-step pattern (PPO compilation.py
                # calls this before every rollout_step).
                torch.compiler.cudagraph_mark_step_begin()
            out = inner_fn(q, m, **init_kwargs)
            if self._uses_outer_compile:
                # Captured graph outputs share static memory across
                # calls — clone every tensor field so subsequent
                # chunks don't overwrite earlier per-chunk results
                # (``_merge_chunk_outputs`` reads them all at the end).
                out = self._clone_grounder_output(out)
            chunk_outputs.append(out)
            chunk_sizes.append(n_real)

        return self._merge_chunk_outputs(chunk_outputs, chunk_sizes, queries)

    @staticmethod
    def _clone_grounder_output(out: GrounderOutput) -> GrounderOutput:
        """Deep-clone every tensor field of a GrounderOutput.

        Used after compiled (CUDA-graph) chunked calls so the per-chunk
        output survives subsequent chunks overwriting the static-address
        buffers the captured graph re-uses.
        """
        def _c(t):
            return t.clone() if isinstance(t, Tensor) else t

        st = out.state
        new_state = ProofState(
            proof_goals=_c(st.proof_goals),
            state_valid=_c(st.state_valid),
            top_ridx=_c(st.top_ridx),
            next_var_indices=_c(st.next_var_indices),
        )
        new_evidence = None
        if out.evidence is not None:
            ev = out.evidence
            new_evidence = ProofEvidence(
                body=_c(ev.body), mask=_c(ev.mask),
                count=_c(ev.count), rule_idx=_c(ev.rule_idx),
                body_count=_c(ev.body_count),
                D=ev.D, M=ev.M, head=_c(ev.head),
            )
        new_rg = None
        if out.rule_groundings is not None:
            rg = out.rule_groundings
            new_rg = RuleGroundings(
                atom_table=_c(rg.atom_table),
                A_in={k: _c(v) for k, v in rg.A_in.items()},
                A_out={k: _c(v) for k, v in rg.A_out.items()},
                num_atoms=rg.num_atoms,
                num_rules=rg.num_rules,
            )
        return GrounderOutput(
            state=new_state,
            evidence=new_evidence,
            rule_groundings=new_rg,
        )

    def _reset_r2g_state(self) -> None:
        if self._collect_rule_groundings:
            self._r2g_buffer: Dict[int, set] = {}
            self._r2g_acc_rule: List[Tensor] = []
            self._r2g_acc_head: List[Tensor] = []
            self._r2g_acc_body: List[Tensor] = []
            self._r2g_skip_per_step = False
        else:
            self._r2g_skip_per_step = False

    def _merge_chunk_outputs(
        self,
        chunk_outputs: List[GrounderOutput],
        chunk_sizes: List[int],
        queries: Tensor,
    ) -> GrounderOutput:
        """Concat per-chunk evidence/state along B; finalise r2g once."""
        evidences = [o.evidence for o in chunk_outputs
                     if o.evidence is not None]
        if evidences:
            # Trim padding rows by chunk_sizes, then cat.
            def _trim_cat(attr):
                parts = []
                for ev, n in zip(evidences, chunk_sizes):
                    t = getattr(ev, attr)
                    if t is None:
                        return None
                    parts.append(t[:n])
                return torch.cat(parts, dim=0)
            body = _trim_cat("body")
            mask = _trim_cat("mask")
            count = mask.sum(dim=1) if mask is not None else None
            rule_idx = _trim_cat("rule_idx")
            body_count = _trim_cat("body_count")
            head = _trim_cat("head")
            evidence = ProofEvidence(
                body=body, mask=mask, count=count, rule_idx=rule_idx,
                body_count=body_count,
                D=evidences[0].D, M=evidences[0].M, head=head,
            )
        else:
            evidence = None

        # State: concat trimmed proof_goals / state_valid / top_ridx.
        # The flat path produces dynamic S_out per chunk, so chunks may
        # have different inner shape on dim=1 (state) and beyond. Pad
        # each chunk to the max-inner-shape before cat. Padding is at
        # ``padding_idx`` for atoms / ``-1`` for ridx / ``False`` for
        # masks / ``0`` for indices — same conventions as ``init_states``
        # / ``pack_states_flat``'s placeholders.
        states = [o.state for o in chunk_outputs]
        pad = self.kb.padding_idx
        def _pad_to(t: Tensor, target_inner: List[int], pad_val) -> Tensor:
            # t shape: [B, *inner]. target_inner is the desired inner.
            cur = list(t.shape[1:])
            if cur == target_inner:
                return t
            out = torch.full(
                (t.size(0), *target_inner), pad_val,
                dtype=t.dtype, device=t.device)
            slices = (slice(None),) + tuple(slice(0, c) for c in cur)
            out[slices] = t
            return out
        _attr_pad = {
            "proof_goals": pad,
            "state_valid": False,
            "top_ridx": -1,
            "next_var_indices": 0,
        }
        def _state_cat(attr):
            parts_raw = []
            for s, n in zip(states, chunk_sizes):
                t = getattr(s, attr)
                if t is None:
                    return None
                parts_raw.append(t[:n])
            # Compute max inner shape across chunks
            max_inner: List[int] = []
            for d in range(1, parts_raw[0].dim()):
                max_inner.append(max(p.size(d) for p in parts_raw))
            pad_val = _attr_pad.get(attr, 0)
            parts = [_pad_to(p, max_inner, pad_val) for p in parts_raw]
            return torch.cat(parts, dim=0)
        state = ProofState(
            proof_goals=_state_cat("proof_goals"),
            state_valid=_state_cat("state_valid"),
            top_ridx=_state_cat("top_ridx"),
            next_var_indices=_state_cat("next_var_indices"),
        )

        # rule_groundings: build once across ALL chunks (the per-chunk
        # ``_forward_one_batch_inner`` skipped finalisation; accumulators
        # already span every chunk).
        rule_groundings = None
        if self._collect_rule_groundings:
            if (not getattr(self, "_r2g_skip_per_step", False)
                    and getattr(self, "_r2g_acc_rule", None)):
                rule_groundings = self._finalize_r2g_tensor()
                if rule_groundings is not None and self.filter_mode == "fp_batch":
                    rule_groundings = self._prune_rule_groundings_tensor(
                        rule_groundings)
            elif hasattr(self, '_r2g_buffer') and self._r2g_buffer:
                from grounder.bc.common import (
                    build_rule_grounding_tensors, prune_rule_groundings)
                r2g = self._r2g_buffer
                if self.filter_mode == "fp_batch":
                    fact_set = set()
                    fi = self.kb.fact_index.facts_idx
                    for f in range(fi.shape[0]):
                        fact_set.add(tuple(fi[f].tolist()))
                    r2g = prune_rule_groundings(
                        r2g, fact_set, max_iterations=self.depth)
                rule_groundings = build_rule_grounding_tensors(
                    r2g, self.kb.num_rules, queries.device)

        return GrounderOutput(state=state, evidence=evidence,
                              rule_groundings=rule_groundings)

    def _forward_one_batch_inner(
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GrounderOutput:
        """Single-batch grounding without resetting / finalising r2g —
        reserved for the chunked path where reset / finalise wrap the
        outer chunk loop.
        """
        states = self.init_states(queries, query_mask, **init_kwargs)
        for d in range(self.depth):
            states = self.step(states, d)
            if self.step_hook is not None:
                cb, cm, cr = self.step_hook.on_step(
                    states["collected_body"], states["collected_mask"],
                    states["collected_ridx"], d)
                states["collected_body"] = cb
                states["collected_mask"] = cm
                states["collected_ridx"] = cr
        evidence = self.filter_terminal(states)
        if isinstance(evidence, dict):
            if self.collect_evidence:
                evidence = ProofEvidence(
                    body=evidence["collected_body"],
                    mask=evidence["collected_mask"],
                    count=evidence["collected_mask"].sum(dim=1),
                    rule_idx=evidence["collected_ridx"],
                    body_count=evidence["collected_bcount"],
                    D=self.depth,
                    M=self.kb.M,
                    head=evidence.get("collected_head"),
                )
            else:
                evidence = None
        if evidence is not None:
            for hook in self.hooks:
                body, mask, ridx = hook.apply(
                    evidence.body_flat, evidence.mask, evidence.rule_idx_top)
                evidence = ProofEvidence(
                    body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx,
                    body_count=evidence.body_count)
        state = ProofState(
            proof_goals=states["proof_goals"],
            state_valid=states["state_valid"],
            top_ridx=states["top_ridx"],
            next_var_indices=(
                states["next_var_indices"]
                if self._standardize_fn is not None else None),
        )
        # No rule_groundings build here — that runs once at the end of
        # the chunked outer call (or in ``_forward_one_batch`` for
        # non-chunked).
        return GrounderOutput(state=state, evidence=evidence,
                              rule_groundings=None)

    def _forward_one_batch(
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GrounderOutput:
        """Single-batch grounder forward: reset r2g, run, finalise."""
        self._reset_r2g_state()
        states = self.init_states(queries, query_mask, **init_kwargs)
        for d in range(self.depth):
            states = self.step(states, d)
            if self.step_hook is not None:
                cb, cm, cr = self.step_hook.on_step(
                    states["collected_body"], states["collected_mask"],
                    states["collected_ridx"], d)
                states["collected_body"] = cb
                states["collected_mask"] = cm
                states["collected_ridx"] = cr
        evidence = self.filter_terminal(states)
        # filter='none' returns raw states dict — wrap in ProofEvidence
        # outside the compiled region (dataclass init breaks fullgraph).
        # Skip when grounding body is not tracked (e.g. RL adapter).
        if isinstance(evidence, dict):
            if self.collect_evidence:
                evidence = ProofEvidence(
                    body=evidence["collected_body"],
                    mask=evidence["collected_mask"],
                    count=evidence["collected_mask"].sum(dim=1),
                    rule_idx=evidence["collected_ridx"],
                    body_count=evidence["collected_bcount"],
                    D=self.depth,
                    M=self.kb.M,
                    head=evidence.get("collected_head"),
                )
            else:
                evidence = None
        if evidence is not None:
            for hook in self.hooks:
                body, mask, ridx = hook.apply(
                    evidence.body_flat, evidence.mask, evidence.rule_idx_top)
                evidence = ProofEvidence(
                    body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx,
                    body_count=evidence.body_count)
        # CUDA-graph-friendly post-forward extraction: when per-step
        # collection was skipped (the default for ``collect_evidence=True``),
        # build _r2g_buffer from the final evidence in one bulk pass.
        if (self._collect_rule_groundings
                and getattr(self, "_r2g_skip_per_step", False)
                and evidence is not None):
            self._build_r2g_from_evidence(evidence)
        # Build RuleGroundings.  Default path: tensorized per-step accumulators
        # → ``_finalize_r2g_tensor`` returns a ready-to-use RuleGroundings.
        # Legacy path: fall back to ``_r2g_buffer`` (Python sets) only when
        # the tensor accumulators are empty (e.g. evidence-based mode).
        rule_groundings = None
        if self._collect_rule_groundings:
            if (not getattr(self, "_r2g_skip_per_step", False)
                    and getattr(self, "_r2g_acc_rule", None)):
                rule_groundings = self._finalize_r2g_tensor()
                if rule_groundings is not None and self.filter_mode == "fp_batch":
                    rule_groundings = self._prune_rule_groundings_tensor(
                        rule_groundings)
            elif hasattr(self, '_r2g_buffer') and self._r2g_buffer:
                from grounder.bc.common import (
                    build_rule_grounding_tensors, prune_rule_groundings)
                r2g = self._r2g_buffer
                if self.filter_mode == "fp_batch":
                    fact_set = set()
                    fi = self.kb.fact_index.facts_idx  # [F, 3]
                    for f in range(fi.shape[0]):
                        fact_set.add(tuple(fi[f].tolist()))
                    r2g = prune_rule_groundings(
                        r2g, fact_set, max_iterations=self.depth)
                rule_groundings = build_rule_grounding_tensors(
                    r2g, self.kb.num_rules, queries.device)

        state = ProofState(
            proof_goals=states["proof_goals"],
            state_valid=states["state_valid"],
            top_ridx=states["top_ridx"],
            next_var_indices=(
                states["next_var_indices"]
                if self._standardize_fn is not None else None),
        )
        return GrounderOutput(state=state, evidence=evidence,
                              rule_groundings=rule_groundings)


    def init_states(
        self, queries: Tensor, query_mask: Tensor,
        *,
        initial_goals: Optional[Tensor] = None,
        next_var_indices: Optional[Tensor] = None,
        excluded_queries: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Build initial states dict for the proof loop.

        Args:
            queries: [B, 3] query atoms.
            query_mask: [B] validity mask.
            initial_goals: [B, M_in, 3] multi-atom goal list to use instead
                of the single query atom (for RL mid-proof entry).
            next_var_indices: [B] pre-allocated variable counters. Defaults
                to ``constant_no + 1`` (fresh).
            excluded_queries: optional tensor for cycle prevention.
        """
        B = queries.size(0)
        dev = queries.device
        pad = self.kb.padding_idx
        G = self.max_goals
        C = self.C
        D = self.depth
        M = self.kb.M  # max body atoms in any single rule
        # M_work: working buffer for the current depth's body atoms.
        # Must fit the widest rule body (``M``) — ``pack_states_flat``
        # writes ``new_body = flat_goals[:, :M_rule, :]`` into this
        # buffer (bc/common.py around line 421), and rules with
        # ``M_rule > 1`` overflow the index target if M_work=1. The
        # earlier ``1 if not self.collect_evidence else M`` shortcut
        # saved a few KB on collect_evidence=False but produced
        # ``shape mismatch: value tensor of shape [N, 2, 3] cannot be
        # broadcast to indexing result of shape [N, 1, 3]`` on any
        # KB with ``kb.M > 1`` (every dataset except trivially-1-body
        # ones — ablation_d2, family, wn18rr, fb15k237, …).
        M_work = M

        # ``init_state_shape``:
        #   "minimal" — S_init=1; smallest possible buffer at d=0.
        #   "full"    — S_init=self.S; same shape as d>=1 so a single
        #               compiled graph covers every depth.
        S_init = 1 if self._init_state_shape == "minimal" else self.S

        proof_goals = torch.full(
            (B, S_init, G, 3), pad, dtype=torch.long, device=dev)
        if initial_goals is not None:
            M_in = initial_goals.shape[1]
            proof_goals[:, 0, :M_in, :] = initial_goals
        else:
            proof_goals[:, 0, 0, :] = queries
        # M-sized working buffer (current depth's rule body atoms)
        grounding_body = torch.full(
            (B, S_init, M_work, 3), pad, dtype=torch.long, device=dev)
        # Structured accumulator: [B, S, D, M, 3] — one slot per depth.
        # ``acc_D``/``acc_M`` collapse to 1 when nothing reads the
        # accumulator, but the ``fp_global`` filter writes witness
        # body atoms into ``accumulated_body`` via
        # ``_inject_witnesses_into_evidence`` and needs the full
        # ``M`` slot count even when ``collect_evidence=False``.
        skip_acc = (not self.collect_evidence
                    and self.filter_mode != "fp_global")
        acc_D = 1 if skip_acc else D
        acc_M = 1 if skip_acc else M
        accumulated_body = torch.full(
            (B, S_init, acc_D, acc_M, 3), pad, dtype=torch.long, device=dev)
        body_count = torch.zeros(B, S_init, acc_D, dtype=torch.long, device=dev)
        ridx_per_depth = torch.full(
            (B, S_init, acc_D), -1, dtype=torch.long, device=dev)
        head_per_depth = torch.full(
            (B, S_init, acc_D, 3), pad, dtype=torch.long, device=dev)
        top_ridx = torch.full((B, S_init), -1, dtype=torch.long, device=dev)
        if S_init == 1:
            state_valid = query_mask.unsqueeze(1)              # [B, 1]
        else:
            # Only slot 0 carries the active query; slots 1.. are inactive.
            state_valid = torch.zeros(
                B, S_init, dtype=torch.bool, device=dev)
            state_valid[:, 0] = query_mask

        if next_var_indices is None:
            E = self.kb.constant_no + 1
            next_var_indices = torch.full(
                (B,), E, dtype=torch.long, device=dev)

        states = {
            "queries": queries,
            "query_mask": query_mask,
            "proof_goals": proof_goals,
            "grounding_body": grounding_body,
            "accumulated_body": accumulated_body,
            "body_count": body_count,
            "ridx_per_depth": ridx_per_depth,
            "head_per_depth": head_per_depth,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "initial_next_var": next_var_indices,
            "collected_body": queries.new_zeros(B, C, acc_D, acc_M, 3),
            "collected_mask": torch.zeros(B, C, dtype=torch.bool, device=dev),
            "collected_ridx": queries.new_full((B, C, acc_D), -1,
                                               dtype=torch.long),
            "collected_bcount": torch.zeros(B, C, acc_D, dtype=torch.long,
                                            device=dev),
            "collected_head": torch.full((B, C, acc_D, 3), pad,
                                         dtype=torch.long, device=dev),
        }
        if initial_goals is not None:
            states["initial_goals"] = initial_goals
        if excluded_queries is not None:
            states["excluded_queries"] = excluded_queries
        return states

    def step(self, states: Dict[str, Tensor], d: int) -> Dict[str, Tensor]:
        """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
        if self.kb.num_rules == 0:
            return states

        # Compiled fast path. The flat path is gated off because its
        # ``torch.nonzero`` produces dynamic shapes incompatible with
        # ``mode='reduce-overhead'``. The dense path compiles every
        # depth, including the last — the ``d == depth-1`` branches in
        # ``resolve_enum_step`` / ``_postprocess`` are static at trace
        # time, so dynamo just specialises an additional graph for
        # them. With ``init_state_shape='full'`` every depth shares the
        # same shape, so the compile cache holds **one** graph regardless
        # of depth.
        if self._compiled:
            flat_step = getattr(self, "_flat_intermediate", False)
            if not flat_step:
                return self._step_compiled(states, d)

        # Capture the goal being resolved at this depth (= head atom)
        if self.collect_evidence or self._collect_rule_groundings:
            states["_selected_goal"] = states["proof_goals"][:, :, 0, :].clone()

        # ── SELECT ──
        goal_queries, remaining, active_mask = self._select(states)

        # ── RESOLVE ──
        resolved = self._resolve(
            goal_queries, remaining,
            states["grounding_body"], states["state_valid"],
            active_mask, states, d,
        )

        # ── SEARCH FILTERS (between RESOLVE and PACK) ──
        resolved = self._apply_search_filters(resolved)

        # ── HOOKS (between RESOLVE and PACK) ──
        resolved = self._apply_hooks(resolved, states)

        # ── COLLECT per-rule groundings (before dedup) ──
        # Tensorized path: appends to per-step GPU tensors with no
        # ``.item()`` / ``.tolist()`` syncs. Final dedup + host transfer
        # happens once after the step loop in ``_finalize_r2g_tensor``.
        if (self._collect_rule_groundings
                and not getattr(self, "_r2g_skip_per_step", False)):
            self._collect_r2g_tensor(resolved, states)

        # ── PACK → returns (states, sync) — no dict pollution ──
        states, sync = self._pack(resolved, states)

        # ── POSTPROCESS ──
        states = self._postprocess(states, sync, d)

        return states

    def filter_terminal(self, states: Dict[str, Tensor]):
        """Apply soundness filter on collected groundings -> ProofEvidence.

        When ``filter='none'``, returns the raw states dict (no collection).
        """
        if self.filter_mode == "none":
            return states

        B = states["collected_body"].size(0)
        C = self.C
        dev = states["collected_body"].device

        body = states["collected_body"]     # [B, C, D, M, 3]
        mask = states["collected_mask"]     # [B, C]
        ridx = states["collected_ridx"]     # [B, C, D]

        if self.kb.num_rules == 0:
            D = body.shape[2]
            M = body.shape[3]
            G_body = D * M
            return self._empty_result(B, C, G_body, dev)

        head = states.get("collected_head")  # [B, C, D, 3] or None

        if self.filter_mode == "fp_batch":
            from grounder.filters.soundness.fp_batch import apply_fp_batch
            body_flat = body.reshape(B, C, -1, 3)
            # Use per-grounding heads if available (grounded collection mode)
            grounding_heads = None
            if head is not None:
                grounding_heads = head  # [B, C, D, 3]
            mask = apply_fp_batch(
                body_flat, mask, states["queries"], self.kb.fact_index,
                self.kb.fact_index.pack_base, self.kb.padding_idx, self.depth,
                grounding_heads=grounding_heads)

        elif self.filter_mode == "fp_global":
            from grounder.filters.soundness.fp_global import apply_fp_global
            body_flat = body.reshape(B, C, -1, 3)
            # NOTE: fp_global_hashes is built by run_forward_chaining with
            # E = num_entities (= constant_no + 1), so body atoms must be
            # hashed with that same base — NOT fact_index.pack_base, which
            # is max(constant_no, padding_idx) + 2 and generally differs.
            # Using pack_base here silently drops every derived atom from
            # the fp_global set, so 2+-hop provable queries emit no grounding.
            mask = apply_fp_global(
                body_flat, mask, self.kb.fact_index,
                self._E_fp_global, self.kb.padding_idx,
                self.fp_global_hashes)

        bcount = states["collected_bcount"]   # [B, C, D]

        # Witness injection for fact-match groundings.
        #
        # When the query atom IS itself in the augmented KB (i.e. it's
        # a closure member), the pack_states initial-depth fact-match
        # path emits a grounding with an empty body. That's a valid
        # Boolean proof but collapses to reasoning_score = 1.0 under
        # fuzzy-AND identity, which kills ranking among multi-valid
        # closure members at the SBR head.
        #
        # Fix: look up each query atom's precomputed witness body
        # (see _build_witness_table) and splice it into the empty
        # fact-match slot. The caller's reasoning head then computes
        # min(KGE(witness body atoms)) just as for any rule match,
        # restoring principled per-atom scoring.
        if (self.filter_mode == "fp_global"
                and getattr(self, "_has_fp_global_witnesses", False)):
            body, ridx, bcount = self._inject_witnesses_into_evidence(
                body, mask, ridx, bcount, states["queries"])

        count = mask.sum(dim=1)
        D_val = self.depth if self.collect_evidence else 0
        M_val = self.kb.M if self.collect_evidence else 0
        return ProofEvidence(
            body=body, mask=mask,
            count=count, rule_idx=ridx,
            body_count=bcount,
            D=D_val, M=M_val,
            head=head,
        )

    def _inject_witnesses_into_evidence(
        self,
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
        E = self._E_fp_global  # Python int captured at init
        E2 = E * E

        q_hash = (queries[..., 0].long() * E2
                  + queries[..., 1].long() * E
                  + queries[..., 2].long())  # [B]

        idx = torch.searchsorted(self.fp_global_hashes, q_hash)  # [B]
        idx_clamped = idx.clamp(0, self._fp_global_last_idx)
        found = self.fp_global_hashes[idx_clamped] == q_hash  # [B]

        wb = self.fp_global_witness_body[idx_clamped]   # [B, M, 3]
        wr = self.fp_global_witness_rule[idx_clamped]   # [B]
        wc = self.fp_global_witness_bcount[idx_clamped]  # [B]

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

    # ==================================================================
    # Phase 1: SELECT (shared)
    # ==================================================================

    def _select(
        self, states: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract first goal from each proof state."""
        proof_goals = states["proof_goals"]
        active_mask = proof_goals[:, :, 0, 0] != self.kb.padding_idx
        queries = proof_goals[:, :, 0, :]
        queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
        remaining = proof_goals.clone()
        remaining[:, :, 0, :] = self.kb.padding_idx
        return queries, remaining, active_mask

    # ==================================================================
    # Phase 2: RESOLVE (dispatches to resolution module)
    # ==================================================================

    def _resolve(
        self,
        queries: Tensor,           # [B, S, 3]
        remaining: Tensor,         # [B, S, G, 3]
        grounding_body: Tensor,    # [B, S, M, 3]
        state_valid: Tensor,       # [B, S]
        active_mask: Tensor,       # [B, S]
        states: Dict[str, Tensor],
        d,                                  # int (eager) or 0-dim Tensor (compiled)
        is_last=None,                       # Optional[Tensor] (compiled path)
        use_hooks: bool = True,
    ) -> ResolvedChildren:
        """Dispatch to resolution strategy. Returns ResolvedChildren.

        ``d`` is a Python int when called from the eager step loop, and
        a 0-dim long tensor when called from the compiled step. The
        ``resolve_enum_step`` downstream accepts both shapes.
        """
        fh = self.fact_hook if use_hooks else None
        rh = self.rule_hook if use_hooks else None

        if self.resolution == "sld":
            return resolve_sld(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.kb.fact_index, facts_idx=self.kb.fact_index.facts_idx,
                rule_index=self.kb.rule_index,
                constant_no=self.kb.constant_no, padding_idx=self.kb.padding_idx,
                K_f=self.kb.K_f, K_r=self.kb.K_r,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.kb.num_rules,
                collect_evidence=self.collect_evidence,
                excluded_queries=states.get("excluded_queries"),
                fact_hook=fh, rule_hook=rh,
            )
        elif self.resolution == "rtf":
            return resolve_rtf(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.kb.fact_index, facts_idx=self.kb.fact_index.facts_idx,
                rule_index=self.kb.rule_index,
                constant_no=self.kb.constant_no, padding_idx=self.kb.padding_idx,
                K_f=self.kb.K_f, K_r=self.kb.K_r, K=self.K,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.kb.num_rules,
                max_fact_pairs_body=self._max_fact_pairs_body,
                collect_evidence=self.collect_evidence,
                fact_hook=fh, rule_hook=rh,
            )
        else:
            return resolve_enum_step(
                queries, remaining, grounding_body, state_valid, active_mask,
                fact_index=self.kb.fact_index,
                d=d, depth=self.depth, width=self.width, is_last=is_last,
                M=self.kb.M, padding_idx=self.kb.padding_idx,
                G_r=self.G_r, K=self.K,
                any_dual=self.any_dual,
                pred_rule_indices=self.pred_rule_indices,
                pred_rule_mask=self.pred_rule_mask,
                has_free=self.has_free,
                body_preds=self.body_preds,
                num_body_atoms=self.num_body_atoms,
                enum_pred_a=self.enum_pred_a,
                enum_bound_binding_a=self.enum_bound_binding_a,
                enum_direction_a=self.enum_direction_a,
                check_arg_source_a=self.check_arg_source_a,
                head_pred_mask=self.head_pred_mask,
                has_dual=getattr(self, "has_dual", None),
                enum_pred_b=getattr(self, "enum_pred_b", None),
                enum_bound_binding_b=getattr(self, "enum_bound_binding_b", None),
                enum_direction_b=getattr(self, "enum_direction_b", None),
                check_arg_source_b=getattr(self, "check_arg_source_b", None),
                collect_evidence=self.collect_evidence,
                cartesian_product=self._enum_cartesian,
                E=self._E,
                w_last_depth=self._w_last_depth,
                fv_enum_pred=getattr(self, "fv_enum_pred", None),
                fv_enum_bound_src=getattr(self, "fv_enum_bound_src", None),
                fv_enum_direction=getattr(self, "fv_enum_direction", None),
                fv_enum_valid=getattr(self, "fv_enum_valid", None),
                V=self.V,
                K_v=self.K_v,
                fv_any_valid=self._fv_any_valid,
                arg_source_dep=getattr(self, "arg_source_dep", None),
                body_preds_dep=getattr(self, "body_preds_dep", None),
                flat_intermediate=getattr(self, "_flat_intermediate", False),
            )

    # ==================================================================
    # Hooks (between RESOLVE and PACK)
    # ==================================================================

    def _apply_hooks(
        self,
        resolved: ResolvedChildren,
        states: Dict[str, Tensor],
    ) -> ResolvedChildren:
        """Apply resolution hooks. Subclasses override for RL filtering."""
        return resolved

    # ==================================================================
    # Per-step search filters (between RESOLVE and PACK)
    # ==================================================================

    def _apply_search_filters(
        self,
        resolved: ResolvedChildren,
    ) -> ResolvedChildren:
        """Per-step search filters. No gradients, zero overhead when disabled."""
        if not self._step_prune_dead and self._step_width is None:
            return resolved

        (fg, fgb, fs, rule_goals, rgb, rule_success, sri,
         f_subs, r_subs) = resolved

        if self._step_prune_dead:
            rule_success = filter_prune_dead(
                rule_goals, rule_success,
                head_pred_mask=self._step_head_pred_mask,
                fact_index=self.kb.fact_index,
                constant_no=self.kb.constant_no,
                padding_idx=self.kb.padding_idx,
                M=self.kb.M,
                a0_lens=self._step_a0_lens if self._step_has_csr else None,
                a1_lens=self._step_a1_lens if self._step_has_csr else None,
                p_lens=getattr(self, '_step_p_lens', None),
                key_scale=self._step_key_scale if self._step_has_csr else 0,
            )

        if self._step_width is not None:
            rule_success = filter_width(
                rule_goals, rule_success,
                fact_index=self.kb.fact_index,
                constant_no=self.kb.constant_no,
                padding_idx=self.kb.padding_idx,
                M=self.kb.M,
                width=self._step_width,
            )

        return ResolvedChildren(fg, fgb, fs, rule_goals, rgb, rule_success,
                                sri, f_subs, r_subs)

    # ==================================================================
    # Phase 3: PACK (shared)
    # ==================================================================

    def _pack(
        self,
        resolved,
        states: Dict[str, Tensor],
    ) -> Tuple[Dict, SyncParams]:
        """Flatten S*K children, propagate grounding body, compact to S.

        Dispatches to pack_states (dense) or pack_states_flat (flat K).
        Returns (states, sync) — no dict pollution with underscore keys.
        """
        if isinstance(resolved, FlatResolvedChildren):
            from grounder.bc.common import pack_states_flat
            packed = pack_states_flat(
                resolved,
                states["top_ridx"], states["grounding_body"],
                states["body_count"],
                self.kb.padding_idx,
                collect_evidence=self.collect_evidence,
                M_rule=self.kb.M,
                dedup=self._pack_dedup,
            )
        else:
            packed = pack_states(
                *resolved,
                states["top_ridx"], states["grounding_body"],
                states["body_count"],
                self.S, self.kb.padding_idx,
                collect_evidence=self.collect_evidence,
                M_rule=self.kb.M,
            )

        states["grounding_body"] = packed.grounding_body
        states["proof_goals"] = packed.proof_goals
        states["top_ridx"] = packed.top_ridx
        states["state_valid"] = packed.state_valid

        sync = SyncParams(
            parent_map=packed.parent_map,
            winning_subs=packed.winning_subs,
            has_new_body=packed.has_new_body,
            parent_bcount=packed.body_count,
            current_ridx=packed.current_ridx,
        )

        S_in = packed.proof_goals.shape[1]  # output S (may differ from input)
        states["next_var_indices"] = (
            states["next_var_indices"] + S_in * self.max_vars_per_rule)
        return states, sync

    def _sync_accumulated(
        self,
        states: Dict[str, Tensor],
        sync: SyncParams,
        d: int,
    ) -> Dict[str, Tensor]:
        """Propagate accumulated_body: gather from parents, apply subs, write at depth d.

        Structured layout: accumulated_body is [B, S, D, M, 3].
        Each depth d writes its body atoms to slot ``[:, :, d, :, :]``.

        Args:
            states: Current states dict with accumulated_body and grounding_body.
            sync: SyncParams with parent_map, winning_subs, has_new_body,
                  parent_bcount, current_ridx.
            d: Current depth index.
        """
        parent_map = sync.parent_map
        winning_subs = sync.winning_subs
        has_new_body = sync.has_new_body
        parent_bcount = sync.parent_bcount

        if not self.collect_evidence:
            states["body_count"] = parent_bcount
            return states

        B, S_out = parent_map.shape
        D_dim = states["accumulated_body"].shape[2]  # D
        M_acc = states["accumulated_body"].shape[3]   # M
        M_work = states["grounding_body"].shape[2]
        pad = self.kb.padding_idx
        dev = parent_map.device

        # a. Gather accumulated_body [B, S_out, D, M, 3] from parents
        pi = parent_map[:, :, None, None, None].expand(-1, -1, D_dim, M_acc, 3)
        acc = states["accumulated_body"].gather(1, pi)

        # b. Gather ridx_per_depth [B, S_out, D] from parents
        rpi = parent_map[:, :, None].expand(-1, -1, D_dim)
        ridx = states["ridx_per_depth"].gather(1, rpi)

        # c. Gather body_count [B, S_out, D] from parents
        bc = states["body_count"].gather(1, rpi)

        # d. Apply substitutions to entire accumulated body
        acc_flat = acc.reshape(B * S_out, D_dim * M_acc, 3)
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
        acc_flat = apply_substitutions(acc_flat, subs_flat, pad)
        acc = acc_flat.reshape(B, S_out, D_dim, M_acc, 3)

        # e. Write new body atoms at depth slot d
        new_atoms = states["grounding_body"]  # [B, S_out, M_work, 3]
        # Truncate or pad to M_acc if needed
        if M_work > M_acc:
            write_atoms = new_atoms[:, :, :M_acc, :]
        elif M_work < M_acc:
            write_atoms = torch.full(
                (B, S_out, M_acc, 3), pad, dtype=torch.long, device=dev)
            write_atoms[:, :, :M_work, :] = new_atoms
        else:
            write_atoms = new_atoms
        write_mask = has_new_body[:, :, None, None]  # [B, S_out, 1, 1]
        # ``d`` may be a Python int (eager) or a 0-dim long tensor
        # (compiled). For the compiled path we use a one-hot
        # broadcast-mask along the D dimension so the write becomes a
        # plain ``torch.where`` (no indexed scatter) — that keeps the
        # graph depth-agnostic and shareable across all d values.
        if isinstance(d, torch.Tensor):
            D_acc = acc.shape[2]
            d_arange = torch.arange(D_acc, device=dev)
            is_slot = (d_arange == d).view(1, 1, D_acc, 1, 1)   # [1,1,D,1,1]
            write_atoms_b = write_atoms.unsqueeze(2)             # [B,S,1,M,3]
            write_mask_b = write_mask.unsqueeze(2)               # [B,S,1,1,1]
            acc = torch.where(
                is_slot & write_mask_b, write_atoms_b, acc)
            # Write ridx and bc at slot d via the same trick:
            #   ridx[:, :, d] ← where(has_new_body, current_ridx, ridx[:, :, d])
            is_slot_2d = (d_arange == d).view(1, 1, D_acc)
            ridx = torch.where(
                is_slot_2d & has_new_body.unsqueeze(-1),
                sync.current_ridx.unsqueeze(-1).expand(-1, -1, D_acc),
                ridx,
            )
            new_active = (write_atoms[:, :, :, 0] != pad)
            new_lens = new_active.long().sum(dim=-1)             # [B, S_out]
            bc = torch.where(
                is_slot_2d & has_new_body.unsqueeze(-1),
                new_lens.unsqueeze(-1).expand(-1, -1, D_acc),
                bc,
            )
        else:
            # Eager fast path — Python int slice.
            acc[:, :, d, :, :] = torch.where(write_mask, write_atoms,
                                             acc[:, :, d, :, :])
            ridx[:, :, d] = torch.where(has_new_body, sync.current_ridx,
                                        ridx[:, :, d])
            new_active = (write_atoms[:, :, :, 0] != pad)
            new_lens = new_active.long().sum(dim=-1)
            bc[:, :, d] = torch.where(has_new_body, new_lens, bc[:, :, d])

        # h. Gather and write head_per_depth at depth d
        hpi = parent_map[:, :, None, None].expand(-1, -1, D_dim, 3)
        head = states["head_per_depth"].gather(1, hpi)
        # Apply substitutions to heads too (variables may get resolved)
        head_flat = head.reshape(B * S_out, D_dim, 3)
        head_flat = apply_substitutions(head_flat, subs_flat, pad)
        head = head_flat.reshape(B, S_out, D_dim, 3)
        # Write the selected goal at depth d
        if "_selected_goal" in states:
            sel = states["_selected_goal"]  # [B, S_in, 3]
            # Gather from parent
            sel_parent = sel.gather(
                1, parent_map.unsqueeze(-1).expand(-1, -1, 3))
            # Apply subs
            sel_flat = sel_parent.reshape(B * S_out, 1, 3)
            sel_flat = apply_substitutions(sel_flat, subs_flat, pad)
            sel_parent = sel_flat.reshape(B, S_out, 3)
            if isinstance(d, torch.Tensor):
                d_arange_h = torch.arange(D_dim, device=dev)
                is_slot_h = (d_arange_h == d).view(1, 1, D_dim, 1)
                head = torch.where(
                    is_slot_h & has_new_body.view(B, S_out, 1, 1),
                    sel_parent.unsqueeze(2),
                    head,
                )
            else:
                head[:, :, d, :] = torch.where(
                    has_new_body.unsqueeze(-1), sel_parent, head[:, :, d, :])

        states["accumulated_body"] = acc
        states["body_count"] = bc
        states["ridx_per_depth"] = ridx
        states["head_per_depth"] = head
        return states

    # ==================================================================
    # Phase 4: POSTPROCESS (shared)
    # ==================================================================

    def _postprocess_goals(self, states: Dict) -> Dict[str, Tensor]:
        """Optionally prune ground facts, compact atoms, and standardize.

        When ``prune_facts=True``, known ground facts are removed from
        proof_goals between steps (compressed depth semantics).
        When ``prune_facts=False`` (default), only compaction is applied
        (standard SLD semantics where every resolution costs 1 depth).

        When ``collect_evidence=False`` and standardization is configured,
        output variables are standardized (proof_goals are the final output).

        Safe for torch.compile — ``self.prune_facts`` is a static Python bool.
        """
        if self.prune_facts:
            proof_goals, _, _ = prune_ground_facts(
                states["proof_goals"], states["state_valid"],
                self.kb.fact_index.fact_hashes, self.kb.fact_index.pack_base,
                self.kb.constant_no, self.kb.padding_idx,
                excluded_queries=states.get("excluded_queries"),
            )
            states["proof_goals"] = compact_atoms(proof_goals, self.kb.padding_idx)
        else:
            states["proof_goals"] = compact_atoms(
                states["proof_goals"], self.kb.padding_idx)

        # Standardize output variables when proof_goals are the final output
        if not self.collect_evidence and self._standardize_fn is not None:
            counts = states["state_valid"].long().sum(dim=1)
            nv = states.get("initial_next_var", states["next_var_indices"])
            inp = states.get("initial_goals", states["proof_goals"].new_zeros(0))
            std, std_nv = self.standardize_output(
                states["proof_goals"], counts, nv, inp)
            # Clone to detach from CUDA graph output buffers — prevents
            # "overwritten by a subsequent run" errors when these tensors
            # are consumed by the next compiled step.
            states["proof_goals"] = std.clone()
            states["next_var_indices"] = std_nv.clone()

        return states

    def _collect_groundings(self, states: Dict) -> Dict[str, Tensor]:
        """Collect completed groundings into output buffer.

        Uses accumulated_body [B, S, D, M, 3] (structured). Called outside
        the compiled step to keep G_body tensors out of the CUDA graph.
        """
        # In "grounded" mode, don't deactivate — states continue to deeper depths
        deactivate = (self._collect_mode != "grounded")
        cb, cm, cr, sv, c_bc, c_hd = collect_groundings(
            states["accumulated_body"], states["proof_goals"],
            states["state_valid"], states["ridx_per_depth"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
            self.kb.constant_no, self.kb.padding_idx, self.C,
            body_count=states["body_count"],
            collected_bcount=states["collected_bcount"],
            collect_mode=self._collect_mode,
            deactivate=deactivate,
            head_per_depth=states.get("head_per_depth"),
            collected_head=states.get("collected_head"),
            variant_to_orig=getattr(self, "_variant_to_orig_t", None),
        )

        states["collected_body"] = cb
        states["collected_mask"] = cm
        states["collected_ridx"] = cr
        states["state_valid"] = sv
        states["collected_bcount"] = c_bc
        if c_hd is not None:
            states["collected_head"] = c_hd
        return states

    def _collect_r2g_tensor(self, resolved, states: Dict[str, Tensor]) -> None:
        """Tensor-only per-step accumulator for rule-application data.

        Replaces the legacy ``_collect_r2g`` Python loop with a pure-tensor
        path: gather (rule_idx, head, sorted_body) for every valid resolved
        child and ``cat`` onto running GPU tensors. No ``.item()``, no
        ``.tolist()``, no Python iteration in the hot path. Dedup runs
        once after the step loop in ``_finalize_r2g_tensor`` via
        ``torch.unique``.
        """
        pad = self.kb.padding_idx
        M = self.kb.M
        sel = states.get("_selected_goal")  # [B, S_in, 3]

        # Bring both ResolvedChildren shapes onto a single flat layout:
        #   rule_idx [T], body [T, M, 3], b_idx [T], s_idx [T], valid [T]
        if isinstance(resolved, FlatResolvedChildren):
            rule_idx = resolved.flat_rule_idx
            body = resolved.flat_goals[:, :M, :]
            b_idx = resolved.flat_b_idx
            s_idx = resolved.flat_s_idx
            valid = torch.ones_like(rule_idx, dtype=torch.bool)
        else:
            ridx = resolved.sub_rule_idx                # [B, S, K_r]
            success = resolved.rule_success             # [B, S, K_r]
            goals = resolved.rule_goals[..., :M, :]     # [B, S, K_r, M, 3]
            B, S, K_r = ridx.shape
            dev = ridx.device
            rule_idx = ridx.reshape(-1)
            body = goals.reshape(-1, M, 3)
            valid = success.reshape(-1)
            bi = (torch.arange(B, device=dev).view(B, 1, 1)
                  .expand(B, S, K_r).reshape(-1))
            si = (torch.arange(S, device=dev).view(1, S, 1)
                  .expand(B, S, K_r).reshape(-1))
            b_idx = bi
            s_idx = si

        # Drop entries with no body atom or invalid mask.
        active_atom = body[..., 0] != pad           # [T, M]
        has_body = active_atom.any(dim=-1)           # [T]
        valid = valid & has_body & (rule_idx >= 0)
        # Note: an early ``if not bool(valid.any()): return`` here
        # used to short-circuit the rest of this function, but that
        # ``bool()`` on a 0-d tensor breaks ``torch.compile`` under
        # ``fullgraph=True`` (Dynamo can't trace the data-dependent
        # Python branch). Downstream tensor ops handle the
        # all-False case correctly — empty selections cascade to
        # zero-size accumulator appends, which is the right result —
        # so we let the rest of the function run unconditionally.

        rule_idx = rule_idx[valid].long()
        body = body[valid]
        b_idx = b_idx[valid].long()
        s_idx = s_idx[valid].long()
        active_atom = active_atom[valid]

        # Map variant → orig.
        v2o = getattr(self, "_variant_to_orig_t", None)
        if v2o is not None:
            rule_idx = v2o[rule_idx.clamp(min=0)]

        # Head: gather selected goal per (b, s).
        if sel is not None:
            head = sel[b_idx, s_idx]                # [T, 3]
        else:
            head = torch.full(
                (rule_idx.size(0), 3), pad,
                dtype=torch.long, device=rule_idx.device)

        # Sort body atoms within each entry by per-atom hash so anchor
        # variants of the same logical rule application share a key.
        # Padding atoms map to a sentinel that sorts past everything.
        P1, P2, P3 = 1_000_003, 999_983, 999_979
        atom_hash = (body[..., 0].long() * P1
                     + body[..., 1].long() * P2
                     + body[..., 2].long() * P3)     # [T, M]
        sentinel = torch.full_like(atom_hash, (2 ** 62) - 1)
        atom_hash_for_sort = torch.where(active_atom, atom_hash, sentinel)
        sort_idx = atom_hash_for_sort.argsort(dim=-1)
        body_sorted = body.gather(
            1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Append to running accumulators (one cat at end of forward).
        self._r2g_acc_rule.append(rule_idx)
        self._r2g_acc_head.append(head)
        self._r2g_acc_body.append(body_sorted)

    def _prune_rule_groundings_tensor(self, rg):
        """fp_batch / Kleene fixed-point pruning over a RuleGroundings.

        Mirrors ``common.prune_rule_groundings`` semantics (snapshot-based,
        ``num_steps`` iterations) but operates on the tensor RuleGroundings
        produced by ``_finalize_r2g_tensor``. Drops apps whose body atoms
        aren't all in the proved-set (facts ∪ heads of proved apps) after
        ``self.depth`` snapshot iterations.
        """
        atom_table = rg.atom_table                              # [num_atoms, 3]
        num_atoms = atom_table.size(0)
        device = atom_table.device
        # Build fact-set membership: which atom_table rows are facts.
        # facts_idx is [F, 3]. Combine atom_table and facts_idx,
        # ``unique`` with ``return_inverse``, then count duplicates.
        fi = self.kb.fact_index.facts_idx                       # [F, 3]
        if fi.numel() > 0:
            fi_dev = fi.to(device)
            joined = torch.cat([atom_table, fi_dev], dim=0)     # [num_atoms+F, 3]
            _, inv = torch.unique(joined, dim=0, return_inverse=True)
            # An atom_table row is a fact iff its unique-bucket is
            # shared by at least one row from the facts side.
            inv_atoms = inv[:num_atoms]
            inv_facts = inv[num_atoms:]
            fact_mask = torch.zeros(
                inv.max().item() + 1, dtype=torch.bool, device=device)
            fact_mask[inv_facts] = True
            is_fact = fact_mask[inv_atoms]                      # [num_atoms]
        else:
            is_fact = torch.zeros(
                num_atoms, dtype=torch.bool, device=device)

        # proved-set: starts as facts, grows with proved heads.
        proved = is_fact.clone()
        # Snapshot iteration: each pass uses a frozen view of
        # ``proved`` from the previous pass.
        for _ in range(max(1, self.depth)):
            new_proved = proved.clone()
            for r, a_in in rg.A_in.items():
                if a_in.numel() == 0:
                    continue
                a_out = rg.A_out[r]                             # [G_r, 1]
                head_idx = a_out[:, 0]
                # For each app, are all body atoms in ``proved``?
                # a_in[g, m] indexes into atom_table; -1/pad-rows are
                # already trimmed by max_body_per_rule, so every
                # column is a valid body atom.
                if a_in.size(1) == 0:
                    body_proved = torch.ones(
                        a_in.size(0), dtype=torch.bool, device=device)
                else:
                    body_proved = proved[a_in].all(dim=-1)      # [G_r]
                new_proved[head_idx[body_proved]] = True
            if torch.equal(new_proved, proved):
                break
            proved = new_proved
        # Final filter: keep apps whose body atoms are all in proved.
        new_A_in: Dict[int, Tensor] = {}
        new_A_out: Dict[int, Tensor] = {}
        for r, a_in in rg.A_in.items():
            if a_in.numel() == 0:
                new_A_in[r] = a_in
                new_A_out[r] = rg.A_out[r]
                continue
            if a_in.size(1) == 0:
                keep = torch.ones(
                    a_in.size(0), dtype=torch.bool, device=device)
            else:
                keep = proved[a_in].all(dim=-1)                 # [G_r]
            new_A_in[r] = a_in[keep].contiguous()
            new_A_out[r] = rg.A_out[r][keep].contiguous()
        from grounder.types import RuleGroundings
        return RuleGroundings(
            atom_table=rg.atom_table,
            A_in=new_A_in,
            A_out=new_A_out,
            num_atoms=rg.num_atoms,
            num_rules=rg.num_rules,
        )

    def _finalize_r2g_tensor(self):
        """Dedup accumulated (rule, head, body) tensors with
        ``torch.unique`` and build a ``RuleGroundings`` directly.

        Returns the RuleGroundings tensor structure (atom_table, A_in,
        A_out) without going through ``_r2g_buffer``. The host transfer
        is one int per (head, body) atom — O(U·(M+1)) — which is
        orders of magnitude smaller than the legacy ``.tolist()`` of
        raw resolved children per step.
        """
        from grounder.types import RuleGroundings

        if not self._r2g_acc_rule:
            return None
        rule_idx = torch.cat(self._r2g_acc_rule, 0)
        head = torch.cat(self._r2g_acc_head, 0)
        body = torch.cat(self._r2g_acc_body, 0)
        T = rule_idx.size(0)
        if T == 0:
            return None
        # Drop sentinel rows (rule_idx == -1) inserted by the per-step
        # path to keep ``_collect_r2g_tensor`` host-sync-free. One sync
        # here at finalize amortises across all steps.
        keep = rule_idx >= 0
        if not bool(keep.any()):
            return None
        if not bool(keep.all()):
            rule_idx = rule_idx[keep]
            head = head[keep]
            body = body[keep]
            T = rule_idx.size(0)
        M = body.size(1)
        device = rule_idx.device
        # Encode each row as a single comparable tensor for unique:
        # [rule, head[3], body[M*3]] = [1 + 3 + 3M].
        combined = torch.cat([
            rule_idx.unsqueeze(-1),
            head.long(),
            body.long().reshape(T, M * 3),
        ], dim=-1)
        uniq = torch.unique(combined, dim=0)          # [U, 1 + 3 + 3M]
        if uniq.size(0) == 0:
            return None
        u_rule = uniq[:, 0].long()
        u_head = uniq[:, 1:4].long()                  # [U, 3]
        u_body = uniq[:, 4:].reshape(-1, M, 3).long() # [U, M, 3]

        # Atom table: union of (head, body) atoms across all rule
        # applications.  Build via a single ``torch.unique`` on the
        # flattened atom set.
        # Drop padding body atoms by tagging them with a sentinel index;
        # they'll get filtered when building per-rule A_in.
        pad = self.kb.padding_idx
        all_atoms = torch.cat([u_head.unsqueeze(1), u_body], dim=1)  # [U, M+1, 3]
        flat_atoms = all_atoms.reshape(-1, 3)                       # [U*(M+1), 3]
        atom_table, inverse = torch.unique(
            flat_atoms, dim=0, return_inverse=True,
        )
        inverse = inverse.reshape(-1, M + 1)                        # [U, M+1]
        head_atom_idx = inverse[:, 0]                               # [U]
        body_atom_idx = inverse[:, 1:]                              # [U, M]
        # Active body-atom mask (drop padding atoms in A_in).
        body_active = (u_body[..., 0] != pad)                       # [U, M]

        # Bucket into per-rule tensors.  Keep this Python loop over
        # ``num_rules`` only — typically tens, not thousands.  The
        # masking + indexing inside is tensor-only.
        num_rules = self.kb.num_rules
        A_in: Dict[int, Tensor] = {}
        A_out: Dict[int, Tensor] = {}
        # Track max body length per rule to size A_in.
        max_body_per_rule = body_active.long().sum(dim=-1)          # [U]
        for r in range(num_rules):
            mask = (u_rule == r)
            if not bool(mask.any()):
                A_in[r] = torch.zeros(
                    0, 0, dtype=torch.long, device=device)
                A_out[r] = torch.zeros(
                    0, 1, dtype=torch.long, device=device)
                continue
            r_body_idx = body_atom_idx[mask]                        # [G_r, M]
            r_head_idx = head_atom_idx[mask].unsqueeze(-1)          # [G_r, 1]
            # Compact: drop padding atoms by per-row slice.
            r_max_m = int(max_body_per_rule[mask].max().item())
            A_in[r] = r_body_idx[:, :r_max_m].contiguous()
            A_out[r] = r_head_idx
        return RuleGroundings(
            atom_table=atom_table.contiguous(),
            A_in=A_in,
            A_out=A_out,
            num_atoms=int(atom_table.size(0)),
            num_rules=num_rules,
        )

    def _build_r2g_from_evidence(self, evidence) -> None:
        """Build self._r2g_buffer from final evidence (post-forward, fast).

        Equivalent to per-step ``_collect_r2g`` but runs ONCE after the
        step loop finishes, with a single bulk ``.cpu()`` transfer
        instead of per-step .item()/.cpu() syncs in the hot path. This
        keeps the step loop CUDA-graph compatible (no host-device sync,
        no Python-side iteration over resolved tensors during step).

        evidence layout (D > 0):
          * body       [B, C, D, M, 3]
          * head       [B, C, D, 3]
          * rule_idx   [B, C, D] — variant index when all_anchors=True
          * mask       [B, C]
          * body_count [B, C, D]
        """
        if evidence is None or evidence.D == 0 or evidence.head is None:
            return
        pad = self.kb.padding_idx
        body_t = evidence.body          # [B, C, D, M, 3]
        head_t = evidence.head          # [B, C, D, 3]
        ridx_t = evidence.rule_idx      # [B, C, D]
        mask_t = evidence.mask          # [B, C]
        bcnt_t = evidence.body_count    # [B, C, D]

        # Per-(b,c,d) validity:
        #   * mask[b, c] (terminal collection accepted the proof tree)
        #   * ridx[b, c, d] >= 0 (a rule fired at this depth)
        # ``head`` may be padding even when ridx>=0 for some legacy
        # paths, so guard on head pred too.
        valid_d = (
            mask_t.unsqueeze(-1)
            & (ridx_t >= 0)
            & (head_t[..., 0] != pad)
        )

        # Single bulk transfer to CPU; the rest is fast Python over
        # numpy arrays. T_max = B*C*D ≤ a few thousand for realistic
        # workloads.
        valid_cpu = valid_d.cpu().numpy()
        body_cpu = body_t.cpu().numpy()
        head_cpu = head_t.cpu().numpy()
        ridx_cpu = ridx_t.cpu().numpy()
        bcnt_cpu = bcnt_t.cpu().numpy()
        v2o = (self._variant_to_orig
               if hasattr(self, "_variant_to_orig") else None)

        B, C, D, M, _ = body_cpu.shape
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    if not valid_cpu[b, c, d]:
                        continue
                    r = int(ridx_cpu[b, c, d])
                    n_atoms = int(bcnt_cpu[b, c, d])
                    body = []
                    for m in range(min(n_atoms, M)):
                        p = int(body_cpu[b, c, d, m, 0])
                        if p == pad:
                            break
                        body.append((p,
                                     int(body_cpu[b, c, d, m, 1]),
                                     int(body_cpu[b, c, d, m, 2])))
                    if not body:
                        continue
                    head = (int(head_cpu[b, c, d, 0]),
                            int(head_cpu[b, c, d, 1]),
                            int(head_cpu[b, c, d, 2]))
                    orig_r = v2o[r] if v2o is not None else r
                    if orig_r not in self._r2g_buffer:
                        self._r2g_buffer[orig_r] = set()
                    self._r2g_buffer[orig_r].add(
                        (head, tuple(sorted(body))))

    def _collect_r2g(self, resolved, states: Dict[str, Tensor]) -> None:
        """Collect per-rule-application groundings before dedup.

        Extracts (head, body) from ALL resolved children and stores in
        self._r2g_buffer (Python dict of sets). Called between RESOLVE
        and PACK so it sees rule applications before per-state packing
        drops them. Works for both FlatResolvedChildren (enum flat path)
        and ResolvedChildren (SLD/RTF + enum dense path with V<2).

        Vectorised: a single bulk ``.tolist()`` transfer to host, then
        pure-Python iteration over the resulting nested lists. This is
        ~20× faster than per-element ``.item()`` calls and keeps the
        host-device sync count at one per call. The function is still
        Python-side and not CUDA-graph compatible; the compiled step
        path (``_step_compiled``) does NOT call it — production
        CUDA-graph callers should consume ``evidence`` instead.
        """
        pad = self.kb.padding_idx
        M = self.kb.M
        sel = states.get("_selected_goal")  # [B, S_in, 3]
        v2o = (self._variant_to_orig
               if hasattr(self, "_variant_to_orig") else None)

        if isinstance(resolved, FlatResolvedChildren):
            T = resolved.flat_rule_idx.size(0)
            if T == 0:
                return
            ridx_l = resolved.flat_rule_idx.tolist()
            goals_l = resolved.flat_goals[:, :M, :].tolist()  # [T][M][3]
            b_idx_l = resolved.flat_b_idx.tolist()
            s_idx_l = resolved.flat_s_idx.tolist()
            sel_l = sel.tolist() if sel is not None else None

            for t in range(T):
                r = ridx_l[t]
                gt = goals_l[t]
                body = []
                for m in range(M):
                    a = gt[m]
                    if a[0] == pad:
                        break
                    body.append((a[0], a[1], a[2]))
                if not body:
                    continue
                if sel_l is not None:
                    h = sel_l[b_idx_l[t]][s_idx_l[t]]
                    head = (h[0], h[1], h[2])
                else:
                    head = (pad, pad, pad)
                orig_r = v2o[r] if v2o is not None else r
                if orig_r not in self._r2g_buffer:
                    self._r2g_buffer[orig_r] = set()
                self._r2g_buffer[orig_r].add(
                    (head, tuple(sorted(body))))
            return

        # ResolvedChildren: SLD/RTF or enum dense path (V<2).
        ridx_l = resolved.sub_rule_idx.tolist()      # [B][S][K_r]
        goals_l = resolved.rule_goals[..., :M, :].tolist()  # [B][S][K_r][M][3]
        success_l = resolved.rule_success.tolist()   # [B][S][K_r]
        sel_l = sel.tolist() if sel is not None else None
        B = len(ridx_l)
        for b in range(B):
            for s in range(len(ridx_l[b])):
                ridx_bs = ridx_l[b][s]
                succ_bs = success_l[b][s]
                goals_bs = goals_l[b][s]
                head_default = (sel_l[b][s] if sel_l is not None
                                else [pad, pad, pad])
                head = (head_default[0], head_default[1], head_default[2])
                for k in range(len(ridx_bs)):
                    if not succ_bs[k]:
                        continue
                    r = ridx_bs[k]
                    gk = goals_bs[k]
                    body = []
                    for m in range(M):
                        a = gk[m]
                        if a[0] == pad:
                            break
                        body.append((a[0], a[1], a[2]))
                    if not body:
                        continue
                    orig_r = v2o[r] if v2o is not None else r
                    if orig_r not in self._r2g_buffer:
                        self._r2g_buffer[orig_r] = set()
                    self._r2g_buffer[orig_r].add(
                        (head, tuple(sorted(body))))

    def _postprocess(self, states: Dict[str, Tensor], sync: SyncParams,
                     d, is_last=None) -> Dict[str, Tensor]:
        """Full postprocess: prune goals + sync accumulated + collect groundings.

        ``d`` is a Python int when called from the eager step loop,
        and a 0-dim long tensor when called from the compiled step.
        ``is_last`` is None in eager (computed from ``d`` directly) and
        a 0-dim bool tensor in compiled mode.
        """
        states = self._postprocess_goals(states)
        states = self._sync_accumulated(states, sync, d)
        # Last step + w_last_depth>0: leftover ground unknowns in
        # proof_goals would block terminal collection. The body atoms
        # are already in accumulated_body; clear proof_goals so the
        # rule application is emitted (matches keras-ns
        # prune_incomplete_proofs=False semantics).
        if self._w_last_depth is not None and self._w_last_depth > 0:
            pad = self.kb.padding_idx
            if is_last is not None:
                # Compiled: tensor select.
                cleared = torch.full_like(states["proof_goals"], pad)
                states["proof_goals"] = torch.where(
                    is_last, cleared, states["proof_goals"])
            elif d == self.depth - 1:
                # Eager: Python int branch.
                states["proof_goals"] = torch.full_like(
                    states["proof_goals"], pad)
        if self.collect_evidence:
            states = self._collect_groundings(states)
        return states

    # ==================================================================
    # Output variable standardization (optional)
    # ==================================================================

    def standardize_output(
        self,
        states: Tensor,
        counts: Tensor,
        next_var_indices: Tensor,
        input_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Standardize runtime variables in output states.

        Renumbers variables in ``states`` using the configured mode
        (``standardization_mode='offset'`` or ``'canonical'``).

        Does nothing if ``standardization_mode=None`` (default).

        Args:
            states: [B, K, M, 3] derived states to standardize.
            counts: [B] valid state count per batch element.
            next_var_indices: [B] current free variable index.
            input_states: [B, ?, 3] parent states (needed for offset mode).

        Returns:
            std_states: [B, K, M, 3] standardized states.
            new_next_var: [B] updated free variable indices.
        """
        if self._standardize_fn is None:
            return states, next_var_indices
        return self._standardize_fn(states, counts, next_var_indices,
                                    input_states if input_states is not None
                                    else states.new_zeros(0))

    # ==================================================================
    # Compiled step (optimization — same semantics as clean path)
    # ==================================================================

    def _fn_step_for_depth(self, d: int):
        """Get the (single, depth-agnostic) compiled step function.

        Static-shape compile: ``dynamic=False``. With
        ``init_state_shape='full'`` every depth has the same input
        shape, and ``d`` enters as a 0-dim tensor (not a Python int)
        so dynamo doesn't specialise the graph on ``d``'s value. One
        graph covers every depth; ``self._fn_steps_by_depth[0]`` is
        the cache slot.

        With ``init_state_shape='minimal'`` (DpRL-friendly), d=0 has
        ``S_in=1`` while d≥1 has ``S_in=max_states``; those two
        shapes still need separate graphs, so the cache holds at most
        2 entries (keyed by ``S_in``).
        """
        # Cache key: state shape kind. 'full' → one shape for all d.
        # 'minimal' → d=0 differs from d>=1.
        if self._init_state_shape == "full" or d > 0:
            key = "main"
        else:
            key = "init"
        if key not in self._fn_steps_by_depth:
            import torch._dynamo as _dynamo
            if getattr(_dynamo.config, "recompile_limit", 0) < 64:
                _dynamo.config.recompile_limit = 64
            self._fn_steps_by_depth[key] = torch.compile(
                self._step_impl, fullgraph=True, mode=self.compile_mode,
                dynamic=False,
            )
        return self._fn_steps_by_depth[key]

    def _step_compiled(self, states: Dict, d: int = 0) -> Dict[str, Tensor]:
        """Compiled step: dict <-> raw tensors.

        ``d`` is converted to a 0-dim long tensor and ``is_last`` to a
        0-dim bool tensor before entering the compiled region — this
        keeps the graph structure depth-agnostic, so a single compile
        replays for every depth.
        """
        if self._clone_between_steps:
            states = {k: v.clone() if isinstance(v, Tensor) else v
                      for k, v in states.items()}

        fn = self._fn_step_for_depth(d)
        dev = states["proof_goals"].device
        # d → 0-dim long tensor; is_last → 0-dim bool tensor. Both are
        # treated as data-dependent inputs by dynamo (no specialisation
        # on their value), so changing d/is_last between calls reuses
        # the same compiled graph.
        d_t = torch.tensor(d, dtype=torch.long, device=dev)
        is_last_t = torch.tensor(
            d == self.depth - 1, dtype=torch.bool, device=dev)

        (gb, ab, bc, rpd, hpd, pg, tr, sv, nvi,
         cb, cm, cr, cbc, chh,
         step_ridx, step_head, step_body, step_valid) = fn(
            states["grounding_body"], states["accumulated_body"],
            states["body_count"], states["ridx_per_depth"],
            states["head_per_depth"],
            states["proof_goals"],
            states["top_ridx"], states["state_valid"],
            states["next_var_indices"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
            states["collected_bcount"],
            states["collected_head"],
            d_t, is_last_t,
        )
        # Clone every output to detach from the CUDA-graph-managed
        # private pool. Without this, the next ``fn`` replay overwrites
        # the buffers while the outer Python still holds references in
        # the ``states`` dict, raising "accessing tensor output of
        # CUDAGraphs that has been overwritten by a subsequent run".
        states["grounding_body"]   = gb.clone()
        states["accumulated_body"] = ab.clone()
        states["body_count"]       = bc.clone()
        states["ridx_per_depth"]   = rpd.clone()
        states["head_per_depth"]   = hpd.clone()
        states["proof_goals"]      = pg.clone()
        states["top_ridx"]         = tr.clone()
        states["state_valid"]      = sv.clone()
        states["next_var_indices"] = nvi.clone()
        states["collected_body"]   = cb.clone()
        states["collected_mask"]   = cm.clone()
        states["collected_ridx"]   = cr.clone()
        states["collected_bcount"] = cbc.clone()
        states["collected_head"]   = chh.clone()
        # Pre-pack rule app collection: append the just-emitted (orig_ridx,
        # head, sorted_body) tuples to the running r2g accumulators. Use
        # the same sentinel-ridx convention as the eager
        # ``_collect_r2g_tensor`` so neither path forces a per-step host
        # sync; the single ``keep = rule_idx >= 0`` sync happens once at
        # ``_finalize_r2g_tensor``. The tensors are cloned to detach
        # from the CUDA graph output buffers — the next step replays the
        # graph and overwrites them, which would corrupt the
        # accumulators.
        if self._collect_rule_groundings:
            pad = self.kb.padding_idx
            active = step_body[..., 0] != pad
            has_body = active.any(dim=-1)
            keep = step_valid & has_body & (step_ridx >= 0)
            ridx_out = torch.where(
                keep, step_ridx.long(),
                step_ridx.new_full((), -1, dtype=torch.long))
            self._r2g_acc_rule.append(ridx_out.clone())
            self._r2g_acc_head.append(step_head.long().clone())
            self._r2g_acc_body.append(step_body.long().clone())
        return states

    def _step_impl(
        self,
        grounding_body: Tensor,
        accumulated_body: Tensor,
        body_count: Tensor,
        ridx_per_depth: Tensor,
        head_per_depth: Tensor,
        proof_goals: Tensor,
        top_ridx: Tensor,
        state_valid: Tensor,
        next_var_indices: Tensor,
        collected_body: Tensor,
        collected_mask: Tensor,
        collected_ridx: Tensor,
        collected_bcount: Tensor,
        collected_head: Tensor,
        d_t: Tensor,
        is_last_t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor]:
        """Raw tensor step for torch.compile -- same phases as clean path.

        Returns the 14 state tensors plus 4 pre-pack rule-application
        tensors used by ``_step_compiled`` to populate the r2g
        accumulators (orig rule index, head, sorted body, valid mask).
        Capturing this data INSIDE the compiled region but appending to
        the Python list OUTSIDE it lets intermediate-step rule
        applications survive ``_pack``'s S-truncation — without any
        ``.item()`` / data-dependent indexing inside the compiled graph.
        """
        states = {
            "grounding_body": grounding_body,
            "accumulated_body": accumulated_body,
            "body_count": body_count,
            "ridx_per_depth": ridx_per_depth,
            "head_per_depth": head_per_depth,
            "proof_goals": proof_goals,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "collected_body": collected_body,
            "collected_mask": collected_mask,
            "collected_ridx": collected_ridx,
            "collected_bcount": collected_bcount,
            "collected_head": collected_head,
        }

        # SELECT -> RESOLVE -> SEARCH FILTERS -> HOOKS -> [collect r2g] -> PACK -> POSTPROCESS
        queries, remaining, active_mask = self._select(states)
        resolved = self._resolve(
            queries, remaining, grounding_body, state_valid,
            active_mask, states, d=d_t, is_last=is_last_t, use_hooks=False,
        )
        resolved = self._apply_search_filters(resolved)
        resolved = self._apply_hooks(resolved, states)

        # Capture pre-pack rule application data for r2g collection.
        # The compiled path is gated to dense ResolvedChildren (flat path
        # is forced eager at ``step``), so this branch is always taken
        # when ``_step_compiled`` runs. Mirrors ``_collect_r2g_tensor``'s
        # canonicalisation (variant→orig + sort body atoms) but stays
        # compile-safe: only tensor ops, no .item()/.tolist(), no
        # data-dependent indexing. The eager wrapper applies the
        # validity mask via boolean indexing (compile-unfriendly) and
        # appends the kept rows to ``_r2g_acc_*``.
        M_collect = self.kb.M
        pad_collect = self.kb.padding_idx
        rule_goals_c = resolved.rule_goals[..., :M_collect, :]   # [B, S, K_r, M, 3]
        sub_ridx_c = resolved.sub_rule_idx                       # [B, S, K_r]
        rule_succ_c = resolved.rule_success                      # [B, S, K_r]
        K_r_c = sub_ridx_c.size(-1)
        v2o_c = self._variant_to_orig_t                          # [num_variants]
        sub_ridx_orig_c = v2o_c[sub_ridx_c.clamp(min=0)]
        P1, P2, P3 = 1_000_003, 999_983, 999_979
        atom_h_c = (rule_goals_c[..., 0].long() * P1
                    + rule_goals_c[..., 1].long() * P2
                    + rule_goals_c[..., 2].long() * P3)
        active_atom_c = rule_goals_c[..., 0] != pad_collect
        sentinel_c = torch.full_like(atom_h_c, (2 ** 62) - 1)
        ah_for_sort_c = torch.where(active_atom_c, atom_h_c, sentinel_c)
        sort_idx_c = ah_for_sort_c.argsort(dim=-1)
        body_sorted_c = rule_goals_c.gather(
            -2, sort_idx_c.unsqueeze(-1).expand(-1, -1, -1, -1, 3))
        sel_c = proof_goals[:, :, 0, :]                          # [B, S, 3]
        head_c = sel_c.unsqueeze(2).expand(-1, -1, K_r_c, -1)    # [B, S, K_r, 3]
        step_ridx = sub_ridx_orig_c.reshape(-1)
        step_head = head_c.reshape(-1, 3)
        step_body = body_sorted_c.reshape(-1, M_collect, 3)
        step_valid = rule_succ_c.reshape(-1)

        states, sync = self._pack(resolved, states)
        states = self._postprocess(states, sync, d_t, is_last_t)

        return (states["grounding_body"], states["accumulated_body"],
                states["body_count"], states["ridx_per_depth"],
                states["head_per_depth"],
                states["proof_goals"],
                states["top_ridx"], states["state_valid"],
                states["next_var_indices"],
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"],
                states["collected_bcount"],
                states["collected_head"],
                step_ridx, step_head, step_body, step_valid)

    # ==================================================================
    # Provability
    # ==================================================================

    def check_known(self, atoms: Tensor) -> Tensor:
        """Check if atoms are known facts or in fp_global set (I_D)."""
        is_fact = self.kb.fact_index.exists(atoms)
        if hasattr(self, "_has_fp_global") and self._has_fp_global:
            E = getattr(self, '_E_fp_global', getattr(self, '_E', self.kb.constant_no + 1))
            h = atoms[..., 0] * (E * E) + atoms[..., 1] * E + atoms[..., 2]
            in_fp_global = check_in_fp_global(h, self.fp_global_hashes)
            return is_fact | in_fp_global
        return is_fact

    def is_provable(self, atoms: Tensor) -> Tensor:
        return self.check_known(atoms)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _empty_result(
        self, B: int, C: int, G_body: int, dev: torch.device,
    ) -> ProofEvidence:
        D = self.depth if self.collect_evidence else 1
        M = self.kb.M if self.collect_evidence else 1
        return ProofEvidence(
            body=torch.zeros(
                B, C, D, M, 3, dtype=torch.long, device=dev),
            mask=torch.zeros(
                B, C, dtype=torch.bool, device=dev),
            count=torch.zeros(
                B, dtype=torch.long, device=dev),
            rule_idx=torch.full(
                (B, C, D), -1, dtype=torch.long, device=dev),
            body_count=torch.zeros(
                B, C, D, dtype=torch.long, device=dev),
            D=D if self.collect_evidence else 0,
            M=M if self.collect_evidence else 0,
        )

    def __repr__(self) -> str:
        return (
            f"BCGrounder(resolution={self.resolution!r}, "
            f"filter={self.filter_mode!r}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_rules={self.kb.num_rules}, "
            f"S={self.S}, C={self.C})")
