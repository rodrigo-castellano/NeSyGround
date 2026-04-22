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
    ResolvedChildren, SyncParams,
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
        filter: str = "fp_batch",
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
        fc_method: str = "join",
        fc_depth: int = 10,
        # Testing/validation enum params (not compile-compatible)
        cartesian_product: bool = False,
        all_anchors: bool = False,
        flat_intermediate: bool = False,
        pack_dedup: bool = True,
        collect_rule_groundings: bool = False,
        w_last_depth: int = 0,
        collect_mode: str = "terminal",
        # Output variable standardization (for consumers of ungrounded states)
        standardization: Optional[StandardizationConfig] = None,
        # Per-step ground-fact pruning: remove known facts from proof goals
        # between resolution steps. Disabled by default (standard SLD semantics
        # where every resolution step costs 1 depth). Enable for "compressed"
        # depth semantics where ground-fact goals are free.
        prune_facts: bool = False,
    ) -> None:
        super().__init__()
        self.kb = kb

        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.filter_mode = filter
        self.compile_mode = compile_mode
        self.hooks = hooks or []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self.step_hook = None  # Optional StepHook (nn.Module), set externally
        self.collect_evidence = collect_evidence
        self.prune_facts = prune_facts
        # Enum defaults: all_anchors + cartesian + collect_rule_groundings
        # for keras-compatible per-rule-application grounding output.
        if resolution == "enum":
            if not all_anchors:
                all_anchors = True
            if not cartesian_product:
                cartesian_product = True
            if not collect_rule_groundings:
                collect_rule_groundings = True
        self._cartesian_product = cartesian_product
        self._all_anchors = all_anchors
        self._flat_intermediate_flag = flat_intermediate
        self._pack_dedup = pack_dedup
        self._collect_rule_groundings = collect_rule_groundings
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
            # When depth > 1 with width > 0, intermediate steps generate many
            # valid children that need state slots for the next step.  Ensure S
            # is at least K so no valid children are lost during packing.
            if self.depth > 1 and self.width is not None and self.width > 0:
                self.S = max(self.S, self.K)
            self.C = meta["C"]
            # (C may be overridden by resolution-specific init)
            self.any_dual = meta["any_dual"]
            self.G_r = meta["G_r"]
            self._enum_cartesian = meta.get("cartesian_product", False)
            self.V = meta.get("V", 1)
            self.K_v = meta.get("K_v", 64)
            self._fv_any_valid = meta.get("fv_any_valid", None)
            self._flat_intermediate = meta.get("flat_intermediate", False)
            # Variant→original rule mapping (for all_anchors)
            if self._all_anchors:
                v2o = []
                for orig_r, blen in enumerate(self.kb.rule_index.rule_lens_sorted.tolist()):
                    for _ in range(blen):
                        v2o.append(orig_r)
                self._variant_to_orig = v2o
            self.fc_method = kwargs["fc_method"]
            self.fc_depth = kwargs["fc_depth"]
            self.max_vars_per_rule = 3  # unused for enum, but keeps state uniform

        else:
            raise ValueError(f"Unknown resolution: {self.resolution}")

        # Compilation (all resolutions — shapes are static)
        self._compiled = False
        self._clone_between_steps = False
        self._fn_steps_by_depth: Dict[int, Any] = {}
        if (self.compile_mode
                and self.depth > 1
                and self.kb.device_.type == "cuda"):
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
        method = getattr(self, 'fc_method', 'dynamic')
        if method in ("join", "spmm"):
            method = "dynamic"
        fc_depth = getattr(self, 'fc_depth', 10)
        fp_global_tensor, n_fp_global = run_forward_chaining(
            compiled_rules=compiled_rules,
            facts_idx=self.kb.fact_index.facts_idx,
            num_entities=E,
            num_predicates=P,
            depth=fc_depth,
            device=str(device),
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
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GrounderOutput:
        if self._collect_rule_groundings:
            self._r2g_buffer: Dict[int, set] = {}  # reset each forward
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
        # Build RuleGroundings from collected per-rule sets
        rule_groundings = None
        if self._collect_rule_groundings and hasattr(self, '_r2g_buffer') and self._r2g_buffer:
            from grounder.bc.common import (
                build_rule_grounding_tensors, prune_rule_groundings)
            r2g = self._r2g_buffer
            # Apply fp_batch-style pruning if soundness filter is enabled
            if self.filter_mode == "fp_batch":
                fact_set = set()
                fi = self.kb.fact_index.facts_idx  # [F, 3]
                for f in range(fi.shape[0]):
                    fact_set.add(tuple(fi[f].tolist()))
                r2g = prune_rule_groundings(r2g, fact_set, max_iterations=self.depth + 1)
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
        M_work = 1 if not self.collect_evidence else M

        proof_goals = torch.full(
            (B, 1, G, 3), pad, dtype=torch.long, device=dev)
        if initial_goals is not None:
            M_in = initial_goals.shape[1]
            proof_goals[:, 0, :M_in, :] = initial_goals
        else:
            proof_goals[:, 0, 0, :] = queries
        # M-sized working buffer (current depth's rule body atoms)
        grounding_body = torch.full(
            (B, 1, M_work, 3), pad, dtype=torch.long, device=dev)
        # Structured accumulator: [B, S, D, M, 3] — one slot per depth
        acc_D = 1 if not self.collect_evidence else D
        acc_M = 1 if not self.collect_evidence else M
        accumulated_body = torch.full(
            (B, 1, acc_D, acc_M, 3), pad, dtype=torch.long, device=dev)
        body_count = torch.zeros(B, 1, acc_D, dtype=torch.long, device=dev)
        ridx_per_depth = torch.full(
            (B, 1, acc_D), -1, dtype=torch.long, device=dev)
        head_per_depth = torch.full(
            (B, 1, acc_D, 3), pad, dtype=torch.long, device=dev)
        top_ridx = torch.full((B, 1), -1, dtype=torch.long, device=dev)
        state_valid = query_mask.unsqueeze(1)

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

        # Compiled fast path (all depths; skip last enum step which needs width=0,
        # and skip flat intermediate steps which have data-dependent shapes).
        if self._compiled:
            last_enum_step = (
                self.resolution == "enum"
                and d == self.depth - 1
                and self.width is not None
            )
            flat_step = getattr(self, "_flat_intermediate", False)
            if not last_enum_step and not flat_step:
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
        if self._collect_rule_groundings:
            self._collect_r2g(resolved, states)

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
        d: int,
        use_hooks: bool = True,
    ) -> ResolvedChildren:
        """Dispatch to resolution strategy. Returns ResolvedChildren."""
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
                d=d, depth=self.depth, width=self.width,
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
        acc[:, :, d, :, :] = torch.where(write_mask, write_atoms,
                                          acc[:, :, d, :, :])

        # f. Write current rule index at depth d
        ridx[:, :, d] = torch.where(has_new_body, sync.current_ridx,
                                     ridx[:, :, d])

        # g. Update per-depth body count at depth d
        new_active = (write_atoms[:, :, :, 0] != pad)  # [B, S_out, M_acc]
        new_lens = new_active.long().sum(dim=-1)       # [B, S_out]
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
        )

        states["collected_body"] = cb
        states["collected_mask"] = cm
        states["collected_ridx"] = cr
        states["state_valid"] = sv
        states["collected_bcount"] = c_bc
        if c_hd is not None:
            states["collected_head"] = c_hd
        return states

    def _collect_r2g(self, resolved, states: Dict[str, Tensor]) -> None:
        """Collect per-rule-application groundings before dedup.

        Extracts (head, body) from ALL resolved children and stores in
        self._r2g_buffer (Python dict of sets). Called between RESOLVE and PACK.
        Works for both FlatResolvedChildren (enum) and ResolvedChildren (SLD/RTF).
        """
        pad = self.kb.padding_idx
        M = self.kb.M
        sel = states.get("_selected_goal")  # [B, S_in, 3]

        if isinstance(resolved, FlatResolvedChildren):
            T = resolved.flat_rule_idx.size(0)
            if T == 0:
                return
            ridx = resolved.flat_rule_idx.cpu()
            goals = resolved.flat_goals.cpu()      # [T, G, 3]
            b_idx = resolved.flat_b_idx.cpu()
            s_idx = resolved.flat_s_idx.cpu()
            sel_cpu = sel.cpu() if sel is not None else None

            for t in range(T):
                r = ridx[t].item()
                # Body atoms = first M slots of goals
                body = []
                for m in range(M):
                    p = goals[t, m, 0].item()
                    if p == pad:
                        break
                    body.append((goals[t, m, 0].item(),
                                 goals[t, m, 1].item(),
                                 goals[t, m, 2].item()))
                if not body:
                    continue
                # Head = selected goal of the parent state
                if sel_cpu is not None:
                    b, s = b_idx[t].item(), s_idx[t].item()
                    head = tuple(sel_cpu[b, s].tolist())
                else:
                    head = (pad, pad, pad)
                # Map variant index → original rule index (for all_anchors)
                orig_r = self._variant_to_orig[r] if hasattr(self, '_variant_to_orig') else r
                if orig_r not in self._r2g_buffer:
                    self._r2g_buffer[orig_r] = set()
                # Sort body for dedup across anchor variants
                self._r2g_buffer[orig_r].add((head, tuple(sorted(body))))

        else:
            # ResolvedChildren (SLD/RTF): dense [B, S, K_r, ...]
            ridx = resolved.sub_rule_idx.cpu()    # [B, S, K_r]
            goals = resolved.rule_goals.cpu()     # [B, S, K_r, G, 3]
            success = resolved.rule_success.cpu() # [B, S, K_r]
            sel_cpu = sel.cpu() if sel is not None else None
            B, S, K_r = ridx.shape

            for b in range(B):
                for s in range(S):
                    for k in range(K_r):
                        if not success[b, s, k]:
                            continue
                        r = ridx[b, s, k].item()
                        body = []
                        for m in range(M):
                            p = goals[b, s, k, m, 0].item()
                            if p == pad:
                                break
                            body.append((goals[b, s, k, m, 0].item(),
                                         goals[b, s, k, m, 1].item(),
                                         goals[b, s, k, m, 2].item()))
                        if not body:
                            continue
                        if sel_cpu is not None:
                            head = tuple(sel_cpu[b, s].tolist())
                        else:
                            head = (pad, pad, pad)
                        # SLD/RTF: rule_idx is already original (no variants)
                        if r not in self._r2g_buffer:
                            self._r2g_buffer[r] = set()
                        self._r2g_buffer[r].add((head, tuple(sorted(body))))

    def _postprocess(self, states: Dict[str, Tensor], sync: SyncParams,
                     d: int = 0) -> Dict[str, Tensor]:
        """Full postprocess: prune goals + sync accumulated + collect groundings.

        Used by the non-compiled path. The compiled path splits this into
        _postprocess_goals (inside compiled) + _sync_and_collect (outside).

        When ``collect_evidence=False``, skips grounding collection (proof_goals
        are the final output, not evidence).

        Args:
            states: Current proof states dict.
            sync: SyncParams from _pack (parent_map, winning_subs, etc.).
            d: Current depth index (for structured accumulation).
        """
        states = self._postprocess_goals(states)
        states = self._sync_accumulated(states, sync, d)
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
        """Get or lazily compile the step function for depth d."""
        if d not in self._fn_steps_by_depth:
            import functools
            fn = functools.partial(self._step_impl, d=d)
            self._fn_steps_by_depth[d] = torch.compile(
                fn, fullgraph=True, mode=self.compile_mode)
        return self._fn_steps_by_depth[d]

    def _step_compiled(self, states: Dict, d: int = 0) -> Dict[str, Tensor]:
        """Compiled step: dict <-> raw tensors."""
        if self._clone_between_steps:
            states = {k: v.clone() if isinstance(v, Tensor) else v
                      for k, v in states.items()}

        # Use per-depth compiled function (d is a trace-time constant)
        fn = self._fn_step_for_depth(d)

        (states["grounding_body"], states["accumulated_body"],
         states["body_count"], states["ridx_per_depth"],
         states["head_per_depth"],
         states["proof_goals"],
         states["top_ridx"], states["state_valid"],
         states["next_var_indices"],
         states["collected_body"], states["collected_mask"],
         states["collected_ridx"],
         states["collected_bcount"],
         states["collected_head"]) = fn(
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
        )
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
        d: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Raw tensor step for torch.compile -- same phases as clean path."""
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

        # SELECT -> RESOLVE -> SEARCH FILTERS -> HOOKS -> PACK -> POSTPROCESS
        queries, remaining, active_mask = self._select(states)
        resolved = self._resolve(
            queries, remaining, grounding_body, state_valid,
            active_mask, states, d=d, use_hooks=False,
        )
        resolved = self._apply_search_filters(resolved)
        resolved = self._apply_hooks(resolved, states)
        states, sync = self._pack(resolved, states)
        states = self._postprocess(states, sync, d)

        return (states["grounding_body"], states["accumulated_body"],
                states["body_count"], states["ridx_per_depth"],
                states["head_per_depth"],
                states["proof_goals"],
                states["top_ridx"], states["state_valid"],
                states["next_var_indices"],
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"],
                states["collected_bcount"],
                states["collected_head"])

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
