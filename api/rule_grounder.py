"""A grounder from rule strings: compile ``(head, body)`` string rules and ``(r, h, t)`` fact ids into a :class:`KB`,
build the backward grounder a type string names (``enum.fp_batch.w1.d2.flat``, ``sld.prune.d1``, ...), and ground
batches of queries into rule firings.

    fact_index = TensorFactIndex(facts, num_predicates, num_entities, device=...)
    grounder = create_grounder("enum.fp_batch.w1.d2.flat", fact_index=fact_index, rules=rules, kb=vocab, ...)
    firings = grounder.run_bc(queries, query_mask)          # RuleGroundings with query_pool_idx

``vocab`` is any object with ``relation2id`` / ``entity2id`` (and optionally ``rule_names``) — the id maps the facts
and queries use. The neural-symbolic reasoners of torch-ns run on it; the guided-search seams
(``set_guided_scorer`` / ``_budget`` / ``_depth`` / ``_capture`` / ``_sample_tau``) serve learned selection policies.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from grounder.data.loader import is_variable
from torch import Tensor

# Named constants for binding indices in CompiledRule.
# Every body-atom argument is encoded as an int indicating what it binds to:
BINDING_HEAD_VAR0 = 0        # Argument is the first head variable
BINDING_HEAD_VAR1 = 1        # Argument is the second head variable
BINDING_FREE_VAR_OFFSET = 2  # Free variables start at index 2 (value = 2 + free_var_i)
BINDING_NO_FREE_VAR = -1     # Sentinel: this body atom introduces no free variable


class CompiledRule:
    """Pre-compiled rule pattern for tensor operations.

    Supports multiple free variables via cascaded enumeration.
    Body atoms are reordered so that each atom has at least one
    already-known argument (topological sort on variable binding).
    """

    def __init__(self, rule_tuple: Tuple, pred_to_idx: Dict[str, int],
                 name: str = None):
        """Compile a rule represented as ``(head_tuple, body_list)``.

        Args:
            rule_tuple: ``(head, body)`` where ``head`` is a single atom
                ``(predicate, var0, var1)`` and ``body`` is a list of
                such atoms.
            pred_to_idx: Predicate-name → predicate-id map.
            name: Optional rule name for debug / metadata. If None, no
                ``self.name`` is set (callers that need names must pass
                them).
        """
        head, body = rule_tuple
        self.name = name

        # Head info
        self.head_pred_idx = pred_to_idx[head[0]]
        self.head_var0 = head[1]  # First variable (e.g., X)
        self.head_var1 = head[2]  # Second variable (e.g., Y)

        # Body info
        self.num_body = len(body)
        self.body_pred_indices = [pred_to_idx[b[0]] for b in body]

        # ── Pass 1: collect all vars, identify free vars ──
        all_vars = set()
        for body_atom in body:
            all_vars.add(body_atom[1])
            all_vars.add(body_atom[2])
        self.free_vars_list = sorted(all_vars - {self.head_var0, self.head_var1})
        self.num_free = len(self.free_vars_list)
        self.free_var_to_idx = {v: i for i, v in enumerate(self.free_vars_list)}

        # ── Pass 2: build body_patterns with extended bindings ──
        # Binding indices: BINDING_HEAD_VAR0, BINDING_HEAD_VAR1, BINDING_FREE_VAR_OFFSET+i
        self.body_patterns = []
        for body_atom in body:
            pattern = {
                'pred_idx': pred_to_idx[body_atom[0]],
                'arg0_var': body_atom[1],
                'arg1_var': body_atom[2],
                'arg0_binding': self._get_binding(body_atom[1]),
                'arg1_binding': self._get_binding(body_atom[2]),
            }
            self.body_patterns.append(pattern)

        # ── Pass 3: compute enumeration order and reorder body atoms ──
        self._compute_enum_order()

    def _get_binding(self, var: str) -> int:
        """Get binding type: BINDING_HEAD_VAR0, BINDING_HEAD_VAR1, or BINDING_FREE_VAR_OFFSET+i."""
        if var == self.head_var0:
            return BINDING_HEAD_VAR0
        elif var == self.head_var1:
            return BINDING_HEAD_VAR1
        else:
            return BINDING_FREE_VAR_OFFSET + self.free_var_to_idx[var]

    def _compute_enum_order(self):
        """Compute processing order for cascaded enumeration.

        Reorders body_patterns so that each atom has at least one known arg.
        Known sources start as {0, 1} (head vars) and grow as free vars
        are introduced by enumeration.

        Stores:
            body_order: list of original body atom indices in processing order
            enum_meta: per-atom metadata for cascaded enumeration
        """
        known_sources = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        remaining = list(range(self.num_body))
        order: List[int] = []
        meta_list: List[dict] = []

        while remaining:
            found = False
            for idx in remaining:
                bp = self.body_patterns[idx]
                b0, b1 = bp['arg0_binding'], bp['arg1_binding']
                a0_known = b0 in known_sources
                a1_known = b1 in known_sources

                if a0_known or a1_known:
                    if a0_known and a1_known:
                        # Both bound — no enumeration needed
                        meta = {
                            'introduces_fv': BINDING_NO_FREE_VAR,
                            'enum_bound_src': BINDING_HEAD_VAR0,
                            'enum_direction': 0,
                            'enum_pred': bp['pred_idx'],
                        }
                    elif a0_known:
                        # arg0 known, arg1 free → enumerate objects
                        fv_idx = b1 - BINDING_FREE_VAR_OFFSET
                        meta = {
                            'introduces_fv': fv_idx,
                            'enum_bound_src': b0,
                            'enum_direction': 0,
                            'enum_pred': bp['pred_idx'],
                        }
                        known_sources.add(b1)
                    else:
                        # arg1 known, arg0 free → enumerate subjects
                        fv_idx = b0 - BINDING_FREE_VAR_OFFSET
                        meta = {
                            'introduces_fv': fv_idx,
                            'enum_bound_src': b1,
                            'enum_direction': 1,
                            'enum_pred': bp['pred_idx'],
                        }
                        known_sources.add(b0)

                    order.append(idx)
                    meta_list.append(meta)
                    remaining.remove(idx)
                    found = True
                    break

            if not found:
                # Fallback: add remaining atoms as fully-bound (shouldn't
                # happen with well-formed rules where head vars cover the
                # connected component)
                for idx in remaining:
                    order.append(idx)
                    meta_list.append({
                        'introduces_fv': BINDING_NO_FREE_VAR,
                        'enum_bound_src': BINDING_HEAD_VAR0,
                        'enum_direction': 0,
                        'enum_pred': self.body_patterns[idx]['pred_idx'],
                    })
                break

        self.body_order = order
        self.enum_meta = meta_list

        # Reorder body_patterns and body_pred_indices to match processing order
        self.body_patterns = [self.body_patterns[i] for i in order]
        self.body_pred_indices = [self.body_pred_indices[i] for i in order]


class TensorFactIndex(nn.Module):
    """Tensorized fact storage feeding the grounder adapter.

    On construction, sorts ``(pred, subj, obj)`` triples by a
    perfect hash and registers four buffers: ``fact_preds``,
    ``fact_subjs``, ``fact_objs``, ``fact_hashes``. Plus the
    ``num_facts`` count.

    Public attributes (read by the adapter):
        fact_preds, fact_subjs, fact_objs: ``[F]`` long tensors —
            sorted facts, used to build the ``[F, 3]`` tensor passed
            to grounder.
        num_predicates, num_entities: scalars (used by adapter).
    """

    def __init__(self,
                 facts: List[Tuple[int, int, int]],
                 num_predicates: int,
                 num_entities: int,
                 max_facts_per_query: int = 64,
                 device: str = 'cuda'):
        super().__init__()

        self.num_predicates = num_predicates
        self.num_entities = num_entities
        self.max_facts_per_query = max_facts_per_query
        self.device = device

        self._build_fact_tensors(facts)

    def _build_fact_tensors(self, facts: List[Tuple[int, int, int]]) -> None:
        """Build the sorted tensor storage."""
        num_facts = len(facts)

        if num_facts == 0:
            self.register_buffer('fact_preds', torch.zeros(1, dtype=torch.long, device=self.device))
            self.register_buffer('fact_subjs', torch.zeros(1, dtype=torch.long, device=self.device))
            self.register_buffer('fact_objs', torch.zeros(1, dtype=torch.long, device=self.device))
            self.register_buffer('fact_hashes', torch.zeros(1, dtype=torch.long, device=self.device))
            self.register_buffer('num_facts', torch.tensor(0, dtype=torch.long, device=self.device))
            return

        fact_preds = torch.tensor([f[0] for f in facts], dtype=torch.long, device=self.device)
        fact_subjs = torch.tensor([f[1] for f in facts], dtype=torch.long, device=self.device)
        fact_objs = torch.tensor([f[2] for f in facts], dtype=torch.long, device=self.device)

        # Hash = pred * (E^2) + subj * E + obj (perfect hash for E entities).
        E = self.num_entities
        fact_hashes = fact_preds * (E * E) + fact_subjs * E + fact_objs

        sorted_indices = torch.argsort(fact_hashes)

        self.register_buffer('fact_preds', fact_preds[sorted_indices])
        self.register_buffer('fact_subjs', fact_subjs[sorted_indices])
        self.register_buffer('fact_objs', fact_objs[sorted_indices])
        self.register_buffer('fact_hashes', fact_hashes[sorted_indices])
        self.register_buffer('num_facts', torch.tensor(num_facts, dtype=torch.long, device=self.device))

    def __repr__(self) -> str:
        return (f"TensorFactIndex(num_facts={self.num_facts.item()}, "
                f"num_predicates={self.num_predicates}, "
                f"num_entities={self.num_entities})")

    def contains_atoms(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return a ``[B]`` bool mask: True where ``atoms[b]`` is in the KB.

        Used by the rule-path reasoner to detect ``fact-only'' queries —
        those whose head atom appears literally in the KB. The keras-ns
        SBR forward returns vacuous truth (1.0) for these because their
        only proof is the trivial fact-itself derivation; the rule-path
        pool-iter loop has no firing for them, so its raw output would
        be the KGE-init pool value. Overriding to 1.0 in
        ``ReasonerModel._reason_impl`` matches the per-tree path's
        BCE behavior on positive training triples.

        Args:
            atoms: ``[B, 3]`` long ``(pred, subj, obj)`` triples.

        Returns:
            ``[B]`` bool tensor.
        """
        if atoms.dim() != 2 or atoms.shape[1] != 3:
            raise ValueError(
                f"atoms must be [B, 3]; got {tuple(atoms.shape)}")
        E = self.num_entities
        atoms = atoms.long()
        atom_hashes = (
            atoms[:, 0] * (E * E) + atoms[:, 1] * E + atoms[:, 2]
        )                                                       # [B]
        pos = torch.searchsorted(self.fact_hashes, atom_hashes)
        pos = pos.clamp(max=int(self.fact_hashes.shape[0]) - 1)
        return self.fact_hashes[pos] == atom_hashes


def _build_rule_tensors(
    compiled_rules: List[CompiledRule],
    num_entities: int,
    max_body: int,
    device: torch.device,
    entity_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Convert CompiledRules to template-variable rule tensors.

    Variable mapping per rule:
      head_var0 → num_entities (template var 0)
      head_var1 → num_entities + 1 (template var 1)
      free_var_i → num_entities + 2 + i

    Body args that are CONSTANTS (e.g. 'high', 'continuous') are encoded
    directly as their entity index (0..num_entities-1), not as template
    vars. CompiledRule treats every arg as a var in its free_vars_list,
    so without this the grounder's FC/BC loses the constant constraint
    and over-generates groundings that ignore constant body atoms.

    Args:
        entity_to_idx: name→idx map for constants. Required to encode
            constant body atoms; if None, constants fall back to template
            var encoding (legacy, buggy behavior).

    Returns:
        rule_heads: [R, 3] head atoms with template vars.
        rule_bodies: [R, M, 3] body atoms with template vars (padded).
        rule_lens: [R] actual body length per rule.
        max_vars: int — max total variables (2 + max_free).
    """
    R = max(len(compiled_rules), 1)
    M = max_body
    V0 = num_entities       # template var for head_var0
    V1 = num_entities + 1   # template var for head_var1

    rule_heads = torch.zeros(R, 3, dtype=torch.long, device=device)
    rule_bodies = torch.zeros(R, M, 3, dtype=torch.long, device=device)
    rule_lens = torch.zeros(R, dtype=torch.long, device=device)
    max_vars = 2  # at least head vars

    def _encode_arg(name: str, binding: int) -> int:
        """Entity index for constants, template var for variables."""
        if (entity_to_idx is not None
                and not is_variable(name)
                and name in entity_to_idx):
            return entity_to_idx[name]
        return num_entities + binding

    for i, cr in enumerate(compiled_rules):
        rule_heads[i, 0] = cr.head_pred_idx
        rule_heads[i, 1] = V0
        rule_heads[i, 2] = V1
        rule_lens[i] = cr.num_body
        n_vars = 2 + cr.num_free
        if n_vars > max_vars:
            max_vars = n_vars

        for j, bp in enumerate(cr.body_patterns):
            rule_bodies[i, j, 0] = bp['pred_idx']
            rule_bodies[i, j, 1] = _encode_arg(
                bp['arg0_var'], bp['arg0_binding'])
            rule_bodies[i, j, 2] = _encode_arg(
                bp['arg1_var'], bp['arg1_binding'])

    return rule_heads, rule_bodies, rule_lens, max_vars


def _grounder_kb(fact_index: TensorFactIndex, rules: list, kb, *, fact_index_type: str,
                 max_facts_per_query: int):
    """The grounder library's ``KB``: the facts ``[F, 3]`` and the rules as template-variable tensors (rules sorted
    by head predicate, as the grounder's rule index sorts them)."""
    from grounder.data import KB
    E = fact_index.num_entities
    dev = fact_index.fact_hashes.device
    rule_names = getattr(kb, "rule_names", None) or [None] * len(rules)
    compiled = sorted((CompiledRule(r, kb.relation2id, name=name) for r, name in zip(rules, rule_names)),
                      key=lambda cr: cr.head_pred_idx)
    max_body = max((cr.num_body for cr in compiled), default=1)
    rule_heads, rule_bodies, rule_lens, max_vars = _build_rule_tensors(
        compiled, E, max_body, dev, entity_to_idx=dict(kb.entity2id))
    facts = torch.stack([fact_index.fact_preds, fact_index.fact_subjs, fact_index.fact_objs], dim=1).long()
    return KB(facts, rule_heads, rule_bodies, rule_lens,
              constant_no=E - 1,                   # entities are 0..E-1
              predicate_no=len(kb.relation2id), padding_idx=E + max_vars,
              device=torch.device(dev) if isinstance(dev, str) else dev,
              fact_index_type=fact_index_type, max_facts_per_query=max_facts_per_query)


# ══════════════════════════════════════════════════════════════════════════════
# Generic adapter
# ══════════════════════════════════════════════════════════════════════════════


class KGEAtomScorer:
    """grounder ``GuidedScorer`` over the reasoner's own KGE model (anything with ``score(h, r, t)``).

    Plain object on purpose (no nn.Module): owns no parameters or buffers, so
    enabling guided grounding never changes the model's state-dict. The KGE ref
    rides a list (not an attribute torch would register). ns and grounder share
    the entity/relation id space, so atoms score directly: ``σ(score(a0, p, a1))``.
    The grounder only calls this on GROUND NON-FACT atoms (facts are exactly 1.0
    fact-index-side; variables are neutral)."""

    def __init__(self, kge_model: nn.Module):
        self._kge_ref = [kge_model]

    @torch.no_grad()
    def score_atoms(self, atoms: Tensor) -> Tensor:   # [N, 3] (p, a0, a1) → [N]
        return torch.sigmoid(
            self._kge_ref[0].score(atoms[:, 1], atoms[:, 0], atoms[:, 2]))


class RandomAtomScorer:
    """grounder ``GuidedScorer`` that scores atoms UNIFORMLY AT RANDOM.

    The control arm for the KGE-guided beam (``guided_tnorm='random'``): same
    top-k grounding-selection machinery, but each ground non-fact atom gets a
    uniform random score in [0,1) instead of ``σ(KGE.score)``. Selecting top-k
    by random scores ≡ keeping k RANDOM groundings — it isolates whether the KGE
    guidance adds value over simply bounding the grounding count to k. Owns no
    parameters (state-dict-neutral); reproducible under a fixed torch seed."""

    @torch.no_grad()
    def score_atoms(self, atoms: Tensor) -> Tensor:   # [N, 3] → [N]
        return torch.rand(atoms.shape[0], device=atoms.device)


class RuleGrounder(nn.Module):
    """A backward grounder over string rules: at init the rules and facts become a :class:`grounder.data.KB` and
    the type string a :class:`grounder.api.config.Backward` config (:func:`_build_backward_config`), built by
    :func:`grounder.api.factory.make_grounder`; :meth:`run_bc` grounds a batch of queries into rule firings.

    Args:
        grounder_type: Type string (e.g. 'sld.prune.d2', 'enum.fp_batch.w1.d2.flat').
        fact_index: TensorFactIndex for fact lookups.
        rules: List of rule tuples ``(head_atom, body_atoms)``.
        kb: the vocabulary (``relation2id``, ``entity2id``, optional ``rule_names``).
        device: Target device.
    """

    def __init__(
        self,
        grounder_type: str,
        fact_index: TensorFactIndex,
        rules: list,
        kb,
        *,
        device: str = 'cuda',
        max_facts_per_query: int = 64,
        max_groundings: int = 32,
        max_total_groundings: int = 64,
        provable_set_method: str = 'join',
        compile_mode=None,
        max_states: Optional[int] = None,
        fact_index_type: str = 'block_sparse',
        chunk_size: int = 0,
        guided_topk: Optional[int] = None,
        guided_tnorm: str = 'min',
        guided_kge: Optional[nn.Module] = None,
    ):
        super().__init__()
        from grounder.api.factory import make_grounder
        kb_obj = _grounder_kb(fact_index, rules, kb, fact_index_type=fact_index_type,
                              max_facts_per_query=max_facts_per_query)

        guided_scorer = None
        gtnorm = guided_tnorm
        if guided_topk is not None:
            if guided_tnorm == "random":
                # Control arm: keep k RANDOM groundings (no KGE guidance). The
                # grounder validates tnorm in {min,product}, so pass "min" — the
                # randomness lives entirely in the scorer's per-atom scores.
                guided_scorer = RandomAtomScorer()
                gtnorm = "min"
            else:
                if guided_kge is None:
                    raise ValueError("guided_topk requires guided_kge (a KGE model with score(h, r, t))")
                guided_scorer = KGEAtomScorer(guided_kge)
        config = _build_backward_config(
            grounder_type, max_groundings=max_groundings,
            max_total_groundings=max_total_groundings, max_states=max_states,
            k_max=_K_MAX, provable_set_method=provable_set_method,
            guided_topk=guided_topk, guided_tnorm=gtnorm,
            guided_scorer=guided_scorer)

        # layout: enum-flat types → flat (eager-only); everything else → dense.
        layout_knob = "flat" if ".flat" in grounder_type else "dense"
        # compile_mode (torch.compile mode) → grounder 'compile' knob; depth-1 and the
        # flat-eager layout stay eager (flat is eager-only; matches the old auto-downgrade).
        _, depth = parse_grounder_type(grounder_type)
        compile_knob = _COMPILE_ALIAS.get(compile_mode, "off")
        if depth <= 1 or layout_knob == "flat":
            compile_knob = "off"
        # chunk_size: 0 → ONE chunk (ns pools are already bounded by
        # batch_size/eval_pool_cap; the lib's ~1 GB auto-budget split family
        # BC12 pools in two for +44% per call); >0 → explicit; <0 → lib auto.
        grounder_chunk = (0 if chunk_size == 0
                          else int(chunk_size) if chunk_size > 0 else None)
        self._inner = make_grounder(kb_obj, config, layout=layout_knob,
                                    compile=compile_knob, chunk_size=grounder_chunk)

        self.fact_index = fact_index
        # Learned-budget provider (rides a list — never registered, so the
        # model state-dict stays byte-identical with/without a budget policy).
        self._guided_budget = [None]
        self._guided_depth = [None]

    # ── runtime guided-search seam (RL / learned-policy experiments) ──
    # The grounder re-snapshots ``guided_scorer`` / ``guided_query_topk`` /
    # ``guided_capture`` from the inner shell on EVERY ground call (same
    # mechanism as the ``guided_stats`` census counters), so these swaps take
    # effect immediately and never touch the constructed config.

    def set_guided_scorer(self, scorer) -> None:
        """Swap the beam's atom prior (a ``grounder.nesy.hooks.GuidedScorer``).

        Replaces the KGE/random scorer chosen at construction — the seam a
        learned selection policy plugs into. The scorer must own no registered
        parameters (ride module refs in a list, like :class:`KGEAtomScorer`).
        """
        self._inner.guided_scorer = scorer

    def set_guided_budget(self, budget) -> None:
        """Attach a per-query budget provider (``None`` disables).

        ``budget.query_topk(queries [B,3]) -> LongTensor [B]`` is evaluated on
        every ``run_bc``/``forward`` call and overrides the scalar
        ``guided_topk`` per query (0 = keep only fact-perfect rows).
        """
        self._guided_budget[0] = budget

    def set_guided_capture(self, capture) -> None:
        """Attach a list collecting state-beam decision records (``None`` off).

        Each record holds the state atoms, pre-chunk query index, t-norm
        scores, and exempt/kept masks — the training signal for
        policy-gradient fine-tuning of a guided scorer.
        """
        self._inner.guided_capture = capture

    def set_guided_sample_tau(self, tau) -> None:
        """Set the stochastic-selection temperature (``None`` = deterministic).

        With a float tau the beam's every top-k (binding + state level) draws
        an exact Plackett-Luce sample of the kept ordered sequence (Gumbel
        top-k on ``log(score)/tau``); with capture attached each record's
        ``order_score`` reproduces the sampled order — together the exact
        density a REINFORCE trainer needs. Training-only; never set for eval.
        """
        self._inner.guided_sample_tau = tau

    def set_guided_depth(self, depth) -> None:
        """Attach a per-query depth-gate provider (``None`` disables).

        ``depth.query_depth(queries [B,3]) -> LongTensor [B]``: query b
        expands only at grounding steps ``d < depth[b]`` — past its gate
        EVERY row dies, fact-perfect included (depth 0 = no grounding,
        KGE-only). Gate D reproduces a plain depth-D run for that query
        exactly (guided_ab contract 13) — the one lever that bounds
        evidence VOLUME, which beam budgets cannot (proof-material rows are
        budget-exempt).
        """
        self._guided_depth[0] = depth

    def _apply_guided_budget(self, queries: Tensor) -> None:
        budget = self._guided_budget[0]
        self._inner.guided_query_topk = (
            budget.query_topk(queries) if budget is not None else None)
        depth = self._guided_depth[0]
        self._inner.guided_query_depth = (
            depth.query_depth(queries) if depth is not None else None)

    def run_bc(
        self,
        queries: Tensor,
        query_mask: Tensor,
        *,
        batch_size: Optional[int] = None,
        pad_outputs: bool = False,
    ):
        """Rule-evidence entry point for SBR/DCR/R2N pool-iter consumers. Width <= 1 pbc grounders take
        :mod:`grounder.backward.fast` (the engine's firings, its atom table compacted to the firings' atoms).

        ``ground()`` for the FIRINGS tier → :class:`grounder.base.types.RuleGroundings`
        with ``query_pool_idx`` populated (the grounder pins every query into the
        atom pool so a reasoner can gather ``pool[query_pool_idx]``).
        ``pad_outputs`` pads each rule slice + the atom table to fixed power-of-2
        shapes (countries_s3/family BC{12,13}) so the compiled reasoner reuses one
        CUDA-graph variant per bucket. (Static pad targets were probed 2026-06-10
        and measured OUT: family BC12's envelope is max-K_r 874 / atoms 102k, so
        padding every call to that ceiling multiplies rule-loop + KGE pool work by
        far more than the ~0.3 ms/call sync it saves.) ``batch_size`` is accepted
        for caller compatibility but unused.
        """
        from grounder.backward import fast
        from grounder.core import GroundRequest, OutputSpec, Tier
        if (fast.supports(self._inner) and self._guided_budget[0] is None
                and self._guided_depth[0] is None):
            # width <= 1 pbc: the fast path — the engine's firings, few host syncs, a compacted atom table
            rg = fast.ground(self._inner, queries, query_mask, chunk_size=self._inner._chunk_size)
        else:
            # FIRINGS-only: the rule path consumes ONLY rule_groundings, and a
            # spec without PROOF_STATE lets the engine skip the final step's
            # pack/postprocess + the GoalState build (~20% of a depth-2 call;
            # firings are captured at resolve time, so rg is byte-identical).
            spec = OutputSpec(frozenset({Tier.FIRINGS}))
            self._apply_guided_budget(queries)
            rg = self._inner.ground(
                GroundRequest(queries=queries, query_mask=query_mask, output_spec=spec)
            ).rule_groundings
        if pad_outputs and rg is not None and rg.rule_offsets.numel() > 1:
            sizes = rg.rule_offsets[1:] - rg.rule_offsets[:-1]
            max_K_r = int(sizes.max().item()) if sizes.numel() else 0
            if max_K_r > 0:
                from grounder.backward.considered import next_pow2, pad_rule_groundings
                rg = pad_rule_groundings(
                    rg, pad_per_rule_to=next_pow2(max(max_K_r, 1)),
                    pad_atom_table_to=next_pow2(max(int(rg.atom_table.size(0)) + 1, 16)),
                    pad_idx_for_atoms=0)
        return rg

    def __repr__(self) -> str:
        return f"RuleGrounder(inner={self._inner!r})"


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

# torch.compile mode string → grounder 'compile' knob ("off"|"graph"|"dynamic").
_COMPILE_ALIAS = {
    None: "off", "none": "off", "off": "off",
    "graph": "graph", "dynamic": "dynamic",
    "default": "graph", "reduce-overhead": "graph", "max-autotune": "graph",
}

# SLD / RTF resolution strategies need the arg-key fact index + the child cap
# (_K_MAX → Backward.max_children).
_K_MAX = 10000
_PROLOG_PREFIXES = ("sld.",)
_RTF_PREFIXES = ("rtf.",)


def parse_grounder_type(grounder_type: str) -> Tuple[int, int]:
    """Parse a grounder name into ``(width, depth)``.

    Dot-separated: ``'sld.prune.d2'`` → ``(1, 2)``, ``'enum.fp_batch.w1.d2'`` →
    ``(1, 2)``. Width/depth default to 1 if the segment is absent.
    """
    m_d = re.search(r"\.d(\d+)", grounder_type)
    m_w = re.search(r"\.w(\d+)", grounder_type)
    return (int(m_w.group(1)) if m_w else 1, int(m_d.group(1)) if m_d else 1)


def _parse_resolution(grounder_type: str) -> str:
    """``'pbc'`` (enum/pbc), ``'sld'``, or ``'rtf'`` from the type string."""
    if grounder_type.startswith(_PROLOG_PREFIXES):
        return "sld"
    if grounder_type.startswith(_RTF_PREFIXES):
        return "rtf"
    return "pbc"   # enum.* / pbc.*


def _build_backward_config(grounder_type: str, *, max_groundings: int,
                           max_total_groundings: int, max_states: Optional[int],
                           k_max: int, provable_set_method: str,
                           guided_topk: Optional[int] = None,
                           guided_tnorm: str = 'min', guided_scorer=None):
    """Type string → typed :class:`grounder.api.config.Backward` config.

    Reproduces the old ``grounder.factory`` shim's parse so grounding stays
    byte-identical: ``max_groundings`` → ``PBC.max_groundings_per_rule`` (per-rule
    Y_r); ``max_total_groundings`` → ``Backward.max_groundings_per_query`` (Y_q
    budget); ``.prune``/``.fp_batch`` → ``filter='fp_batch'`` (PBC u=0 auto-derives
    it, but SLD does not — pass it explicitly or pruning is silently lost);
    ``_K_MAX`` → ``Backward.max_children`` (sld/rtf only).
    """
    from grounder.api.config import PBC, RTF, SLD, Backward
    width, depth = parse_grounder_type(grounder_type)
    res = _parse_resolution(grounder_type)
    common = dict(max_groundings_per_query=max_total_groundings, prune_facts=True)
    if max_states is not None:
        common["max_goals"] = int(max_states)   # grounder renamed the G/states cap max_states→max_goals

    if res == "pbc":
        m_u = re.search(r"\.u(\d+)", grounder_type)
        u = int(m_u.group(1)) if m_u else 0
        # flat layout is selected via the grounder ctor (layout=); "join" provable-set →
        # the in-enumeration width prune (flat_prune), else the one-shot path.
        pbc = PBC(depth=depth, width=width, u=u,
                  max_groundings_per_rule=max_groundings,
                  flat_prune=(provable_set_method == "join"),
                  guided_topk=guided_topk, guided_tnorm=guided_tnorm)
        return Backward(pbc, filter=("fp_batch" if u == 0 else "none"),
                        guided_scorer=guided_scorer, **common)

    if guided_topk is not None:
        raise ValueError(
            f"guided grounding requires an enum/pbc grounder, got {grounder_type!r}")
    common["max_children"] = k_max
    if res == "rtf":
        return Backward(RTF(depth=depth), filter="none", **common)
    # sld: '.prune'/'.fp_batch' → fp_batch filter (else none).
    has_prune = (".prune" in grounder_type) or (".fp_batch" in grounder_type)
    return Backward(SLD(depth=depth), filter=("fp_batch" if has_prune else "none"), **common)


def create_grounder(grounder_type: str, *, fact_index, rules, kb, max_groundings: int,
                    max_total_groundings: int, provable_set_method: str, device, compile_mode=None,
                    max_states: int = None, chunk_size: int = 0, guided_topk: Optional[int] = None,
                    guided_tnorm: str = 'min', guided_kge: Optional[nn.Module] = None) -> "RuleGrounder":
    """A grounder from a type string like ``enum.fp_batch.w1.d2.flat`` behind a :class:`RuleGrounder`.

    ``max_states`` caps the grounder's state budget; ``chunk_size`` is its query chunking (0 = one chunk, >0
    explicit, <0 the library's auto-budget — see ``cfg.grounder_chunk_size``).
    """
    # SLD/RTF use the arg-key fact index; PBC uses the default block-sparse.
    fact_index_type = ("arg_key" if grounder_type.startswith(_PROLOG_PREFIXES + _RTF_PREFIXES)
                       else "block_sparse")
    return RuleGrounder(
        grounder_type, fact_index, rules, kb, device=device,
        max_groundings=max_groundings, max_total_groundings=max_total_groundings,
        provable_set_method=provable_set_method, compile_mode=compile_mode, max_states=max_states,
        fact_index_type=fact_index_type, chunk_size=chunk_size,
        guided_topk=guided_topk, guided_tnorm=guided_tnorm, guided_kge=guided_kge)
