"""RunPlan — the immutable shell→backward snapshot.

``snapshot`` reads every ``grounder.X`` ONCE into this frozen plan; the runtime
then reads ONLY the plan (+ threaded RunState), never the nn.Module — so backward
grounding is reentrant + thread-safe by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, TYPE_CHECKING, Optional

from torch import Tensor

from grounder.core import OutputSpec
from grounder.execution.capability import EAGER, CapabilityRow, Cell
from grounder.execution.chunk_policy import ChunkPolicy
from grounder.execution.strategy import ExecStrategy
from grounder.resolution.pbc import PbcPlan, build_plan
from grounder.vocab.shapes import Shapes
from grounder.base.types import Layout

if TYPE_CHECKING:
    from grounder.data.kb import KB


@dataclass(frozen=True)
class RunPlan:
    """Per-run snapshot handed shell→backward. Immutable; ``for_chunk`` rebatches a copy."""

    shapes: Shapes
    kb: "KB"
    output_spec: OutputSpec
    strategy: ExecStrategy
    pbc: Optional[PbcPlan]               # build_plan(grounder) ONCE; None for sld/rtf
    resolution: str
    filter_mode: str
    depth: int
    width: Optional[int]
    w_last_depth: int
    G: int
    Y_q: int
    max_atoms: int
    max_vars_per_rule: int
    max_fact_pairs_body: int
    flat_prune: bool                     # pbc flat: push width into the enumeration (per-fv prune)
    pack_dedup: bool
    prune_facts: bool
    all_anchors: bool
    init_state_shape: str
    standardize: bool                    # = _standardize_fn is not None
    standardize_fn: Optional[Callable]   # terminal output-var renaming (REFERENCE)
    variant_to_orig: Optional[Tensor]    # REFERENCE, not clone
    fact_hook: object
    rule_hook: object
    guided_topk: Optional[int]           # KGE-guided beam width (join path; None = off)
    guided_tnorm: str
    guided_scorer: object                # GuidedScorer (REFERENCE)
    guided_stats: object                 # GuidedStats census counters (REFERENCE; mutable)

    @staticmethod
    def snapshot(grounder) -> "RunPlan":
        """Read every grounder.X ONCE into a frozen plan (reentrant from here)."""
        kb = grounder.kb
        B = getattr(grounder, "B", 1)
        is_pbc = grounder.resolution == "pbc"
        shapes = Shapes(
            B=B, G=grounder.G, L=grounder.max_atoms, M=kb.M, D=grounder.depth, A=grounder.A,
            Y_q=grounder.Y_q, K_f=kb.K_f, K_r=getattr(grounder, "K_r", kb.K_r),
            Y_r=getattr(grounder, "Y_r", 1), K_v=getattr(grounder, "K_v", 1),
            V=getattr(grounder, "V", 1), E=getattr(grounder, "_E", kb.constant_no),
            N=B * grounder.G, pad=kb.padding_idx)
        return RunPlan(
            shapes=shapes, kb=kb,
            output_spec=grounder.output_spec,  # set in __init__ (no eager default → compile-safe)
            strategy=RunPlan._resolve_strategy(grounder, shapes, is_pbc),
            pbc=build_plan(grounder) if is_pbc else None,
            resolution=grounder.resolution,
            filter_mode=grounder.filter_mode,
            depth=grounder.depth, width=grounder.width,
            w_last_depth=grounder.w_last_depth, G=grounder.G, Y_q=grounder.Y_q,
            max_atoms=grounder.max_atoms, max_vars_per_rule=grounder.max_vars_per_rule,
            max_fact_pairs_body=getattr(grounder, "_max_fact_pairs_body", 0),
            flat_prune=getattr(grounder, "_flat_prune", False),
            pack_dedup=grounder._pack_dedup, prune_facts=grounder.prune_facts,
            all_anchors=grounder._all_anchors,
            init_state_shape=grounder._init_state_shape,
            standardize=grounder._standardize_fn is not None,
            standardize_fn=grounder._standardize_fn,
            variant_to_orig=getattr(grounder, "_variant_to_orig_t", None),
            fact_hook=grounder.fact_hook, rule_hook=grounder.rule_hook,
            guided_topk=getattr(grounder, "guided_topk", None),
            guided_tnorm=getattr(grounder, "guided_tnorm", "min"),
            guided_scorer=getattr(grounder, "guided_scorer", None),
            guided_stats=getattr(grounder, "guided_stats", None))

    @staticmethod
    def _resolve_strategy(grounder, shapes, is_pbc) -> ExecStrategy:
        """Knobs set → explicit cell+chunk. Else auto, then: compile is opt-in (downgrade
        an auto compiled cell to its eager sibling, flat flag picks which), and sld/rtf
        are forced dense (they route layout via strategy.layout(), not flat_intermediate)."""
        row = grounder.capability_row()
        if getattr(grounder, "_knobs_set", False):
            cs = grounder._chunk_size
            chunk = ExecStrategy.default_chunk(shapes) if cs is None else ChunkPolicy(batch_size=int(cs))
            return ExecStrategy.explicit(row, Cell(grounder._exec_layout, grounder._exec_compile), chunk)
        strategy = ExecStrategy.auto(row, shapes=shapes)
        if not strategy.cell.compile.eager and not getattr(grounder, "compile_enabled", False):
            want = Layout.FLAT if is_pbc else Layout.DENSE   # auto flat-pref: pbc→flat, sld/rtf→dense
            return ExecStrategy.explicit(row, Cell(want, EAGER), strategy.chunk)
        if not is_pbc and strategy.cell.layout is Layout.FLAT:
            return ExecStrategy.explicit(row, Cell(Layout.DENSE, EAGER), strategy.chunk)
        return strategy

    @property
    def flat_intermediate(self) -> bool:
        """flat-vs-dense is an EXEC-layer decision — derived from the resolved cell layout,
        not stored. (Kept as a property so the resolver + snapshot tests read one source.)"""
        return self.strategy.cell.layout is Layout.FLAT

    # which evidence tiers to collect — derived from the ONE source (output_spec), not stored.
    @property
    def collect_evidence(self) -> bool:           # TREES tier (CompletedTreeFirings)
        return self.output_spec.trees

    @property
    def collect_rule_groundings(self) -> bool:    # FIRINGS tier (RuleGroundings)
        return self.output_spec.firings

    def for_chunk(self, B: int) -> "RunPlan":
        """New immutable plan with Shapes rebatched to ``B``."""
        return replace(self, shapes=self.shapes.with_batch(B))


__all__ = ["RunPlan"]
