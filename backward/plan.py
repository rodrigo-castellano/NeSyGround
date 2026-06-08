"""RunPlan — the immutable shell->backward snapshot.

The backward runtime reads ONLY this (plus the threaded RunState); it NEVER reads
an attribute off the Grounder/nn.Module. Backward grounding is reentrant +
thread-safe by construction. ``snapshot`` reads every ``grounder.X`` ONCE; the
runtime then redirects every hot-path read through this frozen plan.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, TYPE_CHECKING, Optional

from torch import Tensor

from grounder.core import OutputSpec   # typed tier set (threaded through the plan)
from grounder.execution.capability import EAGER, CapabilityRow, Cell
from grounder.execution.chunk_policy import ChunkPolicy
from grounder.execution.strategy import ExecStrategy
from grounder.resolution.pbc import PbcPlan, build_plan
from grounder.vocab.shapes import Shapes
from grounder.base.types import Layout

if TYPE_CHECKING:                       # cross-layer types
    from grounder.data.kb import KB


def _resolve_cell(grounder) -> Cell:
    return Cell(grounder._exec_layout, grounder._exec_compile)


def _resolve_chunk(grounder, shapes) -> ChunkPolicy:
    cs = grounder._chunk_size
    if cs is None:
        return ExecStrategy.default_chunk(shapes)
    return ChunkPolicy(batch_size=int(cs))


@dataclass(frozen=True)
class RunPlan:
    """Per-run snapshot handed shell->backward. Immutable; ``for_chunk`` returns a
    rebatched copy. The engine never stores any of this on the module."""

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
    S: int
    Y_q: int
    max_goals: int
    max_vars_per_rule: int
    max_fact_pairs_body: int
    collect_evidence: bool
    collect_rule_groundings: bool
    collect_mode: str
    flat_intermediate: bool
    pack_dedup: bool
    prune_facts: bool
    all_anchors: bool
    init_state_shape: str
    standardize: bool                    # = _standardize_fn is not None
    standardize_fn: Optional[Callable]   # terminal output-var renaming (REFERENCE)
    variant_to_orig: Optional[Tensor]    # REFERENCE, not clone
    fact_hook: object
    rule_hook: object

    @staticmethod
    def snapshot(grounder) -> "RunPlan":
        """Read every grounder.X ONCE into a frozen plan (reentrant from here)."""
        kb = grounder.kb
        is_pbc = grounder.resolution == "pbc"
        pbc = build_plan(grounder) if is_pbc else None
        shapes = Shapes(
            B=getattr(grounder, "B", 1), G=grounder.S, L=grounder.max_goals,
            M=kb.M, D=grounder.depth, A=grounder.A,
            Y_q=grounder.Y_q, K_f=kb.K_f, K_r=getattr(grounder, "K_r", kb.K_r),
            Y_r=getattr(grounder, "Y_r", 1), K_v=getattr(grounder, "K_v", 1),
            V=getattr(grounder, "V", 1), E=getattr(grounder, "_E", kb.constant_no),
            N=getattr(grounder, "B", 1) * grounder.S, pad=kb.padding_idx)
        row = grounder.capability_row()
        if getattr(grounder, "_knobs_set", False):
            strategy = ExecStrategy.explicit(row, _resolve_cell(grounder),
                                             _resolve_chunk(grounder, shapes))
        else:
            strategy = ExecStrategy.auto(row, shapes=shapes)
            # Compile is opt-in: downgrade an auto-picked compiled cell to its
            # eager-declared sibling, keyed on the flat flag (PBC declares two eager
            # cells), so wrap_step stays eager.
            if (not strategy.cell.compile.eager
                    and not getattr(grounder, "compile_enabled", False)):
                want = Layout.FLAT if grounder._flat_intermediate else Layout.DENSE
                strategy = ExecStrategy.explicit(row, Cell(want, EAGER), strategy.chunk)
            # sld/rtf route layout via strategy.layout() (not flat_intermediate); under
            # auto they always run dense, so keep the auto cell DENSE for them.
            elif not is_pbc and strategy.cell.layout is Layout.FLAT:
                strategy = ExecStrategy.explicit(row, Cell(Layout.DENSE, EAGER), strategy.chunk)
        return RunPlan(
            shapes=shapes, kb=kb,
            output_spec=grounder.output_spec,  # always set in __init__ (no eager OutputSpec() default → compile-safe)
            strategy=strategy, pbc=pbc,
            resolution=getattr(grounder, "_dispatch_resolution", grounder.resolution),
            filter_mode=grounder.filter_mode,
            depth=grounder.depth, width=grounder.width,
            w_last_depth=grounder.w_last_depth, S=grounder.S, Y_q=grounder.Y_q,
            max_goals=grounder.max_goals, max_vars_per_rule=grounder.max_vars_per_rule,
            max_fact_pairs_body=getattr(grounder, "_max_fact_pairs_body", 0),
            collect_evidence=grounder.collect_evidence,
            collect_rule_groundings=grounder._collect_rule_groundings,
            collect_mode=grounder._collect_mode,
            flat_intermediate=grounder._flat_intermediate,
            pack_dedup=grounder._pack_dedup, prune_facts=grounder.prune_facts,
            all_anchors=grounder._all_anchors,
            init_state_shape=grounder._init_state_shape,
            standardize=grounder._standardize_fn is not None,
            standardize_fn=grounder._standardize_fn,
            variant_to_orig=getattr(grounder, "_variant_to_orig_t", None),
            fact_hook=grounder.fact_hook, rule_hook=grounder.rule_hook)

    def for_chunk(self, B: int) -> "RunPlan":
        """New immutable plan with Shapes rebatched to ``B``."""
        return replace(self, shapes=self.shapes.with_batch(B))

    def needs_provenance(self) -> bool:
        return self.output_spec.needs_provenance()


__all__ = ["RunPlan"]
