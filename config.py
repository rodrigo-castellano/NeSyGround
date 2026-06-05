"""GrounderConfig — the single, validated construction surface for the grounder.

Phase 2a of the restructure: this consolidates the ~30 loose ``BCGrounder``
keyword arguments into one dataclass with explicit grouping and combo
validation, WITHOUT changing any runtime behavior. ``as_kwargs()`` expands
back to the exact constructor arguments ``BCGrounder.__init__`` already
accepts, and ``BCGrounder.from_config(kb, cfg)`` is the clean entry point.

Later phases tighten this (move test-only/forced/computed params off the
public surface, unify the three grounding limits, consolidate hooks/evidence
toggles). For now it is a faithful 1:1 mirror so the migration is provably
behavior-neutral: a grounder built ``from_config`` must produce a byte-
identical grounding fingerprint to one built with the legacy kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

# Legal resolution modes and the (resolution -> allowed filters) matrix.
_RESOLUTIONS = ("sld", "rtf", "enum", "closure")
_FILTERS = ("fp_batch", "fp_global", "none", None)


@dataclass
class GrounderConfig:
    """Configuration for :class:`grounder.bc.bc.BCGrounder`.

    Field groups (faithful mirror of the constructor; see ``BCGrounder``
    for per-field semantics):

    - core:      ``resolution``, ``filter``, ``depth``, ``width``
    - limits:    ``max_total_groundings`` (C), ``max_groundings_per_query``
                 (G_r), ``max_groundings_per_rule``, ``max_goals``,
                 ``max_states``, ``K_MAX``, ``max_derived_per_state``,
                 ``w_last_depth``
    - compile:   ``compile_mode``, ``init_state_shape``, ``bump_s_to_k``
    - evidence:  ``collect_evidence``, ``collect_mode``,
                 ``collect_rule_groundings``, ``pack_dedup``
    - hooks:     ``hooks``, ``fact_hook``, ``rule_hook``
    - enum:      ``fc_method``, ``fc_depth``, ``cartesian_product``,
                 ``all_anchors``, ``flat_intermediate``
    - misc:      ``standardization``, ``prune_facts``, ``step_prune_dead``
    """
    # core
    resolution: str = "enum"
    filter: Optional[str] = None
    depth: int = 2
    width: Optional[int] = 1
    # limits
    max_total_groundings: int = 64
    max_groundings_per_query: int = 32
    max_groundings_per_rule: Optional[int] = None
    max_goals: Optional[int] = None
    max_states: Optional[int] = None
    K_MAX: int = 550
    max_derived_per_state: Optional[int] = None
    w_last_depth: Optional[int] = None
    # compile
    compile_mode: Optional[str] = None
    init_state_shape: str = "minimal"
    bump_s_to_k: bool = True
    # evidence
    collect_evidence: bool = True
    collect_mode: str = "terminal"
    collect_rule_groundings: bool = False
    pack_dedup: bool = True
    # hooks
    hooks: Optional[List] = None
    fact_hook: Optional[Any] = None
    rule_hook: Optional[Any] = None
    # enum
    fc_method: str = "spmm"
    fc_depth: int = 10
    cartesian_product: bool = False
    all_anchors: bool = False
    flat_intermediate: bool = False
    # misc
    standardization: Optional[Any] = None
    prune_facts: bool = False
    step_prune_dead: bool = False

    def __post_init__(self) -> None:
        # ── Validate ──
        if self.resolution not in _RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_RESOLUTIONS}, got {self.resolution!r}")
        if self.filter not in _FILTERS:
            raise ValueError(
                f"filter must be one of {_FILTERS}, got {self.filter!r}")
        # closure has no proof-tree to prune; only 'none'/None is meaningful.
        if self.resolution == "closure" and self.filter not in (None, "none"):
            raise ValueError(
                f"resolution='closure' requires filter in (None,'none'), "
                f"got {self.filter!r}")
        if self.init_state_shape not in ("minimal", "full"):
            raise ValueError(
                f"init_state_shape must be 'minimal' or 'full', "
                f"got {self.init_state_shape!r}")

        # ── Derive the resolved configuration (kb-independent). This is the
        # single home for the forced/defaulted values BCGrounder used to
        # compute inline in its constructor. ──
        # Default filter: paper BC_{w,d,u=0} convention. enum -> 'fp_batch'
        # (keras prune_incomplete_proofs=True); closure/sld/rtf -> 'none'.
        if self.filter is None:
            self.filter = ("none" if self.resolution == "closure"
                           else ("fp_batch" if self.resolution == "enum"
                                 else "none"))
        # all_anchors is forced True for enum: anchoring only on the first
        # body atom misses bindings keras finds (the dedup collapses the
        # K_r anchor variants of one logical application to a single entry).
        if self.resolution == "enum":
            self.all_anchors = True
        # u (= w_last_depth) defaults to 0 — every leaf body atom must be a
        # fact at the terminal step (paper convention).
        if self.w_last_depth is None:
            self.w_last_depth = 0

        # ── Derived (non-field) attributes consumed by BCGrounder ──
        # Per-step width filter applies to SLD/RTF only (enum filters by w).
        self.step_width: Optional[int] = (
            self.width if self.resolution in ("sld", "rtf")
            and self.width is not None else None)
        # prune_dead is meaningful only for SLD/RTF (enum body atoms are all
        # ground), so it's silently a no-op for enum — warn and disable.
        if self.step_prune_dead and self.resolution == "enum":
            import warnings
            warnings.warn(
                "step_prune_dead has no effect with enum resolution "
                "(all body atoms are ground). Ignoring.",
                stacklevel=2,
            )
        self.step_prune_dead_effective: bool = (
            self.step_prune_dead and self.resolution in ("sld", "rtf"))
        # Standardization mode (None when no standardization is configured).
        self.standardization_mode = (
            self.standardization.mode if self.standardization is not None
            else None)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "GrounderConfig":
        """Build from the legacy ``BCGrounder`` keyword names, ignoring any
        the dataclass doesn't model (forward-compatible bridge)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    def as_kwargs(self) -> Dict[str, Any]:
        """Expand to the exact ``BCGrounder.__init__`` keyword arguments."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = ["GrounderConfig"]
