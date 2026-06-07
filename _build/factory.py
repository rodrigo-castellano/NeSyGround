"""Grounder factory — ONE construction entry: ``make_grounder(kb, config, ...)``.

``make_grounder`` dispatches on the config TYPE (the config encodes resolution +
``filter()``), collapsing the old BC/FC string fork:
  SLDConfig / RTFConfig / PBCConfig -> BackwardGrounder
  FCConfig                          -> ForwardGrounder

``create_grounder`` / ``parse_grounder_type`` / ``make_bcwd`` are thin shims that
build a typed config and delegate. They preserve today's defaults exactly
(u->filter, all_anchors forced in the ctor, flat_intermediate).

Type grammar (dot-separated):  {resolution}[.{filter}][.pd][.full|.wW][.dD]
  sld.fp_batch.d2 · rtf.d4 · pbc.fp_batch.w1.d2 · pbc.full · closure
BC_{w,d}[u] shorthand: bc12, bc13, bc12u1 → pbc with the paper-aligned filter.

``enum`` is accepted as a legacy alias for ``pbc`` (the new resolution name).
``make_bcwd(kb, w, d, u=0)`` is the direct BC_{w,d,u} constructor.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.config import FCConfig, PBCConfig, RTFConfig, SLDConfig
from grounder._build.data.kb import KB
from grounder._build.grounder.backward import BackwardGrounder

GrounderConfigT = Union[SLDConfig, RTFConfig, PBCConfig, FCConfig]

_PATTERN = re.compile(
    r"^(?P<resolution>sld|rtf|enum|pbc|closure)"
    r"(\.(?P<filter>fp_batch|fp_global|prune|provset|none))?"
    r"(?P<pd>\.pd)?"
    r"(\.full|\.w(?P<width>\d+))?"
    r"(\.d(?P<depth>\d+))?$"
)
_BCWD_PATTERN = re.compile(r"^bc(?P<width>\d)(?P<depth>\d)(?:u(?P<u>\d))?$")
_FILTER_ALIASES = {"prune": "fp_batch", "provset": "fp_global"}


def _default_pbc_filter(u: int) -> str:
    """u=0 (paper) → fp_batch (keras prune=True); u>0 → none."""
    return "fp_batch" if u == 0 else "none"


def parse_grounder_type(grounder_type: str) -> Dict[str, Any]:
    """Parse a type string into a config dict (resolution uses the new ``pbc`` name)."""
    m_bcwd = _BCWD_PATTERN.match(grounder_type)
    if m_bcwd is not None:
        u = int(m_bcwd.group("u")) if m_bcwd.group("u") is not None else 0
        return {"resolution": "pbc", "filter": _default_pbc_filter(u),
                "depth": int(m_bcwd.group("depth")), "width": int(m_bcwd.group("width")),
                "is_full": False, "step_prune_dead": False,
                "flat_intermediate": True, "u": u}

    clean = grounder_type.replace(".flat", "")
    m = _PATTERN.match(clean)
    if not m:
        raise ValueError(
            f"Unknown grounder type: {grounder_type!r}. Expected "
            f"'{{sld|rtf|pbc|closure}}[.{{filter}}][.wW][.dD]' or BC shorthand 'bcWD'.")

    resolution = m.group("resolution")
    if resolution == "enum":          # legacy alias
        resolution = "pbc"
    is_full = ".full" in grounder_type
    flat_intermediate = ".flat" in grounder_type
    depth = int(m.group("depth")) if m.group("depth") else 1
    width = int(m.group("width")) if m.group("width") else 1
    default_filter = _default_pbc_filter(0) if resolution == "pbc" else "none"
    filter_mode = m.group("filter") or default_filter
    filter_mode = _FILTER_ALIASES.get(filter_mode, filter_mode)
    return {"resolution": resolution, "filter": filter_mode, "depth": depth,
            "width": width, "is_full": is_full,
            "step_prune_dead": bool(m.group("pd")), "flat_intermediate": flat_intermediate, "u": 0}


# ── make_grounder: the single construction entry (dispatch on config TYPE) ──

def _build_backward(kb: KB, config: GrounderConfigT, *, resolution: str,
                    layout: str, compile: str, chunk_size: Optional[int],
                    extra: Dict[str, Any]) -> BackwardGrounder:
    """Translate a backward config + exec knobs into BackwardGrounder ctor kwargs.

    ``filter`` is NOT forced from ``config.filter()`` here: an explicit filter
    (type-string / make_bcwd) rides in ``extra``; otherwise the ctor derives the
    u-based default (pbc u=0 -> fp_batch, u>0 -> none; sld/rtf -> none), which is
    exactly today's behavior and what ``config.filter()`` encodes for u=0/PBC."""
    kw: Dict[str, Any] = dict(
        resolution=resolution, depth=config.depth,
        max_total_groundings=config.max_total_groundings,
        collect_rule_groundings=config.collect_rule_groundings,
        layout=layout, compile=compile, chunk_size=chunk_size)
    if isinstance(config, PBCConfig):
        kw["width"] = config.w
        kw["u"] = config.u
        kw["flat_intermediate"] = config.flat_intermediate
        if config.max_groundings_per_query is not None:
            kw["max_groundings_per_query"] = config.max_groundings_per_query
    kw.update(extra)
    return BackwardGrounder(kb, **kw)


def make_grounder(
    kb: KB,
    config: GrounderConfigT,
    *,
    layout: str = "auto",
    compile: str = "off",
    chunk_size: Optional[int] = None,
    transforms: Sequence = (),
    **extra: Any,
) -> nn.Module:
    """Build the family grounder over ``kb`` by dispatching on ``type(config)``.

    ``transforms`` (Phase A: empty) will wrap the grounder in a ``Pipeline``; the
    seam lands with the transform axis, so a non-empty list raises here for now.
    """
    if transforms:
        from grounder._build.core.pipeline import Pipeline
        base = make_grounder(kb, config, layout=layout, compile=compile,
                             chunk_size=chunk_size, **extra)
        return Pipeline(transforms, base)

    if isinstance(config, FCConfig):
        from grounder._build.forward.grounder import ForwardGrounder
        return ForwardGrounder(kb, method=config.method, depth=config.depth,
                               join_algo=config.join_algo, **extra)
    if isinstance(config, (SLDConfig, RTFConfig, PBCConfig)):
        resolution = ("pbc" if isinstance(config, PBCConfig)
                      else "rtf" if isinstance(config, RTFConfig) else "sld")
        return _build_backward(kb, config, resolution=resolution, layout=layout,
                               compile=compile, chunk_size=chunk_size, extra=extra)
    raise TypeError(f"Unknown grounder config type: {type(config).__name__}")


def create_grounder(
    grounder_type: str,
    *,
    facts_idx: Tensor,
    rule_heads: Tensor,
    rule_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
    padding_idx: int,
    device: torch.device,
    predicate_no: Optional[int] = None,
    max_facts_per_query: int = 64,
    fact_index_type: str = "block_sparse",
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "spmm",
    max_goals: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a KB + grounder from a type string — a shim over ``make_grounder``."""
    cfg = parse_grounder_type(grounder_type)
    kb = KB(facts_idx, rule_heads, rule_bodies, rule_lens,
            constant_no=constant_no, predicate_no=predicate_no, padding_idx=padding_idx,
            device=device, max_facts_per_query=max_facts_per_query, fact_index_type=fact_index_type)

    if cfg["resolution"] == "closure":
        return make_grounder(kb, FCConfig(depth=10, method=fc_method), **kwargs)

    extra: Dict[str, Any] = dict(max_goals=max_goals)
    if cfg["resolution"] == "pbc":
        depth = 1 if cfg["is_full"] else cfg["depth"]
        width = None if cfg["is_full"] else cfg["width"]
        config: GrounderConfigT = PBCConfig(
            depth=depth, w=width, u=cfg["u"], flat_intermediate=cfg["flat_intermediate"],
            max_groundings_per_query=max_groundings,
            max_total_groundings=max_total_groundings)
        extra["fc_method"] = fc_method
        # filter() on PBCConfig is always fp_batch; honor an explicit type-string filter.
        extra["filter"] = cfg["filter"]
    elif cfg["resolution"] == "rtf":
        config = RTFConfig(depth=cfg["depth"], max_total_groundings=max_total_groundings)
        extra["width"] = cfg["width"]
        extra["filter"] = cfg["filter"]
    else:  # sld
        config = SLDConfig(depth=cfg["depth"], max_total_groundings=max_total_groundings)
        extra["width"] = cfg["width"]
        extra["filter"] = cfg["filter"]
    extra.update(kwargs)
    return make_grounder(kb, config, **extra)


def make_bcwd(kb: KB, w: int, d: int, u: int = 0, *,
              flat_intermediate: bool = True, filter: Optional[str] = None,
              **kwargs) -> BackwardGrounder:
    """The paper's BC_{w,d,u} grounder directly — a shim over ``make_grounder``."""
    config = PBCConfig(depth=d, w=w, u=u, flat_intermediate=flat_intermediate,
                       max_total_groundings=kwargs.pop("max_total_groundings", 64))
    extra: Dict[str, Any] = dict(kwargs)
    if filter is not None:                # else PBCConfig.filter()==fp_batch derives it
        extra["filter"] = filter
    return make_grounder(kb, config, **extra)


__all__ = ["parse_grounder_type", "make_grounder", "create_grounder", "make_bcwd"]
