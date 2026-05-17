"""Grounder factory — parse grounder type string and instantiate BCGrounder.

Naming convention (dot-separated):
    {resolution}[.{filter}].d{depth}[.w{width}]

Examples:
    sld.d2              SLD resolution, no filter, depth 2
    sld.fp_batch.d2     SLD + cross-query Kleene T_P, depth 2
    sld.fp_global.d3    SLD + FC provable set filter, depth 3
    rtf.d2              Rule-Then-Fact, depth 2
    enum.fp_batch.w1.d2 Enum resolution, fp_batch, width 1, depth 2
    enum.full           Enum, all entities, depth 1

BC_{w,d} shorthand (paper notation):
    bc01, bc11, bc12, bc13, ...     parametrized BC_{width=W, depth=D};
                                     auto-configures resolution='enum',
                                     all_anchors=True, flat_intermediate=True,
                                     and the keras-prune-aligned filter:
                                       d=1, w>0  → 'none'
                                       else      → 'fp_batch'
                                     ``make_bcwd(kb, w, d, ...)`` is the
                                     equivalent direct constructor.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.kb import KB
from grounder.bc.bc import BCGrounder


# ======================================================================
# Parser
# ======================================================================

_PATTERN = re.compile(
    r"^(?P<resolution>sld|rtf|enum|closure)"
    r"(\.(?P<filter>fp_batch|fp_global|prune|provset|none))?"
    r"(?P<pd>\.pd)?"
    r"(\.full|\.w(?P<width>\d+))?"
    r"(\.d(?P<depth>\d+))?$"
)

# Paper BC_{w,d}[_{u}] shorthand:
#   bcWD       → w=W, d=D, u=0 (paper convention)
#   bcWDuU     → explicit u (rare; non-paper)
# Single digit per param. For multi-digit values use ``make_bcwd`` directly,
# or the long form ``enum.wW.dD``.
_BCWD_PATTERN = re.compile(r"^bc(?P<width>\d)(?P<depth>\d)(?:u(?P<u>\d))?$")


def _default_enum_filter(u: int) -> str:
    """Filter default for enum BC_{w,d,u}, keyed on u (= last-step
    unknown-atom cap, == ``w_last_depth``).

    The paper always uses u=0: at the leaves of the proof tree every
    body atom must be a fact, and ``fp_batch`` prunes any intermediate
    rule application whose unknowns aren't derived through the chain.

    u=0  → 'fp_batch' (paper case; keras ``prune_incomplete_proofs=True``)
    u>0  → 'none'     (admits unknown leaves; keras ``prune=False``)
    """
    return "fp_batch" if u == 0 else "none"


def parse_grounder_type(grounder_type: str) -> Dict[str, Any]:
    """Parse a grounder type string into a config dict.

    Returns dict with keys: resolution, filter, depth, width, is_full,
    step_prune_dead, flat_intermediate.

    Examples:
        'sld.fp_batch.d2'         → {resolution:'sld', filter:'fp_batch', depth:2, ...}
        'enum.fp_batch.w1.d2'     → {resolution:'enum', filter:'fp_batch', depth:2, width:1}
        'enum.full'               → {resolution:'enum', filter:'fp_batch', depth:1, is_full:True}
        'bc11'                    → enum BC_{w=1,d=1}: filter='none', flat_intermediate=True
        'bc12'                    → enum BC_{w=1,d=2}: filter='fp_batch', flat_intermediate=True
    """
    # ── Paper BC_{w,d,u} shorthand ──
    m_bcwd = _BCWD_PATTERN.match(grounder_type)
    if m_bcwd is not None:
        width = int(m_bcwd.group("width"))
        depth = int(m_bcwd.group("depth"))
        # u defaults to 0 (paper convention).
        u_str = m_bcwd.group("u")
        u = int(u_str) if u_str is not None else 0
        cfg = {
            "resolution": "enum",
            "filter": _default_enum_filter(u),
            "depth": depth,
            "width": width,
            "is_full": False,
            "step_prune_dead": False,
            # flat_intermediate is the right default for BC_{w,d} (zero
            # grounding loss when V≥2; falls through to the dense path
            # for V<2 cases).
            "flat_intermediate": True,
            "u": u,
        }
        return cfg

    # Strip known suffixes before regex matching
    clean_type = grounder_type.replace(".flat", "")
    m = _PATTERN.match(clean_type)
    if not m:
        raise ValueError(
            f"Unknown grounder type: {grounder_type!r}. "
            f"Expected format: {{resolution}}[.{{filter}}].d{{depth}}[.w{{width}}]  "
            f"(e.g. 'sld.fp_batch.d2', 'enum.fp_batch.w1.d2', 'rtf.d4'), or "
            f"the BC_{{w,d}} shorthand 'bcWD' (e.g. 'bc11', 'bc13')."
        )

    resolution = m.group("resolution")
    is_full = ".full" in grounder_type
    flat_intermediate = ".flat" in grounder_type

    depth = int(m.group("depth")) if m.group("depth") else 1
    width = int(m.group("width")) if m.group("width") else 1

    # SLD/RTF have no parity story → 'none'. The long-form enum string
    # has no explicit u; default to the paper convention (u=0) so the
    # filter resolves to 'fp_batch' (keras prune=True equivalent).
    if resolution == "enum":
        default_filter = _default_enum_filter(0)
    else:
        default_filter = "none"

    filter_mode = m.group("filter") or default_filter
    # Normalize legacy names
    _FILTER_ALIASES = {"prune": "fp_batch", "provset": "fp_global"}
    filter_mode = _FILTER_ALIASES.get(filter_mode, filter_mode)

    return {
        "resolution": resolution,
        "filter": filter_mode,
        "depth": depth,
        "width": width,
        "is_full": is_full,
        "step_prune_dead": bool(m.group("pd")),
        "flat_intermediate": flat_intermediate,
    }


# ======================================================================
# Factory
# ======================================================================

def create_grounder(
    grounder_type: str,
    *,
    # KB params
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
    # Algorithm params
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "join",
    max_goals: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Create a grounder from a type string.

    Builds a KB from the data params, then instantiates BCGrounder.

    Args:
        grounder_type: e.g. 'sld.fp_batch.d2', 'enum.fp_batch.w1.d2', 'rtf.d4'.

    Returns:
        BCGrounder instance.
    """
    cfg = parse_grounder_type(grounder_type)

    kb = KB(
        facts_idx, rule_heads, rule_bodies, rule_lens,
        constant_no=constant_no, predicate_no=predicate_no,
        padding_idx=padding_idx, device=device,
        max_facts_per_query=max_facts_per_query,
        fact_index_type=fact_index_type,
    )

    bc_kwargs = dict(
        resolution=cfg["resolution"],
        filter=cfg["filter"],
        depth=cfg["depth"],
        max_total_groundings=max_total_groundings,
        max_goals=max_goals,
    )
    bc_kwargs["step_prune_dead"] = cfg["step_prune_dead"]

    if cfg["resolution"] == "enum":
        if cfg["is_full"]:
            bc_kwargs["width"] = None
            bc_kwargs["depth"] = 1
        else:
            bc_kwargs["width"] = cfg["width"]
        bc_kwargs["max_groundings_per_query"] = max_groundings
        bc_kwargs["fc_method"] = fc_method
        bc_kwargs["flat_intermediate"] = cfg["flat_intermediate"]

    bc_kwargs.update(kwargs)

    # Width: always pass to BCGrounder (for SLD/RTF it activates the hook)
    if "width" not in bc_kwargs:
        bc_kwargs["width"] = cfg["width"]

    return BCGrounder(kb, **bc_kwargs)


# ======================================================================
# BC_{w,d} parametrized constructor (paper notation)
# ======================================================================

def make_bcwd(
    kb: KB,
    w: int,
    d: int,
    u: int = 0,
    *,
    flat_intermediate: bool = True,
    filter: Optional[str] = None,
    **kwargs,
) -> BCGrounder:
    """Build the paper's BC_{w,d,u} grounder directly from (w, d, u).

    Paper parametrization (matches `keras-ns
    ApproximateBackwardChainingGrounder` knobs):

      * ``w`` — ``max_unknown_fact_count`` at intermediate steps.
      * ``d`` — ``num_steps`` (proof depth).
      * ``u`` — ``max_unknown_fact_count_last_step`` (last-depth cap).
                The paper convention is ``u=0`` (every leaf body atom
                must be a fact); the IJCAI '25 experiments use this
                everywhere. ``u>0`` admits unknown leaves and is rare.

    Internal mapping: ``u`` is forwarded as ``BCGrounder.w_last_depth``.
    Both names refer to the same per-step unknown cap at ``d == depth-1``.

    Defaults (all overridable via kwargs):
      * ``resolution='enum'``
      * ``all_anchors=True``      (forced by ``BCGrounder.__init__`` for enum)
      * ``flat_intermediate``     defaults to True (zero grounding loss
                                    when V≥2; falls through to the dense
                                    path for V<2)
      * ``filter``                defaults via :func:`_default_enum_filter`:
                                    u=0 → ``'fp_batch'`` (paper / keras
                                          ``prune_incomplete_proofs=True``)
                                    u>0 → ``'none'``    (keras ``prune=False``)

    Args:
        kb: pre-built KB instance.
        w: intermediate-step unknown cap.
        d: proof depth.
        u: last-step unknown cap (default 0 = paper convention).
        flat_intermediate: use the flat intermediate path for V≥2 rules.
        filter: explicit filter override; if None, derived from ``u``.
        **kwargs: forwarded to ``BCGrounder.__init__``.

    Returns:
        Fully-configured ``BCGrounder`` instance.
    """
    if filter is None:
        filter = _default_enum_filter(u)
    return BCGrounder(
        kb,
        resolution="enum",
        depth=d,
        width=w,
        w_last_depth=u,
        filter=filter,
        flat_intermediate=flat_intermediate,
        **kwargs,
    )
