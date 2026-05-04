"""Shared grounder construction helpers for the test sweeps.

Centralises the BCGrounder / make_bcwd dispatch and the keras-ns
ApproximateBackwardChainingGrounder setup so the count sweep
(``test_groundings.py``) and the speed sweep (``test_speed.py``)
don't duplicate the same per-kind boilerplate.

Public API:

  :func:`build_torch_grounder` — for SLD / enum-flat / enum-dense.
  :func:`build_keras_grounder` — keras-ns wrapper.

  The default per-grounder ``compile_mode`` for speed-comparable runs
  is exposed via :data:`DEFAULT_COMPILE_MODES`:

      keras-BC      None  (n/a; keras-ns doesn't use torch.compile)
      SLD           "reduce-overhead"
      enum-flat     None  (flat path uses dynamic shapes; not
                           reduce-overhead-compatible)
      enum-dense    "reduce-overhead"
      FC            None  (closure compute + fp_global filter)

  Test runners that care about wall-clock parity should pass these
  through (see ``test_speed.py``); ``test_groundings.py`` keeps eager
  everywhere because correctness sweeps run many grounders in one
  process and torch's CUDA-graph weakref bookkeeping accumulates.
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from grounder.bc.bc import BCGrounder
from grounder.factory import make_bcwd
from grounder.data.loader import KGDataset


__all__ = [
    "DEFAULT_COMPILE_MODES",
    "build_torch_grounder",
    "build_keras_grounder",
    "HAS_KERAS",
]


# Per-grounder compile mode (the BCGrounder ``compile_mode`` kwarg).
# BCGrounder picks the right compile path internally based on
# ``resolution`` (see bc.py around the ``_uses_outer_compile`` flag):
#
# * ``enum`` (``enum-flat`` / ``enum-dense``) — per-step inner
#   ``torch.compile(self._step_impl, fullgraph=True, mode=...)``.
#   Right pairing for the static-shape dense path; flat path is
#   inherently dynamic and stays eager.
#
# * ``sld`` / ``rtf`` — outer per-batch
#   ``torch.compile(self._forward_one_batch_inner, fullgraph=True,
#   mode=...)``, replayed against padded chunks. Dynamo traces the
#   entire single-batch forward as one graph, which sidesteps the
#   per-step trace failure SLD hits in the inner path
#   (``Unsupported: Observed exception``). Caller MUST pass an
#   explicit ``batch_size`` so the captured CUDA graph sees a
#   stable per-chunk shape — BCGrounder.forward raises ValueError
#   otherwise.
DEFAULT_COMPILE_MODES: Dict[str, Optional[str]] = {
    "keras-BC":    None,
    "SLD":         "reduce-overhead",   # outer compile; needs batch_size
    "enum-flat":   None,                # flat path is dynamic-shape
    "enum-dense":  "reduce-overhead",   # per-step inner compile
    "FC":          None,
}


# ══════════════════════════════════════════════════════════════════════
# Torch grounders (SLD / enum-flat / enum-dense)
# ══════════════════════════════════════════════════════════════════════


def build_torch_grounder(
    kind: str,
    kb,
    cfg: Dict[str, Any],
    *,
    compile_mode: Optional[str] = None,
    max_total_groundings: int = 4096,
    max_states: int = 256,
    fc_method: str = "spmm",
    collect_evidence: bool = True,
    collect_rule_groundings: bool = True,
) -> BCGrounder:
    """Build a torch grounder.

    Args:
        kind: one of ``"SLD"``, ``"enum-flat"``, ``"enum-dense"``.
        kb: a :class:`grounder.data.kb.KB` (already built via
            ``ds.make_kb(...)``).
        cfg: per-kind config dict —
            ``SLD``        → ``{"depth": int}``
            ``enum-*``     → ``{"w": int, "d": int, "flat": bool}``
        compile_mode: forwarded to ``BCGrounder``. ``None`` = eager.
        max_total_groundings, max_states, fc_method,
        collect_evidence, collect_rule_groundings: forwarded
            uniformly. ``collect_*=True`` is the right default for
            both count metrics and FC-closure-driven filtering.
    """
    if kind == "SLD":
        return BCGrounder(
            kb, resolution="sld", filter="none",
            depth=cfg["depth"],
            max_total_groundings=max_total_groundings,
            max_states=max_states,
            prune_facts=True,
            collect_evidence=collect_evidence,
            collect_rule_groundings=collect_rule_groundings,
            compile_mode=compile_mode,
        )
    if kind in ("enum-flat", "enum-dense"):
        flat = cfg.get("flat", kind == "enum-flat")
        return make_bcwd(
            kb, w=cfg["w"], d=cfg["d"],
            flat_intermediate=flat,
            max_groundings_per_query=max_total_groundings,
            max_total_groundings=max_total_groundings,
            max_states=max_states,
            fc_method=fc_method,
            prune_facts=True,
            bump_s_to_k=False,
            init_state_shape="minimal",
            compile_mode=compile_mode,
        )
    raise ValueError(f"unsupported torch grounder kind: {kind!r}")


# ══════════════════════════════════════════════════════════════════════
# keras-ns wrapper
# ══════════════════════════════════════════════════════════════════════


# keras-ns is a sibling-style reference repo (not pip-installed because
# its top-level ``ns_lib`` collides with torch-ns). Resolve via the
# ``KERAS_NS_ROOT`` env var so callers can opt into a non-default
# location.
import sys as _sys
_KERAS_NS_ROOT = Path(os.environ.get(
    "KERAS_NS_ROOT",
    str(Path.home() / "repos" / "keras-ns-swarm" / "main"),
))
if str(_KERAS_NS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_KERAS_NS_ROOT))

# Force TensorFlow off the GPU before any keras-ns import — TF
# eagerly grabs all visible memory, OOM-ing the torch sweep that
# typically runs alongside.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf  # noqa: E402
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

try:
    from ns_lib.logic.commons import Domain, Rule
    from ns_lib.grounding.backward_chaining_grounder import (
        ApproximateBackwardChainingGrounder,
    )
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


_VAR_UPPER = re.compile(r"^[A-Z]")
_VAR_LOWER = re.compile(r"^[a-z]$")


def _is_variable(name: str) -> bool:
    return bool(_VAR_LOWER.match(name) or _VAR_UPPER.match(name))


def build_keras_grounder(
    ds: KGDataset,
    data_dir: Path,
    width: int,
    depth: int,
) -> Tuple["ApproximateBackwardChainingGrounder", list]:
    """Build a keras-ns grounder matching the paper's ``BC_{w, d, u=0}``.

    Returns the constructed grounder plus the keras ``Rule`` list (the
    list is exposed because some downstream tests want to inspect it).

    Paper convention (matches ``test_keras_grounding_comparison`` /
    keras-ns) — every leaf body atom must be a fact (``u=0`` →
    ``max_unknown_fact_count_last_step=0``) and incomplete proofs are
    pruned.
    """
    assert HAS_KERAS, "keras-ns not available (TensorFlow not installed)"

    fact_tuples = list(ds._facts_raw)
    rules_raw = sorted(list(ds._rules_raw))

    # Parse domains
    domain_path = data_dir / "domain2constants.txt"
    domains: Dict[str, Domain] = {}
    if domain_path.exists():
        with open(domain_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    domains[parts[0]] = Domain(parts[0], sorted(parts[1:]))
    if not domains:
        all_ents = sorted(ds.entity2idx.keys())
        domains["entity"] = Domain("entity", all_ents)

    domain_name = next(iter(domains))

    keras_rules = []
    for i, (head, body_atoms) in enumerate(rules_raw):
        all_vars: set = set()
        for atom in [head] + list(body_atoms):
            for arg in atom[1:]:
                if _is_variable(arg):
                    all_vars.add(arg)
        var2domain = {v: domain_name for v in all_vars}
        keras_rules.append(Rule(
            name=f"r{i}", head_atoms=[head], body_atoms=body_atoms,
            var2domain=var2domain,
        ))

    # Paper BC_{w, d, u=0}: u=0 always; every leaf body atom must be
    # a fact; incomplete proofs are pruned. ``BC_{w, 1, 0}`` collapses
    # to ``BC_{0, 1, 0}`` for every w because at d=1 the only step IS
    # the last step (capped at u=0). They differ only for d ≥ 2.
    return ApproximateBackwardChainingGrounder(
        rules=keras_rules, facts=fact_tuples, domains=domains,
        num_steps=depth, max_unknown_fact_count=width,
        max_unknown_fact_count_last_step=0,
        prune_incomplete_proofs=True,
    ), keras_rules
