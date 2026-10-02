"""Parametrized backward chaining, BC_{w,d} (IJCAI-25): backward chaining over ground atoms.

    g = PBC(kb, depth=2, width=1)                          # fp_batch: only the groundings of provable atoms
    g = PBC.parse(kb, "enum.keras.w1.d2.r2.flat")          # the type strings of torch-ns's configs
    out = g.ground(queries)                                # [S, Q, 3] pools (or [Q, 3]) -> Groundings

Each step grounds the unique goal atoms of each pool — the queries, then the unknown body atoms of the last step's
groundings — with every rule variant of their predicate (see ``pbc.tables``): the head binds its variables, a body atom
with one bound argument enumerates its free variable through the facts, and a grounding is kept when at most ``width``
of its body atoms are not facts (``last_width`` at the last step), none is its own goal, and each such atom's predicate
is some rule's head (it can still be proved). Then ``prune``: ``"fp_batch"`` keeps the groundings whose body is proved
within ``depth`` rounds from the facts; ``"keras"`` prunes as keras-ns's BC_{w,d} does (the papers' protocol); None
keeps them all. See ``pbc.engine``. Runs on the GPU (Triton kernels).
"""
from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.pbc import engine
from grounder.pbc.guide import Guide, Scorer
from grounder.pbc.tables import Tables
from grounder.types import Groundings


class PBC(nn.Module):
    """BC_{w,d} over ``kb`` (see the module docstring): ``depth`` steps of ``width`` (``last_width`` at the last:
    default 0, keras ``width``), ``prune`` ``"fp_batch"`` | ``"keras"`` | None, ``rounds`` keras-ns's proof-walk rounds
    (default ``depth - 1``; the later keras-ns of XAI-25 / NeSy-25 walks ``depth``); ``bind`` ``"facts"`` (a free
    variable ranges over what a fact lookup returns) | ``"all"`` (over every entity: IJCAI-25's Full grounder);
    ``guide``: a ``Guide`` keeps, per goal and step, the all-fact groundings and the ``k`` best others by a model's
    scores."""

    def __init__(self, kb, *, depth: int, width: int = 1, last_width: Optional[int] = None,
                 prune: Optional[str] = "fp_batch", rounds: Optional[int] = None, bind: str = "facts",
                 guide: Optional[Guide] = None) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if prune not in ("fp_batch", "keras", None):
            raise ValueError(f"prune: 'fp_batch', 'keras' or None, got {prune!r}")
        if bind not in ("facts", "all"):
            raise ValueError(f"bind: 'facts' or 'all', got {bind!r}")
        last_width = (width if prune == "keras" else 0) if last_width is None else last_width
        top = 2 if prune == "keras" else 1
        if width > top or last_width > (top if prune == "keras" else 0):
            raise NotImplementedError(f"width {width}, last width {last_width}: PBC grounds width <= {top} and a last "
                                      f"width of {'at most ' + str(top) if prune == 'keras' else '0'}")
        if prune == "keras" and (rounds or depth - 1):
            h, b, n = kb.rules.heads, kb.rules.bodies, kb.rules.lens
            own = ((b[..., 1:, None] == h[:, None, None, 1:]).any(-1).all(-1)
                   & (torch.arange(b.shape[1], device=b.device) < n.unsqueeze(1)))
            if bool((own.any(1) & (n > 1)).any()):
                # keras-ns takes such an atom, bound by the head alone, as known without looking it up; with no proof
                # rounds (IJCAI-25's depth 1: FB15k-237's AMIE rules) its pruning keeps a grounding only when every
                # body atom is a fact, as here
                raise NotImplementedError("the keras prune: a rule with more than one body atom has one whose "
                                          "arguments are all head variables")
        self.kb, self.depth, self.width, self.last_width = kb, depth, width, last_width
        self.prune, self.rounds, self.bind, self.guide = prune, rounds, bind, guide
        self.tables = Tables(kb)
        # the body atoms a lookup draws from the facts are facts, never tested — unless every entity is drawn
        self.enumerated = self.tables.enumerated if bind == "facts" else torch.zeros_like(self.tables.enumerated)
        self._facts: Optional[engine.FactTable] = None
        self._live = None

    def facts(self) -> engine.FactTable:
        """The fact lookups and hash set, built on first use (on the GPU)."""
        if self._facts is None:
            if self.kb.device.type != "cuda":
                raise RuntimeError("PBC grounds on the GPU (its kernels are Triton's): build the KB on a CUDA device")
            self._facts = engine.FactTable(self.kb.facts, every=self.bind == "all", n_entities=self.kb.E)
        return self._facts

    def ground(self, queries: Tensor, mask: Optional[Tensor] = None, *, depth: Optional[int] = None,
               stats: Optional[list] = None) -> Groundings:
        """The groundings of the pools ``queries [S, Q, 3]`` (or one pool ``[Q, 3]``; ``mask``: the real queries);
        ``depth`` steps (default the grounder's); ``stats`` collects each step's ``(step, goals, kept groundings)``."""
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
            mask = None if mask is None else mask.unsqueeze(0)
        if mask is None:
            mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        return engine.ground(self, queries, mask, self.depth if depth is None else int(depth), stats)

    @classmethod
    def parse(cls, kb, type_: str) -> "PBC":
        """A grounder from a type string: ``enum.fp_batch.w1.d2.flat`` (fp_batch, width 1, depth 2),
        ``enum.keras.w1.d2.r2.flat`` (keras-ns's prune, 2 walk rounds), ``.u<N>`` a last width (fp_batch: a last width
        above 0 prunes nothing)."""
        if type_.startswith("sld."):
            raise ValueError(f"{type_!r} is an SLD grounder: grounder.sld.SLD")

        def num(c: str, default):
            m = re.search(rf"\.{c}(\d+)(?:\.|$)", type_)
            return int(m[1]) if m else default
        width, depth = num("w", 1), num("d", 1)
        if ".keras" in type_:
            return cls(kb, depth=depth, width=width, last_width=num("u", width), prune="keras",
                       rounds=num("r", None))
        u = num("u", 0)
        return cls(kb, depth=depth, width=width, last_width=u, prune="fp_batch" if u == 0 else None)

    def __repr__(self) -> str:
        return (f"PBC(depth={self.depth}, width={self.width}, last_width={self.last_width}, prune={self.prune!r}"
                + (f", rounds={self.rounds}" if self.rounds else "") + (", bind='all'" if self.bind == "all" else "")
                + ")")


__all__ = ["PBC", "Guide", "Scorer"]
