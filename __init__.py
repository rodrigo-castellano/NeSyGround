"""grounder: grounding techniques for neural-symbolic reasoning.

    from grounder import KB, PBC, SLD, Forward
    kb = KB.from_strings(facts, rules, entity2id, relation2id)
    groundings = PBC(kb, depth=2, width=1, prune="keras").ground(queries)
    proofs = SLD(kb, depth=3).prove(queries)
    closure = Forward(kb, depth=10).closure()

See README.md and docs/design.md.
"""
from grounder.forward import Forward
from grounder.kb import KB
from grounder.pbc import PBC, Guide
from grounder.sld import SLD
from grounder.types import Closure, Groundings, Proofs

__all__ = ["KB", "PBC", "SLD", "Forward", "Guide", "Groundings", "Proofs", "Closure"]
