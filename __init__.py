"""grounder: grounding techniques for neural-symbolic reasoning.

    from grounder import KB, SLD, Forward
    kb = KB.from_strings(facts, rules, entity2id, relation2id)
    proofs = SLD(kb, depth=3).prove(queries)
    closure = Forward(kb, depth=10).closure()

See docs/design.md.
"""
from grounder.forward import Forward
from grounder.kb import KB
from grounder.sld import SLD
from grounder.types import Closure, Proofs

__all__ = ["KB", "SLD", "Forward", "Proofs", "Closure"]
