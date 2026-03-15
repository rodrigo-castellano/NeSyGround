# resolution/ — resolution strategies for backward chaining
#
# sld.py             — SLD: parallel fact + rule (K_f + K_r)
# rtf.py             — RTF: cascade rule-then-fact (K_f × K_r)
# enum.py            — Compiled enumeration (no MGU, pre-compiled bindings)
# mgu.py             — Shared MGU primitives (resolve_facts, resolve_rules)
# standardization.py — Variable standardization (MGU only)

from grounder.resolution.sld import resolve_sld
from grounder.resolution.rtf import resolve_rtf
from grounder.resolution.mgu import resolve_facts, resolve_rules, init_mgu
from grounder.resolution.enum import (
    resolve_enum,
    resolve_enum_full,
    resolve_enum_step,
    init_enum,
)
from grounder.resolution.standardization import (
    standardize_vars_canonical,
    standardize_vars_offset,
)
