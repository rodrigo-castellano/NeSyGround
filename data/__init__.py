# data/ — knowledge base, fact/rule indexing, dataset loading
from grounder.data.kb import KB
from grounder.data.fact_index import (
    ArgKeyFactIndex,
    BlockSparseFactIndex,
    FactIndex,
    InvertedFactIndex,
    fact_contains,
    pack_triples_64,
)
from grounder.data.rule_index import RuleIndex, RuleIndexEnum, RulePattern, compile_rules
from grounder.data.loader import KGDataset

__all__ = [
    "KB",
    "ArgKeyFactIndex",
    "BlockSparseFactIndex",
    "FactIndex",
    "InvertedFactIndex",
    "fact_contains",
    "pack_triples_64",
    "RuleIndex",
    "RuleIndexEnum",
    "RulePattern",
    "compile_rules",
    "KGDataset",
]
