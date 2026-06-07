"""Data layer — id encoding, dataset loading, KB construction, fact/rule indexing.

    from grounder.data import KGDataset, KB, Encoding
    from grounder.data import fact_index, rule_index
"""
from __future__ import annotations

from grounder.data import fact_index, rule_index
from grounder.data.dataset import KGDataset
from grounder.data.encoding import Encoding
from grounder.data.kb import KB

__all__ = ["KGDataset", "KB", "Encoding", "fact_index", "rule_index"]
