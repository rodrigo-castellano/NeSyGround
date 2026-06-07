"""Data layer — id encoding, dataset loading, KB construction, fact/rule indexing.

    from grounder._build.data import KGDataset, KB, Encoding
    from grounder._build.data import fact_index, rule_index
"""
from __future__ import annotations

from grounder._build.data import fact_index, rule_index
from grounder._build.data.dataset import KGDataset
from grounder._build.data.encoding import Encoding
from grounder._build.data.kb import KB

__all__ = ["KGDataset", "KB", "Encoding", "fact_index", "rule_index"]
