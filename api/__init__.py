"""torch-ns's entry point: ``create_grounder`` and ``RuleGrounder`` (``rule_grounder``)."""
from grounder.api.rule_grounder import RuleGrounder, RuleGroundings, TensorFactIndex, create_grounder

__all__ = ["RuleGrounder", "RuleGroundings", "TensorFactIndex", "create_grounder"]
