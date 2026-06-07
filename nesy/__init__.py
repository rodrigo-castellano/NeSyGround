# nesy/ — neural-symbolic hooks and scoring (new _build layer)
#
# hooks.py   — ResolutionFactHook, ResolutionRuleHook, GroundingHook, StepHook
# scoring.py — PartialScorer, LazyPartialScorer
# kge.py     — KGEScorer, KGEFactFilter, KGERuleFilter, KGEStepFilter
# neural.py  — NeuralScorer: learned attention + topk
# soft.py    — SoftScorer: soft provability + topk
# sampler.py — RandomSampler: random subsampling

import torch
from torch import Tensor

from grounder.nesy.hooks import (
    GroundingHook,
    ResolutionFactHook,
    ResolutionRuleHook,
    StepHook,
)
from grounder.nesy.scoring import (
    LazyPartialScorer,
    PartialScorer,
)
from grounder.nesy.kge import KGEScorer, KGEFactFilter, KGERuleFilter, KGEStepFilter
from grounder.nesy.neural import GroundingAttention, NeuralScorer
from grounder.nesy.soft import ProvabilityMLP, SoftScorer
from grounder.nesy.sampler import RandomSampler
from grounder.nesy._util import _topk_select


__all__ = [
    # Hook protocols
    "GroundingHook",
    "ResolutionFactHook",
    "ResolutionRuleHook",
    "StepHook",
    # Scoring primitives
    "LazyPartialScorer",
    "PartialScorer",
    # Hook implementations
    "KGEScorer",
    "KGEFactFilter",
    "KGERuleFilter",
    "KGEStepFilter",
    "NeuralScorer",
    "GroundingAttention",
    "SoftScorer",
    "ProvabilityMLP",
    "RandomSampler",
    "_topk_select",
]
