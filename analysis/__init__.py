"""Analysis tools for depth generation, comparison, and gold standard verification."""

from grounder.analysis.generate_depths_dynamic import generate_depths_dynamic
from grounder.analysis.generate_depths_static import generate_depths_static

__all__ = [
    "generate_depths_dynamic",
    "generate_depths_static",
]
