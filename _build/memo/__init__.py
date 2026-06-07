"""Memoization seam — only ``NullMemo`` ships (tabling/subgoal descoped)."""
from grounder._build.memo.store import NullMemo

__all__ = ["NullMemo"]
