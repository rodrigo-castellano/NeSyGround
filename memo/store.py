"""Memoization store — descoped to a no-op (NullMemo).

Per-query tabling and per-subgoal memo were default-OFF in every consumer
(DpRL uses depth-1 RLGrounder, torch-ns/probfol never enable them), so the
redesign ships only the null implementation. The seam exists so the engine can
call ``memo.active()`` / ``memo.route(...)`` uniformly; a real cache would
subclass this and the engine wiring stays unchanged.
"""
from __future__ import annotations


class NullMemo:
    """No-op memo: never active, routes nothing. The only shipped store."""

    def active(self) -> bool:
        return False

    def reset(self) -> None:
        pass


__all__ = ["NullMemo"]
