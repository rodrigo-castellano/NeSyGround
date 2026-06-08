"""Staged ragged trie-join FC engine (the ``StagedMethod`` impl).

  leapfrog.py — EXPERIMENTAL worst-case-optimal join (Leapfrog Triejoin)

The live ``FCDynamic`` engine currently lives in ``forward/fc.py``; it is split
into this subpackage in a follow-up step. ``leapfrog.py`` is parked here first:
its step-0 join is implemented and correct, but it is NOT wired into the engine
(the supported ``join_algo`` values are ``staged`` / ``chunked``).
"""
