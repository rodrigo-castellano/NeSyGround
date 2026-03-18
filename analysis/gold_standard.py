"""Gold standard SWI-Prolog prover using KGDataset.

Uses pyswip to delegate resolution to SWI-Prolog.
Encodes entities as integer IDs for efficient Prolog indexing.

Two depth semantics:
  - "sld" (default): every resolution step (fact or rule) costs 1.
    Plain 2-arity predicates p_N/2 with call_with_depth_limit/3 at query time.
    Full first-argument indexing, zero Prolog-level depth-tracking overhead.

  - "rule_only": only rule applications cost depth, facts are free.
    3-arity predicates with depth counter only in rules.

Query exclusion: when a query IS a base fact, it is retracted before proving
and re-asserted after.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Set, Tuple

from grounder.data.loader import KGDataset


class PrologProver:
    """Prover using SWI-Prolog with configurable depth semantics."""

    def __init__(
        self,
        dataset: KGDataset,
        max_depth: int = 2,
        inference_limit: int = 100_000_000,
        depth_semantics: str = "sld",
    ):
        if depth_semantics not in ("sld", "rule_only"):
            raise ValueError(f"depth_semantics must be 'sld' or 'rule_only', got '{depth_semantics}'")

        try:
            from pyswip import Prolog  # noqa: F401
        except ImportError:
            raise ImportError(
                "PrologProver requires pyswip + SWI-Prolog. Install with:\n"
                "  conda install -c conda-forge swi-prolog && pip install pyswip"
            )

        self.dataset = dataset
        self.max_depth = max_depth
        self.inference_limit = inference_limit
        self.depth_semantics = depth_semantics
        self._prolog = Prolog()

        self._pred2idx: Dict[str, int] = dict(dataset.pred2idx)
        self._entity2idx: Dict[str, int] = dict(dataset.entity2idx)

        # Collect all predicate indices used in rules and facts
        all_pred_idx: Set[int] = set()
        for head, body_atoms in dataset._rules_raw:
            all_pred_idx.add(self._pred2idx[head[0]])
            for atom in body_atoms:
                all_pred_idx.add(self._pred2idx[atom[0]])
        for pred, _, _ in dataset._facts_raw:
            all_pred_idx.add(self._pred2idx[pred])
        self._all_pred_idx = sorted(all_pred_idx)

        # Build fact set for exclude-self logic
        self._fact_set_idx: Set[Tuple[int, int, int]] = set()
        for pred, a0, a1 in dataset._facts_raw:
            pi = self._pred2idx[pred]
            si = self._entity2idx[a0]
            oi = self._entity2idx[a1]
            self._fact_set_idx.add((pi, si, oi))

        self._sld = depth_semantics == "sld"
        self._setup_prolog()

    def _setup_prolog(self) -> None:
        """Write a depth-limited Prolog program and consult it."""
        lines: List[str] = []

        if self._sld:
            self._setup_sld(lines)
        else:
            self._setup_rule_only(lines)

        fd, tmp_path = tempfile.mkstemp(suffix='.pl', prefix='prolog_prover_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            self._prolog.consult(tmp_path)
        finally:
            os.unlink(tmp_path)

        n_facts = len(self.dataset._facts_raw)
        n_rules = len(self.dataset._rules_raw)
        print(f"  PrologProver: {n_facts} facts, {n_rules} rules, "
              f"depth={self.max_depth}, {len(self._all_pred_idx)} predicates, "
              f"semantics={self.depth_semantics}")

    def _setup_sld(self, lines: List[str]) -> None:
        """SLD semantics using native call_with_depth_limit/3.

        Plain 2-arity predicates p_N(S, O) with facts as unit clauses
        and rules as standard Prolog clauses. Depth limiting is handled
        by SWI-Prolog's built-in call_with_depth_limit/3 at query time,
        which counts every resolution step (fact or rule) natively in C.

        This gives the best performance: full first-argument indexing on
        p_N/2 and zero Prolog-level depth-tracking overhead.
        """
        for pi in self._all_pred_idx:
            lines.append(f":- discontiguous p_{pi}/2.")
            lines.append(f":- dynamic p_{pi}/2.")

        # Assert facts
        for pred, a0, a1 in self.dataset._facts_raw:
            pi = self._pred2idx[pred]
            si = self._entity2idx[a0]
            oi = self._entity2idx[a1]
            lines.append(f"p_{pi}({si}, {oi}).")

        # Assert rules (non-recursive first for better indexing)
        non_recursive = []
        recursive = []
        for head, body_atoms in self.dataset._rules_raw:
            head_pi = self._pred2idx[head[0]]
            body_pis = {self._pred2idx[a[0]] for a in body_atoms}
            if head_pi in body_pis:
                recursive.append((head, body_atoms))
            else:
                non_recursive.append((head, body_atoms))
        for head, body_atoms in non_recursive:
            lines.append(self._rule_to_prolog_2arity(head, body_atoms))
        for head, body_atoms in recursive:
            lines.append(self._rule_to_prolog_2arity(head, body_atoms))

    def _setup_rule_only(self, lines: List[str]) -> None:
        """Rule-only semantics: 3-arity predicates, facts free."""
        for pi in self._all_pred_idx:
            lines.append(f":- discontiguous p_{pi}/3.")
            lines.append(f":- dynamic p_{pi}/3.")

        # Assert facts (depth argument is anonymous — free)
        for pred, a0, a1 in self.dataset._facts_raw:
            pi = self._pred2idx[pred]
            si = self._entity2idx[a0]
            oi = self._entity2idx[a1]
            lines.append(f"p_{pi}({si}, {oi}, _).")

        # Assert rules with depth counter
        for head, body_atoms in self.dataset._rules_raw:
            lines.append(self._rule_to_prolog_3arity(head, body_atoms))

    def _rule_to_prolog_2arity(
        self,
        head: Tuple[str, str, str],
        body_atoms: List[Tuple[str, str, str]],
    ) -> str:
        """Convert a rule to a plain 2-arity Prolog clause."""
        var_map, _var = self._build_var_map(head, body_atoms)

        hp = self._pred2idx[head[0]]
        h0 = _var(head[1])
        h1 = _var(head[2])
        head_str = f"p_{hp}({h0}, {h1})"

        body_parts: List[str] = []
        for atom in body_atoms:
            bp = self._pred2idx[atom[0]]
            a0 = _var(atom[1])
            a1 = _var(atom[2])
            body_parts.append(f"p_{bp}({a0}, {a1})")

        return f"{head_str} :- {', '.join(body_parts)}."

    def _rule_to_prolog_3arity(
        self,
        head: Tuple[str, str, str],
        body_atoms: List[Tuple[str, str, str]],
    ) -> str:
        """Convert a rule to a 3-arity depth-limited Prolog clause."""
        var_map, _var = self._build_var_map(head, body_atoms)

        hp = self._pred2idx[head[0]]
        h0 = _var(head[1])
        h1 = _var(head[2])
        head_str = f"p_{hp}({h0}, {h1}, D)"

        body_parts: List[str] = ["D > 0", "D1 is D-1"]
        for atom in body_atoms:
            bp = self._pred2idx[atom[0]]
            a0 = _var(atom[1])
            a1 = _var(atom[2])
            body_parts.append(f"p_{bp}({a0}, {a1}, D1)")

        return f"{head_str} :- {', '.join(body_parts)}."

    def _build_var_map(
        self,
        head: Tuple[str, str, str],
        body_atoms: List[Tuple[str, str, str]],
    ):
        """Build variable mapping and singleton-aware _var function for a rule.

        Variables are detected by: single lowercase letter or uppercase-starting name.
        Constants are mapped to their integer index.
        """
        var_map: Dict[str, str] = {}
        head_args = (head[1], head[2])
        same_head = (head_args[0] == head_args[1])
        var_map[head_args[0]] = "H0"
        var_map[head_args[1]] = "H0" if same_head else "H1"

        free_idx = 0
        for atom in body_atoms:
            for arg in (atom[1], atom[2]):
                if self._is_variable(arg) and arg not in var_map:
                    var_map[arg] = f"F{free_idx}"
                    free_idx += 1

        # Count occurrences to detect singletons
        var_counts: Dict[str, int] = {}
        for a in head_args:
            v = var_map.get(a, a)
            var_counts[v] = var_counts.get(v, 0) + 1
        for atom in body_atoms:
            for arg in (atom[1], atom[2]):
                v = var_map.get(arg, arg)
                var_counts[v] = var_counts.get(v, 0) + 1

        entity2idx = self._entity2idx

        def _var(arg: str) -> str:
            if self._is_variable(arg):
                v = var_map.get(arg, arg)
                if v[0].isupper() and var_counts.get(v, 0) <= 1:
                    return f"_{v}"
                return v
            # Constant: map to integer ID
            return str(entity2idx[arg])

        return var_map, _var

    @staticmethod
    def _is_variable(name: str) -> bool:
        """Check if a name is a logical variable."""
        if len(name) == 1 and name.islower():
            return True
        if name[0].isupper():
            return True
        return False

    def _query_to_idx(self, query: Tuple[str, str, str]) -> Tuple[int, int, int]:
        pi = self._pred2idx[query[0]]
        si = self._entity2idx[query[1]]
        oi = self._entity2idx[query[2]]
        return pi, si, oi

    def _limited_goal_str(self, pi: int, si: int, oi: int) -> str:
        """Return goal wrapped with depth + inference limits."""
        if self._sld:
            depth_goal = (
                f"call_with_depth_limit(p_{pi}({si}, {oi}), "
                f"{self.max_depth}, Rd), Rd \\== depth_limit_exceeded"
            )
            return (
                f"call_with_inference_limit(({depth_goal}), "
                f"{self.inference_limit}, Ri), "
                f"Ri \\== inference_limit_exceeded"
            )
        else:
            goal = f"p_{pi}({si}, {oi}, {self.max_depth})"
            return (
                f"call_with_inference_limit({goal}, "
                f"{self.inference_limit}, Ri), "
                f"Ri \\== inference_limit_exceeded"
            )

    def _raw_query(self, pi: int, si: int, oi: int) -> bool:
        """Execute a Prolog query with depth + inference limits."""
        q = self._limited_goal_str(pi, si, oi)
        try:
            return bool(list(self._prolog.query(q, maxresult=1)))
        except Exception:
            try:
                list(self._prolog.query("abolish_all_tables"))
                return bool(list(self._prolog.query(q, maxresult=1)))
            except Exception:
                return False

    def prove(
        self,
        queries: List[Tuple[str, str, str]],
        exclude_self: bool = True,
    ) -> Tuple[Set[int], float]:
        """Prove queries, return (provable_indices, elapsed_seconds).

        Args:
            queries: List of (pred, arg0, arg1) string tuples.
            exclude_self: If True, retract a query from the fact base before
                proving (to test derivability without the trivial self-proof).

        Returns:
            Tuple of (set of provable query indices, elapsed time in seconds).
        """
        N = len(queries)
        provable: Set[int] = set()
        t0 = time.perf_counter()

        non_fact_idx: List[int] = []
        fact_idx: List[int] = []
        query_idx_cache: List[Tuple[int, int, int]] = []
        for i, q in enumerate(queries):
            pso = self._query_to_idx(q)
            query_idx_cache.append(pso)
            if exclude_self and pso in self._fact_set_idx:
                fact_idx.append(i)
            else:
                non_fact_idx.append(i)

        # Non-fact queries can be proven directly
        if non_fact_idx:
            provable.update(self._individual_prove(non_fact_idx, query_idx_cache))

        # Fact queries need retract-prove-reassert
        for i in fact_idx:
            pi, si, oi = query_idx_cache[i]

            if self._sld:
                retract_goal = f"retract(p_{pi}({si}, {oi}))"
            else:
                retract_goal = f"retract(p_{pi}({si}, {oi}, _))"
            try:
                list(self._prolog.query(retract_goal))
                list(self._prolog.query("abolish_all_tables"))
            except Exception:
                pass

            if self._raw_query(pi, si, oi):
                provable.add(i)

            if self._sld:
                assert_goal = f"assertz(p_{pi}({si}, {oi}))"
            else:
                assert_goal = f"assertz(p_{pi}({si}, {oi}, _))"
            try:
                list(self._prolog.query(assert_goal))
                list(self._prolog.query("abolish_all_tables"))
            except Exception:
                pass

        elapsed = time.perf_counter() - t0
        return provable, elapsed

    def _individual_prove(
        self,
        indices: List[int],
        query_idx_cache: List[Tuple[int, int, int]],
    ) -> Set[int]:
        """Prove queries one at a time."""
        provable: Set[int] = set()
        for i in indices:
            pi, si, oi = query_idx_cache[i]
            if self._raw_query(pi, si, oi):
                provable.add(i)
        return provable

    def verify(self, n_samples: int = 5) -> None:
        """Sanity check on known facts (without exclusion)."""
        print("  Verifying on known facts...")
        for pred, a0, a1 in self.dataset._facts_raw[:n_samples]:
            pi, si, oi = self._query_to_idx((pred, a0, a1))
            status = "OK" if self._raw_query(pi, si, oi) else "FAIL"
            print(f"    {pred}({a0},{a1}) -> {status}")
