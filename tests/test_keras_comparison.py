"""Cross-system comparison: torch-ns enum (cartesian+all_anchors) vs keras-ns.

Verifies that torch-ns with ``cartesian_product=True`` and ``all_anchors=True``
produces the same step-0 body groundings as the keras-ns
``ApproximateBackwardChainingGrounder`` on the family dataset.

Requires TensorFlow (keras-ns dependency). Skipped gracefully if unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest
import torch

# Guard keras-ns TensorFlow dependency
tf = pytest.importorskip("tensorflow")

TESTS_DIR = Path(__file__).resolve().parent
GROUNDER_ROOT = TESTS_DIR.parent
# keras-ns is a sibling-style reference repo (not pip-installed because its
# top-level `ns_lib/` collides with torch-ns).  Default to ~/repos/keras-ns-swarm/main/
# but allow override via KERAS_NS_ROOT env var.
KERAS_NS_ROOT = Path(os.environ.get(
    "KERAS_NS_ROOT",
    str(Path.home() / "repos" / "keras-ns-swarm" / "main"),
))
DATA_DIR = GROUNDER_ROOT / "data" / "family"

# Add keras-ns to sys.path
if str(KERAS_NS_ROOT) not in sys.path:
    sys.path.insert(0, str(KERAS_NS_ROOT))

from ns_lib.logic.commons import Domain, Rule  # noqa: E402
from ns_lib.grounding.backward_chaining_grounder import (  # noqa: E402
    ApproximateBackwardChainingGrounder,
)

from grounder.data.loader import KGDataset, _parse_rules_arrow, _parse_triples  # noqa: E402
from grounder.bc.bc import BCGrounder  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_domain_file(path: Path) -> Dict[str, Domain]:
    """Parse ``domain2constants.txt`` → ``{name: Domain}``."""
    domains: Dict[str, Domain] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0]
            constants = sorted(set(parts[1:]))
            domains[name] = Domain(name, constants)
    return domains


def _build_keras_rules(
    rules_raw: List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]],
    domain_name: str = "people",
) -> List[Rule]:
    """Convert parsed arrow-format rules to keras-ns Rule objects."""
    rules: List[Rule] = []
    for i, (head, body_atoms) in enumerate(rules_raw):
        all_vars: set[str] = set()
        for atom in [head] + body_atoms:
            for arg in atom[1:]:
                if len(arg) == 1 and arg.islower():
                    all_vars.add(arg)
        var2domain = {v: domain_name for v in all_vars}
        rules.append(
            Rule(
                name=f"r{i}",
                head_atoms=[head],
                body_atoms=body_atoms,
                var2domain=var2domain,
            )
        )
    return rules


def _torch_step0_bodies(
    evidence, ds: KGDataset, M: int, padding_idx: int,
) -> Set[Tuple[Tuple[str, str, str], ...]]:
    """Extract step-0 body atom sets from torch-ns ProofEvidence (B=1)."""
    bodies: set[tuple[tuple[str, str, str], ...]] = set()
    if evidence is None:
        return bodies
    mask = evidence.mask[0]        # [tG]
    body = evidence.body_flat[0]   # [tG, G_body, 3]
    for g in range(mask.shape[0]):
        if not mask[g].item():
            continue
        atoms: list[tuple[str, str, str]] = []
        for j in range(M):
            p, s, o = body[g, j].tolist()
            if p == padding_idx or p == 0:
                continue
            p_str = ds.idx2pred.get(p)
            s_str = ds.idx2entity.get(s)
            o_str = ds.idx2entity.get(o)
            if p_str is None or s_str is None or o_str is None:
                continue
            atoms.append((p_str, s_str, o_str))
        if atoms:
            bodies.add(tuple(sorted(atoms)))
    return bodies


def _keras_step0_bodies(
    rule2groundings: dict, query_str: Tuple[str, str, str],
) -> Set[Tuple[Tuple[str, str, str], ...]]:
    """Extract step-0 body atom sets from keras-ns output."""
    bodies: set[tuple[tuple[str, str, str], ...]] = set()
    for rg in rule2groundings.values():
        for head_tuple, body_tuple in rg.groundings:
            if len(head_tuple) == 1 and head_tuple[0] == query_str:
                if body_tuple:
                    bodies.add(tuple(sorted(body_tuple)))
    return bodies


# ── Test ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestKerasComparison:
    """Compare torch-ns enum (all_anchors) vs keras-ns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Family dataset not found: {DATA_DIR}")
        if not KERAS_NS_ROOT.exists():
            pytest.skip(f"keras-ns not found: {KERAS_NS_ROOT}")

    def _setup_shared(self):
        """Parse the family dataset for both systems."""
        facts_raw = sorted(_parse_triples(DATA_DIR / "train.txt"))
        rules_raw = sorted(_parse_rules_arrow(DATA_DIR / "rules.txt"))
        domains = _parse_domain_file(DATA_DIR / "domain2constants.txt")
        fact_tuples = [(p, a0, a1) for p, a0, a1 in facts_raw]

        ds = KGDataset(str(DATA_DIR), device="cpu")
        kb = ds.make_kb(fact_index_type="block_sparse")

        keras_rules = _build_keras_rules(rules_raw)

        return ds, kb, fact_tuples, domains, keras_rules

    def test_depth1_equivalence(self):
        """Depth-1: all body atoms must be known facts (width_d=0)."""
        ds, kb, fact_tuples, domains, keras_rules = self._setup_shared()
        M = kb.M
        pad = ds.padding_idx

        # all_anchors=True: every body atom tried as anchor.
        # With width≤1, fact-anchored enum covers all valid candidates
        # (at least 1 body atom must be a fact → anchor covers it).
        # No cartesian_product needed.
        grounder_torch = BCGrounder(
            kb,
            resolution="enum",
            filter="fp_batch",
            depth=1,
            width=1,
            max_total_groundings=4096,
            max_groundings_per_query=4096,
            all_anchors=True,
            prune_facts=True,
        )

        # num_steps=1, max_unknown_fact_count=0: all body atoms must be facts.
        keras_grounder = ApproximateBackwardChainingGrounder(
            rules=keras_rules,
            facts=fact_tuples,
            domains=domains,
            num_steps=1,
            max_unknown_fact_count=0,
            max_unknown_fact_count_last_step=0,
            prune_incomplete_proofs=False,
        )

        self._compare(grounder_torch, keras_grounder, ds, fact_tuples, M, pad)

    def test_depth2_equivalence(self):
        """Depth-2: step-0 allows 1 unknown, step-1 requires all facts.

        Uses filter='none': fp_batch's cross-query pool at B=1 can't
        verify intermediate atoms proved at step-1.  filter='none' is
        still sound (last step forces width_d=0 → no circularity).
        """
        ds, kb, fact_tuples, domains, keras_rules = self._setup_shared()
        M = kb.M
        pad = ds.padding_idx

        grounder_torch = BCGrounder(
            kb,
            resolution="enum",
            filter="none",
            depth=2,
            width=1,
            max_total_groundings=4096,
            max_groundings_per_query=4096,
            all_anchors=True,
            prune_facts=True,
        )

        keras_grounder = ApproximateBackwardChainingGrounder(
            rules=keras_rules,
            facts=fact_tuples,
            domains=domains,
            num_steps=2,
            max_unknown_fact_count=1,
            max_unknown_fact_count_last_step=0,
            prune_incomplete_proofs=True,
        )

        self._compare(grounder_torch, keras_grounder, ds, fact_tuples, M, pad)

    def test_w0_d1(self):
        """width=0, depth=1: only groundings where all body atoms are facts."""
        ds, kb, fact_tuples, domains, keras_rules = self._setup_shared()
        M = kb.M
        pad = ds.padding_idx

        grounder_torch = BCGrounder(
            kb, resolution="enum", filter="fp_batch",
            depth=1, width=0,
            max_total_groundings=4096, max_groundings_per_query=4096,
            all_anchors=True, prune_facts=True,
        )
        keras_grounder = ApproximateBackwardChainingGrounder(
            rules=keras_rules, facts=fact_tuples, domains=domains,
            num_steps=1, max_unknown_fact_count=0,
            max_unknown_fact_count_last_step=0,
            prune_incomplete_proofs=False,
        )
        self._compare(grounder_torch, keras_grounder, ds, fact_tuples, M, pad)

    @pytest.mark.skip(reason="cartesian_product + depth>1 requires too much "
                      "memory at step-1 (S * R_eff * E tensors)")
    def test_w2_d2(self):
        """width=2, depth=2: requires cartesian_product (width >= M) which
        is too memory-intensive for multi-depth on CPU."""
        pass

    def test_w1_d3(self):
        """width=1, depth=3: deeper multi-step proofs."""
        ds, kb, fact_tuples, domains, keras_rules = self._setup_shared()
        M = kb.M
        pad = ds.padding_idx

        grounder_torch = BCGrounder(
            kb, resolution="enum", filter="none",
            depth=3, width=1,
            max_total_groundings=16384, max_groundings_per_query=16384,
            all_anchors=True, prune_facts=True,
        )
        keras_grounder = ApproximateBackwardChainingGrounder(
            rules=keras_rules, facts=fact_tuples, domains=domains,
            num_steps=3, max_unknown_fact_count=1,
            max_unknown_fact_count_last_step=0,
            prune_incomplete_proofs=True,
        )
        self._compare(grounder_torch, keras_grounder, ds, fact_tuples, M, pad,
                       n_queries=20)

    def _compare(
        self, grounder_torch, keras_grounder, ds, fact_tuples, M, pad,
        n_queries: int = 100,
    ):
        """Run queries through both grounders and assert body sets match."""

        # ── Get test queries ──
        test_triples = sorted(_parse_triples(DATA_DIR / "test.txt"))
        n_queries = min(n_queries, len(test_triples))

        mismatches: list[dict] = []
        for qi in range(n_queries):
            pred_s, arg0_s, arg1_s = test_triples[qi]
            query_str = (pred_s, arg0_s, arg1_s)

            # ── torch-ns: single query (B=1) ──
            q_tensor = torch.tensor(
                [[ds.pred2idx[pred_s], ds.entity2idx[arg0_s],
                  ds.entity2idx[arg1_s]]],
                dtype=torch.long,
            )
            q_mask = torch.ones(1, dtype=torch.bool)
            result = grounder_torch(q_tensor, q_mask)
            torch_bodies = _torch_step0_bodies(result.evidence, ds, M, pad)

            # ── keras-ns: single query ──
            keras_result = keras_grounder.ground(
                facts=fact_tuples,
                queries=[query_str],
                deterministic=True,
            )
            keras_bodies = _keras_step0_bodies(keras_result, query_str)

            if torch_bodies != keras_bodies:
                mismatches.append(
                    {
                        "query": query_str,
                        "torch_only": torch_bodies - keras_bodies,
                        "keras_only": keras_bodies - torch_bodies,
                    }
                )

        if mismatches:
            msg = f"{len(mismatches)}/{n_queries} queries differ:\n"
            for m in mismatches[:5]:
                msg += (
                    f"  {m['query']}: "
                    f"torch_only={len(m['torch_only'])}, "
                    f"keras_only={len(m['keras_only'])}\n"
                )
                if m["torch_only"]:
                    msg += f"    sample torch_only: {list(m['torch_only'])[:2]}\n"
                if m["keras_only"]:
                    msg += f"    sample keras_only: {list(m['keras_only'])[:2]}\n"
            pytest.fail(msg)
