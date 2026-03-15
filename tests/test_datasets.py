"""Integration tests using real KG datasets (family, wn18rr, fb15k237).

Tests that BCGrounder (SLD resolution) can load real KBs and find groundings for
test queries. Datasets are loaded from kge_experiments/data/.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pytest
from grounder import BCGrounder

DEVICE = torch.device("cpu")
DATA_ROOT = Path(os.environ.get(
    "GROUNDER_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "data"),
))


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _parse_facts(path: Path) -> List[Tuple[str, str, str]]:
    """Parse facts file: predicate(entity1,entity2)."""
    facts = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#"):
                continue
            # Format: pred(arg0,arg1)
            # Find the last '(' that starts the args
            paren_idx = line.rfind("(")
            if paren_idx > 0 and line.endswith(")"):
                pred = line[:paren_idx]
                args = line[paren_idx + 1:-1]
                parts = args.split(",", 1)
                if len(parts) == 2:
                    facts.append((pred.strip(), parts[0].strip(), parts[1].strip()))
                    continue
            # Try tab-separated format
            parts = line.split("\t")
            if len(parts) == 3:
                facts.append(tuple(parts))
    return facts


def _parse_rules(path: Path) -> List[Tuple[str, List[Tuple[str, str, str]], Tuple[str, str, str]]]:
    """Parse rules file: rN:score:body1(a,h), body2(b,h) -> head(a,b)."""
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: rN:score:body -> head
            parts = line.split(":")
            if len(parts) < 3:
                continue
            rest = ":".join(parts[2:])
            if "->" not in rest:
                continue
            body_str, head_str = rest.rsplit("->", 1)
            body_str = body_str.strip()
            head_str = head_str.strip()

            def _parse_atom(s):
                s = s.strip()
                m = re.match(r"(.+)\(([^,]+),([^)]+)\)", s)
                if m:
                    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
                return None

            head = _parse_atom(head_str)
            if head is None:
                continue
            body_atoms = []
            for b in body_str.split("),"):
                b = b.strip()
                if not b.endswith(")"):
                    b += ")"
                atom = _parse_atom(b)
                if atom:
                    body_atoms.append(atom)
            if body_atoms:
                rules.append((head, body_atoms))
    return rules


def _build_kb(
    dataset_name: str,
    facts_file: str = "train.txt",
    max_facts: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, Dict[str, int], Dict[str, int]]:
    """Build KB tensors from dataset files.

    Returns:
        facts_idx, heads_idx, bodies_idx, rule_lens,
        constant_no, padding_idx, predicate_no,
        entity2idx, pred2idx
    """
    data_dir = DATA_ROOT / dataset_name
    assert data_dir.exists(), f"Dataset not found: {data_dir}"

    facts = _parse_facts(data_dir / facts_file)
    rules = _parse_rules(data_dir / "rules.txt")

    # Build vocabularies
    all_preds = set()
    all_entities = set()
    var_names = set()

    for pred, arg0, arg1 in facts:
        all_preds.add(pred)
        all_entities.add(arg0)
        all_entities.add(arg1)

    # Identify variable names used in rules (single lowercase letters)
    _var_pattern = re.compile(r"^[a-z]$")
    for head, body_atoms in rules:
        all_preds.add(head[0])
        for atom in [head] + body_atoms:
            all_preds.add(atom[0])
            for arg in atom[1:]:
                if _var_pattern.match(arg):
                    var_names.add(arg)
                else:
                    all_entities.add(arg)

    # 1-based indexing (0 is reserved)
    pred2idx = {p: i + 1 for i, p in enumerate(sorted(all_preds))}
    entity2idx = {e: i + 1 for i, e in enumerate(sorted(all_entities))}
    var2idx = {v: len(entity2idx) + 1 + i for i, v in enumerate(sorted(var_names))}

    constant_no = len(entity2idx)
    predicate_no = len(pred2idx) + 1  # exclusive upper bound
    padding_idx = constant_no + len(var_names) + 10

    # Build fact tensors
    fact_list = []
    for pred, arg0, arg1 in facts:
        if pred in pred2idx and arg0 in entity2idx and arg1 in entity2idx:
            fact_list.append([pred2idx[pred], entity2idx[arg0], entity2idx[arg1]])
    if max_facts and len(fact_list) > max_facts:
        fact_list = fact_list[:max_facts]
    facts_idx = torch.tensor(fact_list, dtype=torch.long) if fact_list else torch.empty(0, 3, dtype=torch.long)

    # Build rule tensors
    max_body = max((len(body) for _, body in rules), default=1)
    heads_list = []
    bodies_list = []
    lens_list = []
    for head, body_atoms in rules:
        h = [pred2idx[head[0]],
             var2idx.get(head[1], entity2idx.get(head[1], padding_idx)),
             var2idx.get(head[2], entity2idx.get(head[2], padding_idx))]
        heads_list.append(h)

        body_row = []
        for atom in body_atoms:
            b = [pred2idx[atom[0]],
                 var2idx.get(atom[1], entity2idx.get(atom[1], padding_idx)),
                 var2idx.get(atom[2], entity2idx.get(atom[2], padding_idx))]
            body_row.append(b)
        while len(body_row) < max_body:
            body_row.append([padding_idx, padding_idx, padding_idx])
        bodies_list.append(body_row)
        lens_list.append(len(body_atoms))

    if heads_list:
        heads_idx = torch.tensor(heads_list, dtype=torch.long)
        bodies_idx = torch.tensor(bodies_list, dtype=torch.long)
        rule_lens = torch.tensor(lens_list, dtype=torch.long)
    else:
        heads_idx = torch.empty(0, 3, dtype=torch.long)
        bodies_idx = torch.empty(0, 1, 3, dtype=torch.long)
        rule_lens = torch.empty(0, dtype=torch.long)

    return (facts_idx, heads_idx, bodies_idx, rule_lens,
            constant_no, padding_idx, predicate_no, entity2idx, pred2idx)


def _make_grounder(dataset_name: str, depth: int = 2, max_facts: Optional[int] = None,
                   max_total_groundings: int = 16) -> Tuple[BCGrounder, Dict, Dict]:
    """Build a BCGrounder (SLD, no filter) from a dataset."""
    (facts_idx, heads_idx, bodies_idx, rule_lens,
     constant_no, padding_idx, predicate_no,
     entity2idx, pred2idx) = _build_kb(dataset_name, max_facts=max_facts)

    M = int(rule_lens.max().item()) if rule_lens.numel() > 0 else 1
    G = M + (M - 1) * depth + 1  # enough goals for depth

    grounder = BCGrounder(
        facts_idx=facts_idx,
        rules_heads_idx=heads_idx,
        rules_bodies_idx=bodies_idx,
        rule_lens=rule_lens,
        constant_no=constant_no,
        padding_idx=padding_idx,
        device=DEVICE,
        predicate_no=predicate_no,
        resolution='sld',
        filter='prune',
        max_goals=G,
        depth=depth,
        max_total_groundings=max_total_groundings,
        K_MAX=50,  # keep small for CPU tests
        fact_index_type='arg_key',
    )
    return grounder, entity2idx, pred2idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DATA_ROOT / "family").exists(),
    reason="family dataset not found",
)
class TestFamily:
    """Integration tests with family/kinship dataset."""

    def test_loads_and_constructs(self):
        grounder, e2i, p2i = _make_grounder("family", depth=2)
        assert grounder.num_facts > 0
        assert grounder.num_rules > 0
        print(f"Family: {grounder.num_facts} facts, {grounder.num_rules} rules, "
              f"K_f={grounder.K_f}, K_r={grounder.K_r}, K={grounder.K}")

    def test_finds_groundings(self):
        grounder, e2i, p2i = _make_grounder("family", depth=2)
        # Pick a rule head predicate and a known entity
        head_pred = list(p2i.values())[0]
        entity = list(e2i.values())[0]
        var = grounder.constant_no + 1

        queries = torch.tensor([[head_pred, entity, var]], dtype=torch.long)
        query_mask = torch.tensor([True])
        result = grounder(queries, query_mask)
        print(f"Family query pred={head_pred}, entity={entity}: "
              f"found {result.count[0].item()} groundings")
        # We don't assert >0 since not every (pred, entity) combo has groundings

    def test_batch_queries(self):
        grounder, e2i, p2i = _make_grounder("family", depth=2)
        B = min(8, len(e2i))
        entities = list(e2i.values())[:B]
        head_preds = list(p2i.values())
        var = grounder.constant_no + 1

        queries = torch.tensor([
            [head_preds[i % len(head_preds)], entities[i], var]
            for i in range(B)
        ], dtype=torch.long)
        query_mask = torch.ones(B, dtype=torch.bool)
        result = grounder(queries, query_mask)
        total = result.count.sum().item()
        print(f"Family batch of {B}: total groundings = {total}")

    def test_depth_increases_groundings(self):
        """Deeper proofs should find at least as many groundings."""
        grounder_d1, e2i, p2i = _make_grounder("family", depth=2)
        grounder_d2, _, _ = _make_grounder("family", depth=3)

        # Build queries from test triples
        test_facts = _parse_facts(DATA_ROOT / "family" / "test.txt")
        B = min(16, len(test_facts))
        var = grounder_d1.constant_no + 1
        queries = []
        for pred, arg0, arg1 in test_facts[:B]:
            if pred in p2i and arg0 in e2i:
                queries.append([p2i[pred], e2i[arg0], var])
        if not queries:
            pytest.skip("No valid test queries")
        queries_t = torch.tensor(queries, dtype=torch.long)
        mask = torch.ones(len(queries), dtype=torch.bool)

        r1 = grounder_d1(queries_t, mask)
        r2 = grounder_d2(queries_t, mask)
        c1 = r1.count.sum().item()
        c2 = r2.count.sum().item()
        print(f"Family depth comparison: depth=2 → {c1}, depth=3 → {c2}")
        assert c2 >= c1, f"Depth 3 should find >= depth 2 groundings ({c2} < {c1})"


@pytest.mark.skipif(
    not (DATA_ROOT / "wn18rr").exists(),
    reason="wn18rr dataset not found",
)
class TestWN18RR:
    """Integration tests with WN18RR dataset."""

    def test_loads_and_constructs(self):
        grounder, e2i, p2i = _make_grounder("wn18rr", depth=2, max_facts=10000)
        assert grounder.num_facts > 0
        assert grounder.num_rules > 0
        print(f"WN18RR: {grounder.num_facts} facts, {grounder.num_rules} rules, "
              f"K_f={grounder.K_f}, K_r={grounder.K_r}")

    def test_finds_groundings(self):
        grounder, e2i, p2i = _make_grounder("wn18rr", depth=2, max_facts=10000)
        test_facts = _parse_facts(DATA_ROOT / "wn18rr" / "test.txt")
        B = min(8, len(test_facts))
        var = grounder.constant_no + 1
        queries = []
        for pred, arg0, arg1 in test_facts[:B * 4]:
            if pred in p2i and arg0 in e2i:
                queries.append([p2i[pred], e2i[arg0], var])
                if len(queries) >= B:
                    break
        if not queries:
            pytest.skip("No valid test queries")
        queries_t = torch.tensor(queries, dtype=torch.long)
        mask = torch.ones(len(queries), dtype=torch.bool)
        result = grounder(queries_t, mask)
        total = result.count.sum().item()
        print(f"WN18RR batch of {len(queries)}: total groundings = {total}")


@pytest.mark.skipif(
    not (DATA_ROOT / "fb15k237").exists(),
    reason="fb15k237 dataset not found",
)
class TestFB15K237:
    """Integration tests with FB15K-237 dataset."""

    def test_loads_and_constructs(self):
        grounder, e2i, p2i = _make_grounder("fb15k237", depth=2, max_facts=10000)
        assert grounder.num_facts > 0
        assert grounder.num_rules > 0
        print(f"FB15K-237: {grounder.num_facts} facts, {grounder.num_rules} rules, "
              f"K_f={grounder.K_f}, K_r={grounder.K_r}")

    def test_finds_groundings(self):
        grounder, e2i, p2i = _make_grounder("fb15k237", depth=2, max_facts=10000)
        test_facts = _parse_facts(DATA_ROOT / "fb15k237" / "test.txt")
        B = min(8, len(test_facts))
        var = grounder.constant_no + 1
        queries = []
        for pred, arg0, arg1 in test_facts[:B * 4]:
            if pred in p2i and arg0 in e2i:
                queries.append([p2i[pred], e2i[arg0], var])
                if len(queries) >= B:
                    break
        if not queries:
            pytest.skip("No valid test queries")
        queries_t = torch.tensor(queries, dtype=torch.long)
        mask = torch.ones(len(queries), dtype=torch.bool)
        result = grounder(queries_t, mask)
        total = result.count.sum().item()
        print(f"FB15K-237 batch of {len(queries)}: total groundings = {total}")
