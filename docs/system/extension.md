# Extension Guide

Step-by-step guide to adding a new grounder to NeSyGround.

---

## 1. Choose a Base Class

| If you want to... | Extend |
|-------------------|--------|
| Modify BC goal selection, resolution, or packing | `BCGrounder` |
| Add pruning/filtering to BC groundings | `ParametrizedBCGrounder` |
| Add scoring/ranking to BC groundings | `ParametrizedBCGrounder` |
| Implement a new FC strategy | `FCSemiNaiveGrounder` |
| Wrap an existing grounder with pre/post-processing | `nn.Module` (composition) |

Most new grounders extend `ParametrizedBCGrounder` — it provides the full BC pipeline with width/depth control, fact-anchoring, dual anchoring, and rule clustering out of the box.

---

## 2. Override Table

Which methods to override for which behavior:

| Behavior | Method to override | Example |
|----------|-------------------|---------|
| Change goal selection strategy | `_select` | Priority-based goal selection |
| Change fact resolution | `_resolve_facts` | Custom fact matching |
| Change rule resolution | `_resolve_rules` | Custom unification |
| Add post-grounding filtering | `_postprocess` | BCPruneGrounder |
| Add pre-grounding computation | `pre_ground` | BCProvsetGrounder (runs FC) |
| Add scoring + selection | `forward` | KGEGrounder, SamplerGrounder |
| Change FC iteration | `compute_provable_set` | Custom T_P operator |

---

## 3. Skeleton: Scored Grounder

The most common extension pattern — oversample from parent, score, select top-k:

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class MyGrounder(ParametrizedBCGrounder):
    """Custom scored grounder."""

    def __init__(
        self,
        fact_index: FactIndex,
        rules: List[Rule],
        kb: KnowledgeBase,
        depth: int = 2,
        width: Optional[int] = 1,
        max_groundings_per_query: int = 32,
        max_total_groundings: int = 64,
        prune_incomplete_proofs: bool = True,
        provable_set_method: str = "join",
        device: str = "cuda",
        # --- custom args ---
        output_budget: Optional[int] = None,
        my_param: float = 1.0,
    ) -> None:
        # Oversample: give parent 2x the output budget
        actual_budget = output_budget or max_total_groundings
        super().__init__(
            fact_index=fact_index,
            rules=rules,
            kb=kb,
            depth=depth,
            width=width,
            max_groundings_per_query=max_groundings_per_query,
            max_total_groundings=actual_budget * 2,  # 2x oversample
            prune_incomplete_proofs=prune_incomplete_proofs,
            provable_set_method=provable_set_method,
            device=device,
        )
        self._output_budget = min(actual_budget, self.effective_total_G)
        self.effective_total_G = self._output_budget
        self._my_param = my_param

    def forward(
        self,
        queries: torch.Tensor,      # [B, 3]
        query_mask: torch.Tensor,    # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Step 1: get parent groundings (oversampled)
        body, mask, count, ridx = super().forward(queries, query_mask)
        # body: [B, tG_in, M, 3], mask: [B, tG_in]

        B, tG_in, M, _ = body.shape
        budget = self._output_budget

        # Step 2: score each grounding
        scores = self._score(body, mask)  # [B, tG_in]

        # Step 3: select top-k
        _, indices = scores.topk(budget, dim=1)  # [B, budget]

        # Step 4: gather selected groundings
        out_body = body.gather(1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
        out_mask = mask.gather(1, indices)
        out_ridx = ridx.gather(1, indices)
        out_count = out_mask.sum(dim=1)

        return out_body, out_mask, out_count, out_ridx

    def _score(
        self,
        body: torch.Tensor,          # [B, tG, M, 3]
        mask: torch.Tensor,          # [B, tG]
    ) -> torch.Tensor:               # [B, tG]
        """Compute scores for each grounding. Override this."""
        # Example: random scoring
        scores = torch.rand_like(mask.float()) * mask.float()
        return scores
```

### CUDA graph constraints

Your `_score` method must follow these rules:

- No `.item()` calls
- No `if tensor.sum() > 0:` or similar data-dependent branching
- No dynamic tensor shapes — output shape must be deterministic
- Use `torch.where()` instead of Python `if/else` on tensor values

---

## 4. Skeleton: Pipeline Override Grounder

For modifying the BC pipeline itself (not just scoring):

```python
class MyBCGrounder(BCGrounder):
    """Custom BC grounder with modified resolution."""

    def __init__(
        self,
        fact_index: FactIndex,
        rules: List[Rule],
        kb: KnowledgeBase,
        depth: int = 2,
        max_total_groundings: int = 64,
        device: str = "cuda",
        # --- custom args ---
        my_flag: bool = False,
    ) -> None:
        super().__init__(
            fact_index=fact_index,
            rules=rules,
            kb=kb,
            depth=depth,
            max_total_groundings=max_total_groundings,
            device=device,
        )
        self._my_flag = my_flag

    def _resolve_rules(self, query, remaining):
        """Custom rule resolution logic."""
        # Call parent for default behavior
        children, valid, rule_idx, body = super()._resolve_rules(query, remaining)

        # Apply custom filtering/modification
        # ... (must maintain fixed tensor shapes)

        return children, valid, rule_idx, body
```

---

## 5. Register in Factory

Add your grounder to the factory registry:

```python
# In factory.py

def create_grounder(grounder_type: str, ...) -> Grounder:
    ...
    if grounder_type.startswith("mygrounder_"):
        w, d = parse_grounder_type(grounder_type)
        return MyGrounder(
            fact_index=fact_index,
            rules=rules,
            kb=kb,
            depth=d,
            width=w,
            max_groundings_per_query=max_groundings,
            max_total_groundings=max_total_groundings,
            provable_set_method=provable_set_method,
            device=device,
        )
    ...
```

Naming convention: `mygrounder_{W}_{D}` (e.g., `mygrounder_1_2`).

---

## 6. Add Tests

Every new grounder needs three test types:

### Unit test

Verify basic functionality with a small KB:

```python
def test_mygrounder_basic():
    kb = make_test_kb()  # Small family KB
    grounder = MyGrounder(kb=kb, depth=2, ...)
    queries = torch.tensor([[3, 0, 3]])  # grandparent(john, sue)
    query_mask = torch.ones(1, dtype=torch.bool)
    body, mask, count, ridx = grounder(queries, query_mask)
    assert body.shape == (1, grounder.effective_total_G, grounder.max_body_atoms, 3)
    assert mask.shape == (1, grounder.effective_total_G)
    assert count.sum() > 0  # At least one grounding found
```

### Soundness test

Verify that every returned grounding is valid:

```python
def test_mygrounder_soundness():
    kb = make_test_kb()
    grounder = MyGrounder(kb=kb, depth=2, ...)
    body, mask, count, ridx = grounder(queries, query_mask)
    for b in range(B):
        for g in range(count[b]):
            assert is_valid_grounding(body[b, g], ridx[b, g], kb)
```

### Regression test

Compare against stored baselines:

```python
def test_mygrounder_regression():
    body, mask, count, ridx = grounder(queries, query_mask)
    baseline = load_baseline("mygrounder")
    assert count.tolist() == baseline["counts"]
```

---

## 7. Document Soundness

Add a row to the properties table in [soundness.md](soundness.md):

| Grounder | Sound | Complete | Bounds | Notes |
|----------|-------|----------|--------|-------|
| MyGrounder | Yes/No | Yes/No | ... | ... |

For scored grounders (subclasses of ParametrizedBCGrounder that only select from parent output): soundness is inherited from the parent. Document what the scoring function optimizes and what it might miss.

---

## Checklist

- [ ] Choose base class
- [ ] Implement constructor with typed args
- [ ] Override the minimal set of methods needed
- [ ] Ensure all tensor shapes are fixed (no dynamic shapes in `forward()`)
- [ ] No `.item()`, no data-dependent branching
- [ ] Register in factory with naming convention
- [ ] Unit test: basic functionality
- [ ] Soundness test: all groundings are valid
- [ ] Regression test: matches baseline
- [ ] Document soundness properties
- [ ] Set `effective_total_G` correctly (must reflect actual output capacity)
