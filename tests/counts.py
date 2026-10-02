"""Grounding counts on real KGs: what each grounder type grounds for fixed query batches, against a recorded reference
(``tests/baselines/grounding_counts.json``). Optional (not a pytest file), GPU, under 90 s.

    python tests/counts.py                    # every cell: counts and content hashes exactly
    python tests/counts.py --cell family      # the cells whose name contains "family"
    python tests/counts.py --update           # record this run as the reference

A cell is a dataset, a grounder type and a query set, grounded as torch-ns grounds it: ids alphabetical over every
split and the grounding facts of ``torch_kge.KnowledgeBase`` (train.txt, or facts.txt plus the train triples of
relations no rule concludes), the rules whose predicates the KB has, ``create_grounder`` with torch-ns's defaults, and
the queries in fixed batches of ``BATCH`` (the keras filter's pruning depends on the batch). Per cell it checks each
step's goals, the kept firings and atoms, the firings per rule, and a hash of the whole output (atom tables, firings,
query rows); each step's raw groundings (what it wrote before the pruning, which a faster grounder may shrink) are
reported, not checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from grounder.api.rule_grounder import TensorFactIndex, create_grounder
from grounder.kb import parse_rules, parse_triples

REFERENCE = Path(__file__).with_name("baselines") / "grounding_counts.json"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "repos/data-swarm/main"))
BATCH, GROUP = 256, 4
FP = ["enum.fp_batch.w0.d1.flat", "enum.fp_batch.w1.d2.flat", "enum.fp_batch.w1.d3.flat"]
KERAS = ["enum.keras.w0.d1.flat", "enum.keras.w1.d1.flat", "enum.keras.w1.d2.flat", "enum.keras.w1.d2.r2.flat",
         "enum.keras.w1.d3.flat", "enum.keras.w1.d3.r3.flat"]
KERAS_W2 = ["enum.keras.w2.d1.flat", "enum.keras.w2.d2.flat", "enum.keras.w2.d3.flat"]     # IJCAI-25's Countries BC2x
# (dataset, query split, first n queries or None, grounder types)
CELLS = [("family", "test", None, FP + KERAS), ("family", "train", 2048, FP[1:2] + KERAS[2:4]),
         ("wn18rr", "test", None, FP + KERAS), ("countries_s3", "test", None, FP + KERAS + KERAS_W2),
         ("family", "test", 1024, KERAS_W2[1:2])]
INFO = ("raw", "seconds")      # reported, not checked
FIELDS = ("atom_table", "body_pool_idx", "body_atom_valid", "head_pool_idx", "rule_idx", "rule_offsets",
          "query_pool_idx")


class KB:
    """A dataset as torch_kge.KnowledgeBase reads it: ids, splits, rules and the grounding facts."""

    def __init__(self, name: str, device) -> None:
        path = DATA_ROOT / name
        read = lambda f: parse_triples(path / f) if (path / f).is_file() else []  # noqa: E731
        splits = {s: read(f"{s}.txt") for s in ("train", "valid", "test", "facts")}
        every = [t for s in splits.values() for t in s]
        self.relation2id = {n: i for i, n in enumerate(sorted({r for r, _, _ in every}))}
        self.entity2id = {n: i for i, n in enumerate(sorted({e for _, h, t in every for e in (h, t)}))}
        r, e = self.relation2id, self.entity2id
        self.split = {s: torch.tensor([[r[p], e[h], e[t]] for p, h, t in ts], dtype=torch.long).reshape(-1, 3)
                      for s, ts in splits.items()}
        self.rules = [r for r in parse_rules(path / "rules.txt")
                      if all(a[0] in self.relation2id for a in (r[0], *r[1]))]
        self.rule_names = [f"r{i}" for i in range(len(self.rules))]
        heads = torch.tensor(sorted({self.relation2id[h[0]] for h, _ in self.rules}))
        train, facts = self.split["train"], self.split["facts"]
        facts = torch.cat([facts, train[~torch.isin(train[:, 0], heads)]]) if len(facts) else train
        self.facts = [tuple(f) for f in torch.unique(facts, dim=0).tolist()]
        self.device = device

    def grounder(self, grounder_type: str):
        fi = TensorFactIndex(facts=self.facts, num_predicates=len(self.relation2id),
                             num_entities=len(self.entity2id), max_facts_per_query=64, device=self.device)
        max_states = max(32, min(256, (int(getattr(fi, "K", 0)) or 64) + len(self.rules)))
        return create_grounder(grounder_type, fact_index=fi, rules=self.rules, kb=self, max_groundings=32,
                               max_total_groundings=64, provable_set_method="spmm", device=self.device,
                               max_states=max_states)


def _digest(outputs) -> str:
    h = hashlib.sha256()
    for rg in outputs:
        for f in FIELDS:
            h.update(getattr(rg, f).detach().to("cpu", torch.int64).numpy().tobytes())
    return h.hexdigest()[:16]


def run_cell(kb: KB, dataset: str, split: str, n, grounder_type: str) -> dict:
    queries = kb.split[split][:n].to(kb.device)
    Q = queries.shape[0]
    S = -(-Q // BATCH)
    padded = torch.zeros(S * BATCH, 3, dtype=torch.long, device=kb.device)
    padded[:Q] = queries
    mask = torch.arange(S * BATCH, device=kb.device) < Q
    rg = kb.grounder(grounder_type)
    torch.cuda.synchronize()
    t = time.perf_counter()
    stats: list = []
    outs = [o for s in range(0, S, GROUP)                  # a few batches a call (each batch's output is unchanged)
            for o in rg.run_bc_many(padded.view(S, BATCH, 3)[s:s + GROUP], mask.view(S, BATCH)[s:s + GROUP],
                                    stats=stats)]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    goals, raw = [0] * rg._inner.depth, [0] * rg._inner.depth
    for d, g, k in stats:
        goals[d] += g
        raw[d] += k
    per_rule = torch.zeros(len(kb.rules), dtype=torch.long)
    for o in outs:
        per_rule += o.rule_offsets.diff().cpu()
    got = {"queries": Q, "goals": goals, "raw": raw, "firings": int(per_rule.sum()),
           "atoms": sum(int(o.atom_table.shape[0]) for o in outs),
           "proved_queries": sum(int(torch.isin(o.query_pool_idx, o.head_pool_idx).sum()) for o in outs),
           "per_rule": per_rule.tolist(), "sha": _digest(outs), "seconds": round(seconds, 2)}
    return got


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="", help="run the cells whose name contains this")
    p.add_argument("--update", action="store_true", help="record this run as the reference")
    a = p.parse_args()
    if not torch.cuda.is_available():
        print("counts: needs a GPU (the fast path is Triton)")
        return 2
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else {}
    results, drift, wall = {}, [], time.perf_counter()
    kbs = {}
    for dataset, split, n, types in CELLS:
        for grounder_type in types:
            name = f"{dataset}:{split}{'' if n is None else f'[:{n}]'}:{grounder_type}"
            if a.cell not in name:
                continue
            kb = kbs.setdefault(dataset, KB(dataset, "cuda"))
            got = results[name] = run_cell(kb, dataset, split, n, grounder_type)
            want = ref.get(name)
            bad = [] if want is None else [k for k in got if k not in INFO and got[k] != want.get(k)]
            raw_was = ("" if want is None or got["raw"] == want["raw"]
                       else " (raw was " + ">".join(map(str, want["raw"])) + ")")
            status = "NEW" if want is None else "DRIFT " + ",".join(bad) if bad else "ok" + raw_was
            if want is not None and bad:
                drift.append((name, {k: (want.get(k), got[k]) for k in bad}))
            print(f"{name:52s} goals {'>'.join(map(str, got['goals'])):22s} raw {'>'.join(map(str, got['raw'])):24s}"
                  f" firings {got['firings']:9d} atoms {got['atoms']:9d} {got['seconds']:6.2f}s  {status}")
    print(f"wall {time.perf_counter() - wall:.1f} s")
    for name, diff in drift:
        print(f"DRIFT {name}: " + "; ".join(f"{k} {w} -> {g}" for k, (w, g) in diff.items()))
    if a.update:
        ref.update(results)
        REFERENCE.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(results)} cells in {REFERENCE}")
        return 0
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
