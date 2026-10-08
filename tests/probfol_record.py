"""probfol-llm's forward-chaining and SLD calls into the grounder: record them on main, check a branch against them.

    python tests/probfol_record.py           # check the grounder on the path against the recording
    python tests/probfol_record.py record    # record a new reference (after an intended change)
    python tests/probfol_record.py rekey     # key the recorded SLD calls by the SLD's inputs (the recording was
                                             # taken through the former grounder's backward shell)

Runs probfol-llm's own test suite in process, then ``GrounderExact`` on Industrial IoT (the staged join: rules of 3 to
8 body atoms, constants) and on MetaQA (spmm, 3-body chains), with the grounder's entry points wrapped: forward
chaining (``run_forward_chaining``) and SLD (``SLD.prove``). Each call is keyed by a digest of
its inputs (facts, rules, sizes, depth, queries) and recorded by a digest of its outputs, raw (the tensors probfol
reads, as laid out) and canonical (forward chaining: the derived atoms, sorted; SLD: each query's valid proofs, sorted).
Check mode requires every recorded call to recur with the same outputs, and probfol's tests to pass as they did.
The inputs and outputs themselves go to ``~/tmp/grounder-recordings/probfol.pt`` (for debugging a difference).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

PROBFOL = Path(os.environ.get("PROBFOL_ROOT", Path.home() / "repos/probfol-llm-swarm/main"))
REFERENCE = Path(__file__).with_name("baselines") / "probfol.json"
TENSORS = Path.home() / "tmp/grounder-recordings/probfol.pt"
EXTRA = {   # GrounderExact workloads probfol's tests do not reach
    "iot": PROBFOL / "datasets/industrial_iot/data/industrial_v3_clean",
    "metaqa": PROBFOL / "datasets/metaqa/data/metaqa",
}


def _digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t = t.detach().to("cpu")
            h.update(f"{tuple(t.shape)}{t.dtype}".encode())
            h.update((t.to(torch.int64) if t.dtype == torch.bool else t).contiguous().numpy().tobytes())
        else:
            h.update(repr(t).encode())
    return h.hexdigest()[:16]


def _rules_of(compiled) -> list:
    """The rules of compiled ``RulePattern`` s as plain tuples: (head predicate, head vars, body atoms)."""
    return [(cr.head_pred_idx, cr.head_var0, cr.head_var1,
             tuple((bp["pred_idx"], bp["arg0_var"], bp["arg1_var"])
                   for bp in getattr(cr, "_orig_body_patterns", cr.body_patterns)[:cr.num_body]))
            for cr in compiled]


class Recorder:
    def __init__(self) -> None:
        self.calls: list = []
        self.tensors: list = []
        self.node = "?"

    def add(self, kind: str, inputs: tuple, raw: tuple, canonical: tuple) -> None:
        self.calls.append({"node": self.node, "kind": kind, "in": _digest(*inputs), "raw": _digest(*raw),
                           "canonical": _digest(*canonical)})
        self.tensors.append({"node": self.node, "kind": kind, "inputs": inputs, "raw": raw})

    # ── the grounder's entry points as probfol calls them ──
    def install(self) -> None:
        import grounder.forward as forward
        import grounder.forward.router as router
        from grounder.sld import SLD
        rec, run_fc, prove = self, router.run_forward_chaining, SLD.prove

        def run_forward_chaining(compiled_rules, facts_idx, num_entities, num_predicates, depth=10, device="cpu",
                                 **kw):
            out = run_fc(compiled_rules, facts_idx, num_entities, num_predicates, depth, device, **kw)
            hashes, n = out
            E = num_entities
            keys = hashes[:n].to("cpu")
            atoms = torch.stack([keys // (E * E), keys // E % E, keys % E], 1)
            rec.add("forward", (facts_idx, repr(_rules_of(compiled_rules)), num_entities, num_predicates, depth,
                                repr(sorted(kw.items()))),
                    (keys, n), (atoms[torch.argsort(atoms[:, 0] * E * E + atoms[:, 1] * E + atoms[:, 2])],))
            return out

        def prove_(self_, queries, mask=None, excluded=None):
            p = prove(self_, queries, mask, excluded)
            proofs = []
            for b in range(p.body.shape[0]):           # each query's valid proofs, as sorted tuples
                v = p.mask[b]
                proofs.append(sorted(zip(p.rule[b][v].tolist(), p.head[b][v].tolist(), p.body[b][v].tolist(),
                                         p.count[b][v].tolist())))
            rec.add("sld", _sld_inputs(self_, queries, mask),
                    (p.body, p.mask, p.mask.sum(1), p.rule, p.count, p.head, self_.depth),
                    (repr(proofs), self_.depth))
            return p

        router.run_forward_chaining = forward.run_forward_chaining = run_forward_chaining
        SLD.prove = prove_


def _sld_inputs(sld, queries, mask) -> tuple:
    """An SLD call's inputs: its program, sizes and queries."""
    kb = sld.kb
    return (kb.facts.atoms, kb.rules.heads, kb.rules.bodies, kb.rules.lens, kb.E, kb.pad, sld.depth, sld.prune,
            sld.drop_facts, sld.S, sld.P, sld.L, sld.max_children, queries, mask)


def rekey() -> None:
    """The recorded SLD calls (made through the former grounder's backward shell) keyed by ``_sld_inputs`` of the
    SLD probfol-llm now builds from the same tensors: ``KB(facts, rules, E=constant_no + 1, pad)``, the depth,
    prune, proofs and atoms of the recorded config."""
    import re
    from grounder.kb import KB
    from grounder.sld import SLD
    ref = json.loads(REFERENCE.read_text())
    tensors = torch.load(TENSORS, weights_only=False)
    calls = [c for c in ref["calls"]]
    for c, t in zip(calls, tensors):
        if c["kind"] != "sld":
            continue
        facts, heads, bodies, lens, cno, _pno, pad, cfg, q, m = t["inputs"]
        L = re.search(r"max_atoms=(\d+|None)", cfg)[1]
        prune = re.search(r"filter='?(\w+)'?", cfg)[1]
        sld = SLD(KB(facts, heads, bodies, lens, E=cno + 1, pad=pad), depth=int(re.search(r"SLD\(depth=(\d+)", cfg)[1]),
                  prune="fp_batch" if prune == "fp_batch" else None,
                  max_proofs=int(re.search(r"max_groundings_per_query=(\d+)", cfg)[1]),
                  max_atoms=None if L == "None" else int(L))
        c["in"] = _digest(*_sld_inputs(sld, q, m))
    REFERENCE.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"rekeyed {sum(c['kind'] == 'sld' for c in calls)} SLD calls in {REFERENCE}")


def run(rec: Recorder) -> dict:
    import pytest

    class Plugin:
        outcomes: dict = {}

        def pytest_runtest_setup(self, item):
            rec.node = item.nodeid

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.outcome != "passed":
                self.outcomes[report.nodeid] = report.outcome

    plugin = Plugin()
    os.chdir(PROBFOL)
    sys.path.insert(0, str(PROBFOL))
    code = pytest.main(["-q", "-p", "no:cacheprovider", "tests"], plugins=[plugin])
    from prover.grounder_exact import GrounderExact
    for name, path in EXTRA.items():
        rec.node = f"extra::{name}"
        t = time.perf_counter()
        GrounderExact(path / "facts.txt", path / "rules.txt")
        print(f"{name}: {time.perf_counter() - t:.1f} s")
    return {"exit": int(code), "outcomes": plugin.outcomes}


def main() -> int:
    if sys.argv[1:] == ["rekey"]:
        rekey()
        return 0
    record = sys.argv[1:] == ["record"]
    import grounder
    print(f"grounder: {Path(grounder.__file__).parent}")
    rec = Recorder()
    rec.install()
    tests = run(rec)
    got = {"tests": tests, "calls": rec.calls}
    if record:
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(got, indent=1) + "\n")
        TENSORS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec.tensors, TENSORS)
        print(f"recorded {len(rec.calls)} calls ({sum(c['kind'] == 'sld' for c in rec.calls)} SLD), "
              f"probfol tests exit {tests['exit']}: {REFERENCE}")
        return 0
    ref = json.loads(REFERENCE.read_text())
    want = {(c["kind"], c["in"]): c for c in ref["calls"]}
    seen = {(c["kind"], c["in"]): c for c in got["calls"]}
    bad = []
    for key, c in want.items():
        g = seen.get(key)
        if g is None:
            bad.append(f"{c['node']} {c['kind']}: not called")
        elif g["canonical"] != c["canonical"]:
            bad.append(f"{c['node']} {c['kind']}: outputs differ")
        elif g["raw"] != c["raw"]:
            bad.append(f"{c['node']} {c['kind']}: same outputs, laid out differently")
    new = [f"{c['node']} {c['kind']}: not in the recording" for k, c in seen.items() if k not in want]
    changed = {n: (o, tests["outcomes"].get(n)) for n, o in ref["tests"]["outcomes"].items()
               if tests["outcomes"].get(n) != o}
    for line in bad + new + [f"test {n}: {a} -> {b}" for n, (a, b) in changed.items()]:
        print("DIFF", line)
    print(f"{len(want)} recorded calls, {len(bad)} differ, {len(new)} new; {len(changed)} tests changed outcome")
    return 1 if bad or new or changed else 0


if __name__ == "__main__":
    sys.exit(main())
