"""CONTRACTS for the execution layer (AST-based; scans LIBRARY code only).

- Compiler.wrap is the ONLY torch.compile call site.
- cudagraph.mark_step_begin is the only torch.compiler.cudagraph_mark_step_begin call.
- detach_from_pool deep-clones an arbitrary pytree.
- validate() rejects globally-illegal and undeclared cells.
- CompileSpec self-validates illegal axis combos (cudagraphs ⊥ dynamic=True).
- Integration / CompileBackend enums match the vocab registries.
- the d/is_last duality lives only in DepthSelector (no isinstance(_, Tensor) in lib code).
- auto_select returns a cell the row declares (fan-out keyed).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from grounder.data.kb import KB
from grounder.errors import ConfigError, StrategyError
from grounder.execution.auto_select import auto_select
from grounder.execution.capability import (
    COMPILED_DYNAMIC, COMPILED_STEP, EAGER, OUTER_REDUCE_OVERHEAD,
    CapabilityRow, Cell, CompileBackend, CompileSpec, Integration, validate,
)
from grounder.execution.cudagraph import detach_from_pool
from grounder.execution.depth import DepthSelector
from grounder.config import FCConfig, PBCConfig, RTFConfig, SLDConfig
from grounder.factory import create_grounder, make_bcwd, make_grounder
from grounder.glossary import COMPILE_BACKENDS, INTEGRATIONS
from grounder.grounder.backward import BackwardGrounder
from grounder.backward.plan import RunPlan
from grounder.types import Layout, ProofState

_BUILD = Path(__file__).parent.parent
_LIB_PY = [p for p in _BUILD.rglob("*.py")
           if "__pycache__" not in p.parts and p.parent.name != "contracts"]


def _calls(predicate) -> list[tuple[str, int]]:
    out = []
    for p in _LIB_PY:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call) and predicate(node):
                out.append((p.name, node.lineno))
    return out


def test_single_compile_site() -> None:
    def is_torch_compile(n: ast.Call) -> bool:
        f = n.func
        return (isinstance(f, ast.Attribute) and f.attr == "compile"
                and isinstance(f.value, ast.Name) and f.value.id == "torch")
    sites = _calls(is_torch_compile)
    assert len(sites) == 1 and sites[0][0] == "compiler.py", f"torch.compile sites: {sites}"


def test_single_cudagraph_marker_site() -> None:
    def is_marker(n: ast.Call) -> bool:
        return isinstance(n.func, ast.Attribute) and n.func.attr == "cudagraph_mark_step_begin"
    sites = _calls(is_marker)
    assert len(sites) == 1 and sites[0][0] == "cudagraph.py", f"marker sites: {sites}"


def test_no_isinstance_on_tensor() -> None:
    def is_isinstance_tensor(n: ast.Call) -> bool:
        if not (isinstance(n.func, ast.Name) and n.func.id == "isinstance" and len(n.args) == 2):
            return False
        second = n.args[1]
        name = (second.attr if isinstance(second, ast.Attribute)
                else second.id if isinstance(second, ast.Name) else None)
        return name == "Tensor"
    sites = _calls(is_isinstance_tensor)
    assert not sites, f"isinstance(_, Tensor) in lib code (use DepthSelector/is_tensor): {sites}"


def test_layout_values_match_vocab() -> None:
    from grounder.glossary import LAYOUTS
    assert {m.value for m in Layout} == set(LAYOUTS)


def test_integration_values_match_vocab() -> None:
    assert {m.value for m in Integration} == set(INTEGRATIONS)


def test_compile_backend_values_match_vocab() -> None:
    assert {m.value for m in CompileBackend} == set(COMPILE_BACKENDS)


def test_compilespec_rejects_illegal_axis_combo() -> None:
    # CUDA-graph backend + dynamic=True is illegal -> rejected at construction
    with pytest.raises(StrategyError):
        CompileSpec(eager=False, backend=CompileBackend.REDUCE_OVERHEAD, dynamic=True)
    # but a non-cudagraph dynamic spec IS constructible (the granularity works)
    CompileSpec(eager=False, backend=CompileBackend.DEFAULT, dynamic=True, fullgraph=False)


def test_compiled_dynamic_preset_legal_dense_only() -> None:
    # COMPILED_DYNAMIC is constructible/legal (DEFAULT backend, not cudagraphs)
    assert not COMPILED_DYNAMIC.eager and COMPILED_DYNAMIC.dynamic is True
    assert not COMPILED_DYNAMIC.uses_cudagraphs()
    row = CapabilityRow(frozenset({Cell(Layout.DENSE, COMPILED_DYNAMIC),
                                   Cell(Layout.FLAT, EAGER)}))
    validate(row, Cell(Layout.DENSE, COMPILED_DYNAMIC))      # dense-compiled is legal
    with pytest.raises(StrategyError):
        validate(row, Cell(Layout.FLAT, COMPILED_DYNAMIC))   # flat + compiled rejected


def test_detach_from_pool_deep_clones() -> None:
    t = torch.zeros(2, 2)
    ps = ProofState(proof_goals=t, state_valid=t, top_rule_idx=t)
    cloned = detach_from_pool((t, [t], {"a": t}, ps))
    assert cloned[0] is not t and torch.equal(cloned[0], t)
    assert cloned[1][0] is not t
    assert cloned[2]["a"] is not t
    assert isinstance(cloned[3], ProofState) and cloned[3].proof_goals is not t


def test_strategy_validate_rejects_illegal_and_undeclared() -> None:
    flat_eager = Cell(Layout.FLAT, EAGER)
    dense_compiled = Cell(Layout.DENSE, COMPILED_STEP)
    row = CapabilityRow(frozenset({flat_eager, dense_compiled}))
    with pytest.raises(StrategyError):
        validate(row, Cell(Layout.FLAT, COMPILED_STEP))      # flat + compiled
    with pytest.raises(StrategyError):
        validate(row, Cell(Layout.SPARSE, COMPILED_STEP))    # sparse + compiled
    with pytest.raises(StrategyError):
        validate(row, Cell(Layout.DENSE, EAGER))             # undeclared
    validate(row, flat_eager)
    validate(row, dense_compiled)


def test_auto_select_returns_declared_cell() -> None:
    pbc = CapabilityRow(frozenset({Cell(Layout.FLAT, EAGER),
                                   Cell(Layout.DENSE, COMPILED_STEP)}))
    assert auto_select(pbc, K_f=28, K_r=16, G_r=64, M=2, B=4).layout is Layout.FLAT
    assert auto_select(pbc, K_f=3612, K_r=59, G_r=64, M=2, B=4) == Cell(Layout.DENSE, COMPILED_STEP)
    fwd = CapabilityRow(frozenset({Cell(Layout.SPARSE, EAGER)}))
    assert auto_select(fwd, K_f=999, K_r=1, G_r=1, M=1, B=1).layout is Layout.SPARSE
    sld = CapabilityRow(frozenset({Cell(Layout.DENSE, EAGER),
                                   Cell(Layout.DENSE, OUTER_REDUCE_OVERHEAD)}))
    assert auto_select(sld, K_f=28, K_r=16, G_r=64, M=2, B=4) == Cell(Layout.DENSE, EAGER)


def _pbc_grounder(**knobs) -> BackwardGrounder:
    facts = torch.tensor([[1, 1, 2], [1, 2, 3]])            # p(1,2), p(2,3)
    H = torch.tensor([[2, 4, 5]])                           # q(x,y)
    B = torch.tensor([[[1, 4, 6], [1, 6, 5]]])             # :- p(x,z), p(z,y)
    L = torch.tensor([2])
    kb = KB(facts, H, B, L, constant_no=3, predicate_no=12,
            padding_idx=9, device=torch.device("cpu"))
    return BackwardGrounder(kb, resolution="pbc", depth=2, **knobs)


def test_explicit_knobs_select_declared_cell() -> None:
    # Explicit layout/compile knobs resolve to the declared cell on the strategy.
    assert RunPlan.snapshot(_pbc_grounder(layout="dense")).strategy.cell == Cell(Layout.DENSE, EAGER)
    assert RunPlan.snapshot(_pbc_grounder(layout="flat")).strategy.cell == Cell(Layout.FLAT, EAGER)
    assert RunPlan.snapshot(_pbc_grounder(compile="graph")).strategy.cell == Cell(Layout.DENSE, COMPILED_STEP)
    assert RunPlan.snapshot(_pbc_grounder(compile="dynamic")).strategy.cell == Cell(Layout.DENSE, COMPILED_DYNAMIC)
    # chunk_size threads through as an explicit batch_size (single dep on the knob).
    assert RunPlan.snapshot(_pbc_grounder(chunk_size=8)).strategy.chunk.batch_size == 8
    # Bad knob strings raise ConfigError at construction.
    with pytest.raises(ConfigError):
        _pbc_grounder(layout="sideways")
    with pytest.raises(ConfigError):
        _pbc_grounder(compile="jit")


def test_explicit_flat_compiled_rejected() -> None:
    # FLAT + compiled is globally illegal; explicit() validate() rejects it at snapshot.
    with pytest.raises(StrategyError):
        RunPlan.snapshot(_pbc_grounder(layout="flat", compile="graph"))
    with pytest.raises(StrategyError):
        RunPlan.snapshot(_pbc_grounder(layout="flat", compile="dynamic"))


def test_snapshot_lockstep_default_and_layout_knobs() -> None:
    # Default ctor (knobs unset) -> auto path, byte-identical to today: eager FLAT cell,
    # legacy flat_intermediate stays False, one chunk for the whole query batch.
    default = RunPlan.snapshot(_pbc_grounder())
    assert default.strategy.cell.compile.eager  # auto-downgraded to eager
    assert default.strategy.cell == Cell(Layout.FLAT, EAGER)
    assert default.flat_intermediate is False
    assert len(list(default.strategy.chunk.slices(50))) == 1
    # layout='dense' -> Cell(DENSE,EAGER) AND flat_intermediate False.
    dense = RunPlan.snapshot(_pbc_grounder(layout="dense"))
    assert dense.strategy.cell == Cell(Layout.DENSE, EAGER) and dense.flat_intermediate is False
    # layout='flat' -> cell.layout FLAT AND (pbc) flat_intermediate True.
    flat = RunPlan.snapshot(_pbc_grounder(layout="flat"))
    assert flat.strategy.cell.layout is Layout.FLAT and flat.flat_intermediate is True


def test_compiled_seam_inert_on_default() -> None:
    # compile='off' (default) -> eager cell -> wrap_step is identity (never torch.compile).
    plan = RunPlan.snapshot(_pbc_grounder(layout="dense"))
    strat = plan.strategy
    assert strat.cell.compile.eager
    def step(x):
        return x
    assert strat.wrap_step(step, plan.shapes) is step
    assert strat.wrap_batch(step, plan.shapes) is step


def test_depth_selector_duality() -> None:
    eager = DepthSelector.make(1, 3, compiled=False)
    assert eager.d == 1 and eager.is_last is False and eager.width_for(7, 0) == 7
    assert DepthSelector.make(2, 3, compiled=False).width_for(7, 0) == 0
    comp = DepthSelector.make(1, 3, compiled=True)
    assert torch.is_tensor(comp.d) and torch.is_tensor(comp.is_last)
    acc = torch.zeros(1, 1, 3, 2)
    out = eager.write_slot(acc, torch.ones(1, 1, 2))
    assert torch.equal(out[:, :, 1], torch.ones(1, 1, 2)) and torch.equal(out[:, :, 0], torch.zeros(1, 1, 2))


# ── make_grounder: typed-config dispatch + construction-path equivalence ──

def _bc_kb() -> KB:
    facts = torch.tensor([[1, 1, 2], [1, 2, 3]])
    H = torch.tensor([[2, 4, 5]])
    B = torch.tensor([[[1, 4, 6], [1, 6, 5]]])
    L = torch.tensor([2])
    return KB(facts, H, B, L, constant_no=3, predicate_no=12,
              padding_idx=9, device=torch.device("cpu"))


def _bc_fingerprint(g: BackwardGrounder) -> tuple:
    """The resolved knobs that define a backward grounder's behavior."""
    return (g.resolution, g.filter_mode, g.depth, g.width, g._w_last_depth,
            g._all_anchors, g._flat_intermediate, g._collect_rule_groundings,
            g.S, g.C, g.max_goals)


def test_make_grounder_dispatches_on_config_type() -> None:
    kb = _bc_kb()
    sld = make_grounder(kb, SLDConfig(depth=2, max_total_groundings=4096),
                        max_derived_per_state=64)
    rtf = make_grounder(kb, RTFConfig(depth=2, max_total_groundings=4096),
                        max_derived_per_state=64)
    assert isinstance(sld, BackwardGrounder) and sld.resolution == "sld"
    assert rtf.resolution == "rtf"
    assert make_grounder(kb, PBCConfig(depth=2, w=1, max_total_groundings=64)).resolution == "pbc"
    from grounder.forward.grounder import ForwardGrounder
    assert isinstance(make_grounder(kb, FCConfig(depth=3)), ForwardGrounder)


def test_make_grounder_pbc_forces_invariants() -> None:
    # PBC forces all_anchors + (u=0) fp_batch filter — today's defaults, via config.
    g = make_grounder(_bc_kb(), PBCConfig(depth=2, w=1, max_total_groundings=64))
    assert g._all_anchors is True
    assert g.filter_mode == "fp_batch"
    # u>0 -> none filter, derived by the ctor (config.filter() not forced).
    g_u = make_grounder(_bc_kb(), PBCConfig(depth=2, w=1, u=1, max_total_groundings=64))
    assert g_u.filter_mode == "none"


def test_construction_paths_equivalent() -> None:
    """type-string create_grounder, make_bcwd, and direct PBCConfig produce the
    same resolved backward grounder (the BC/FC fork collapsed to one factory)."""
    raw = dict(facts_idx=torch.tensor([[1, 1, 2], [1, 2, 3]]),
               rule_heads=torch.tensor([[2, 4, 5]]),
               rule_bodies=torch.tensor([[[1, 4, 6], [1, 6, 5]]]),
               rule_lens=torch.tensor([2]), constant_no=3, predicate_no=12,
               padding_idx=9, device=torch.device("cpu"))
    via_str = create_grounder("bc12", **raw)
    via_bcwd = make_bcwd(_bc_kb(), w=1, d=2, u=0)
    via_cfg = make_grounder(_bc_kb(), PBCConfig(depth=2, w=1, u=0, flat_intermediate=True,
                                                max_groundings_per_query=32,
                                                max_total_groundings=64))
    assert _bc_fingerprint(via_str) == _bc_fingerprint(via_bcwd) == _bc_fingerprint(via_cfg)


def test_make_grounder_rejects_unknown_config() -> None:
    with pytest.raises(TypeError):
        make_grounder(_bc_kb(), object())
