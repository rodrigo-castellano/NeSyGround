"""Execution layer — every torch.compile / CUDA-graph / layout / chunk decision.

The engine (backward/, forward/) names none of these directly; it asks an
``ExecStrategy`` and gets back a policy. One file per decision:

  capability   — Cell (layout + CompileSpec) + CapabilityRow + validate
  auto_select  — pick a Cell from a grounder's declared row (fan-out heuristics)
  strategy     — ExecStrategy: the resolved policy + all chunk mechanics
  chunk_policy — ChunkPolicy: static memory math → padded query chunks
  compiler     — THE single torch.compile call site (Compiler.wrap)
  depth        — DepthSelector: per-depth width/write-slot bookkeeping
  cudagraph    — mark_step_begin + the single clone-to-detach-from-pool seam

Submodule-imported only (no package-level re-exports by design).
"""
