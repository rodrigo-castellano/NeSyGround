"""Shared logging adapter for standalone grounder analysis runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping

from kge_kernels.logging import LoggingConfig, ModelConfig, OutputConfig, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "output")


class _AnalysisSpec:
    def __init__(
        self,
        *,
        default_experiment_name: str,
        default_signature: str,
        run_fn: Callable[[Any, Any], Mapping[str, Any]],
        model_mode: str = 'none',
        model_filename: str = 'model.safetensors',
    ) -> None:
        self._default_experiment_name = default_experiment_name
        self._default_signature = default_signature
        self._run_fn = run_fn
        self._model_mode = model_mode
        self._model_filename = model_filename

    def resolve_config(self, raw_config: Any) -> Any:
        return copy.deepcopy(raw_config)

    def logging_config(self, config: Any) -> LoggingConfig:
        return LoggingConfig(
            output=OutputConfig(
                output_root=str(getattr(config, 'output_root', DEFAULT_OUTPUT_ROOT)),
            ),
            model=ModelConfig(
                mode=self._model_mode,
                filename=self._model_filename,
            ),
        )

    def family(self, config: Any) -> str:
        return str(getattr(config, 'experiment_name', self._default_experiment_name))

    def signature(self, config: Any) -> str:
        return str(getattr(config, 'run_signature', self._default_signature))

    def run(self, ctx, config: Any) -> Mapping[str, Any]:
        return self._run_fn(ctx, config)


def run_logged_analysis(
    raw_config: Any,
    *,
    default_experiment_name: str,
    default_signature: str,
    run_fn: Callable[[Any, Any], Mapping[str, Any]],
    model_mode: str = 'none',
    model_filename: str = 'model.safetensors',
) -> Mapping[str, Any]:
    """Run one standalone analysis script inside the canonical output bundle."""
    spec = _AnalysisSpec(
        default_experiment_name=default_experiment_name,
        default_signature=default_signature,
        run_fn=run_fn,
        model_mode=model_mode,
        model_filename=model_filename,
    )
    return run_experiment(raw_config, spec)


__all__ = [
    'DEFAULT_OUTPUT_ROOT',
    'run_logged_analysis',
]
