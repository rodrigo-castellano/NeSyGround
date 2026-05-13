"""Shared logging adapter for standalone grounder analysis runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping

from kge_kernels.runs import LoggingConfig, ModelConfig, OutputConfig, run_one

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "output")


def _make_logging_config(
    config: Any,
    *,
    model_mode: str,
    model_filename: str,
) -> LoggingConfig:
    return LoggingConfig(
        output=OutputConfig(
            output_root=str(getattr(config, 'output_root', DEFAULT_OUTPUT_ROOT)),
        ),
        model=ModelConfig(
            mode=model_mode,
            filename=model_filename,
        ),
    )


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
    return run_one(
        copy.deepcopy(raw_config),
        config_cls=type(raw_config),
        run_experiment=run_fn,
        family_fn=lambda c: str(
            getattr(c, 'experiment_name', default_experiment_name)
        ),
        signature_fn=lambda c: str(
            getattr(c, 'run_signature', default_signature)
        ),
        logging_config_fn=lambda c: _make_logging_config(
            c, model_mode=model_mode, model_filename=model_filename,
        ),
    )


__all__ = [
    'DEFAULT_OUTPUT_ROOT',
    'run_logged_analysis',
]
