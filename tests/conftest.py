"""Shared fixtures for grounder tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
GROUNDER_ROOT = TESTS_DIR.parent
BASELINES_DIR = TESTS_DIR / "baselines"
DATA_ROOT = Path(os.environ.get(
    "GROUNDER_DATA_ROOT",
    str(GROUNDER_ROOT / "data"),
))


def pytest_addoption(parser):
    parser.addoption("--dataset", default=None, help="Dataset name")
    parser.addoption("--grounder-type", default=None, help="Grounder resolution type (sld/rtf)")
    parser.addoption("--depth", type=int, default=None, help="Grounder depth")
    parser.addoption("--generate-baseline", action="store_true", default=False,
                     help="Generate baseline instead of testing against it")


_DEFAULTS = {
    "test_groundings.py": {
        "dataset": "family",
        "grounder_type": "sld",
        "depth": 4,
    },
}


def _get_default(request, key: str):
    cli_val = request.config.getoption(f"--{key.replace('_', '-')}", default=None)
    if cli_val is not None:
        return cli_val
    test_file = Path(request.fspath).name
    return _DEFAULTS.get(test_file, {}).get(key)


@pytest.fixture
def dataset(request):
    return _get_default(request, "dataset")


@pytest.fixture
def grounder_type(request):
    return _get_default(request, "grounder_type")


@pytest.fixture
def depth(request):
    return _get_default(request, "depth")


@pytest.fixture
def baseline_dir():
    return BASELINES_DIR


@pytest.fixture
def data_root():
    return DATA_ROOT


@pytest.fixture
def generate_baseline(request):
    return request.config.getoption("--generate-baseline")
