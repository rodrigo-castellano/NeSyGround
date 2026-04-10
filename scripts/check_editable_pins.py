#!/usr/bin/env python3
"""Verify SHA pins in pyproject.toml match the editable install of each dep.

Used as a pre-commit hook to keep `<name> @ git+<url>@<sha>` pins truthful
when shared dependencies are installed editable. The pin in pyproject.toml is
the historical record of "what version this commit was tested against." With
editable installs the live import always wins, so the pin can silently drift
relative to what was actually exercised. This hook catches that drift before
a misleading commit lands.

For each git+ dependency in [project].dependencies:
  - if the dep is not editable in the current env, skip (fresh-install case;
    by definition the pin is what got installed)
  - otherwise, find its source repo and HEAD SHA
  - require: editable HEAD matches the pinned SHA (prefix match either way,
    so 7-char short SHAs in pyproject still validate)
  - require: editable HEAD is reachable from at least one remote-tracking
    branch (otherwise pinning to it would 404 on a fresh install)

Exit 0 on success, 1 on any drift.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# name [optional-extras]? @ git+<url>@<sha>
# The URL is matched greedily so SSH-style URLs containing '@' (e.g.
# git+ssh://git@github.com/...) still parse — the trailing SHA must be 7-40
# hex chars at end of string, which forces backtracking onto the URL portion.
PIN_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?"
    r"\s*@\s*git\+(?P<url>.+)@(?P<sha>[0-9a-f]{7,40})\s*$"
)


def find_pyproject(start: Path | None = None) -> Path | None:
    """Walk upward from start (or CWD) until a pyproject.toml is found."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        cand = parent / "pyproject.toml"
        if cand.exists():
            return cand
    return None


def editable_path(pkg: str) -> Path | None:
    """Return the editable source path for pkg, or None if not editable."""
    res = subprocess.run(
        ["pip", "show", pkg],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        if line.startswith("Editable project location:"):
            return Path(line.split(":", 1)[1].strip())
    return None


def head_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def head_is_pushed(path: Path) -> bool:
    """True iff HEAD is reachable from at least one remote-tracking branch."""
    out = subprocess.run(
        ["git", "-C", str(path), "branch", "-r", "--contains", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return bool(out)


def main() -> int:
    pyproject = find_pyproject()
    if pyproject is None:
        print("check_editable_pins: no pyproject.toml found", file=sys.stderr)
        return 0

    data = tomllib.loads(pyproject.read_text())
    deps = data.get("project", {}).get("dependencies", []) or []

    failed = False
    for dep in deps:
        m = PIN_RE.match(dep)
        if not m:
            continue
        name, pin_sha = m["name"], m["sha"]

        ed = editable_path(name)
        if ed is None:
            # not editable in this env; the pin will be what gets installed.
            # nothing to verify here.
            continue

        actual = head_sha(ed)
        if not (actual.startswith(pin_sha) or pin_sha.startswith(actual)):
            print(f"STALE PIN: {name}")
            print(f"  pyproject.toml pins: {pin_sha}")
            print(f"  editable HEAD at {ed}: {actual}")
            print(f"  fix: push {name} and bump the pin to {actual}")
            failed = True
            continue

        if not head_is_pushed(ed):
            print(f"UNPUSHED HEAD: {name}")
            print(f"  editable {ed} is at {actual}, but no remote-tracking branch contains it.")
            print(f"  fix: cd {ed} && git push, then re-run this commit.")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
