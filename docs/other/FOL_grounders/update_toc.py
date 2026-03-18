#!/usr/bin/env python3
"""Auto-update the ## Index section of forward_chaining.md (or any similar file).

Scans the file for all  <a id="sec-..."></a>  anchors (skipping "sec-1",
the top-level section header) and rebuilds the index in the ## Index block
in-place, with subsections visually nested under their parent sections.

Heading level determines nesting:
  ### (level 3)  →  top-level numbered entry:   "N. [text](#anchor)"
  #### (level 4) →  indented bullet sub-entry:  "   - [text](#anchor)"
  ##### (level 5) → doubly-indented sub-entry:  "      - [text](#anchor)"
  ... and so on (3 extra spaces per additional level).

Usage:
    python update_toc.py [file.md]          # defaults to forward_chaining.md
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1 – collect (anchor_id, heading_text, level) triples
# ---------------------------------------------------------------------------

_ANCHOR_RE  = re.compile(r'<a id="(sec-[^"]+)"></a>')
_HEADING_RE = re.compile(r'(#{2,})\s+(.*)')


def collect_entries(lines: list[str]) -> list[tuple[str, str, int]]:
    """Return [(anchor_id, heading_text, heading_level), ...] in document order.

    heading_level is the number of '#' characters (e.g. ### → 3, #### → 4).
    Skips the top-level section anchor "sec-1".
    """
    entries = []
    for i, line in enumerate(lines):
        m = _ANCHOR_RE.match(line.strip())
        if not m:
            continue
        anchor = m.group(1)
        if anchor == "sec-1":
            continue
        # Find the heading on the next non-blank line
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j >= len(lines):
            continue
        hm = _HEADING_RE.match(lines[j].strip())
        if hm:
            level = len(hm.group(1))          # number of '#' chars
            text  = hm.group(2).strip()
            entries.append((anchor, text, level))
    return entries


# ---------------------------------------------------------------------------
# Step 2 – build the nested TOC
# ---------------------------------------------------------------------------

def build_toc(entries: list[tuple[str, str, int]]) -> str:
    if not entries:
        return ""

    # The minimum level among all entries is the "root" level (numbered).
    # Everything deeper gets indented bullet points.
    root_level = min(level for _, _, level in entries)

    lines = []
    counter = 0
    for anchor, text, level in entries:
        depth = level - root_level          # 0 = top-level, 1 = one indent, ...
        if depth == 0:
            counter += 1
            lines.append(f"{counter}. [{text}](#{anchor})")
        else:
            indent = "    " * depth         # 4 spaces per extra level
            lines.append(f"{indent}- [{text}](#{anchor})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3 – splice the new TOC into the file
# ---------------------------------------------------------------------------

# Matches from the start of the Index content to (but not including) the
# blank line + <a id="sec-1"> that ends the index block.
_TOC_BLOCK_RE = re.compile(
    r"(## Index\n\n)"
    r".*?"
    r"(\n\n<a id=\"sec-1\">)",
    re.DOTALL,
)


def update_file(path: Path) -> None:
    text = path.read_text()
    lines = text.splitlines()

    entries = collect_entries(lines)
    if not entries:
        print("No <a id='sec-...'> anchors found; nothing to do.")
        return

    toc = build_toc(entries)

    new_text = _TOC_BLOCK_RE.sub(
        lambda m: m.group(1) + toc + m.group(2),
        text,
        count=1,
    )

    if new_text == text:
        print(f"TOC already up to date ({len(entries)} entries).")
        return

    path.write_text(new_text)
    n_top  = sum(1 for _, _, lvl in entries if lvl == min(e[2] for e in entries))
    n_sub  = len(entries) - n_top
    print(f"Updated TOC: {n_top} top-level + {n_sub} sub-entries → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("forward_chaining.md")
    if not target.exists():
        sys.exit(f"File not found: {target}")
    update_file(target)
