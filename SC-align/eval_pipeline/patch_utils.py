"""Lightweight utilities for parsing unified diffs and scoping coverage — no heavy
dependencies.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_mutant_lines(patch: str, side: str = "new") -> Dict[str, List[int]]:
    """Extract per-file changed line numbers from a unified diff."""
    if side not in ("new", "old"):
        raise ValueError(f"side must be 'new' or 'old', got {side!r}")
    result: Dict[str, List[int]] = defaultdict(list)
    current_file: Optional[str] = None
    line_no = 0
    mine, theirs = ("+", "-") if side == "new" else ("-", "+")

    for raw in patch.split("\n"):
        if raw.startswith("+++ b/"):
            current_file = raw[6:]
            continue
        m = _HUNK_RE.match(raw)
        if m:
            line_no = int(m.group(2 if side == "new" else 1))
            continue
        if current_file is None:
            continue
        if raw.startswith(mine) and not raw.startswith(mine * 3):
            result[current_file].append(line_no)
            line_no += 1
        elif raw.startswith(theirs) and not raw.startswith(theirs * 3):
            pass
        else:
            line_no += 1

    return dict(result)


def fold_lines_to_statements(source: str, lines: List[int]) -> Dict[int, Optional[int]]:
    """Map each line to the first line of the statement it belongs to."""
    try:
        from coverage.parser import PythonParser

        parser = PythonParser(text=source, filename="<mutant>")
        parser.parse_source()
        statements = set(parser.statements)
        out: Dict[int, Optional[int]] = {}
        for line in lines:
            folded = parser.first_lines([line])
            first = next(iter(folded)) if folded else None
            out[line] = first if first in statements else None
        return out
    except ImportError:
        pass

    import ast

    tree = ast.parse(source)
    best: Dict[int, tuple] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            end = node.lineno
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.stmt, ast.excepthandler)):
                    continue
                child_end = getattr(child, "end_lineno", None)
                if child_end:
                    end = max(end, child_end)
        else:
            end = getattr(node, "end_lineno", None) or node.lineno
        for line in range(node.lineno, end + 1):
            candidate = (end - node.lineno, node.lineno)
            if line not in best or candidate < best[line]:
                best[line] = candidate
    return {line: (best[line][1] if line in best else None) for line in lines}


def mutant_line_hit_rate(
    coverage: Dict[str, Any],
    mutant_lines: Dict[str, List[int]],
    anchors: Optional[Dict[str, Dict[int, Optional[int]]]] = None,
) -> Dict[str, Any]:
    """Compute whether the cert touched the lines a mutant patch modifies."""
    if not mutant_lines:
        return {
            "hit_rate": None,
            "total": 0,
            "hit": 0,
            "detail": {},
        }

    files_data = coverage.get("files") or {}

    def _match(rel_path: str) -> Optional[Dict[str, Any]]:
        suffix = "/" + rel_path.lstrip("/")
        for abs_path, fdata in files_data.items():
            if abs_path == rel_path or abs_path.endswith(suffix):
                return fdata
        return None

    total = 0
    hit = 0
    detail: Dict[str, Any] = {}
    for rel_path, lines in mutant_lines.items():
        fd = _match(rel_path)
        executed = set(fd.get("executed_lines", [])) if fd else set()
        file_anchors = (anchors or {}).get(rel_path) or {}
        anchor_of = {ln: file_anchors.get(ln, ln) for ln in lines}
        code_lines = [ln for ln in lines if anchor_of[ln] is not None]
        not_code = [ln for ln in lines if anchor_of[ln] is None]
        hit_lines = [ln for ln in code_lines if anchor_of[ln] in executed]
        missed_lines = [ln for ln in code_lines if anchor_of[ln] not in executed]
        total += len(code_lines)
        hit += len(hit_lines)
        detail[rel_path] = {
            "lines": sorted(lines),
            "hit": sorted(hit_lines),
            "missed": sorted(missed_lines),
            "not_code": sorted(not_code),
            "anchors": {ln: anchor_of[ln] for ln in sorted(code_lines) if anchor_of[ln] != ln},
        }
        if fd is None and lines:
            logger.warning(
                "mutant_lines path %r not found in coverage data; "
                "treating all %d lines as missed", rel_path, len(lines),
            )

    return {
        "hit_rate": (hit / total) if total else None,
        "total": total,
        "hit": hit,
        "detail": detail,
    }
