"""Function-level coverage scoping and gold-patch function extraction."""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .call_chain_resolver import resolve_scope
from .models import canonical_testbed_path

logger = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(")
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z_0-9]*)\b")
_HUNK_CLASS_RE = re.compile(r"^@@.*@@\s+class\s+([A-Za-z_][A-Za-z_0-9]*)\b")
_EXIT = "__exit__"


def _match_file(
    files_data: Dict[str, Any], target_file: str
) -> Optional[Dict[str, Any]]:
    """Find the coverage.py file block for ``target_file``."""
    if target_file in files_data:
        return files_data[target_file]
    norm = canonical_testbed_path(target_file)
    if not norm:
        return None
    for key, fdata in files_data.items():
        if canonical_testbed_path(key) == norm:
            return fdata
    suffix = "/" + norm
    for key, fdata in files_data.items():
        if key.endswith(suffix):
            return fdata
    return None


def _arc_tuple(raw: Any) -> Optional[Tuple[Any, Any]]:
    """Normalize a two-item arc/edge payload to a tuple."""
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (raw[0], raw[1])
    return None


def _serialise_edges(edges: Set[Tuple[Any, Any]]) -> List[List[Any]]:
    """Stable JSON-serializable representation for edge sets."""
    return [
        [src, dst]
        for src, dst in sorted(edges, key=lambda e: (str(e[0]), str(e[1])))
    ]


def _line_edge_stats(func_block: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback edge coverage for function blocks without branches."""
    executed_branch_edges = {
        arc for arc in (_arc_tuple(a) for a in func_block.get("executed_branches", []))
        if arc is not None
    }
    missing_branch_edges = {
        arc for arc in (_arc_tuple(a) for a in func_block.get("missing_branches", []))
        if arc is not None
    }
    branch_edges = executed_branch_edges | missing_branch_edges
    if branch_edges:
        possible_edges = branch_edges
        covered_edges = executed_branch_edges & possible_edges
    else:
        executed_lines = {
            int(x) for x in func_block.get("executed_lines", [])
            if isinstance(x, int)
        }
        all_lines = sorted(
            executed_lines
            | {
                int(x) for x in func_block.get("missing_lines", [])
                if isinstance(x, int)
            }
        )
        possible_edges: Set[Tuple[Any, Any]] = set()
        covered_edges: Set[Tuple[Any, Any]] = set()
        if all_lines:
            for src, dst in zip(all_lines, all_lines[1:]):
                edge = (src, dst)
                possible_edges.add(edge)
                if src in executed_lines and dst in executed_lines:
                    covered_edges.add(edge)
            exit_edge = (all_lines[-1], _EXIT)
            possible_edges.add(exit_edge)
            if all_lines[-1] in executed_lines:
                covered_edges.add(exit_edge)

    total = len(possible_edges)
    covered = len(covered_edges)
    return {
        "num_edges": total,
        "covered_edges": covered,
        "edge_coverage_pct": (covered / total * 100) if total else None,
        "executed_edges": _serialise_edges(covered_edges),
        "missing_edges": _serialise_edges(possible_edges - covered_edges),
    }


def _function_edge_stats(
    func_block: Dict[str, Any],
    file_block: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute function edge coverage."""
    if not file_block:
        return _line_edge_stats(func_block)

    possible_file_edges = {
        arc for arc in (_arc_tuple(a) for a in file_block.get("possible_arcs", []))
        if arc is not None
    }
    executed_file_edges = {
        arc for arc in (_arc_tuple(a) for a in file_block.get("executed_arcs", []))
        if arc is not None
    }
    if not possible_file_edges:
        return _line_edge_stats(func_block)

    statement_lines = {
        int(x) for x in (
            list(func_block.get("executed_lines", []))
            + list(func_block.get("missing_lines", []))
        )
        if isinstance(x, int)
    }
    start_line = func_block.get("start_line")
    if not isinstance(start_line, int):
        return _line_edge_stats(func_block)

    def _belongs(edge: Tuple[Any, Any]) -> bool:
        src, dst = edge
        if isinstance(src, int) and isinstance(dst, int):
            if src in statement_lines and dst in statement_lines:
                return True
            if src in statement_lines and dst == -start_line:
                return True
        return False

    possible_edges = {edge for edge in possible_file_edges if _belongs(edge)}
    if not possible_edges:
        return _line_edge_stats(func_block)
    covered_edges = possible_edges & executed_file_edges
    total = len(possible_edges)
    covered = len(covered_edges)
    return {
        "num_edges": total,
        "covered_edges": covered,
        "edge_coverage_pct": (covered / total * 100) if total else None,
        "executed_edges": _serialise_edges(covered_edges),
        "missing_edges": _serialise_edges(possible_edges - covered_edges),
    }


def function_edge_stats(
    func_block: Dict[str, Any],
    file_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Public wrapper for deriving edge coverage from a function block."""
    return _function_edge_stats(func_block, file_block)


def scope_to_function(
    coverage: Optional[Dict[str, Any]],
    file: str,
    func_name: str,
) -> Optional[Dict[str, Any]]:
    """Return per-function coverage scoped to ``file::func_name``."""
    if not coverage:
        return None
    files_data = coverage.get("files") or {}
    fdata = _match_file(files_data, file)
    if fdata is None:
        return {
            "file": file,
            "function": func_name,
            "num_statements": 0,
            "covered_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "num_edges": 0,
            "covered_edges": 0,
            "line_coverage_pct": None,
            "branch_coverage_pct": None,
            "edge_coverage_pct": None,
            "executed_lines": [],
            "executed_branches": [],
            "missing_branches": [],
            "executed_edges": [],
            "missing_edges": [],
            "source": "not_found",
        }

    functions_block = fdata.get("functions")
    if functions_block is None:
        raise RuntimeError(
            f"coverage data for {file!r} has no per-file 'functions' block; "
            "this pipeline requires coverage>=7.0 with `coverage json --pretty-print`. "
            "If you see this against fresh data, the in-process driver or "
            "_extract_coverage was misconfigured."
        )
    resolved_func_name = func_name
    func_block = functions_block.get(func_name)
    if func_block is None:
        suffix = "." + func_name
        candidates = [
            k for k in functions_block
            if k == func_name or k.endswith(suffix)
        ]
        if len(candidates) == 1:
            resolved_func_name = candidates[0]
            func_block = functions_block[resolved_func_name]
        elif len(candidates) > 1:
            logger.warning(
                "scope_to_function: ambiguous qualname for %r in %r among "
                "%s; reporting not_found",
                func_name, file, candidates,
            )
    if func_block is None:
        return {
            "file": file,
            "function": func_name,
            "num_statements": 0,
            "covered_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "num_edges": 0,
            "covered_edges": 0,
            "line_coverage_pct": None,
            "branch_coverage_pct": None,
            "edge_coverage_pct": None,
            "executed_lines": [],
            "executed_branches": [],
            "missing_branches": [],
            "executed_edges": [],
            "missing_edges": [],
            "source": "not_found",
        }

    summary = func_block.get("summary", {})
    ns = int(summary.get("num_statements") or 0)
    cl = int(summary.get("covered_lines") or 0)
    nb = int(summary.get("num_branches") or 0)
    cb = int(summary.get("covered_branches") or 0)
    edge_stats = _function_edge_stats(func_block, fdata)
    return {
        "file": file,
        "function": resolved_func_name,
        "num_statements": ns,
        "covered_lines": cl,
        "num_branches": nb,
        "covered_branches": cb,
        "num_edges": edge_stats["num_edges"],
        "covered_edges": edge_stats["covered_edges"],
        "line_coverage_pct": (cl / ns * 100) if ns else None,
        "branch_coverage_pct": (cb / nb * 100) if nb else None,
        "edge_coverage_pct": edge_stats["edge_coverage_pct"],
        "executed_lines": list(func_block.get("executed_lines", [])),
        "executed_branches": list(func_block.get("executed_branches", [])),
        "missing_branches": list(func_block.get("missing_branches", [])),
        "executed_edges": edge_stats["executed_edges"],
        "missing_edges": edge_stats["missing_edges"],
        "source": "coverage_functions",
    }


@dataclass
class FunctionInfo:
    """One function identified by its source path and qualified name."""

    file: str
    name: str

    def key(self) -> Tuple[str, str]:
        return (self.file, self.name)

    def as_dict(self) -> Dict[str, str]:
        return {"file": self.file, "name": self.name}


def extract_gold_patch_functions(
    gold_patch: str,
    pre_image_root: Optional[str] = None,
) -> List[FunctionInfo]:
    """Parse a unified-diff gold patch; return newly-added functions and methods."""
    found: Set[Tuple[str, str]] = set()
    current_file: Optional[str] = None
    class_stack: List[Tuple[int, str]] = []
    for raw in gold_patch.split("\n"):
        if raw.startswith("+++ b/"):
            current_file = raw[6:]
            class_stack = []
            continue
        if raw.startswith("+++ a/"):
            current_file = raw[6:]
            class_stack = []
            continue
        if raw.startswith("@@"):
            class_stack = []
            hunk_class = _HUNK_CLASS_RE.match(raw)
            if hunk_class:
                class_stack.append((0, hunk_class.group(1)))
            continue
        if current_file is None or not raw or raw[0] not in {" ", "+"}:
            continue
        if raw.startswith("+++"):
            continue

        source_line = raw[1:]
        if not source_line.strip() or source_line.lstrip().startswith("@"):
            continue
        indent = len(source_line) - len(source_line.lstrip())
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()

        class_match = _CLASS_RE.match(source_line)
        if class_match:
            class_stack.append((indent, class_match.group(1)))
            continue

        if raw.startswith("+"):
            def_match = _DEF_RE.match(source_line)
            if def_match:
                qualname = ".".join(
                    [scope_name for _, scope_name in class_stack]
                    + [def_match.group(1)]
                )
                found.add((current_file, qualname))

    if not found:
        return []

    if pre_image_root is not None:
        filtered: List[FunctionInfo] = []
        for file, name in sorted(found):
            src_path = os.path.join(pre_image_root, file)
            if not os.path.isfile(src_path):
                filtered.append(FunctionInfo(file=file, name=name))
                continue
            preexisting = _ast_function_names(src_path)
            if name not in preexisting:
                filtered.append(FunctionInfo(file=file, name=name))
        return filtered

    return [FunctionInfo(file=f, name=n) for f, n in sorted(found)]


_NAMES_CACHE: Dict[str, Set[str]] = {}


def _ast_function_names(source_path: str) -> Set[str]:
    """Return the set of top-level + nested function names defined in a file."""
    if source_path in _NAMES_CACHE:
        return _NAMES_CACHE[source_path]
    names: Set[str] = set()
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=source_path)
    except (OSError, SyntaxError) as e:
        logger.warning("AST parse failed for %s: %s", source_path, e)
        _NAMES_CACHE[source_path] = names
        return names
    def collect(nodes: Iterable[ast.AST], class_path: Tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(".".join((*class_path, node.name)))
            elif isinstance(node, ast.ClassDef):
                collect(ast.iter_child_nodes(node), (*class_path, node.name))

    collect(ast.iter_child_nodes(tree))
    names.update(
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    _NAMES_CACHE[source_path] = names
    return names


def resolve_scope_via_call_chain(
    declared: List[FunctionInfo],
    gold_patch_funcs: List[FunctionInfo],
    testbed_root: str,
) -> Dict[str, Any]:
    """Resolve the gold-only transitive call scope with diagnostics."""
    return resolve_scope(
        [item.as_dict() for item in declared],
        [item.as_dict() for item in gold_patch_funcs],
        testbed_root,
    )


def extend_scope_via_call_chain(
    declared: List[FunctionInfo],
    gold_patch_funcs: List[FunctionInfo],
    testbed_root: str,
) -> List[FunctionInfo]:
    """Return the resolved scope."""
    result = resolve_scope_via_call_chain(
        declared, gold_patch_funcs, testbed_root
    )
    return [
        FunctionInfo(item["file"], item["name"])
        for item in result["scope"]
    ]


def aggregate_gold_patch_coverage(
    trials: Iterable[Dict[str, Any]],
    gold_funcs: List[FunctionInfo],
) -> Dict[str, Any]:
    """Compute metric D over a set of trials for a single (instance, state)."""
    per_function: Dict[str, Dict[str, Any]] = {}
    func_covered_edges: Dict[Tuple[str, str], Set[Tuple[Any, Any]]] = {
        f.key(): set() for f in gold_funcs
    }
    func_covered_arcs: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {
        f.key(): set() for f in gold_funcs
    }
    func_covered_lines: Dict[Tuple[str, str], Set[int]] = {
        f.key(): set() for f in gold_funcs
    }
    func_totals: Dict[Tuple[str, str], Dict[str, int]] = {}
    func_trials_with_data: Dict[Tuple[str, str], int] = {
        f.key(): 0 for f in gold_funcs
    }

    trials = list(trials)
    for f in gold_funcs:
        for trial in trials:
            scoped = scope_to_function(
                trial.get("coverage"),
                f.file, f.name,
            )
            if scoped is None or scoped.get("source") == "not_found":
                continue

            ftot = func_totals.setdefault(f.key(), {
                "num_edges": 0,
                "num_branches": 0,
                "num_statements": 0,
            })
            ftot["num_edges"] = max(
                ftot["num_edges"], int(scoped.get("num_edges") or 0)
            )
            ftot["num_branches"] = max(
                ftot["num_branches"], int(scoped.get("num_branches") or 0)
            )
            ftot["num_statements"] = max(
                ftot["num_statements"], int(scoped.get("num_statements") or 0)
            )

            executed_edges = scoped.get("executed_edges", [])
            executed_arcs = scoped.get("executed_branches", [])
            executed_lines_list = scoped.get("executed_lines", [])
            edges_added = 0
            arcs_added = 0
            lines_added = 0
            for edge in executed_edges:
                normalized = _arc_tuple(edge)
                if normalized is not None:
                    func_covered_edges[f.key()].add(normalized)
                    edges_added += 1
            for arc in executed_arcs:
                if isinstance(arc, (list, tuple)) and len(arc) >= 2:
                    func_covered_arcs[f.key()].add((arc[0], arc[1]))
                    arcs_added += 1
            for ln in executed_lines_list:
                if isinstance(ln, int):
                    func_covered_lines[f.key()].add(ln)
                    lines_added += 1
            if edges_added > 0 or arcs_added > 0 or lines_added > 0:
                func_trials_with_data[f.key()] += 1

    total_edges = 0
    covered_edges = 0
    total_branches = 0
    covered_branches = 0
    total_lines = 0
    covered_lines = 0
    for f in gold_funcs:
        totals = func_totals.get(f.key(), {
            "num_edges": 0,
            "num_branches": 0,
            "num_statements": 0,
        })
        ne = totals["num_edges"]
        nb = totals["num_branches"]
        ns = totals["num_statements"]
        ce = len(func_covered_edges[f.key()])
        cb = len(func_covered_arcs[f.key()])
        cl = len(func_covered_lines[f.key()])
        total_edges += ne
        covered_edges += ce
        total_branches += nb
        covered_branches += cb
        total_lines += ns
        covered_lines += cl
        per_function[f"{f.file}::{f.name}"] = {
            "covered_edges": ce,
            "total_edges": ne,
            "covered_branches": cb,
            "total_branches": nb,
            "covered_lines": cl,
            "total_lines": ns,
            "trials_with_data": func_trials_with_data[f.key()],
        }

    return {
        "edge_coverage_pct": (
            (covered_edges / total_edges * 100) if total_edges else None
        ),
        "branch_coverage_pct": (
            (covered_branches / total_branches * 100) if total_branches else None
        ),
        "line_coverage_pct": (
            (covered_lines / total_lines * 100) if total_lines else None
        ),
        "covered_edges": covered_edges,
        "total_edges": total_edges,
        "covered_branches": covered_branches,
        "total_branches": total_branches,
        "covered_lines": covered_lines,
        "total_lines": total_lines,
        "per_function": per_function,
    }


def spec_breadth(
    gold_funcs: List[FunctionInfo],
    targeted_funcs: Iterable[Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute metric C: which gold-patch functions did *any* spec target?"""
    gold_set = {(canonical_testbed_path(f.file), f.name) for f in gold_funcs}
    targeted_set: Set[Tuple[str, str]] = set()
    for entry in targeted_funcs:
        if entry and len(entry) == 2 and entry[0] and entry[1]:
            targeted_set.add((canonical_testbed_path(entry[0]), entry[1]))

    hit = gold_set & targeted_set
    missing = sorted(gold_set - targeted_set)
    return {
        "gold_patch_functions_total": len(gold_set),
        "gold_patch_functions_targeted": len(hit),
        "ratio": (len(hit) / len(gold_set)) if gold_set else None,
        "missing": [{"file": f, "name": n} for f, n in missing],
        "targeted_but_not_gold": sorted(
            [{"file": f, "name": n} for f, n in (targeted_set - gold_set)],
            key=lambda d: (d["file"], d["name"]),
        ),
    }
