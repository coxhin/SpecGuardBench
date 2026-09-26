"""Import-aware static call-chain resolver for coverage scoping."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
}


@dataclass(frozen=True)
class FunctionRef:
    file: str
    name: str

    def key(self) -> Tuple[str, str]:
        return (self.file, self.name)

    def as_dict(self) -> Dict[str, str]:
        return {"file": self.file, "name": self.name}


@dataclass(frozen=True)
class ImportBinding:
    kind: str
    module: str
    symbol: Optional[str] = None


@dataclass
class SourceInfo:
    file: str
    module: str
    is_package: bool
    tree: ast.Module
    bodies: Dict[str, ast.AST]
    classes: Set[str]
    module_imports: Dict[str, ImportBinding]


class _ScopedVisitor(ast.NodeVisitor):
    """Visit one lexical scope without descending into nested definitions."""

    def __init__(self) -> None:
        self.calls: List[ast.Call] = []
        self.import_nodes: List[ast.AST] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.import_nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_nodes.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class ImportAwareCallResolver:
    def __init__(self, testbed_root: str, gold_funcs: Iterable[FunctionRef]) -> None:
        self.root = Path(testbed_root).resolve()
        self.gold_funcs = list(gold_funcs)
        self.gold_keys = {ref.key() for ref in self.gold_funcs}
        self.gold_by_leaf: Dict[str, List[FunctionRef]] = {}
        for ref in self.gold_funcs:
            self.gold_by_leaf.setdefault(ref.name.rsplit(".", 1)[-1], []).append(ref)
        self.module_to_file, self.file_to_module = self._build_module_index()
        self._source_cache: Dict[str, Optional[SourceInfo]] = {}

    def _build_module_index(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        module_to_file: Dict[str, str] = {}
        file_to_module: Dict[str, str] = {}
        python_files: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            python_files.extend(
                Path(dirpath) / filename
                for filename in filenames
                if filename.endswith(".py")
            )

        for path in python_files:
            rel_path = path.relative_to(self.root)
            rel = rel_path.as_posix()
            stem_parts = list(rel_path.with_suffix("").parts)
            is_package = stem_parts[-1] == "__init__"
            if is_package:
                stem_parts.pop()
            if not stem_parts:
                continue

            full_module = ".".join(stem_parts)
            package_dirs = list(rel_path.parent.parts)
            start = len(package_dirs)
            while start > 0:
                package_path = self.root.joinpath(*package_dirs[:start])
                if not (package_path / "__init__.py").is_file():
                    break
                start -= 1
            if start < len(package_dirs):
                canonical_parts = package_dirs[start:]
                if not is_package:
                    canonical_parts = [*canonical_parts, rel_path.stem]
                canonical_module = ".".join(canonical_parts)
            else:
                canonical_module = full_module

            for module in {canonical_module, full_module}:
                if module:
                    module_to_file.setdefault(module, rel)
            file_to_module[rel] = canonical_module
        return module_to_file, file_to_module

    @staticmethod
    def _collect_definitions(tree: ast.Module) -> Tuple[Dict[str, ast.AST], Set[str]]:
        bodies: Dict[str, ast.AST] = {}
        classes: Set[str] = set()

        def collect(nodes: Iterable[ast.AST], class_path: Tuple[str, ...] = ()) -> None:
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bodies[".".join((*class_path, node.name))] = node
                elif isinstance(node, ast.ClassDef):
                    qualname = ".".join((*class_path, node.name))
                    classes.add(qualname)
                    collect(ast.iter_child_nodes(node), (*class_path, node.name))

        collect(ast.iter_child_nodes(tree))
        return bodies, classes

    @staticmethod
    def _module_scope_nodes(tree: ast.Module) -> List[ast.AST]:
        visitor = _ScopedVisitor()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            visitor.visit(node)
        return visitor.import_nodes

    def _source_info(self, file: str) -> Optional[SourceInfo]:
        norm = file.lstrip("/")
        if norm in self._source_cache:
            return self._source_cache[norm]
        path = self.root / norm
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            self._source_cache[norm] = None
            return None
        module = self.file_to_module.get(norm)
        if not module:
            self._source_cache[norm] = None
            return None
        bodies, classes = self._collect_definitions(tree)
        info = SourceInfo(
            file=norm,
            module=module,
            is_package=Path(norm).name == "__init__.py",
            tree=tree,
            bodies=bodies,
            classes=classes,
            module_imports={},
        )
        info.module_imports = self._bindings_from_imports(
            self._module_scope_nodes(tree), info
        )
        self._source_cache[norm] = info
        return info

    def _absolute_import_module(
        self, node: ast.ImportFrom, info: SourceInfo
    ) -> Optional[str]:
        if node.level == 0:
            return node.module or ""
        package = info.module if info.is_package else info.module.rpartition(".")[0]
        if not package:
            return None
        relative = "." * node.level + (node.module or "")
        try:
            return importlib.util.resolve_name(relative, package)
        except (ImportError, ValueError):
            return None

    def _bindings_from_imports(
        self, import_nodes: Iterable[ast.AST], info: SourceInfo
    ) -> Dict[str, ImportBinding]:
        bindings: Dict[str, ImportBinding] = {}

        def bind(name: str, value: ImportBinding) -> None:
            previous = bindings.get(name)
            if previous is None or previous == value:
                bindings[name] = value
            else:
                bindings[name] = ImportBinding("ambiguous", "")

        for node in import_nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bind(alias.asname, ImportBinding("module", alias.name))
                    else:
                        root_name = alias.name.split(".", 1)[0]
                        bind(root_name, ImportBinding("module", root_name))
            elif isinstance(node, ast.ImportFrom):
                base_module = self._absolute_import_module(node, info)
                if base_module is None:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    submodule = ".".join(
                        part for part in (base_module, alias.name) if part
                    )
                    if submodule in self.module_to_file:
                        bind(local_name, ImportBinding("module", submodule))
                    else:
                        bind(
                            local_name,
                            ImportBinding("symbol", base_module, alias.name),
                        )
        return bindings

    @staticmethod
    def _function_scope_nodes(func_node: ast.AST) -> _ScopedVisitor:
        visitor = _ScopedVisitor()
        for node in getattr(func_node, "body", []):
            visitor.visit(node)
        return visitor

    @staticmethod
    def _dotted_name(node: ast.AST) -> Optional[List[str]]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            prefix = ImportAwareCallResolver._dotted_name(node.value)
            if prefix:
                return [*prefix, node.attr]
        return None

    def _function_in_module(
        self, module: str, qualname: str
    ) -> Optional[FunctionRef]:
        file = self.module_to_file.get(module)
        if not file:
            return None
        info = self._source_info(file)
        if info is None or qualname not in info.bodies:
            return None
        return FunctionRef(file, qualname)

    def _resolve_qualified_export(
        self,
        module: str,
        qualname: str,
        seen: Optional[Set[Tuple[str, str]]] = None,
    ) -> Optional[FunctionRef]:
        seen = seen or set()
        key = (module, qualname)
        if key in seen:
            return None
        seen.add(key)

        direct = self._function_in_module(module, qualname)
        if direct:
            return direct
        file = self.module_to_file.get(module)
        info = self._source_info(file) if file else None
        if info is None:
            return None

        root, separator, remainder = qualname.partition(".")
        binding = info.module_imports.get(root)
        if binding is None:
            return None
        if binding.kind == "symbol" and binding.symbol:
            forwarded = binding.symbol
            if separator:
                forwarded = f"{forwarded}.{remainder}"
            return self._resolve_qualified_export(binding.module, forwarded, seen)
        if binding.kind == "module" and separator:
            return self._resolve_qualified_export(binding.module, remainder, seen)
        return None

    def _resolve_export(
        self, module: str, symbol: str
    ) -> Optional[FunctionRef]:
        return self._resolve_qualified_export(module, symbol)

    def _resolve_dotted_call(
        self,
        parts: List[str],
        current: FunctionRef,
        info: SourceInfo,
        bindings: Dict[str, ImportBinding],
    ) -> Optional[FunctionRef]:
        if not parts:
            return None
        if len(parts) == 1:
            name = parts[0]
            binding = bindings.get(name)
            if binding and binding.kind == "symbol" and binding.symbol:
                return self._resolve_export(binding.module, binding.symbol)
            if name in info.bodies:
                return FunctionRef(info.file, name)
            current_class = current.name.rpartition(".")[0]
            class_member = f"{current_class}.{name}" if current_class else ""
            if class_member in info.bodies:
                return FunctionRef(info.file, class_member)
            return None

        root, *tail = parts
        if root in {"self", "cls"}:
            current_class = current.name.rpartition(".")[0]
            if current_class and len(tail) == 1:
                qualname = f"{current_class}.{tail[0]}"
                if qualname in info.bodies:
                    return FunctionRef(info.file, qualname)
            return None

        if root in info.classes:
            qualname = ".".join(parts)
            if qualname in info.bodies:
                return FunctionRef(info.file, qualname)

        binding = bindings.get(root)
        if binding is None:
            module_prefix = ".".join(parts[:-1])
            return self._function_in_module(module_prefix, parts[-1])

        if binding.kind == "module":
            module = binding.module
            if len(tail) > 1:
                module = ".".join([module, *tail[:-1]])
            return self._function_in_module(module, tail[-1])

        if binding.kind == "symbol" and binding.symbol:
            qualname = ".".join([binding.symbol, *tail])
            return self._resolve_qualified_export(binding.module, qualname)
        return None

    def _resolve_getattr_call(
        self,
        call: ast.Call,
        current: FunctionRef,
        info: SourceInfo,
        bindings: Dict[str, ImportBinding],
    ) -> Tuple[Optional[FunctionRef], Optional[str]]:
        if not isinstance(call.func, ast.Call):
            return None, None
        inner = call.func
        if not isinstance(inner.func, ast.Name) or inner.func.id != "getattr":
            return None, None
        if len(inner.args) < 2:
            return None, "getattr call has no literal attribute"
        attr = inner.args[1]
        if not isinstance(attr, ast.Constant) or not isinstance(attr.value, str):
            return None, "getattr attribute is dynamic"
        base_parts = self._dotted_name(inner.args[0])
        if not base_parts:
            return None, "getattr receiver is dynamic"
        return (
            self._resolve_dotted_call(
                [*base_parts, attr.value], current, info, bindings
            ),
            None,
        )

    @staticmethod
    def _call_text(call: ast.Call) -> str:
        try:
            return ast.unparse(call.func)
        except Exception:
            return type(call.func).__name__

    def _candidate_dicts(self, leaf_name: Optional[str]) -> List[Dict[str, str]]:
        if not leaf_name:
            return []
        return [ref.as_dict() for ref in self.gold_by_leaf.get(leaf_name, [])]

    def _canonicalize(self, ref: FunctionRef) -> FunctionRef:
        info = self._source_info(ref.file)
        if info is None or ref.name in info.bodies:
            return ref
        suffix = "." + ref.name
        matches = [name for name in info.bodies if name.endswith(suffix)]
        if len(matches) == 1:
            return FunctionRef(ref.file, matches[0])
        return ref

    def resolve(self, declared: Iterable[FunctionRef]) -> Dict[str, object]:
        visited: Set[Tuple[str, str]] = set()
        scope: List[FunctionRef] = []
        unresolved: List[Dict[str, object]] = []
        queue = [self._canonicalize(ref) for ref in declared]

        while queue:
            current = self._canonicalize(queue.pop(0))
            if current.key() in visited:
                continue
            visited.add(current.key())
            scope.append(current)

            info = self._source_info(current.file)
            func_node = info.bodies.get(current.name) if info else None
            if info is None or func_node is None:
                unresolved.append({
                    "caller": current.as_dict(),
                    "line": None,
                    "expression": current.name,
                    "status": "unresolved",
                    "reason": "caller function body not found",
                    "candidates": [],
                })
                continue

            visitor = self._function_scope_nodes(func_node)
            local_bindings = self._bindings_from_imports(visitor.import_nodes, info)
            bindings = {**info.module_imports, **local_bindings}
            for call in visitor.calls:
                resolved, dynamic_reason = self._resolve_getattr_call(
                    call, current, info, bindings
                )
                parts = None
                if resolved is None and dynamic_reason is None:
                    parts = self._dotted_name(call.func)
                    if parts:
                        resolved = self._resolve_dotted_call(
                            parts, current, info, bindings
                        )
                leaf_name = parts[-1] if parts else None
                if resolved is not None:
                    if resolved.key() in self.gold_keys and resolved.key() not in visited:
                        queue.append(resolved)
                    continue

                candidates = self._candidate_dicts(leaf_name)
                if dynamic_reason is None and not candidates:
                    if parts and len(parts) == 1 and parts[0] in dir(builtins):
                        continue
                    continue
                if dynamic_reason is not None or candidates:
                    unresolved.append({
                        "caller": current.as_dict(),
                        "line": getattr(call, "lineno", None),
                        "expression": self._call_text(call),
                        "status": "ambiguous" if len(candidates) > 1 else "unresolved",
                        "reason": dynamic_reason or "callee could not be resolved exactly",
                        "candidates": candidates,
                    })

        scope.sort(key=lambda ref: ref.key())
        return {
            "scope": [ref.as_dict() for ref in scope],
            "complete": not unresolved,
            "unresolved_calls": unresolved,
        }


def resolve_scope(
    declared: Iterable[Dict[str, str]],
    gold_funcs: Iterable[Dict[str, str]],
    testbed_root: str,
) -> Dict[str, object]:
    resolver = ImportAwareCallResolver(
        testbed_root,
        [FunctionRef(item["file"], item["name"]) for item in gold_funcs],
    )
    return resolver.resolve(
        FunctionRef(item["file"], item["name"]) for item in declared
    )


def main() -> None:
    payload = json.loads(sys.stdin.read())
    result = resolve_scope(
        payload["declared"], payload["gold_funcs"], payload["testbed_root"]
    )
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
