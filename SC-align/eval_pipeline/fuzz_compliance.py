"""AST linter for the ``hypofuzz`` harness mode."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Violation:
    """One rule failure on one test function (or module-level)."""

    rule_id: str
    test_func: Optional[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _decorator_callable_name(dec: ast.expr) -> Optional[str]:
    """Return the dotted name of the callable being applied as a decorator."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    parts: List[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
        return ".".join(reversed(parts))
    return None


def _short_name(name: Optional[str]) -> str:
    """Last component of a dotted name."""
    if not name:
        return ""
    return name.rsplit(".", 1)[-1]


def _is_given_decorator(dec: ast.expr) -> bool:
    return isinstance(dec, ast.Call) and _short_name(_decorator_callable_name(dec)) == "given"


def _is_settings_decorator(dec: ast.expr) -> bool:
    return isinstance(dec, ast.Call) and _short_name(_decorator_callable_name(dec)) == "settings"


def _strategy_callable_short_name(strategy_expr: ast.expr) -> Optional[str]:
    """For ``st.just(1)`` → ``"just"``; for ``foo()`` → ``"foo"``; else None."""
    if not isinstance(strategy_expr, ast.Call):
        return None
    name = _decorator_callable_name(strategy_expr)
    return _short_name(name) if name else None


def _is_just_call(expr: ast.expr) -> bool:
    return _strategy_callable_short_name(expr) == "just"


_TINY_SAMPLED_FROM_THRESHOLD = 20


def _classify_strategy(expr: ast.expr) -> str:
    """Classify one strategy expression for R3/R4 enforcement."""
    if not isinstance(expr, ast.Call):
        return "unknown"
    short = _strategy_callable_short_name(expr)
    if short == "just":
        return "just"
    if short == "sampled_from":
        if not expr.args:
            return "real"
        arg0 = expr.args[0]
        if isinstance(arg0, (ast.List, ast.Tuple, ast.Set)):
            if len(arg0.elts) < _TINY_SAMPLED_FROM_THRESHOLD:
                return "tiny_sampled_from"
        return "real"
    return "real"


def _check_function(func: ast.FunctionDef) -> List[Violation]:
    violations: List[Violation] = []
    fname = func.name

    given_decorators: List[ast.Call] = [
        d for d in func.decorator_list if _is_given_decorator(d)
    ]
    settings_decorators: List[ast.Call] = [
        d for d in func.decorator_list if _is_settings_decorator(d)
    ]

    if not given_decorators:
        violations.append(Violation(
            rule_id="R1_has_given",
            test_func=fname,
            message=f"test function '{fname}' has no @given decorator "
                    f"(line {func.lineno})",
        ))

    for sd in settings_decorators:
        for kw in sd.keywords:
            if kw.arg != "phases":
                continue
            phases_val = kw.value
            if isinstance(phases_val, (ast.List, ast.Tuple, ast.Set)):
                for elt in phases_val.elts:
                    if isinstance(elt, ast.Attribute) and elt.attr == "explicit":
                        violations.append(Violation(
                            rule_id="R2_no_explicit_phase",
                            test_func=fname,
                            message=f"@settings(phases=[..., Phase.explicit, ...]) "
                                    f"disables fuzzing in '{fname}' (line {sd.lineno})",
                        ))
                        break

    for gd in given_decorators:
        classes = [
            _classify_strategy(s)
            for s in list(gd.args) + [kw.value for kw in gd.keywords]
        ]
        if classes and all(c == "just" for c in classes):
            violations.append(Violation(
                rule_id="R3_no_just_only",
                test_func=fname,
                message=f"every @given strategy in '{fname}' is st.just(...) "
                        f"— no real input generation (line {gd.lineno})",
            ))
        if classes and all(c == "tiny_sampled_from" for c in classes):
            violations.append(Violation(
                rule_id="R4_no_tiny_sampled_from",
                test_func=fname,
                message=f"every @given strategy in '{fname}' is "
                        f"st.sampled_from(<list of <"
                        f"{_TINY_SAMPLED_FROM_THRESHOLD} elements>) — "
                        f"input space too small (line {gd.lineno})",
            ))

    if given_decorators:
        bound_names: List[str] = []
        for gd in given_decorators:
            for kw in gd.keywords:
                if kw.arg:
                    bound_names.append(kw.arg)
        if bound_names:
            referenced: set = set()
            for stmt in func.body:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name):
                        referenced.add(node.id)
            for name in bound_names:
                if name not in referenced:
                    violations.append(Violation(
                        rule_id="R5_no_unused_given_arg",
                        test_func=fname,
                        message=f"parameter '{name}' bound by @given in '{fname}' "
                                f"is unused in the test body (cosmetic @given) "
                                f"(line {func.lineno})",
                    ))

    if not settings_decorators:
        violations.append(Violation(
            rule_id="R6_required_settings",
            test_func=fname,
            message=f"test function '{fname}' missing required @settings "
                    f"decorator (deadline=None expected) (line {func.lineno})",
        ))
    else:
        deadline_ok = False
        for sd in settings_decorators:
            for kw in sd.keywords:
                if kw.arg == "deadline":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                        deadline_ok = True
                        break
            if deadline_ok:
                break
        if not deadline_ok:
            violations.append(Violation(
                rule_id="R6_required_settings",
                test_func=fname,
                message=f"@settings in '{fname}' must set deadline=None "
                        f"(found other / no value) (line {settings_decorators[0].lineno})",
            ))

    return violations


def validate_fuzz_compliance(cert_code: str) -> List[Violation]:
    """Audit cert source for fuzz-compliance."""
    if not cert_code.strip():
        return [Violation(
            rule_id="R1_has_given",
            test_func=None,
            message="certificate is empty",
        )]

    try:
        tree = ast.parse(cert_code)
    except SyntaxError as e:
        return [Violation(
            rule_id="R1_has_given",
            test_func=None,
            message=f"certificate has SyntaxError ({e.msg}) at line {e.lineno}",
        )]

    test_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]

    if not test_funcs:
        return [Violation(
            rule_id="R1_has_given",
            test_func=None,
            message="certificate has no def test_* functions",
        )]

    violations: List[Violation] = []
    for f in test_funcs:
        violations.extend(_check_function(f))
    return violations
