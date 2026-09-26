"""Run a frozen certificate oracle against Python case factories."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from types import ModuleType
from typing import Any, Dict, Iterable, Optional


RESULT_MARKER = "__FACTORY_ORACLE_DRIVER_RESULT__"

DRIVER_VERSION = 5


def _clean_coverage(testbed: str) -> None:
    for stale in (".coverage", ".pytest_cache", ".hypothesis"):
        path = os.path.join(testbed, stale)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    for entry in os.listdir(testbed):
        if entry.startswith(".coverage."):
            try:
                os.remove(os.path.join(testbed, entry))
            except OSError:
                pass


def _load_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _iter_cases(factory: Any) -> Iterable[Any]:
    cases = factory()
    if cases is None:
        raise ValueError("build_cases() returned None")
    return iter(cases)


def _validate_call_case(case: Any) -> None:
    """Validate the call-case shape every certificate must yield."""
    if not isinstance(case, dict):
        raise TypeError("case must be a dict")
    if set(case) != {"args", "kwargs"}:
        raise ValueError("case must contain exactly 'args' and 'kwargs'")
    if not isinstance(case["args"], (tuple, list)):
        raise TypeError("case['args'] must be a tuple or list")
    if not isinstance(case["kwargs"], dict):
        raise TypeError("case['kwargs'] must be a dict")
    if not all(isinstance(key, str) for key in case["kwargs"]):
        raise TypeError("case['kwargs'] keys must be strings")


def replay(
    cert_path: str,
    testbed: str = "/testbed",
    budget_seconds: Optional[float] = None,
    run_all_cases: bool = False,
) -> Dict[str, Any]:
    """Execute ``check_case`` over the cases the certificate generates."""
    started = time.time()
    result: Dict[str, Any] = {
        "cert_path": cert_path,
        "driver_version": DRIVER_VERSION,
        "cases_run": 0,
        "cases_seen": 0,
        "cases_failed": 0,
        "first_failure_index": None,
        "cases_truncated_by_budget": False,
        "exit_code": 0,
        "kind": "passed",
        "coverage_data_path": None,
        "elapsed": None,
        "error": None,
    }
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    _clean_coverage(testbed)

    try:
        import coverage
    except Exception as exc:
        result["exit_code"] = 3
        result["kind"] = "infra_error"
        result["error"] = f"coverage_import_failed: {type(exc).__name__}: {exc}"
        result["elapsed"] = round(time.time() - started, 2)
        return result

    cov = coverage.Coverage(
        branch=True,
        source=[testbed],
        data_file=os.path.join(testbed, ".coverage"),
        config_file=False,
    )
    cov.start()
    try:
        testbed_root = os.path.abspath(testbed)
        if testbed_root not in sys.path:
            sys.path.insert(0, testbed_root)
        try:
            cert = _load_module("_featurebench_certificate", cert_path)
        except Exception as exc:
            result["exit_code"] = 3
            result["kind"] = "collection_error"
            result["error"] = (
                f"certificate_load_failed: {type(exc).__name__}: {exc}"
            )
            result["elapsed"] = round(time.time() - started, 2)
            return result
        check_case = getattr(cert, "check_case", None)
        if not callable(check_case):
            result["exit_code"] = 3
            result["kind"] = "invalid_certificate"
            result["error"] = "certificate must define callable check_case(case)"
            result["elapsed"] = round(time.time() - started, 2)
            return result
        build_cases = getattr(cert, "build_cases", None)
        if not callable(build_cases):
            result["exit_code"] = 3
            result["kind"] = "invalid_certificate"
            result["error"] = "certificate must define callable build_cases()"
            result["elapsed"] = round(time.time() - started, 2)
            return result

        cases = _iter_cases(build_cases)
        deadline = (started + budget_seconds) if budget_seconds else None

        def _fail(kind: str, exit_code: int, message: str) -> bool:
            """Record the failure; return whether the loop should keep going."""
            result["cases_failed"] += 1
            if result["kind"] == "passed":
                result["exit_code"] = exit_code
                result["kind"] = kind
                result["error"] = message
                result["first_failure_index"] = result["cases_seen"] - 1
            return run_all_cases

        try:
            for case in cases:
                if deadline is not None and time.time() >= deadline:
                    result["cases_truncated_by_budget"] = True
                    break
                result["cases_seen"] += 1
                try:
                    _validate_call_case(case)
                except Exception as exc:
                    if not _fail(
                        "invalid_case", 3,
                        f"invalid_case: {type(exc).__name__}: {exc}",
                    ):
                        break
                    continue
                try:
                    check_case(case)
                except AssertionError as exc:
                    if not _fail("assertion", 1, f"assertion_failed: {exc}"):
                        break
                    continue
                except Exception as exc:
                    if not _fail(
                        "test_exception", 1,
                        f"test_exception: {type(exc).__name__}: {exc}",
                    ):
                        break
                    continue
                result["cases_run"] += 1
        finally:
            close = getattr(cases, "close", None)
            if callable(close):
                close()
        if result["exit_code"] == 0 and result["cases_run"] == 0:
            result["exit_code"] = 3
            result["kind"] = "invalid_certificate"
            result["error"] = "build_cases() yielded no cases"
    except Exception as exc:
        result["exit_code"] = 3
        result["kind"] = "infra_error"
        result["error"] = f"execution_failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            cov.stop()
            cov.save()
            result["coverage_data_path"] = os.path.join(testbed, ".coverage")
        except Exception as exc:
            result["exit_code"] = 3
            result["kind"] = "infra_error"
            suffix = f"coverage_save_failed: {type(exc).__name__}: {exc}"
            result["error"] = (
                f"{result['error']} ; {suffix}" if result["error"] else suffix
            )

    result["elapsed"] = round(time.time() - started, 2)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(
            "usage: factory_oracle_driver.py <cert_path> [<testbed>]",
            file=sys.stderr,
        )
        return 3
    cert_path = argv[0]
    testbed = argv[1] if len(argv) >= 2 else "/testbed"
    raw_budget = os.environ.get("FACTORY_ORACLE_BUDGET_SECONDS", "")
    try:
        budget = float(raw_budget) if raw_budget else None
    except ValueError:
        budget = None
    run_all = os.environ.get("FACTORY_ORACLE_RUN_ALL_CASES", "") == "1"
    summary = replay(
        cert_path,
        testbed=testbed,
        budget_seconds=budget,
        run_all_cases=run_all,
    )
    print(f"{RESULT_MARKER}{json.dumps(summary, ensure_ascii=False)}", flush=True)
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
