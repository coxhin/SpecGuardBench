"""Replay agent-saved inputs under coverage."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional


RESULT_MARKER = "__SAVED_INPUTS_DRIVER_RESULT__"


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


def _count_inputs(inputs_path: str) -> Optional[int]:
    try:
        with open(inputs_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        count = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                return None
            count += 1
        return count
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("inputs", "cases", "saved_inputs"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def replay(
    cert_path: str,
    inputs_path: str,
    testbed: str = "/testbed",
    run_all_tests: bool = False,
) -> Dict[str, Any]:
    """Run ``pytest cert_path`` with ``AGENT_SAVED_INPUTS_PATH`` set."""
    t0 = time.time()
    result: Dict[str, Any] = {
        "cert_path": cert_path,
        "inputs_path": inputs_path,
        "inputs_count": _count_inputs(inputs_path),
        "pytest_exit_code": None,
        "coverage_data_path": None,
        "elapsed": None,
        "error": None,
    }

    os.environ["AGENT_SAVED_INPUTS_PATH"] = inputs_path
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    _clean_coverage(testbed)

    try:
        import coverage
        import pytest
    except Exception as e:
        result["error"] = f"import_failed: {type(e).__name__}: {e}"
        result["pytest_exit_code"] = 3
        result["elapsed"] = round(time.time() - t0, 2)
        return result

    cov = coverage.Coverage(
        branch=True,
        source=[testbed],
        data_file=os.path.join(testbed, ".coverage"),
        config_file=False,
    )

    exit_code = 3
    try:
        cov.start()
        args = [cert_path, "-v", "--tb=long"]
        if not run_all_tests:
            args.insert(1, "-x")
        exit_code = pytest.main(args)
    except SystemExit as e:
        raw_code = e.code
        exit_code = raw_code if isinstance(raw_code, int) else 1
    except Exception as e:
        result["error"] = f"pytest_failed: {type(e).__name__}: {e}"
        exit_code = 3
    finally:
        try:
            cov.stop()
            cov.save()
            result["coverage_data_path"] = os.path.join(testbed, ".coverage")
        except Exception as e:
            result["error"] = (
                (result["error"] + " ; " if result["error"] else "")
                + f"cov_save_failed: {type(e).__name__}: {e}"
            )

    result["pytest_exit_code"] = int(exit_code)
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) < 2:
        print(
            "usage: saved_inputs_driver.py <cert_path> <inputs_path> [<testbed>]",
            file=sys.stderr,
        )
        return 3
    cert_path = argv[0]
    inputs_path = argv[1]
    testbed = argv[2] if len(argv) >= 3 else "/testbed"
    summary = replay(
        cert_path,
        inputs_path,
        testbed=testbed,
        run_all_tests=os.environ.get("SAVED_INPUTS_RUN_ALL_TESTS", "") == "1",
    )
    print(f"{RESULT_MARKER}{json.dumps(summary, ensure_ascii=False)}", flush=True)
    return int(summary.get("pytest_exit_code") or 0)


if __name__ == "__main__":
    sys.exit(main())
