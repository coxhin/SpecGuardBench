"""In-process hypofuzz driver — runs FuzzWorker without multiprocessing."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from typing import Any, Dict, Optional, Tuple


RESULT_MARKER = "__HYPOFUZZ_DRIVER_RESULT__"


def _coveragerc(testbed: str) -> str:
    """Inline coveragerc string."""
    return (
        "[run]\n"
        f"source = {testbed}\n"
        "branch = True\n"
        "[report]\n"
        "ignore_errors = True\n"
    )


def drive_one(
    cert_path: str,
    timeout_seconds: int,
    testbed: str = "/testbed",
) -> Dict[str, Any]:
    """Run one hypofuzz campaign for ``cert_path`` for at most ``timeout_seconds``."""
    t0 = time.time()
    result: Dict[str, Any] = {
        "cert_path": cert_path,
        "timeout_seconds": timeout_seconds,
        "fuzz_elapsed": None,
        "timed_out": None,
        "examples_count": None,
        "coverage_data_path": None,
        "error": None,
        "n_targets": None,
    }

    for stale in (".coverage", ".hypothesis"):
        p = os.path.join(testbed, stale)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    for entry in os.listdir(testbed):
        if entry.startswith(".coverage."):
            try:
                os.remove(os.path.join(testbed, entry))
            except OSError:
                pass

    import coverage
    cov = coverage.Coverage(
        branch=True,
        source=[testbed],
        data_file=os.path.join(testbed, ".coverage"),
        config_file=False,
    )
    cov.start()

    try:
        try:
            from hypofuzz import detection
            from hypofuzz.collection import collect_tests
            from hypofuzz.hypofuzz import FuzzWorker
            import hypofuzz.coverage as hf_cov
        except Exception as e:
            result["error"] = f"import_failed: {type(e).__name__}: {e}"
            return result

        detection._in_hypofuzz_run = True

        if sys.version_info[:2] < (3, 12):
            _orig_trace = hf_cov.CoverageCollector.trace_pre_312

            def _chained_trace(self, frame, event, arg):
                if self.prev_trace is not None:
                    try:
                        self.prev_trace(frame, event, arg)
                    except Exception:
                        pass
                return _orig_trace(self, frame, event, arg)

            hf_cov.CoverageCollector.trace_pre_312 = _chained_trace
        try:
            collected = collect_tests((cert_path,))
        except Exception as e:
            result["error"] = f"collect_failed: {type(e).__name__}: {e}"
            return result

        nodeids = {t.nodeid for t in collected.fuzz_targets}
        result["n_targets"] = len(nodeids)
        if not nodeids:
            result["error"] = "no_fuzz_targets"
            return result

        shared_state: Dict[str, Any] = {
            "hub_state": {"nodeids": nodeids},
            "worker_state": {
                "nodeids": {},
                "current_lifetime": 0.0,
                "expected_lifetime": 0.0,
                "valid_nodeids": [],
            },
        }
        worker = FuzzWorker(pytest_args=(cert_path,), shared_state=shared_state)

        timed_out = False
        cov_saved = [False]

        def _save_coverage_idempotent():
            if cov_saved[0]:
                return
            try:
                cov.stop()
                cov.save()
                cov_saved[0] = True
            except Exception as e:
                result["error"] = (
                    (result["error"] + " ; " if result["error"] else "")
                    + f"cov_save_in_alarm_failed: {type(e).__name__}: {e}"
                )

        def _alarm_soft(_signum, _frame):
            nonlocal timed_out
            timed_out = True
            _save_coverage_idempotent()
            signal.alarm(5)
            signal.signal(signal.SIGALRM, _alarm_hard)
            raise KeyboardInterrupt("hypofuzz_timeout")

        def _alarm_hard(_signum, _frame):
            _save_coverage_idempotent()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(124)

        signal.signal(signal.SIGALRM, _alarm_soft)
        signal.alarm(int(timeout_seconds))
        fuzz_start = time.time()
        try:
            worker.start()
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        except Exception as e:
            result["error"] = f"worker_exception: {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
        result["fuzz_elapsed"] = round(time.time() - fuzz_start, 2)
        result["timed_out"] = timed_out
    finally:
        _save_coverage_idempotent_outer = locals().get(
            "_save_coverage_idempotent"
        )
        if _save_coverage_idempotent_outer is not None:
            _save_coverage_idempotent_outer()
        else:
            try:
                cov.stop()
                cov.save()
            except Exception as e:
                result["error"] = (
                    (result["error"] + " ; " if result["error"] else "")
                    + f"cov_save_failed: {type(e).__name__}: {e}"
                )
        result["coverage_data_path"] = os.path.join(testbed, ".coverage")

    hyp_dir = os.path.join(testbed, ".hypothesis")
    n = 0
    if os.path.isdir(hyp_dir):
        for _root, _dirs, files in os.walk(hyp_dir):
            n += len(files)
    result["examples_count"] = n
    result["total_elapsed"] = round(time.time() - t0, 2)
    return result


def main(argv: Optional[list] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) < 2:
        print(
            "usage: hypofuzz_driver.py <timeout_seconds> <cert_path> [<testbed>]",
            file=sys.stderr,
        )
        return 1
    timeout_s = int(argv[0])
    cert_path = argv[1]
    testbed = argv[2] if len(argv) >= 3 else "/testbed"

    summary = drive_one(cert_path, timeout_s, testbed=testbed)
    print(f"{RESULT_MARKER}{json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
