import json
from pathlib import Path

from eval_pipeline.factory_oracle_driver import replay


def test_generated_factory_passes(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'args': (2,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    assert case['args'][0] % 2 == 0\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 0
    assert result["kind"] == "passed"
    assert result["cases_run"] == 1


def test_missing_contract_function_is_invalid_certificate(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text("def build_cases():\n    yield {}\n", encoding="utf-8")
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 3
    assert result["kind"] == "invalid_certificate"
    assert "check_case" in result["error"]


def test_empty_factory_is_invalid_certificate(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    return iter(())\n"
        "def check_case(case):\n"
        "    assert True\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 3
    assert result["kind"] == "invalid_certificate"
    assert "yielded no cases" in result["error"]


def test_non_assertion_check_failure_is_test_exception(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'args': (), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    raise RuntimeError('broken oracle')\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 1
    assert result["kind"] == "test_exception"
    assert "test_exception" in result["error"]


def test_invalid_call_case_shape_is_invalid_case(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'value': 1}\n"
        "def check_case(case):\n"
        "    assert True\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 3
    assert result["kind"] == "invalid_case"
    assert "exactly 'args' and 'kwargs'" in result["error"]


def test_kwargs_keys_must_be_strings(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'args': (), 'kwargs': {1: 'bad'}}\n"
        "def check_case(case):\n"
        "    assert True\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path))
    assert result["exit_code"] == 3
    assert result["kind"] == "invalid_case"
    assert "keys must be strings" in result["error"]


def test_certificate_load_failure_is_collection_error(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text("def broken(:\n", encoding="utf-8")

    result = replay(str(cert), testbed=str(tmp_path))

    assert result["exit_code"] == 3
    assert result["kind"] == "collection_error"
    assert "certificate_load_failed" in result["error"]


def test_checkout_root_is_importable_while_the_certificate_loads(
    tmp_path: Path,
) -> None:
    import sys

    testbed = tmp_path / "testbed"
    (testbed / "fb_probe_tests" / "protos").mkdir(parents=True)
    (testbed / "fb_probe_tests" / "protos" / "helper.py").write_text(
        "SAMPLE = 'from-tests-dir'\n", encoding="utf-8"
    )
    (testbed / "fb_probe_src" / "fb_probe_pkg").mkdir(parents=True)
    (testbed / "fb_probe_src" / "fb_probe_pkg" / "__init__.py").write_text(
        "def impl(x):\n    return x + 1\n", encoding="utf-8"
    )
    cert = tmp_path / "cert.py"
    cert.write_text(
        "import sys\n"
        "PATH_HEAD = sys.path[0]\n"
        "from fb_probe_tests.protos.helper import SAMPLE\n"
        "from fb_probe_src.fb_probe_pkg import impl\n"
        "target = impl\n"
        "def build_cases():\n"
        "    yield {'args': (1,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    assert SAMPLE == 'from-tests-dir'\n"
        "    assert target(*case['args'], **case['kwargs']) == 2\n",
        encoding="utf-8",
    )
    root = str(testbed.resolve())
    try:
        result = replay(str(cert), testbed=str(testbed))
        assert result["kind"] == "passed", result
        assert result["cases_run"] == 1
        assert sys.modules["_featurebench_certificate"].PATH_HEAD == root
    finally:
        for name in list(sys.modules):
            if name.startswith(("fb_probe_tests", "fb_probe_src")):
                del sys.modules[name]
        while root in sys.path:
            sys.path.remove(root)


def test_certificate_can_assert_unexpected_target_exception(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'args': (), 'kwargs': {}}\n"
        "def target():\n"
        "    raise ValueError('boom')\n"
        "def check_case(case):\n"
        "    try:\n"
        "        target(*case['args'], **case['kwargs'])\n"
        "    except Exception as exc:\n"
        "        raise AssertionError(type(exc).__name__) from exc\n",
        encoding="utf-8",
    )

    result = replay(str(cert), testbed=str(tmp_path))

    assert result["exit_code"] == 1
    assert result["kind"] == "assertion"
    assert "ValueError" in result["error"]


def test_coverage_counts_implementation_run_at_certificate_import(
    tmp_path: Path,
) -> None:
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "impl.py").write_text(
        "def gold_fn(x):\n"
        "    if x > 0:\n"
        "        a = 1\n"
        "    else:\n"
        "        a = 2\n"
        "    return a\n",
        encoding="utf-8",
    )
    cert = tmp_path / "cert.py"
    cert.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(testbed)!r})\n"
        "from impl import gold_fn\n"
        "_setup = gold_fn(1)\n"
        "target = gold_fn\n"
        "def build_cases():\n"
        "    yield {'args': (-5,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    target(*case['args'], **case['kwargs'])\n",
        encoding="utf-8",
    )

    result = replay(str(cert), testbed=str(testbed))
    assert result["kind"] == "passed"

    import coverage as coverage_mod

    cov = coverage_mod.Coverage(data_file=str(testbed / ".coverage"))
    cov.load()
    cov.json_report(outfile=str(tmp_path / "report.json"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    fn = next(
        blk["functions"]["gold_fn"]
        for name, blk in report["files"].items()
        if name.endswith("impl.py")
    )

    assert fn["missing_lines"] == []
    assert fn["summary"]["num_statements"] == 4
    assert fn["summary"]["covered_lines"] == 4


def _cert_failing_on_second_of_four(tmp_path: Path) -> Path:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def _impl(x):\n"
        "    return x\n"
        "target = _impl\n"
        "def build_cases():\n"
        "    for i in range(4):\n"
        "        yield {'args': (i,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    x = target(*case['args'], **case['kwargs'])\n"
        "    assert x != 1, 'boom'\n",
        encoding="utf-8",
    )
    return cert


def test_cases_behind_a_failure_still_run(tmp_path: Path) -> None:
    result = replay(
        str(_cert_failing_on_second_of_four(tmp_path)),
        testbed=str(tmp_path), run_all_cases=True,
    )

    assert result["kind"] == "assertion"
    assert result["exit_code"] == 1
    assert "boom" in result["error"]
    assert result["first_failure_index"] == 1

    assert result["cases_seen"] == 4
    assert result["cases_run"] == 3
    assert result["cases_failed"] == 1
    assert result["cases_truncated_by_budget"] is False


def test_first_failure_wins_when_several_cases_fail(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def build_cases():\n"
        "    yield {'args': (0,), 'kwargs': {}}\n"
        "    yield {'args': (1,), 'kwargs': {}}\n"
        "    yield {'args': (2,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    x = case['args'][0]\n"
        "    if x == 1:\n"
        "        assert False, 'assertion first'\n"
        "    if x == 2:\n"
        "        raise RuntimeError('exception second')\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path), run_all_cases=True)

    assert result["kind"] == "assertion"
    assert result["first_failure_index"] == 1
    assert result["cases_failed"] == 2
    assert result["cases_seen"] == 3
    assert result["cases_run"] == 1


def test_malformed_case_does_not_stop_the_well_formed_ones(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "def _impl(x):\n"
        "    return x\n"
        "target = _impl\n"
        "def build_cases():\n"
        "    yield {'nonsense': 1}\n"
        "    yield {'args': (7,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    target(*case['args'], **case['kwargs'])\n",
        encoding="utf-8",
    )
    result = replay(str(cert), testbed=str(tmp_path), run_all_cases=True)

    assert result["kind"] == "invalid_case"
    assert result["exit_code"] == 3
    assert result["cases_seen"] == 2
    assert result["cases_run"] == 1


def test_budget_stops_iteration_without_disturbing_the_verdict(tmp_path: Path) -> None:
    cert = tmp_path / "cert.py"
    cert.write_text(
        "import time\n"
        "def build_cases():\n"
        "    for i in range(50):\n"
        "        yield {'args': (i,), 'kwargs': {}}\n"
        "def check_case(case):\n"
        "    if case['args'][0] == 0:\n"
        "        assert False, 'early'\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    result = replay(
        str(cert), testbed=str(tmp_path),
        budget_seconds=0.2, run_all_cases=True,
    )

    assert result["kind"] == "assertion"
    assert result["first_failure_index"] == 0
    assert result["cases_truncated_by_budget"] is True
    assert result["cases_seen"] < 50


def test_no_budget_means_no_truncation(tmp_path: Path) -> None:
    result = replay(
        str(_cert_failing_on_second_of_four(tmp_path)),
        testbed=str(tmp_path), run_all_cases=True,
    )
    assert result["cases_truncated_by_budget"] is False
    assert result["cases_seen"] == 4


def test_default_still_stops_at_the_first_failure(tmp_path: Path) -> None:
    result = replay(str(_cert_failing_on_second_of_four(tmp_path)), testbed=str(tmp_path))

    assert result["kind"] == "assertion"
    assert result["first_failure_index"] == 1
    assert result["cases_seen"] == 2
    assert result["cases_run"] == 1
    assert result["cases_failed"] == 1


def test_verdict_is_identical_in_both_modes(tmp_path: Path) -> None:
    cert = _cert_failing_on_second_of_four(tmp_path)
    stopped = replay(str(cert), testbed=str(tmp_path))
    exhaustive = replay(str(cert), testbed=str(tmp_path), run_all_cases=True)

    for field in ("kind", "exit_code", "error", "first_failure_index"):
        assert stopped[field] == exhaustive[field], field
    assert exhaustive["cases_seen"] > stopped["cases_seen"]
