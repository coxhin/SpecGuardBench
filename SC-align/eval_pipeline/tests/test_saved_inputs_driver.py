from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_saved_inputs_driver_replays_cases_and_writes_coverage(tmp_path: Path):
    pytest.importorskip("coverage")

    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "impl.py").write_text(
        "def double(x):\n    return x * 2\n",
        encoding="utf-8",
    )
    cert_path = tmp_path / "cert.py"
    cert_path.write_text(
        textwrap.dedent(
            """
            import json
            import os

            from impl import double


            def test_saved_inputs_replay():
                with open(os.environ["AGENT_SAVED_INPUTS_PATH"], "r", encoding="utf-8") as f:
                    cases = [json.loads(line) for line in f if line.strip()]
                for case in cases:
                    x = case["x"]
                    assert double(x) == case["expected"]
            """
        ),
        encoding="utf-8",
    )
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(
        '{"x": 2, "expected": 4}\n{"x": -3, "expected": -6}\n',
        encoding="utf-8",
    )

    driver = Path(__file__).resolve().parent.parent / "saved_inputs_driver.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(testbed) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(driver), str(cert_path), str(inputs_path), str(testbed)],
        cwd=testbed,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "__SAVED_INPUTS_DRIVER_RESULT__" in proc.stdout
    assert (testbed / ".coverage").exists()


def test_saved_inputs_driver_returns_pytest_failure(tmp_path: Path):
    pytest.importorskip("coverage")

    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "impl.py").write_text(
        "def double(x):\n    return x * 2\n",
        encoding="utf-8",
    )
    cert_path = tmp_path / "cert.py"
    cert_path.write_text(
        textwrap.dedent(
            """
            import json
            import os

            from impl import double


            def test_saved_inputs_replay():
                with open(os.environ["AGENT_SAVED_INPUTS_PATH"], "r", encoding="utf-8") as f:
                    cases = [json.loads(line) for line in f if line.strip()]
                for case in cases:
                    assert double(case["x"]) == case["expected"]
            """
        ),
        encoding="utf-8",
    )
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(
        '{"x": 2, "expected": 5}\n',
        encoding="utf-8",
    )

    driver = Path(__file__).resolve().parent.parent / "saved_inputs_driver.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(testbed) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(driver), str(cert_path), str(inputs_path), str(testbed)],
        cwd=testbed,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 1
    assert "__SAVED_INPUTS_DRIVER_RESULT__" in proc.stdout
    assert (testbed / ".coverage").exists()
