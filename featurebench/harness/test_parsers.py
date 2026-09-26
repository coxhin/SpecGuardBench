"""Test log parsers for FeatureBench evaluation harness."""

import re

from featurebench.harness.constants import TestStatus


def parse_log_pytest(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework."""
    test_status_map = {}

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1
            break

    if summary_start_idx == -1:
        return test_status_map

    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        if line.startswith('==='):
            break

        if not line:
            continue

        if line.startswith(TestStatus.PASSED.value + ' '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.PASSED.value

        elif line.startswith(TestStatus.FAILED.value + ' '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.FAILED.value

        elif line.startswith(TestStatus.ERROR.value + ' '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.ERROR.value

        elif line.startswith(TestStatus.XFAIL.value + ' '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.XFAIL.value

        elif line.startswith(TestStatus.SKIPPED.value + ' '):
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_path = match.group(1).strip()
                test_status_map[test_path] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_pytest_with_subfailed(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework, handling SUBFAILED
    duplicates.
    """
    test_status_map = {}
    subfailed_counts = {}

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1
            break

    if summary_start_idx == -1:
        return test_status_map

    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        if line.startswith('==='):
            break

        if not line:
            continue

        test_name = None
        status = None
        is_subfailed = False

        if line.startswith(TestStatus.PASSED.value + ' '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.PASSED.value

        elif line.startswith(TestStatus.FAILED.value + ' '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value

        elif line.startswith('SUBFAILED'):
            match = re.match(r'SUBFAILED(?:\(<subtest>\))?\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value
                is_subfailed = True

        elif line.startswith(TestStatus.ERROR.value + ' '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.ERROR.value

        elif line.startswith(TestStatus.XFAIL.value + ' '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.XFAIL.value

        elif line.startswith(TestStatus.SKIPPED.value + ' '):
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.SKIPPED.value

        if test_name and status:
            if is_subfailed:
                if test_name in subfailed_counts:
                    subfailed_counts[test_name] += 1
                else:
                    subfailed_counts[test_name] = 1
                unique_test_name = f"{test_name}[{subfailed_counts[test_name]}]"
            else:
                unique_test_name = test_name

            test_status_map[unique_test_name] = status

    return test_status_map


def parse_log_pytest_verbose(log: str) -> dict[str, str]:
    """Parser for verbose test logs generated with PyTest framework."""
    test_status_map = {}

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    statuses = [
        TestStatus.PASSED.value,
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
        TestStatus.XFAIL.value,
        TestStatus.SKIPPED.value
    ]

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith('===') and  ('test session starts' not in line_stripped):
            break

        if not line_stripped:
            continue

        for status in statuses:
            if f' {status} ' in line_stripped or f' {status}\t' in line_stripped:
                parts = line_stripped.split(status, 1)
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    if '::' in test_name:
                        test_status_map[test_name] = status
                        break

    return test_status_map


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_pytest,
    "fastapi/fastapi": parse_log_pytest,
    "huggingface/accelerate": parse_log_pytest,
    "huggingface/transformers": parse_log_pytest,
    "huggingface/trl": parse_log_pytest,
    "Lightning-AI/pytorch-lightning": parse_log_pytest,
    "linkedin/Liger-Kernel": parse_log_pytest,
    "matplotlib/matplotlib": parse_log_pytest,
    "mesonbuild/meson": parse_log_pytest_with_subfailed,
    "mlflow/mlflow": parse_log_pytest,
    "mwaskom/seaborn": parse_log_pytest,
    "Netflix/metaflow": parse_log_pytest,
    "optuna/optuna": parse_log_pytest,
    "pandas-dev/pandas": parse_log_pytest,
    "pydantic/pydantic": parse_log_pytest_verbose,
    "pydata/xarray": parse_log_pytest,
    "pypa/hatch": parse_log_pytest,
    "pypa/packaging": parse_log_pytest,
    "pypa/setuptools": parse_log_pytest,
    "pytest-dev/pytest": parse_log_pytest,
    "python/mypy": parse_log_pytest,
    "scikit-learn/scikit-learn": parse_log_pytest,
    "sphinx-doc/sphinx": parse_log_pytest,
    "sympy/sympy": parse_log_pytest,
}

MAP_REPO_TO_TEST_CMD = {
    "pydantic/pydantic": "pytest -rA -v --color=no",
}
