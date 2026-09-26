"""Constants for FeatureBench evaluation harness."""

from enum import Enum
from pathlib import Path

FB_BENCH_DIR = Path(__file__).parent.parent

DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"

LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
LOG_REPORT = "report.json"
LOG_PATCH = "patch.diff"

KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"
KEY_N_ATTEMPT = "n_attempt"

APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"

UTF8 = "utf-8"

DEFAULT_PYTEST_CMD = "pytest -rA -p no:cacheprovider --color=no"


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class EvalType(Enum):
    PASS_AND_FAIL = "pass_and_fail"
    FAIL_ONLY = "fail_only"


FAIL_ONLY_REPOS: set[str] = {
}
