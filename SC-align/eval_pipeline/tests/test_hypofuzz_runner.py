from eval_pipeline.models import (
    KIND_ASSERTION,
    KIND_COLLECTION_ERROR,
    KIND_NO_TESTS,
    KIND_PASSED,
    classify_hypofuzz_output,
    classify_pytest_output,
)


def test_stage2_passed_maps_to_passed():
    output = "1 passed in 0.56s"
    assert classify_hypofuzz_output(0, output) == KIND_PASSED
    assert classify_hypofuzz_output(0, output) == classify_pytest_output(0, output)


def test_stage2_falsifying_example_maps_to_assertion():
    output = (
        "Falsifying example: test_x(\n    x=1,  # or any other generated value\n)\n"
        "FAILED cert.py::test_x - assert 1 < 0\n"
    )
    assert classify_hypofuzz_output(1, output) == KIND_ASSERTION


def test_stage2_no_tests_collected_maps_to_no_tests():
    output = "no tests ran in 0.05s"
    assert classify_hypofuzz_output(5, output) == KIND_NO_TESTS


def test_stage2_collection_error_maps_to_collection_error():
    output = (
        "ERROR collecting cert.py\n"
        "ImportError while importing test module 'cert.py'\n"
    )
    assert classify_hypofuzz_output(2, output) == KIND_COLLECTION_ERROR


def test_classifier_alias_full_passthrough():
    cases = [
        (0, ""),
        (0, "1 passed"),
        (1, "AssertionError"),
        (1, "Falsifying example: ..."),
        (1, "DID NOT RAISE"),
        (2, "ERROR collecting"),
        (3, "INTERNALERROR"),
        (4, "usage error"),
        (5, "no tests collected"),
    ]
    for exit_code, output in cases:
        assert classify_hypofuzz_output(exit_code, output) == classify_pytest_output(
            exit_code, output
        ), f"mismatch for exit={exit_code}, output={output!r}"
