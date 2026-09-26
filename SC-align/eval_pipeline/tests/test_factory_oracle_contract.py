from __future__ import annotations

from eval_pipeline.stages.stage3_proof_execution import _factory_oracle_code_error

CHECK_CASE = "def check_case(case):\n    assert target(*case['args'], **case['kwargs'])\n"


def _cert(build_cases_body: str) -> str:
    return f"def build_cases():\n{build_cases_body}\n\n{CHECK_CASE}"


def test_forbidden_key_inside_kwargs_is_allowed() -> None:
    code = _cert(
        "    yield {\n"
        "        'args': (receiver,),\n"
        "        'kwargs': {'exc': ValueError, 'expected': 'an exception type'},\n"
        "    }"
    )
    assert _factory_oracle_code_error(code) is None


def test_forbidden_key_inside_args_is_allowed() -> None:
    code = _cert("    yield {'args': ({'expected': 3},), 'kwargs': {}}")
    assert _factory_oracle_code_error(code) is None


def test_forbidden_key_nested_deep_in_payload_is_allowed() -> None:
    code = _cert(
        "    yield {'args': (), 'kwargs': {'cfg': {'inner': {'should_raise': True}}}}"
    )
    assert _factory_oracle_code_error(code) is None


def test_forbidden_key_at_case_level_is_rejected() -> None:
    code = _cert("    yield {'args': (2,), 'kwargs': {}, 'expected': 4}")
    error = _factory_oracle_code_error(code)
    assert error is not None and "'expected'" in error


def test_forbidden_key_in_starred_case_dict_is_rejected() -> None:
    code = _cert(
        "    base = {'args': (2,), 'kwargs': {}}\n"
        "    yield {**base, 'expected': 4}"
    )
    error = _factory_oracle_code_error(code)
    assert error is not None and "'expected'" in error


def test_every_forbidden_key_is_rejected_at_case_level() -> None:
    for key in ("expected", "expected_result", "expected_calls", "should_raise"):
        code = _cert(f"    yield {{'args': (), 'kwargs': {{}}, {key!r}: 1}}")
        error = _factory_oracle_code_error(code)
        assert error is not None and repr(key) in error, key


def test_assert_in_build_cases_is_rejected() -> None:
    code = _cert("    assert True\n    yield {'args': (), 'kwargs': {}}")
    assert _factory_oracle_code_error(code) == (
        "build_cases() must not contain assertions"
    )


def test_missing_required_function_is_rejected() -> None:
    code = "def build_cases():\n    yield {'args': (), 'kwargs': {}}\n"
    assert _factory_oracle_code_error(code) == (
        "factory_oracle certificate must define check_case()"
    )


def test_plain_certificate_passes() -> None:
    code = _cert("    yield {'args': (2,), 'kwargs': {'strict': True}}")
    assert _factory_oracle_code_error(code) is None


def test_syntax_error_is_not_a_contract_error() -> None:
    assert _factory_oracle_code_error("def build_cases(:\n") is None
