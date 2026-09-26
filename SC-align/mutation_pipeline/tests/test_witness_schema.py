from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mutation_pipeline.models import (
    CALL_CASE_CONTRACT,
    OBSERVATION_LINE_PREFIX,
    AtomicSpec,
    FilterSurvival,
    MutationResult,
    PipelineInput,
    Proof,
    SpecCategory,
    SpecSource,
)
from mutation_pipeline.steps.step2_mutation_generation import (
    MutationGenerator,
    WitnessUnobservable,
    _validate_factory_module,
)
from mutation_pipeline.steps.step4_witness_execution import (
    WitnessExecutor,
    _MARKER_BEGIN,
    _MARKER_END,
    _extract_stray_output,
    _extract_observation,
    _wrap_with_markers,
)
from mutation_pipeline.docker_runner import _filter_diff_to_files


GOOD_WITNESS = """
def test_spec_violation():
    pass
"""

GOOD_FACTORY = (
    "def build_cases():\n"
    "    yield {'args': (-3,), 'kwargs': {}}\n"
)
GOOD_OBSERVER = (
    "target = abs\n"
    "def observe_case(case):\n"
    "    return {'value': target(*case['args'], **case['kwargs'])}\n"
)


def _payload(**overrides) -> dict:
    payload = {
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "mutation_strategy": "swap",
        "witness_factory": GOOD_FACTORY,
        "witness_observer": GOOD_OBSERVER,
        "trigger_condition": "any negative input",
    }
    payload.update(overrides)
    return payload


def _make_spec() -> AtomicSpec:
    return AtomicSpec(
        spec_id="S1",
        text="example spec",
        category=SpecCategory.BEHAVIORAL,
        sources=[SpecSource.DOCSTRING],
        code_mapping=[],
    )


def _make_pipeline_input() -> PipelineInput:
    return PipelineInput(
        instance_id="example.instance.lv1",
        level=1,
        task_statement="t",
        interface_descriptions=[],
        gold_patch="gold",
        image_name="img",
    )


def test_proof_roundtrip_v3() -> None:
    proof = Proof(
        witness_script=GOOD_WITNESS,
        actual_gold_output="42\n",
        actual_mutant_output="43\n",
        divergence_type="value",
    )
    payload = proof.to_dict()
    assert set(payload) == {
        "witness_script",
        "actual_gold_output",
        "actual_mutant_output",
        "divergence_type",
    }
    for legacy in (
        "in_function_assertions",
        "harness_assertions",
        "hypothesis_harness",
        "failing_input",
        "triggered_assertion",
        "assertion_output",
        "failure_type",
        "exception_info",
    ):
        assert legacy not in payload

    restored = Proof.from_dict(payload)
    assert restored == proof


def test_factory_protocol_proof_roundtrip() -> None:
    proof = Proof(
        witness_script="def test_factory_witness(): pass",
        actual_gold_output="1",
        actual_mutant_output="2",
        divergence_type="value",
        case_contract=CALL_CASE_CONTRACT,
        witness_factory=(
            "def build_cases():\n"
            "    yield {'args': (1,), 'kwargs': {}}\n"
        ),
        witness_observer=(
            "def observe_case(case):\n"
            "    return case['args'][0]\n"
        ),
    )
    payload = proof.to_dict()
    assert payload["case_contract"] == CALL_CASE_CONTRACT
    assert Proof.from_dict(payload) == proof


def test_filter_survival_v3_fields() -> None:
    fs = FilterSurvival(tests_passed=True, witness_diverged=True, verifier_approved=True)
    payload = fs.to_dict()
    assert set(payload) == {"tests_passed", "witness_diverged", "verifier_approved"}
    assert "sanity_check_passed" not in payload
    assert "proof_triggered" not in payload
    assert FilterSurvival.from_dict(payload) == fs


def test_mutation_result_roundtrip_with_witness_proof() -> None:
    proof = Proof(
        witness_script=GOOD_WITNESS,
        actual_gold_output="84",
        actual_mutant_output="42",
        divergence_type="value",
    )
    result = MutationResult(
        schema_version=3,
        instance_id="example.instance.lv1",
        spec=_make_spec(),
        mask_patch="mask.patch",
        gold_patch="gold.patch",
        mutant_patch="mutant.patch",
        test_patch="test.patch",
        proof=proof,
        metadata={"mutations": [{"file": "x.py"}]},
    )
    payload = json.loads(result.to_jsonl())
    assert payload["schema_version"] == 3
    assert "test_spec_violation" in payload["proof"]["witness_script"]

    restored = MutationResult.from_dict(payload)
    assert restored.proof == proof
    assert restored.schema_version == 3


@pytest.mark.parametrize("source, match", [
    (
        "import inspect\n"
        "def build_cases():\n"
        "    yield {'args': (inspect.getsource(int),), 'kwargs': {}}\n",
        "getsource",
    ),
    (
        "import linecache\n"
        "def build_cases():\n"
        "    yield {'args': (1,), 'kwargs': {}}\n",
        "linecache",
    ),
    (
        "def build_cases():\n"
        "    yield {'args': (open('/testbed/foo.py').read(),), 'kwargs': {}}\n",
        r"\.py source file",
    ),
])
def test_factory_module_rejects_source_introspection(source, match) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_factory_module(source, "build_cases")


def test_factory_module_rejects_unparseable_python() -> None:
    with pytest.raises(ValueError, match="not valid Python"):
        _validate_factory_module("def build_cases(:\n    pass", "build_cases")


def test_factory_module_requires_its_entry_point() -> None:
    with pytest.raises(ValueError, match="must define build_cases"):
        _validate_factory_module("def other():\n    pass\n", "build_cases")


def _parser() -> MutationGenerator:
    return MutationGenerator(llm_caller=MagicMock(), prompt_env=MagicMock())


def test_step2_parser_accepts_full_witness_payload() -> None:
    raw = json.dumps(_payload(divergence_type="value"))
    out = _parser()._parse_mutation_response(raw)
    assert "test_factory_witness" in out["witness_script"]
    assert out["divergence_type"] == "value"
    assert out["mutations"][0]["file"] == "f.py"


def test_step2_parser_rejects_a_witness_script_without_the_protocol() -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "witness_script": "def test_x():\n    print(1)\n",
    })
    with pytest.raises(WitnessUnobservable):
        _parser()._parse_mutation_response(raw)


def test_step2_parser_accepts_factory_protocol() -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "mutation_strategy": "guard one valid input",
        "case_contract": "LLM-authored contracts are ignored.",
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'args': (60,), 'kwargs': {}}\n"
        ),
        "witness_observer": (
            "target = abs\n"
            "def observe_case(case):\n"
            "    return target(*case['args'], **case['kwargs'])\n"
        ),
        "witness_script": "",
        "divergence_type": "value",
        "trigger_condition": "any input whose absolute value is requested",
    })
    out = _parser()._parse_mutation_response(raw)
    assert out["case_contract"] == CALL_CASE_CONTRACT
    assert out["trigger_condition"].startswith("any input")
    assert "test_factory_witness" in out["witness_script"]
    assert "_validate_call_case" in out["witness_script"]
    assert "build_cases" in out["witness_factory"]

    namespace = {}
    exec(out["witness_script"], namespace)
    namespace["test_factory_witness"]()


def test_factory_witness_emits_prefixed_observation_lines(capsys) -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'args': (-3,), 'kwargs': {}}\n"
        ),
        "witness_observer": (
            "target = abs\n"
            "def observe_case(case):\n"
            "    return {'value': target(*case['args'], **case['kwargs'])}\n"
        ),
        "witness_script": "",
        "trigger_condition": "any negative input",
    })
    out = _parser()._parse_mutation_response(raw)
    namespace = {}
    exec(out["witness_script"], namespace)
    namespace["test_factory_witness"]()

    printed = capsys.readouterr().out.strip()
    assert printed == f'{OBSERVATION_LINE_PREFIX}{{"value": 3}}'


@pytest.mark.parametrize("module_key, source", [
    (
        "witness_factory",
        "def build_cases():\n"
        "    print('building')\n"
        "    yield {'args': (1,), 'kwargs': {}}\n",
    ),
    (
        "witness_observer",
        "target = abs\n"
        "def observe_case(case):\n"
        "    result = target(*case['args'], **case['kwargs'])\n"
        "    print(result)\n"
        "    return {'value': result}\n",
    ),
    (
        "witness_observer",
        "import sys\n"
        "target = abs\n"
        "def observe_case(case):\n"
        "    result = target(*case['args'], **case['kwargs'])\n"
        "    sys.stderr.write(str(result))\n"
        "    return {'value': result}\n",
    ),
])
def test_step2_parser_rejects_stray_writes(module_key, source) -> None:
    payload = {
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'args': (1,), 'kwargs': {}}\n"
        ),
        "witness_observer": (
            "target = abs\n"
            "def observe_case(case):\n"
            "    return {'value': target(*case['args'], **case['kwargs'])}\n"
        ),
        "witness_script": "",
        "trigger_condition": "any input",
    }
    payload[module_key] = source

    with pytest.raises(ValueError, match="must not (call print|write to)"):
        _parser()._parse_mutation_response(json.dumps(payload))


def test_step2_parser_allows_capturing_a_side_effect_and_returning_it() -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'args': (-3,), 'kwargs': {}}\n"
        ),
        "witness_observer": (
            "import warnings\n"
            "target = abs\n"
            "def observe_case(case):\n"
            "    with warnings.catch_warnings(record=True) as caught:\n"
            "        warnings.simplefilter('always')\n"
            "        value = target(*case['args'], **case['kwargs'])\n"
            "    return {\n"
            "        'value': value,\n"
            "        'warnings': sorted(\n"
            "            (w.category.__name__, str(w.message)) for w in caught\n"
            "        ),\n"
            "    }\n"
        ),
        "witness_script": "",
        "trigger_condition": "any negative input",
    })
    out = _parser()._parse_mutation_response(raw)
    namespace = {}
    exec(out["witness_script"], namespace)
    namespace["test_factory_witness"]()


def test_factory_witness_runtime_rejects_nonstandard_case_shape() -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'value': 60}\n"
        ),
        "witness_observer": (
            "target = abs\n"
            "def observe_case(case):\n"
            "    return target(*case.get('args', ()), **case.get('kwargs', {}))\n"
        ),
        "witness_script": "",
        "trigger_condition": "any input at all",
    })
    out = _parser()._parse_mutation_response(raw)
    namespace = {}
    exec(out["witness_script"], namespace)

    with pytest.raises(ValueError, match="exactly 'args' and 'kwargs'"):
        namespace["test_factory_witness"]()


def test_step2_parser_rejects_oracle_data_in_factory_case() -> None:
    raw = json.dumps({
        "mutations": [{"file": "f.py", "find": "a", "replace": "b"}],
        "case_contract": "case",
        "witness_factory": (
            "def build_cases():\n"
            "    yield {'value': 1, 'expected': 2}\n"
        ),
        "witness_observer": (
            "def observe_case(case):\n"
            "    return case['value']\n"
        ),
        "witness_script": "",
    })
    with pytest.raises(ValueError, match="forbidden oracle key"):
        _parser()._parse_mutation_response(raw)


def test_step2_parser_defaults_missing_divergence_type() -> None:
    out = _parser()._parse_mutation_response(json.dumps(_payload()))
    assert out["divergence_type"] == ""


def test_step2_parser_treats_empty_witness_as_unobservable() -> None:
    raw = json.dumps(_payload(witness_factory="   \n  ", witness_observer=""))
    with pytest.raises(WitnessUnobservable, match="no observable witness"):
        _parser()._parse_mutation_response(raw)


def test_step2_parser_rejects_half_a_protocol() -> None:
    raw = json.dumps(_payload(witness_observer=""))
    with pytest.raises(ValueError, match="must both be"):
        _parser()._parse_mutation_response(raw)


def test_step2_parser_rejects_missing_mutations() -> None:
    payload = _payload()
    del payload["mutations"]
    raw = json.dumps(payload)
    with pytest.raises(ValueError, match="mutations"):
        _parser()._parse_mutation_response(raw)


def test_step2_parser_skips_python_fence_picks_json_fence() -> None:
    response = (
        "Looking at the spec...\n\n"
        "```python\n"
        "def gold_impl():\n"
        "    return list(model.__fields__.values())\n"
        "```\n\n"
        "Now my answer:\n\n"
        "```json\n"
        + json.dumps(_payload(divergence_type="value"))
        + "\n```\n"
    )
    out = _parser()._parse_mutation_response(response)
    assert out["divergence_type"] == "value"
    assert "test_factory_witness" in out["witness_script"]


def test_step2_parser_handles_untagged_fence() -> None:
    response = (
        "Here's the answer:\n\n"
        "```\n"
        + json.dumps(_payload())
        + "\n```\n"
    )
    out = _parser()._parse_mutation_response(response)
    assert "test_factory_witness" in out["witness_script"]


def test_step2_parser_prefers_last_json_block_when_llm_revises() -> None:
    draft = json.dumps(_payload(
        mutations=[{"file": "f.py", "find": "a", "replace": "wrong"}],
        mutation_strategy="draft",
        divergence_type="value",
    ))
    final = json.dumps(_payload(
        mutation_strategy="guarded variant — only fires on non-default args",
        divergence_type="value",
    ))
    response = (
        "Let me draft a mutation:\n\n"
        "```json\n" + draft + "\n```\n\n"
        "Wait, that breaks F2P. Let me try a guarded variant:\n\n"
        "```json\n" + final + "\n```\n"
    )
    out = _parser()._parse_mutation_response(response)
    assert out["mutation_strategy"].startswith("guarded variant")
    assert out["mutations"][0]["replace"] == "b"
    assert "test_factory_witness" in out["witness_script"]


def test_wrap_with_markers_inserts_begin_end_around_test_body() -> None:
    src = """
from hypothesis import given, example, settings, Phase, HealthCheck, strategies as st
@settings(phases=[Phase.explicit], deadline=None, database=None, derandomize=True, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(x=st.integers())
@example(x=42)
def test_x(x):
    print(repr(x))
"""
    wrapped = _wrap_with_markers(src)
    assert _MARKER_BEGIN in wrapped
    assert _MARKER_END in wrapped
    assert "print(repr(x))" in wrapped
    assert "finally:" in wrapped


def test_wrap_with_markers_does_not_touch_helper_functions() -> None:
    src = """
def helper():
    print("not wrapped")
def test_x():
    print("wrapped")
"""
    wrapped = _wrap_with_markers(src)
    assert wrapped.count(_MARKER_BEGIN) == 1


def test_wrap_with_markers_appends_a_driver_that_runs_the_tests() -> None:
    src = (
        "def test_one():\n"
        f"    print({OBSERVATION_LINE_PREFIX!r} + 'first')\n"
        "def test_two():\n"
        f"    print({OBSERVATION_LINE_PREFIX!r} + 'second')\n"
    )
    wrapped = _wrap_with_markers(src)
    assert "__main__" in wrapped

    completed = subprocess.run(
        [sys.executable, "-c", wrapped], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert _extract_observation(completed.stdout) == "first\n----\nsecond"


def test_wrap_with_markers_emits_no_driver_without_a_test() -> None:
    wrapped = _wrap_with_markers("def helper():\n    pass\n")
    assert "__main__" not in wrapped


def test_extract_observation_returns_the_prefixed_lines() -> None:
    pytest_output = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.11\n"
        "collected 1 item\n\n"
        f"test.py::test_x {_MARKER_BEGIN}\n"
        f'{OBSERVATION_LINE_PREFIX}{{"value": 42}}\n'
        f"{_MARKER_END}\n"
        "PASSED                                                                   [100%]\n\n"
        "============================== 1 passed in 0.05s ==============================="
    )
    assert _extract_observation(pytest_output) == '{"value": 42}'


def test_extract_observation_empty_when_no_markers() -> None:
    assert _extract_observation("============================= no markers here ===") == ""


def test_extract_observation_drops_stray_lines_inside_the_fence() -> None:
    pytest_output = (
        f"{_MARKER_BEGIN}\n"
        "[0/1] Regenerating build files.\n"
        "core.py:88: RuntimeWarning: invalid value encountered\n"
        f'{OBSERVATION_LINE_PREFIX}{{"value": 42}}\n'
        f"{_MARKER_END}\n"
    )
    assert _extract_observation(pytest_output) == '{"value": 42}'


def test_extract_observation_ignores_stray_lines_that_happen_to_be_json() -> None:
    pytest_output = (
        f"{_MARKER_BEGIN}\n"
        "3\n"
        f'{OBSERVATION_LINE_PREFIX}{{"value": 42}}\n'
        f"{_MARKER_END}\n"
    )
    assert _extract_observation(pytest_output) == '{"value": 42}'


def test_extract_observation_empty_when_only_stray_output() -> None:
    pytest_output = f"{_MARKER_BEGIN}\nhello 42\n{_MARKER_END}\n"
    assert _extract_observation(pytest_output) == ""


def test_extract_observation_joins_multiple_test_outputs() -> None:
    pytest_output = (
        f"{_MARKER_BEGIN}\n"
        f"{OBSERVATION_LINE_PREFIX}first\n"
        f"{_MARKER_END}\n"
        "test.py::test_y "
        f"{_MARKER_BEGIN}\n"
        f"{OBSERVATION_LINE_PREFIX}second\n"
        f"{_MARKER_END}\n"
    )
    extracted = _extract_observation(pytest_output)
    assert "first" in extracted
    assert "second" in extracted


def test_extract_stray_output_keeps_what_the_observation_drops() -> None:
    pytest_output = (
        "collected 1 item\n"
        f"{_MARKER_BEGIN}\n"
        "[0/1] Regenerating build files.\n"
        f'{OBSERVATION_LINE_PREFIX}{{"value": 42}}\n'
        f"{_MARKER_END}\n"
        "1 passed in 0.05s\n"
    )
    stray = _extract_stray_output(pytest_output)
    assert stray == "[0/1] Regenerating build files."
    assert "collected 1 item" not in stray


class _FakeRunner:
    def __init__(self, mutant_result, gold_result):
        self._mutant_result = mutant_result
        self._gold_result = gold_result
        self.calls: list[str] = []
        self.env_seen: list[dict | None] = []
        self._idx = 0

    def run_witness_script(self, container, code, timeout, env=None):
        self.calls.append("run_witness_script")
        self.env_seen.append(env)
        idx = self._idx
        self._idx += 1
        return self._mutant_result if idx == 0 else self._gold_result

    def reset_testbed(self, container, pipeline_input):
        self.calls.append("reset_testbed")


def _markered(payload: str, stray: str = "") -> str:
    body = f"{stray}\n" if stray else ""
    return (
        f"{_MARKER_BEGIN}\n{body}{OBSERVATION_LINE_PREFIX}{payload}\n"
        f"{_MARKER_END}\nPASSED\n"
    )


def test_witness_executor_succeeds_when_observations_diverge() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered("mutant-obs")),
        gold_result=(False, _markered("gold-obs")),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is True
    assert result.data["mutant_output"] == "mutant-obs"
    assert result.data["gold_output"] == "gold-obs"
    assert runner.calls == [
        "run_witness_script", "reset_testbed", "run_witness_script",
    ]


def test_witness_executor_mutes_bind_preflight_on_mutant_only() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered("mutant-obs")),
        gold_result=(False, _markered("gold-obs")),
    )
    executor = WitnessExecutor(runner)

    executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert runner.env_seen == [
        {"WITNESS_ALLOW_BIND_MISMATCH": "1"},
        None,
    ]


def test_witness_executor_runs_gold_once() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered("mutant-obs")),
        gold_result=(False, _markered("gold-obs")),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is True
    assert runner.calls.count("run_witness_script") == 2
    assert "gold_observation_repeat" not in result.data


def test_witness_executor_fails_when_observations_match() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered("identical")),
        gold_result=(False, _markered("identical")),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is False
    assert "identical" in (result.error or "").lower()
    assert runner.calls == ["run_witness_script", "reset_testbed", "run_witness_script"]


def test_witness_executor_ignores_a_divergence_that_is_only_stray() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered(
            "identical", stray="[1/1] Generating write_version_file",
        )),
        gold_result=(False, _markered(
            "identical", stray="[0/1] Regenerating build files.\nCleaning... 0 files.",
        )),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is False
    assert result.data["gold_stray_output"] == (
        "[0/1] Regenerating build files.\nCleaning... 0 files."
    )
    assert result.data["mutant_stray_output"] == "[1/1] Generating write_version_file"


def test_witness_executor_fails_when_mutant_pytest_exits_nonzero() -> None:
    runner = _FakeRunner(
        mutant_result=(True, "Traceback..."),
        gold_result=(False, _markered("gold-obs")),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is False
    assert "mutant" in (result.error or "").lower()
    assert runner.calls == ["run_witness_script"]


def test_witness_executor_fails_when_gold_pytest_exits_nonzero() -> None:
    runner = _FakeRunner(
        mutant_result=(False, _markered("mutant-obs")),
        gold_result=(True, "Traceback..."),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is False
    assert "gold" in (result.error or "").lower()
    assert runner.calls == ["run_witness_script", "reset_testbed", "run_witness_script"]


def test_witness_executor_fails_when_markers_missing() -> None:
    runner = _FakeRunner(
        mutant_result=(False, "no markers here\n1 passed in 0.05s"),
        gold_result=(False, "no markers here\n1 passed in 0.05s"),
    )
    executor = WitnessExecutor(runner)

    result = executor.execute(
        _make_pipeline_input(),
        witness_script=GOOD_WITNESS,
        timeout=30,
        container=object(),
    )

    assert result.success is False
    assert "marker" in (result.error or "").lower()


def test_filter_diff_keeps_only_whitelisted_files() -> None:
    raw_diff = (
        "diff --git a/pydantic/_internal/x.py b/pydantic/_internal/x.py\n"
        "index aaa..bbb 100644\n"
        "--- a/pydantic/_internal/x.py\n"
        "+++ b/pydantic/_internal/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/tests/pydantic_core b/tests/pydantic_core\n"
        "deleted file mode 120000\n"
        "index 1234567..0000000\n"
        "--- a/tests/pydantic_core\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-../some/path\n"
        "\\ No newline at end of file\n"
    )
    allowed = {"pydantic/_internal/x.py"}
    out = _filter_diff_to_files(raw_diff, allowed)
    assert "pydantic/_internal/x.py" in out
    assert "tests/pydantic_core" not in out


def test_filter_diff_returns_empty_when_no_overlap() -> None:
    raw_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    out = _filter_diff_to_files(raw_diff, allowed_files={"bar.py"})
    assert out.strip() == ""


def test_filter_diff_keeps_multiple_whitelisted_files() -> None:
    raw_diff = (
        "diff --git a/a.py b/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
        "diff --git a/b.py b/b.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-p\n"
        "+q\n"
        "diff --git a/c.py b/c.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-m\n"
        "+n\n"
    )
    out = _filter_diff_to_files(raw_diff, allowed_files={"a.py", "c.py"})
    assert "a.py" in out
    assert "c.py" in out
    assert "b.py" not in out
