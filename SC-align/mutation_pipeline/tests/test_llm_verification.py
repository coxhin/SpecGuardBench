from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import jinja2
import pytest

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mutation_pipeline.models import (
    AtomicSpec,
    PipelineInput,
    Proof,
    SpecCategory,
    SpecSource,
)
from mutation_pipeline.steps.step6_llm_verification import (
    LLMVerifier,
    _extract_json,
    _parse_verdict,
)


def _make_spec(sid: str = "S1", text: str = "example spec") -> AtomicSpec:
    return AtomicSpec(
        spec_id=sid,
        text=text,
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


def _make_proof() -> Proof:
    return Proof(
        witness_script="def test_x(): print('ok')",
        actual_gold_output="A",
        actual_mutant_output="B",
        divergence_type="value",
    )


def _prompt_env() -> jinja2.Environment:
    prompts = _SC_ALIGN / "mutation_pipeline" / "prompts"
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(prompts)),
        undefined=jinja2.StrictUndefined,
    )


def test_parser_accepts_valid_approve() -> None:
    out = _parse_verdict(
        '```json\n{"verdict": "approve", "category": "valid", '
        '"reasoning": "div follows from spec text"}\n```'
    )
    assert out["verdict"] == "approve"
    assert out["category"] == "valid"
    assert out["reasoning"] == "div follows from spec text"


def test_parser_accepts_reject_overreach() -> None:
    out = _parse_verdict(
        '```json\n{"verdict": "reject", "category": "overreach", '
        '"reasoning": "spec silent on order"}\n```'
    )
    assert out["verdict"] == "reject"
    assert out["category"] == "overreach"


def test_parser_rejects_bad_verdict() -> None:
    with pytest.raises(ValueError, match="verdict"):
        _parse_verdict('{"verdict": "maybe", "category": "valid", "reasoning": "x"}')


def test_parser_rejects_bad_category() -> None:
    with pytest.raises(ValueError, match="category"):
        _parse_verdict('{"verdict": "reject", "category": "bogus", "reasoning": "x"}')


def test_parser_rejects_approve_with_non_valid_category() -> None:
    with pytest.raises(ValueError, match="approve.*requires.*valid"):
        _parse_verdict(
            '{"verdict": "approve", "category": "overreach", "reasoning": "x"}'
        )


def test_parser_prefers_last_json_block() -> None:
    response = (
        "Initial thought:\n"
        '```json\n'
        '{"verdict": "approve", "category": "valid", "reasoning": "first"}\n'
        '```\n\n'
        "Wait, on reread the spec is silent on this:\n"
        '```json\n'
        '{"verdict": "reject", "category": "overreach", "reasoning": "final"}\n'
        '```'
    )
    out = _parse_verdict(response)
    assert out["verdict"] == "reject"
    assert out["reasoning"] == "final"


def test_extract_json_skips_non_json_fence() -> None:
    response = (
        "```python\n"
        "def example(): pass\n"
        "```\n"
        "```json\n"
        '{"verdict": "reject", "category": "noise", "reasoning": "x"}\n'
        "```"
    )
    out = _parse_verdict(response)
    assert out["verdict"] == "reject"
    assert out["category"] == "noise"


def test_extract_json_falls_back_to_first_brace_block() -> None:
    response = 'no fence here {"verdict": "approve", "category": "valid", "reasoning": "x"} trailing'
    raw = _extract_json(response)
    assert json.loads(raw)["verdict"] == "approve"


def test_pipeline_refuses_to_start_without_llm_b(tmp_path) -> None:
    from mutation_pipeline.config import MutationPipelineConfig
    from mutation_pipeline.pipeline import MutationPipeline

    run_dir = tmp_path / "run"
    config = MutationPipelineConfig(
        llm_a_name="some-llm", llm_b_name=None, output_dir=run_dir,
    )
    with pytest.raises(ValueError, match="llm_b_name is required"):
        MutationPipeline(config, config_loader=None)
    assert not run_dir.exists()


class _MockLLMCaller:
    def __init__(self, response_text: str):
        self._response = response_text
        self.calls = 0

    def call_llm(self, messages):
        self.calls += 1
        return {"_response": self._response}

    def get_llm_response_text(self, completion):
        return completion["_response"]


class _ScriptedLLMCaller:
    def __init__(self, *verdicts):
        self._script = list(verdicts)
        self.calls = 0

    @staticmethod
    def _body(verdict):
        if verdict == "approve":
            return ('{"verdict": "approve", "category": "valid", '
                    '"reasoning": "ok"}')
        return (f'{{"verdict": "reject", "category": "{verdict}", '
                f'"reasoning": "not ok"}}')

    def call_llm(self, messages):
        verdict = self._script[self.calls]
        self.calls += 1
        if verdict is None:
            raise RuntimeError("network timeout")
        return {"_response": self._body(verdict)}

    def get_llm_response_text(self, completion):
        return completion["_response"]


def _target(llm, votes=3):
    return LLMVerifier(
        llm_caller=llm, prompt_env=_prompt_env(), votes=votes,
    ).verify_target(
        spec=_make_spec(),
        mutant_patch="diff",
        proof=_make_proof(),
        pipeline_input=_make_pipeline_input(),
    )


def test_majority_approve_outvotes_one_reject() -> None:
    llm = _ScriptedLLMCaller("approve", "overreach", "approve")
    result = _target(llm)
    assert result.success is True
    assert llm.calls == 3
    assert result.data["vote_tally"] == {
        "votes_requested": 3, "approve": 2, "reject": 1,
        "infra_error": 0, "unanimous": False,
    }


def test_majority_reject_outvotes_one_approve() -> None:
    llm = _ScriptedLLMCaller("overreach", "approve", "overreach")
    result = _target(llm)
    assert result.success is False
    assert result.data["category"] == "overreach"


def test_every_ballot_is_kept_for_agreement_analysis() -> None:
    llm = _ScriptedLLMCaller("approve", "overreach", "approve")
    result = _target(llm)
    assert [b["category"] for b in result.data["ballots"]] == [
        "valid", "overreach", "valid",
    ]
    assert [b["verdict"] for b in result.data["ballots"]] == [
        "approve", "reject", "approve",
    ]

    unanimous = _target(_ScriptedLLMCaller("approve", "approve", "approve"))
    assert len(unanimous.data["ballots"]) == 3
    assert unanimous.data["vote_tally"]["unanimous"] is True


def test_a_failed_ballot_is_an_absent_rater_not_a_reject_vote() -> None:
    llm = _ScriptedLLMCaller("approve", None, None)
    result = _target(llm)
    assert result.success is True
    assert result.data["vote_tally"]["approve"] == 1
    assert result.data["vote_tally"]["reject"] == 0
    assert result.data["vote_tally"]["infra_error"] == 2


def test_a_split_panel_fails_closed() -> None:
    llm = _ScriptedLLMCaller("approve", "overreach", None)
    result = _target(llm)
    assert result.success is False
    assert result.data["vote_tally"] == {
        "votes_requested": 3, "approve": 1, "reject": 1,
        "infra_error": 1, "unanimous": False,
    }


def test_all_ballots_failing_stays_no_evidence() -> None:
    result = _target(_ScriptedLLMCaller(None, None, None))
    assert result.success is False
    assert result.data["category"] == "infra_error"
    assert result.data["vote_tally"]["infra_error"] == 3


def test_one_vote_is_the_old_single_call_behaviour() -> None:
    llm = _ScriptedLLMCaller("approve")
    result = _target(llm, votes=1)
    assert result.success is True
    assert llm.calls == 1


def test_ballots_for_one_verdict_are_cast_serially() -> None:
    import threading

    in_flight = []
    peak = [0]
    lock = threading.Lock()

    class _CountingLLM(_ScriptedLLMCaller):
        def call_llm(self, messages):
            with lock:
                in_flight.append(1)
                peak[0] = max(peak[0], len(in_flight))
            time.sleep(0.02)
            try:
                return super().call_llm(messages)
            finally:
                with lock:
                    in_flight.pop()

    _target(_CountingLLM("approve", "approve", "approve"))
    assert peak[0] == 1


def test_verifier_approves_valid_response() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", '
        '"reasoning": "divergence follows from spec"}\n```'
    )
    v = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = v.verify_target(
        spec=_make_spec(),
        mutant_patch="diff",
        proof=_make_proof(),
        pipeline_input=_make_pipeline_input(),
        mutation_strategy="reverse the order",
    )
    assert result.success is True
    assert result.data["verdict"] == "approve"
    assert result.data["category"] == "valid"
    assert llm.calls == 3
    assert result.data["vote_tally"]["approve"] == 3


def test_verifier_rejects_when_llm_b_says_reject() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "reject", "category": "overreach", '
        '"reasoning": "spec is silent on ordering"}\n```'
    )
    v = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = v.verify_target(
        spec=_make_spec(),
        mutant_patch="diff",
        proof=_make_proof(),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.success is False
    assert result.data["verdict"] == "reject"
    assert result.data["category"] == "overreach"
    assert "overreach" in (result.error or "").lower()


def test_verifier_rejects_on_malformed_response() -> None:
    llm = _MockLLMCaller("not json at all, no fenced block either")
    v = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = v.verify_target(
        spec=_make_spec(),
        mutant_patch="diff",
        proof=_make_proof(),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.success is False
    assert result.data["category"] == "infra_error"


def test_verifier_rejects_on_llm_call_exception() -> None:
    class _ExplodingLLM:
        def call_llm(self, messages):
            raise RuntimeError("network timeout")
        def get_llm_response_text(self, completion):
            raise AssertionError("should not be reached")

    v = LLMVerifier(llm_caller=_ExplodingLLM(), prompt_env=_prompt_env())
    result = v.verify_target(
        spec=_make_spec(),
        mutant_patch="diff",
        proof=_make_proof(),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.success is False
    assert result.data["category"] == "infra_error"
    assert "network timeout" in (result.error or "")


def test_both_templates_render_with_required_context() -> None:
    env = _prompt_env()
    shared = dict(
        spec=_make_spec("S5", "the function returns an empty dict"),
        mutation_strategy="return defaultdict(list) instead of {}",
        mutant_patch="--- a.py\n+++ b.py\n",
        witness_script="def test_x(): print(1)",
        actual_gold_output="True",
        actual_mutant_output="False",
    )
    target = env.get_template("llm_verification_target.j2").render(
        other_specs=[], **shared
    )
    preservation = env.get_template("llm_verification_preservation.j2").render(
        other_specs=[_make_spec("S6", "must accept any arg")],
        probe=None,
        **shared,
    )
    for rendered in (target, preservation):
        assert "the function returns an empty dict" in rendered
        assert "return defaultdict(list) instead of {}" in rendered
    assert "S6" in preservation
    assert "S6" not in target


def test_parser_accepts_non_behavioral() -> None:
    out = _parse_verdict(
        '{"verdict": "reject", "category": "non_behavioral", '
        '"reasoning": "the observations differ only in the class docstring"}'
    )
    assert out["verdict"] == "reject"
    assert out["category"] == "non_behavioral"


def test_non_behavioral_drops_the_spec_without_retrying() -> None:
    from types import SimpleNamespace

    from mutation_pipeline.models import StepResult
    from mutation_pipeline.pipeline import MutationPipeline, SpecInfeasible

    pipe = MutationPipeline.__new__(MutationPipeline)
    pipe.config = SimpleNamespace(step2_max_attempts=3)
    pipe.output_dir = Path(tempfile.mkdtemp())
    attempts: list = []

    def fake_step2(*args, **kwargs):
        return StepResult(
            success=True, error=None, data={}, step_name="mutation_generation",
        ), 0.0

    def fake_verify_attempt(*args, **kwargs):
        attempts.append(1)
        raise SpecInfeasible("prose-only divergence")

    pipe._run_step2_single = fake_step2
    pipe._verify_attempt = fake_verify_attempt
    result, counts = pipe._verify_spec(
        _make_spec(), _make_pipeline_input(), [],
    )
    assert result is None
    assert len(attempts) == 1
    assert counts["specs_infeasible"] == 1
