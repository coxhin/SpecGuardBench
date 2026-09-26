import sys
from pathlib import Path

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_ROOT = _SC_ALIGN.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_pipeline.models import BaselineResult
from eval_pipeline.stages.stage5_metrics import _aggregate_agent_steps


def _br(instance_id="inst1", cost=None, attempts=None) -> BaselineResult:
    return BaselineResult(
        instance_id=instance_id, baseline="codex", model="gpt-5.6-sol",
        tool_version="0.117.0", raw_output="", certificates={},
        cost=cost if cost is not None else {},
        wall_time_seconds=0.0,
        attempts=attempts or [],
    )


def test_successful_llm_calls_is_summarised_across_runs():
    brs = [
        _br(cost={"successful_llm_calls": 11, "api_requests": 11}),
        _br(cost={"successful_llm_calls": 25, "api_requests": 25}),
    ]

    block = _aggregate_agent_steps(brs)

    assert block["successful_llm_calls"]["min"] == 11
    assert block["successful_llm_calls"]["max"] == 25
    assert block["counts"]["successful_llm_calls_total"] == 36
    assert block["counts"]["runs_with_call_count"] == 2


def test_gap_between_requests_and_completions_is_surfaced():
    brs = [_br(cost={"successful_llm_calls": 8, "api_requests": 10})]

    counts = _aggregate_agent_steps(brs)["counts"]

    assert counts["requests_without_completion"] == 2


def test_a_clean_run_reports_a_zero_gap():
    brs = [_br(cost={"successful_llm_calls": 18, "api_requests": 18})]

    assert _aggregate_agent_steps(brs)["counts"]["requests_without_completion"] == 0


def test_steps_are_summed_over_the_attempt_chain():
    brs = [_br(attempts=[
        {"attempt": 1, "steps": {"tool_calls": 4, "agent_messages": 2}},
        {"attempt": 2, "steps": {"tool_calls": 6, "agent_messages": 3}},
    ])]

    block = _aggregate_agent_steps(brs)

    assert block["steps"]["tool_calls"]["median"] == 10
    assert block["steps"]["agent_messages"]["median"] == 5
    assert block["attempts"]["median"] == 2


def test_adapter_specific_step_keys_are_kept_apart():
    brs = [
        _br(attempts=[{"attempt": 1, "steps": {"tool_calls": 5, "reasoning_items": 9}}]),
        _br(attempts=[{"attempt": 1, "steps": {"tool_calls": 7}}]),
    ]

    steps = _aggregate_agent_steps(brs)["steps"]

    assert steps["tool_calls"]["n"] == 2
    assert steps["reasoning_items"]["n"] == 1


def test_runs_without_a_call_count_do_not_fabricate_one():
    brs = [_br(cost={}), _br(cost={"successful_llm_calls": 5, "api_requests": 5})]

    block = _aggregate_agent_steps(brs)

    assert block["successful_llm_calls"]["n"] == 1
    assert block["counts"]["runs"] == 2
    assert block["counts"]["runs_with_call_count"] == 1


def test_runs_stopped_by_the_step_cap_are_counted():
    brs = [
        _br(attempts=[{"attempt": 1, "steps": {}, "step_cap": None}]),
        _br(attempts=[{"attempt": 1, "steps": {},
                       "step_cap": {"unit": "codex", "limit": 50, "count": 50}}]),
        _br(attempts=[
            {"attempt": 1, "steps": {},
             "step_cap": {"unit": "codex", "limit": 50, "count": 50}},
            {"attempt": 2, "steps": {}, "step_cap": None},
        ]),
    ]

    counts = _aggregate_agent_steps(brs)["counts"]

    assert counts["runs_stopped_by_step_cap"] == 2
    assert counts["runs"] == 3
