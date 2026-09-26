import json
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple
from unittest import mock

import pytest

_mock_docker = ModuleType("docker")
_mock_docker_models = ModuleType("docker.models")
_mock_docker_containers = ModuleType("docker.models.containers")
_mock_docker_containers.Container = type("Container", (), {})
sys.modules.setdefault("docker", _mock_docker)
sys.modules.setdefault("docker.models", _mock_docker_models)
sys.modules.setdefault("docker.models.containers", _mock_docker_containers)

_mock_featurebench = ModuleType("featurebench")
_mock_harness = ModuleType("featurebench.harness")
_mock_harness_container = ModuleType("featurebench.harness.container")
_mock_harness_container.copy_to_container = mock.MagicMock()
_mock_harness_container.exec_run_with_timeout = mock.MagicMock(return_value=(0, b""))
_mock_harness_constants = ModuleType("featurebench.harness.constants")
_mock_harness_constants.UTF8 = "utf-8"
sys.modules.setdefault("featurebench", _mock_featurebench)
sys.modules.setdefault("featurebench.harness", _mock_harness)
sys.modules.setdefault("featurebench.harness.container", _mock_harness_container)
sys.modules.setdefault("featurebench.harness.constants", _mock_harness_constants)

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.adapters.base import EvalBaselineAdapter
from eval_pipeline.adapters.claude_code import ClaudeCodeAdapter
from eval_pipeline.adapters.codex import CodexAdapter
from eval_pipeline.models import BaselineResult, EvalTask
from eval_pipeline.trajectory import (
    chain_events,
    joined_trajectory,
    session_chains,
    total_steps,
)


def _claude_stream(
    session: str, turns: int, out_tokens: int, tools: int = 1,
    aborted: bool = False,
) -> str:
    lines = [json.dumps({"type": "system", "subtype": "init", "session_id": session})]
    for i in range(tools):
        lines.append(json.dumps({
            "type": "assistant", "session_id": session,
            "message": {"id": f"msg_{i}", "content": [
                {"type": "thinking", "thinking": "...", "signature": "sig"}]},
        }))
        lines.append(json.dumps({
            "type": "assistant", "session_id": session,
            "message": {"id": f"msg_{i}", "content": [
                {"type": "tool_use", "name": "Bash", "input": {}}]},
        }))
        lines.append(json.dumps({
            "type": "user", "session_id": session,
            "message": {"content": [{"type": "tool_result", "content": "ok"}]},
        }))
    if aborted:
        lines.append(json.dumps({
            "type": "assistant", "session_id": session,
            "message": {"id": "msg_cut", "content": [
                {"type": "thinking", "thinking": "cut off mid-"}]},
        }))
    lines.append(json.dumps({
        "type": "result", "session_id": session, "num_turns": turns,
        "result": "" if aborted else "done",
        "modelUsage": {"opus": {
            "inputTokens": 10, "outputTokens": out_tokens,
            "cacheReadInputTokens": 1000, "cacheCreationInputTokens": 100}},
    }))
    return "\n".join(lines)


def _codex_stream(input_tokens: int, cached: int, commands: int = 2) -> str:
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"}),
             json.dumps({"type": "turn.started"})]
    for i in range(commands):
        lines.append(json.dumps({
            "type": "item.completed",
            "item": {"id": f"r{i}", "type": "reasoning", "text": "..."}}))
        lines.append(json.dumps({
            "type": "item.completed",
            "item": {"id": f"c{i}", "type": "command_execution", "command": "ls"}}))
    lines.append(json.dumps({
        "type": "item.completed",
        "item": {"id": "a0", "type": "agent_message", "text": "done"}}))
    lines.append(json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": input_tokens, "cached_input_tokens": cached,
                  "output_tokens": 500}}))
    return "\n".join(lines)


class _ScriptedClaudeAdapter(ClaudeCodeAdapter):
    def __init__(self, outputs: List[Tuple[int, str]], certs_after: int):
        super().__init__(
            name="claude_code", model="claude-opus-5", tool_version="2.1.77",
            agent_config={"price_input_per_mtok": 15.0,
                          "price_output_per_mtok": 75.0,
                          "price_cache_read_per_mtok": 1.5,
                          "price_cache_write_per_mtok": 18.75},
            env_vars={},
        )
        self._outputs = list(outputs)
        self._certs_after = certs_after
        self._calls = 0

    def setup_container(self, container, task, prompt_text):
        return True

    def _run_cli(self, container, cmd, timeout):
        if "claude -p" not in cmd:
            return 0, b""
        self._calls += 1
        exit_code, text = self._outputs[self._calls - 1]
        return exit_code, text.encode("utf-8")

    def _parse_output(self, container, raw_output):
        certs = {"S1": mock.MagicMock()} if self._calls >= self._certs_after else {}
        return certs, {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}

    def _extract_cost(self, container):
        return {"input_tokens": -1, "output_tokens": -1, "total_cost_usd": -1.0}


class _ScriptedCodexAdapter(CodexAdapter):
    def __init__(self, outputs: List[Tuple[int, str]], certs_after: int):
        super().__init__(
            name="codex", model="gpt-5.6-sol", tool_version="0.117.0",
            agent_config={}, env_vars={},
        )
        self._outputs = list(outputs)
        self._certs_after = certs_after
        self._calls = 0

    def setup_container(self, container, task, prompt_text):
        return True

    def _run_cli(self, container, cmd, timeout):
        if "codex exec" not in cmd:
            return 0, b""
        self._calls += 1
        exit_code, text = self._outputs[self._calls - 1]
        return exit_code, text.encode("utf-8")

    def _parse_output(self, container, raw_output):
        certs = {"S1": mock.MagicMock()} if self._calls >= self._certs_after else {}
        return certs, {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}

    def _extract_cost(self, container):
        return {"input_tokens": -1, "output_tokens": -1, "total_cost_usd": -1.0}


def _make_task(iid: str) -> EvalTask:
    return EvalTask(
        instance_id=iid, image_name="img", surviving_specs=[],
        f2p_tests=[], p2p_tests=[], repo_settings={},
    )


def _run_scripted(outputs, certs_after) -> BaselineResult:
    adapter = _ScriptedClaudeAdapter(outputs, certs_after)
    adapter.RETRY_WAIT_SECONDS = 0
    task = _make_task("repo__proj.abc.test_x.def.lv1")
    return _drive(adapter, task)


def _run_scripted_codex(outputs, certs_after) -> BaselineResult:
    adapter = _ScriptedCodexAdapter(outputs, certs_after)
    adapter.RETRY_WAIT_SECONDS = 0
    return _drive(adapter, _make_task("repo__proj.abc.test_x.def.lv1"))


def _drive(adapter, task) -> BaselineResult:
    with mock.patch(
        "eval_pipeline.adapters.base.exec_run_with_timeout",
        side_effect=lambda c, cmd, timeout=None: adapter._run_cli(c, cmd, timeout),
    ), mock.patch.object(adapter, "_write_file_to_container"), mock.patch.object(
        adapter, "_log_tool_filter_audit", create=True
    ):
        return adapter.run(mock.MagicMock(short_id="abc"), task, "prompt", 600)


def test_fresh_retry_keeps_only_the_last_attempt():
    failed = json.dumps({
        "type": "error",
        "message": "Selected model is at capacity, try a different model",
    })
    succeeded = _codex_stream(1000, 200, commands=3)

    result = _run_scripted_codex([(0, failed), (0, succeeded)], certs_after=2)

    assert len(result.attempts) == 1
    assert result.attempts[0]["attempt"] == 2
    assert "at capacity" not in result.attempts[0]["raw_output"]
    assert [e.error_type for e in result.llm_errors] == ["capacity"]


def test_resumed_chain_keeps_both_halves():
    first = _claude_stream("sess-A", turns=10, out_tokens=1000, aborted=True)
    second = _claude_stream("sess-A", turns=30, out_tokens=2000)

    result = _run_scripted([(0, first), (0, second)], certs_after=2)

    assert [a["attempt"] for a in result.attempts] == [1, 2]
    assert result.attempts[0]["resumed"] is False
    assert result.attempts[1]["resumed"] is True
    assert result.attempts[1]["resume_of_session"] == "sess-A"


def test_resumed_chain_reads_as_one_timeline():
    first = _claude_stream("sess-A", turns=10, out_tokens=1000, tools=2,
                           aborted=True)
    second = _claude_stream("sess-A", turns=30, out_tokens=2000, tools=3)

    result = _run_scripted([(0, first), (0, second)], certs_after=2)
    events = joined_trajectory(result)

    assert {e["_attempt"] for e in events} == {1, 2}
    attempts_in_order = [e["_attempt"] for e in events]
    assert attempts_in_order == sorted(attempts_in_order)
    assert len(events) > len(list(chain_events([result.attempts[1]])))


def test_billing_sums_the_resumed_chain_but_not_abandoned_attempts():
    interrupted = _claude_stream("sess-A", turns=10, out_tokens=1000,
                                 aborted=True)
    resumed = _claude_stream("sess-A", turns=30, out_tokens=2000)
    result = _run_scripted([(0, interrupted), (0, resumed)], certs_after=2)
    assert result.cost["output_tokens"] == 3000

    failed = json.dumps({"type": "error", "message": "at capacity"})
    result = _run_scripted_codex(
        [(0, failed), (0, _codex_stream(1000, 200))], certs_after=2)
    assert result.cost["input_tokens"] == 1000


def test_single_attempt_run_records_one_entry():
    result = _run_scripted([(0, _claude_stream("sess-A", 12, 900))], certs_after=1)
    assert len(result.attempts) == 1
    assert result.attempts[0]["resumed"] is False
    assert result.attempts[0]["cost"]["output_tokens"] == 900


def test_cost_falls_back_when_the_adapter_cannot_derive_it():
    class _NoStreamCost(_ScriptedClaudeAdapter):
        def _cost_from_stream(self, raw_output):
            return None

    adapter = _NoStreamCost([(0, _claude_stream("sess-A", 12, 900))], certs_after=1)
    task = _make_task("i")
    with mock.patch(
        "eval_pipeline.adapters.base.exec_run_with_timeout",
        side_effect=lambda c, cmd, timeout=None: adapter._run_cli(c, cmd, timeout),
    ), mock.patch.object(adapter, "_write_file_to_container"):
        result = adapter.run(mock.MagicMock(short_id="abc"), task, "p", 600)
    assert result.cost["output_tokens"] == -1


def test_session_chains_does_not_join_unrelated_sessions():
    attempts = [
        {"attempt": 1, "session_id": "A", "resume_of_session": None},
        {"attempt": 2, "session_id": "B", "resume_of_session": None},
    ]
    assert [len(c) for c in session_chains(attempts)] == [1, 1]


def test_session_chains_joins_a_resume_of_the_previous_session():
    attempts = [
        {"attempt": 1, "session_id": "A", "resume_of_session": None},
        {"attempt": 2, "session_id": "A", "resume_of_session": "A"},
    ]
    assert [len(c) for c in session_chains(attempts)] == [2]


def test_session_chains_does_not_join_a_resume_of_some_other_session():
    attempts = [
        {"attempt": 1, "session_id": "A", "resume_of_session": None},
        {"attempt": 2, "session_id": "C", "resume_of_session": "B"},
    ]
    assert [len(c) for c in session_chains(attempts)] == [1, 1]


def test_refused_resume_stays_with_the_session_it_tried_to_continue():
    attempts = [
        {"attempt": 1, "session_id": "A", "resume_of_session": None},
        {"attempt": 1, "session_id": None, "resume_of_session": "A",
         "resume_rejected": True, "raw_output": ""},
        {"attempt": 1, "session_id": "B", "resume_of_session": None},
    ]
    chains = session_chains(attempts)
    assert [len(c) for c in chains] == [2, 1]
    assert chains[0][1]["resume_rejected"] is True


def test_claude_steps_count_api_responses_not_assistant_events():
    adapter = ClaudeCodeAdapter("claude_code", "m", "v", {}, {})
    steps = adapter._extract_steps(_claude_stream("s", turns=7, out_tokens=1, tools=3))
    assert steps["api_responses"] == 3
    assert steps["assistant_events"] == 6
    assert steps["tool_calls"] == 3
    assert steps["num_turns"] == 7


def test_codex_steps_count_items_since_turn_completed_is_always_one():
    adapter = CodexAdapter("codex", "m", "v", {}, {})
    steps = adapter._extract_steps(_codex_stream(100, 50, commands=4))
    assert steps["tool_calls"] == 4
    assert steps["reasoning_items"] == 4
    assert steps["agent_messages"] == 1


def test_total_steps_sums_across_a_chain():
    attempts = [
        {"steps": {"tool_calls": 3, "api_responses": 5}},
        {"steps": {"tool_calls": 4, "api_responses": 9}},
    ]
    assert total_steps(attempts) == {"tool_calls": 7, "api_responses": 14}


def test_base_adapter_reports_no_steps_and_no_stream_cost():
    assert EvalBaselineAdapter._extract_steps(mock.MagicMock(), "{}") == {}
    assert EvalBaselineAdapter._cost_from_stream(mock.MagicMock(), "{}") is None


def test_codex_cost_records_cached_input_tokens():
    adapter = CodexAdapter("codex", "m", "v", {}, {})
    cost = adapter._cost_from_stream(_codex_stream(563987, 494080))
    assert cost["input_tokens"] == 563987
    assert cost["cached_input_tokens"] == 494080
    assert cost["output_tokens"] == 500


def test_claude_cost_splits_cached_and_uncached_totals():
    adapter = ClaudeCodeAdapter(
        "claude_code", "m", "v",
        {"price_input_per_mtok": 15.0, "price_output_per_mtok": 75.0,
         "price_cache_read_per_mtok": 1.5, "price_cache_write_per_mtok": 18.75}, {})
    cost = adapter._cost_from_stream(_claude_stream("s", 5, out_tokens=2000))
    assert cost["cache_read_input_tokens"] == 1000
    assert cost["cache_creation_input_tokens"] == 100
    assert cost["total_cost_usd"] < cost["total_cost_usd_no_cache"]


def test_attempts_survive_a_json_round_trip():
    result = _run_scripted([(0, _claude_stream("sess-A", 12, 900))], certs_after=1)
    result.certificates = {}
    restored = BaselineResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.attempts == result.attempts


def test_results_written_before_attempt_recording_load_as_empty():
    legacy = {
        "instance_id": "i", "baseline": "claude_code", "model": "m",
        "tool_version": "v", "raw_output": "x", "certificates": {},
        "cost": {}, "wall_time_seconds": 1.0,
    }
    assert BaselineResult.from_dict(legacy).attempts == []


_AUDIT_LOG = (
    "stripped 1 hosted tool(s) from POST /openai/v1/responses "
    "(total 10 in 10/11 requests)\n"
    "stripped 0 hosted tool(s) from POST /openai/v1/responses "
    "(total 10 in 10/12 requests)\n"
)


class _RealCostCodexAdapter(_ScriptedCodexAdapter):
    _extract_cost = CodexAdapter._extract_cost


def _codex_with_audit(audit_text, stream):
    adapter = _RealCostCodexAdapter([(0, stream)], certs_after=1)
    adapter.RETRY_WAIT_SECONDS = 0

    def _exec(container, cmd, timeout=None):
        if "hosted_tool_filter.log" in cmd:
            return 0, audit_text.encode("utf-8")
        if "codex_output.txt" in cmd:
            return 0, stream.encode("utf-8")
        return adapter._run_cli(container, cmd, timeout)

    with mock.patch(
        "eval_pipeline.adapters.base.exec_run_with_timeout", side_effect=_exec
    ), mock.patch(
        "eval_pipeline.adapters.codex.exec_run_with_timeout", side_effect=_exec
    ), mock.patch(
        "eval_pipeline.adapters.tool_filter.exec_run_with_timeout",
        side_effect=_exec,
    ), mock.patch.object(adapter, "_write_file_to_container"):
        return adapter.run(mock.MagicMock(short_id="abc"),
                           _make_task("i"), "prompt", 600)


def test_request_count_comes_from_the_last_audit_line():
    adapter = CodexAdapter("codex", "m", "v", {}, {})
    assert adapter._AUDIT_REQUESTS_RE.search(
        _AUDIT_LOG.splitlines()[-1]).group(2) == "12"


def test_request_count_survives_a_trailing_unstripped_request():
    adapter = CodexAdapter("codex", "m", "v", {}, {})
    last = _AUDIT_LOG.splitlines()[-1]
    stripped, total = adapter._AUDIT_REQUESTS_RE.search(last).groups()
    assert (stripped, total) == ("10", "12")


def test_api_requests_reaches_the_result_alongside_per_attempt_cost():
    result = _codex_with_audit(_AUDIT_LOG, _codex_stream(1000, 200, commands=3))
    assert result.cost["api_requests"] == 12
    assert result.cost["input_tokens"] == 1000
    assert result.cost["cached_input_tokens"] == 200


def test_a_missing_audit_log_leaves_the_count_absent_rather_than_wrong():
    result = _codex_with_audit("", _codex_stream(1000, 200))
    assert "api_requests" not in result.cost


def test_only_model_invocations_are_counted_as_llm_calls():
    import importlib.util
    import os

    os.environ.setdefault("UPSTREAM_BASE_URL", "http://upstream.invalid/v1")
    src = (Path(__file__).resolve().parents[3]
           / "featurebench" / "resources" / "hosted_tool_filter.py")
    spec = importlib.util.spec_from_file_location("_htf", src)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    counted = [("POST", "/openai/v1/responses"),
               ("POST", "/openai/v1/responses?stream=true"),
               ("POST", "/v1/chat/completions")]
    ignored = [("GET", "/openai/v1/models"),
               ("DELETE", "/openai/v1/responses/resp_123"),
               ("POST", "/openai/v1/responses/resp_123/cancel")]
    for method, path in counted:
        assert shim.is_model_invocation(method, path), (method, path)
    for method, path in ignored:
        assert not shim.is_model_invocation(method, path), (method, path)


def test_claude_separates_complete_responses_from_truncated_ones():
    adapter = ClaudeCodeAdapter("claude_code", "m", "v", {}, {})
    steps = adapter._extract_steps(
        _claude_stream("s", turns=7, out_tokens=1, tools=3, aborted=True))
    assert steps["api_responses"] == 4
    assert steps["successful_llm_calls"] == 3


def test_claude_complete_equals_total_when_nothing_was_cut():
    adapter = ClaudeCodeAdapter("claude_code", "m", "v", {}, {})
    steps = adapter._extract_steps(
        _claude_stream("s", turns=7, out_tokens=1, tools=3))
    assert steps["api_responses"] == steps["successful_llm_calls"] == 3


def test_codex_reports_both_attempted_and_successful_calls():
    audit = (
        "stripped 1 hosted tool(s) from POST /openai/v1/responses "
        "(total 1 in 1/1 requests; ok=1 failed=0)\n"
        "stripped 1 hosted tool(s) from POST /openai/v1/responses "
        "(total 2 in 2/3 requests; ok=2 failed=1)\n"
    )
    result = _codex_with_audit(audit, _codex_stream(1000, 200))
    assert result.cost["api_requests"] == 3
    assert result.cost["successful_llm_calls"] == 2


def test_logs_without_outcome_tracking_omit_the_success_count():
    legacy = ("stripped 1 hosted tool(s) from POST /openai/v1/responses "
              "(total 11 in 11/11 requests)\n")
    result = _codex_with_audit(legacy, _codex_stream(1000, 200))
    assert result.cost["api_requests"] == 11
    assert "successful_llm_calls" not in result.cost


def test_shim_counts_a_dropped_stream_as_a_failed_call():
    import importlib.util
    import os

    os.environ.setdefault("UPSTREAM_BASE_URL", "http://upstream.invalid/v1")
    src = (Path(__file__).resolve().parents[3]
           / "featurebench" / "resources" / "hosted_tool_filter.py")
    spec = importlib.util.spec_from_file_location("_htf2", src)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim.AUDIT_PATH = str(Path(os.environ.get("TMPDIR", "/tmp")) / "_htf_test.log")

    shim._record_outcome("POST", "/v1/responses", 1, ok=True)
    shim._record_outcome("POST", "/v1/responses", 1, ok=False)
    shim._record_outcome("POST", "/v1/responses", 1, ok=True)
    assert shim._stats["ok"] == 2
    assert shim._stats["failed"] == 1


def _install_outcomes(monkeypatch, exit_codes):
    from eval_pipeline.adapters import base as base_mod

    calls: list = []
    outcomes = list(exit_codes)

    def fake_exec(container, cmd, timeout=None):
        calls.append(cmd)
        code = outcomes.pop(0)
        return code, (b"ok" if code == 0 else b"npm error network")

    monkeypatch.setattr(base_mod, "exec_run_with_timeout", fake_exec)
    monkeypatch.setattr(base_mod, "copy_to_container", lambda *a, **k: None)
    monkeypatch.setattr(base_mod.time, "sleep", lambda _s: None)
    return calls


def test_tool_install_retries_a_registry_failure(monkeypatch):
    calls = _install_outcomes(monkeypatch, [1, 0])
    adapter = ClaudeCodeAdapter("claude_code", "m", "v", {}, {})
    assert adapter._install_tool(mock.MagicMock()) is True
    assert len(calls) == 2


def test_tool_install_gives_up_after_the_last_attempt(monkeypatch):
    calls = _install_outcomes(monkeypatch, [1, 1, 1])
    adapter = ClaudeCodeAdapter("claude_code", "m", "v", {}, {})
    assert adapter._install_tool(mock.MagicMock()) is False
    assert len(calls) == adapter.INSTALL_ATTEMPTS
