import json
import sys
from pathlib import Path
from types import ModuleType
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

from eval_pipeline.adapters.base import SUCCESSFUL_LLM_CALLS
from eval_pipeline.adapters.opencode import OpenCodeAdapter

SESSION = "ses_fa6de6955ffewfXjEmuT8UMSc2"

PRICES = {
    "price_input_per_mtok": 10.0,
    "price_output_per_mtok": 100.0,
    "price_cache_read_per_mtok": 1.0,
    "price_cache_write_per_mtok": 20.0,
}


def _adapter(max_steps=None, **env) -> OpenCodeAdapter:
    env_vars = {
        "OPENCODE_API_KEY": "sk-test",
        "OPENCODE_BASE_URL": "https://proxy.example/v1",
        "OPENCODE_MODEL": "deepseek-v4-flash",
    }
    env_vars.update(env)
    return OpenCodeAdapter(
        name="opencode",
        model="deepseek-v4-flash",
        tool_version="1.18.25",
        agent_config=dict(PRICES),
        env_vars=env_vars,
        max_steps=max_steps,
    )


def _step_finish(reason="stop", inp=0, out=0, reasoning=0, read=0, write=0) -> str:
    return json.dumps({
        "type": "step_finish",
        "sessionID": SESSION,
        "part": {
            "type": "step-finish",
            "reason": reason,
            "cost": 0,
            "tokens": {
                "total": inp + out + reasoning + read + write,
                "input": inp, "output": out, "reasoning": reasoning,
                "cache": {"read": read, "write": write},
            },
        },
    })


def _tool_use(tool="bash", status="completed") -> str:
    return json.dumps({
        "type": "tool_use",
        "sessionID": SESSION,
        "part": {"type": "tool", "tool": tool, "state": {"status": status}},
    })


def _text(text="DONE") -> str:
    return json.dumps({
        "type": "text", "sessionID": SESSION,
        "part": {"type": "text", "text": text},
    })


def _step_start() -> str:
    return json.dumps({
        "type": "step_start", "sessionID": SESSION,
        "part": {"type": "step-start"},
    })


def _clean_run() -> str:
    return "\n".join([
        _step_start(),
        _tool_use(),
        _step_finish(reason="tool-calls", inp=7054, out=48, reasoning=18),
        _step_start(),
        _text(),
        _step_finish(reason="stop", inp=221, out=4, reasoning=16, read=6912),
    ])


def test_log_lines_sharing_the_stream_are_not_read_as_events():
    noisy = "\n".join([
        'timestamp=2026-08-31T18:41:32.159Z level=INFO message="creating instance"',
        _step_start(),
        'timestamp=2026-08-31T18:41:34.105Z level=ERROR message="stream error"',
        _step_finish(reason="stop", inp=10, out=2),
    ])

    events = OpenCodeAdapter._parse_events(noisy)

    assert [e["type"] for e in events] == ["step_start", "step_finish"]


def test_session_id_is_read_off_any_event():
    assert _adapter()._extract_session_id(_clean_run()) == SESSION


def test_session_id_is_absent_when_nothing_was_emitted():
    assert _adapter()._extract_session_id("") is None


def test_closed_responses_are_the_shared_call_count():
    steps = _adapter()._extract_steps(_clean_run())

    assert steps["model_responses"] == 2
    assert steps[SUCCESSFUL_LLM_CALLS] == 2


def test_only_completed_tools_count_as_tool_calls():
    stream = "\n".join([
        _tool_use(status="completed"),
        _tool_use(status="running"),
        _tool_use(status="error"),
        _step_finish(reason="stop"),
    ])

    assert _adapter()._extract_steps(stream)["tool_calls"] == 1


def test_cache_tokens_are_additional_not_a_subset_of_input():
    cost = _adapter()._cost_from_stream(_clean_run())

    assert cost["input_tokens"] == 7054 + 221
    assert cost["cache_read_input_tokens"] == 6912
    assert cost["cache_creation_input_tokens"] == 0


def test_reasoning_tokens_are_reported_apart_and_billed_as_output():
    cost = _adapter()._cost_from_stream(_clean_run())

    assert cost["output_tokens"] == 52
    assert cost["reasoning_tokens"] == 34
    expected = (7275 * 10 + 86 * 100 + 6912 * 1) / 1_000_000
    assert cost["total_cost_usd"] == pytest.approx(expected)


def test_the_no_cache_figure_bills_cache_reads_at_the_input_rate():
    cost = _adapter()._cost_from_stream(_clean_run())

    expected = ((7275 + 6912) * 10 + 86 * 100) / 1_000_000
    assert cost["total_cost_usd_no_cache"] == pytest.approx(expected)


def test_a_transcript_without_usage_yields_no_cost():
    assert _adapter()._cost_from_stream(_text()) is None


def test_a_run_that_closed_its_turn_is_not_an_abort():
    assert _adapter()._detect_stream_abort(_clean_run()) is None


def test_an_empty_transcript_is_an_abort():
    error = _adapter()._detect_stream_abort("")

    assert error is not None
    assert error.error_type == "stream_abort"


def test_a_run_cut_at_the_output_limit_is_an_abort():
    stream = "\n".join([_step_start(), _step_finish(reason="length", inp=10, out=64000)])

    error = _adapter()._detect_stream_abort(stream)

    assert error is not None
    assert "length" in error.sample_message


def test_a_run_left_mid_tool_call_is_an_abort():
    stream = "\n".join([
        _step_start(), _tool_use(), _step_finish(reason="tool-calls", inp=10, out=2),
    ])

    assert _adapter()._detect_stream_abort(stream) is not None


def test_the_config_pins_a_default_model_as_well_as_the_flag():
    config = json.loads(_adapter()._config_json())

    assert config["model"] == "proxy/deepseek-v4-flash"
    assert config["small_model"] == "proxy/deepseek-v4-flash"


def test_the_config_declares_the_model_the_registry_does_not_know():
    config = json.loads(_adapter(OPENCODE_CONTEXT_LIMIT="128000")._config_json())

    model = config["provider"]["proxy"]["models"]["deepseek-v4-flash"]
    assert model["limit"] == {"context": 128000, "output": 64000}


def test_the_key_reaches_the_cli_through_the_environment():
    config = json.loads(_adapter()._config_json())

    assert config["provider"]["proxy"]["options"]["apiKey"] == "{env:OPENCODE_PROXY_API_KEY}"
    assert "sk-test" not in json.dumps(config)


def test_snapshots_and_sharing_are_off():
    config = json.loads(_adapter()._config_json())

    assert config["snapshot"] is False
    assert config["share"] == "disabled"
    assert config["autoupdate"] is False


def test_the_networked_tool_is_disabled_to_match_the_other_baselines():
    assert json.loads(_adapter()._config_json())["tools"]["webfetch"] is False


def test_the_wire_selects_the_sdk_package():
    anthropic = json.loads(_adapter(OPENCODE_WIRE_API="anthropic")._config_json())
    compatible = json.loads(_adapter(OPENCODE_WIRE_API="openai-compatible")._config_json())

    assert anthropic["provider"]["proxy"]["npm"] == "@ai-sdk/anthropic"
    assert compatible["provider"]["proxy"]["npm"] == "@ai-sdk/openai-compatible"


def test_the_default_wire_is_anthropic():
    config = json.loads(_adapter()._config_json())

    assert config["provider"]["proxy"]["npm"] == "@ai-sdk/anthropic"


def test_an_unknown_wire_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="OPENCODE_WIRE_API"):
        _adapter(OPENCODE_WIRE_API="grpc")._config_json()


def test_the_run_command_bypasses_the_permission_prompt():
    cmd = _adapter().get_run_command("/tmp/eval_prompt.txt")

    assert "--auto" in cmd
    assert "--format json" in cmd
    assert "--model proxy/deepseek-v4-flash" in cmd


def test_the_run_command_keeps_provider_errors_in_the_transcript():
    cmd = _adapter().get_run_command("/tmp/eval_prompt.txt")

    assert "--print-logs --log-level ERROR" in cmd
    assert "2>&1 | tee /tmp/opencode_output.txt" in cmd


def test_the_variant_flag_is_omitted_unless_asked_for():
    assert "--variant" not in _adapter().get_run_command("/tmp/p.txt")
    assert "--variant high" in _adapter(OPENCODE_VARIANT="high").get_run_command("/tmp/p.txt")


def test_the_install_pins_the_version_and_proves_the_model_resolves():
    script = _adapter().get_install_script()

    assert "npm install -g opencode-ai@1.18.25" in script
    assert "opencode models | grep -qx proxy/deepseek-v4-flash" in script


def test_the_baseline_model_wins_over_the_adapters_own():
    adapter = _adapter(OPENCODE_MODEL="claude-opus-5")

    assert "--model proxy/claude-opus-5" in adapter.get_run_command("/tmp/p.txt")


CHILD = "ses_fa4f81155ffevkXDCUG5bmB8iH"
GRANDCHILD = "ses_grandchild0000000000000"


def _task_call(child_session: str) -> str:
    return json.dumps({
        "type": "tool_use",
        "sessionID": SESSION,
        "part": {
            "type": "tool",
            "tool": "task",
            "state": {
                "status": "completed",
                "input": {"subagent_type": "explore"},
                "metadata": {
                    "parentSessionId": SESSION,
                    "sessionId": child_session,
                    "model": {"modelID": "deepseek-v4-flash", "providerID": "proxy"},
                },
            },
        },
    })


def _exported(steps, task_children=()) -> dict:
    parts = [
        {"type": "step-finish", "reason": "stop", "cost": 0, "tokens": {
            "input": inp, "output": out, "reasoning": 0,
            "cache": {"read": 0, "write": 0}}}
        for inp, out in steps
    ]
    for child in task_children:
        parts.append({
            "type": "tool", "tool": "task",
            "state": {"status": "completed",
                      "metadata": {"sessionId": child}},
        })
    return {"info": {"id": "ses_x"}, "messages": [{"parts": parts}]}


class _container_returning:
    def __init__(self, transcript: str, exports: dict, audit: str = None):
        self.transcript = transcript
        self.exports = exports
        self.audit = audit
        self.commands: list = []

    def fake_exec(self, container, cmd, timeout=None):
        self.commands.append(cmd)
        if cmd.startswith("cat /tmp/opencode_output.txt"):
            return 0, self.transcript.encode()
        if cmd.startswith("cat /tmp/hosted_tool_filter.log"):
            if self.audit is None:
                return 1, b"cat: /tmp/hosted_tool_filter.log: No such file"
            return 0, self.audit.encode()
        for session_id, document in self.exports.items():
            if f"opencode export {session_id}" in cmd:
                if document is None:
                    return 1, b"no such session"
                return 0, json.dumps(document).encode()
        return 1, b"unknown command"

    def __enter__(self):
        self._patches = [
            mock.patch(
                "eval_pipeline.adapters.opencode.exec_run_with_timeout",
                self.fake_exec,
            ),
            mock.patch(
                "eval_pipeline.adapters.tool_filter.exec_run_with_timeout",
                self.fake_exec,
            ),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()
        return False


def test_task_session_ids_are_found_in_the_printed_stream():
    events = OpenCodeAdapter._parse_events(_task_call(CHILD))

    assert OpenCodeAdapter._task_session_ids(events) == [CHILD]


def test_task_session_ids_are_found_in_an_exported_document():
    document = _exported([(10, 2)], task_children=[GRANDCHILD])

    assert OpenCodeAdapter._task_session_ids(document) == [GRANDCHILD]


def test_subagent_calls_and_tokens_are_folded_into_the_run():
    transcript = "\n".join([
        _step_start(),
        _task_call(CHILD),
        _step_finish(reason="tool-calls", inp=100, out=10),
        _step_finish(reason="stop", inp=200, out=20),
    ])
    exports = {CHILD: _exported([(1000, 100), (2000, 200), (3000, 300)])}

    with _container_returning(transcript, exports):
        cost = _adapter()._extract_cost(object())

    assert cost[SUCCESSFUL_LLM_CALLS] == 5
    assert cost["subagent_llm_calls"] == 3
    assert cost["subagent_sessions"] == 1
    assert cost["input_tokens"] == 300 + 6000
    assert cost["output_tokens"] == 30 + 600


def test_a_subagent_that_delegates_again_is_walked():
    transcript = "\n".join([_task_call(CHILD), _step_finish(reason="stop", inp=1, out=1)])
    exports = {
        CHILD: _exported([(10, 1)], task_children=[GRANDCHILD]),
        GRANDCHILD: _exported([(20, 2), (30, 3)]),
    }

    with _container_returning(transcript, exports):
        cost = _adapter()._extract_cost(object())

    assert cost["subagent_sessions"] == 2
    assert cost[SUCCESSFUL_LLM_CALLS] == 1 + 1 + 2


def test_a_session_is_never_visited_twice():
    transcript = "\n".join([
        _task_call(CHILD), _task_call(CHILD), _step_finish(reason="stop", inp=1, out=1),
    ])
    exports = {CHILD: _exported([(10, 1)])}

    with _container_returning(transcript, exports):
        cost = _adapter()._extract_cost(object())

    assert cost["subagent_sessions"] == 1
    assert cost[SUCCESSFUL_LLM_CALLS] == 2


def test_an_unreadable_subagent_leaves_the_parent_figure_standing():
    transcript = "\n".join([_task_call(CHILD), _step_finish(reason="stop", inp=100, out=10)])

    with _container_returning(transcript, {CHILD: None}):
        cost = _adapter()._extract_cost(object())

    assert cost[SUCCESSFUL_LLM_CALLS] == 1
    assert cost["subagent_llm_calls"] == 0
    assert cost["input_tokens"] == 100


def test_a_stream_that_carried_no_steps_falls_back_to_the_store():
    transcript = _text()
    exports = {SESSION: _exported([(500, 50)])}

    with _container_returning(transcript, exports):
        cost = _adapter()._extract_cost(object())

    assert cost[SUCCESSFUL_LLM_CALLS] == 1
    assert cost["input_tokens"] == 500


def test_per_attempt_cost_does_not_win_over_the_container_figure():
    attempts = [{"cost": {"input_tokens": 5, "output_tokens": 1}}]

    assert _adapter()._chain_cost(attempts) is None


def _audit(ok, requests=None, refused=0):
    requests = ok if requests is None else requests
    return "\n".join([
        "hosted_tool_filter listening on 127.0.0.1:8123 -> https://proxy.example "
        "(stripping: []; budget: 80 model calls)",
        f"stripped 0 hosted tool(s) from POST /v1/chat/completions (tools=9; "
        f"total 0 in 0/{requests} requests; ok={ok} failed={requests - ok}; "
        f"refused={refused})",
    ])


def test_the_provider_is_pointed_at_the_shim():
    config = json.loads(_adapter()._config_json())
    assert config["provider"]["proxy"]["options"]["baseURL"] == (
        "http://127.0.0.1:8123/v1"
    )


def test_the_install_writes_the_shim_before_proving_the_model_resolves():
    script = _adapter().get_install_script()
    assert "cat > /opt/eval/hosted_tool_filter.py" in script
    assert "export UPSTREAM_BASE_URL=https://proxy.example/v1" in script
    assert script.index("hosted_tool_filter.py") < script.index("opencode models")


def test_a_capped_run_starts_a_fresh_shim_with_the_budget_and_polls_it():
    cmd = _adapter(max_steps=50).get_run_command("/tmp/eval_prompt.txt")

    assert 'pkill -f "[h]osted_tool_filter.py"' in cmd
    assert "export MAX_MODEL_CALLS=50; " in cmd
    assert cmd.index("MAX_MODEL_CALLS=50") < cmd.index("start_tool_filter.sh")
    assert cmd.index("start_tool_filter.sh") < cmd.index("opencode run")
    assert (
        "step_gate.py 50 opencode_wire /tmp/eval_agent.pid "
        "/tmp/hosted_tool_filter.log" in cmd
    )


def test_the_session_is_titled_so_no_call_is_spent_generating_one():
    cmd = _adapter().get_run_command("/tmp/eval_prompt.txt")
    assert "--title 'sc-align eval'" in cmd


def test_an_uncapped_run_still_goes_through_the_shim():
    cmd = _adapter().get_run_command("/tmp/eval_prompt.txt")
    assert "unset MAX_MODEL_CALLS; " in cmd
    assert "start_tool_filter.sh" in cmd
    assert "step_gate" not in cmd


def test_without_an_upstream_there_is_no_shim_and_the_stream_unit_stands():
    adapter = _adapter(max_steps=50, OPENCODE_BASE_URL="")
    assert adapter.step_gate_unit() == "opencode"
    assert adapter.step_gate_source() is None
    assert "hosted_tool_filter" not in adapter.get_install_script()
    cmd = adapter.get_run_command("/tmp/eval_prompt.txt")
    assert "start_tool_filter.sh" not in cmd
    assert "step_gate.py 50 opencode /tmp/eval_agent.pid" in cmd


def test_the_wire_count_supersedes_the_one_assembled_from_sessions():
    transcript = "\n".join([
        _step_start(),
        _task_call(CHILD),
        _step_finish(reason="tool-calls", inp=100, out=10),
        _step_finish(reason="stop", inp=200, out=20),
    ])
    exports = {CHILD: None}

    with _container_returning(transcript, exports, audit=_audit(ok=9, requests=10)):
        cost = _adapter()._extract_cost(object())

    assert cost[SUCCESSFUL_LLM_CALLS] == 9
    assert cost["api_requests"] == 10
    assert cost["refused_llm_calls"] == 0
    assert cost["session_llm_calls"] == 2
    assert cost["subagent_llm_calls"] == 0
    assert cost["subagent_sessions"] == 1


def test_a_missing_audit_log_leaves_the_session_count_standing():
    transcript = "\n".join([
        _step_start(),
        _task_call(CHILD),
        _step_finish(reason="tool-calls", inp=100, out=10),
        _step_finish(reason="stop", inp=200, out=20),
    ])
    exports = {CHILD: _exported([(1000, 100), (2000, 200), (3000, 300)])}

    with _container_returning(transcript, exports, audit=None):
        cost = _adapter()._extract_cost(object())

    assert cost[SUCCESSFUL_LLM_CALLS] == 5
    assert "session_llm_calls" not in cost
    assert "api_requests" not in cost


def test_the_export_is_read_back_from_a_file_without_its_banner():
    transcript = "\n".join([_task_call(CHILD), _step_finish(reason="stop", inp=1, out=1)])
    with _container_returning(transcript, {CHILD: _exported([(1, 1)])}) as fake:
        _adapter()._extract_cost(object())

    export = [c for c in fake.commands if f"opencode export {CHILD}" in c]
    assert len(export) == 1
    assert (
        f"opencode export {CHILD} > /tmp/opencode_export.json 2>/dev/null "
        f"&& cat /tmp/opencode_export.json" in export[0]
    )
