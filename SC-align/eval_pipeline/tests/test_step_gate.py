import json
import subprocess
import sys
import textwrap
import time
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SC_ALIGN = Path(__file__).resolve().parents[2]
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from featurebench.resources import step_gate
from eval_pipeline.adapters import base as base_mod
from eval_pipeline.adapters.claude_code import ClaudeCodeAdapter
from eval_pipeline.adapters.codex import CodexAdapter
from eval_pipeline.adapters.opencode import OpenCodeAdapter
from eval_pipeline.models import EvalTask


GATE = str(base_mod.STEP_GATE_SRC)


def test_marker_name_agrees_across_the_container_boundary():
    assert base_mod.STEP_CAP_EVENT_TYPE == step_gate.STEP_CAP_EVENT_TYPE


def test_the_gate_source_the_adapters_ship_exists():
    assert base_mod.STEP_GATE_SRC.is_file()
    assert Path(step_gate.__file__).resolve() == base_mod.STEP_GATE_SRC


def _count(unit: str, events: List[dict]) -> int:
    counter = step_gate.COUNTERS[unit]
    state = set()
    return sum(counter(event, state) for event in events)


def test_claude_code_counts_responses_not_events():
    events = []
    for i in range(3):
        events.append({"type": "assistant", "message": {
            "id": f"msg_{i}", "content": [{"type": "thinking"}]}})
        events.append({"type": "assistant", "message": {
            "id": f"msg_{i}", "content": [{"type": "tool_use"}]}})
        events.append({"type": "user", "message": {"content": []}})
    assert _count("claude_code", events) == 3


def test_codex_counts_tool_calls_and_messages_but_not_reasoning():
    events = [
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {"type": "item.completed", "item": {"type": "agent_message"}},
        {"type": "item.completed", "item": {"type": "todo_list"}},
        {"type": "turn.completed", "usage": {}},
    ]
    assert _count("codex", events) == 2


def test_opencode_counts_step_finish():
    events = [
        {"type": "text", "part": {}},
        {"type": "step_finish", "part": {"reason": "tool-calls"}},
        {"type": "tool_use", "part": {"state": {"status": "completed"}}},
        {"type": "step_finish", "part": {"reason": "stop"}},
    ]
    assert _count("opencode", events) == 2


def test_every_adapter_that_declares_a_unit_has_a_rule_for_it():
    adapters = [
        ClaudeCodeAdapter, CodexAdapter, OpenCodeAdapter,
    ]
    for cls in adapters:
        adapter = cls(
            name=cls.__name__, model="m", tool_version="v",
            agent_config={}, env_vars={}, max_steps=50,
        )
        assert adapter.step_gate_unit() in step_gate.COUNTERS


def _fake_cli(tmp_path: Path, events: int) -> Path:
    script = tmp_path / "fake_cli.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        for i in $(seq 1 {events}); do
            printf '{{"type":"assistant","message":{{"id":"msg_%s","content":[]}}}}\\n' "$i"
            sleep 0.05
        done
        # Past the cap: an agent that would keep going forever.
        while true; do sleep 1; done
    """))
    script.chmod(0o755)
    return script


def _run_gated_pipeline(tmp_path: Path, max_steps: int, events: int = 20):
    cli = _fake_cli(tmp_path, events)
    transcript = tmp_path / "transcript.txt"
    pid_file = tmp_path / "agent.pid"
    cmd = (
        f"( {cli} 2>&1 & echo $! > {pid_file}; wait $! ) "
        f"| tee {transcript} "
        f"| {sys.executable} {GATE} {max_steps} claude_code {pid_file}"
    )
    started = time.time()
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=90,
    )
    return proc, transcript, pid_file, time.time() - started


def test_the_gate_stops_a_cli_that_would_run_forever(tmp_path):
    proc, transcript, pid_file, elapsed = _run_gated_pipeline(
        tmp_path, max_steps=4, events=20,
    )

    assert proc.returncode == step_gate.STEP_CAP_EXIT_CODE
    assert elapsed < 30

    pid = int(pid_file.read_text().strip())
    with pytest.raises(OSError):
        import os
        os.kill(pid, 0)

    marker = [
        json.loads(line) for line in proc.stdout.splitlines()
        if line.startswith("{") and "step_cap_reached" in line
    ]
    assert len(marker) == 1
    assert marker[0] == {
        "type": "step_cap_reached", "unit": "claude_code",
        "limit": 4, "count": 4,
    }

    assert "step_cap_reached" not in transcript.read_text()


def test_an_agent_that_finishes_under_the_cap_is_untouched(tmp_path):
    cli = tmp_path / "short_cli.sh"
    cli.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        printf '{"type":"assistant","message":{"id":"msg_1","content":[]}}\\n'
        printf '{"type":"result","result":"done"}\\n'
    """))
    cli.chmod(0o755)
    pid_file = tmp_path / "agent.pid"
    transcript = tmp_path / "t.txt"
    proc = subprocess.run(
        ["bash", "-c",
         f"( {cli} 2>&1 & echo $! > {pid_file}; wait $! ) | tee {transcript} "
         f"| {sys.executable} {GATE} 50 claude_code {pid_file}"],
        capture_output=True, text=True, timeout=60,
    )

    assert proc.returncode == 0
    assert "step_cap_reached" not in proc.stdout
    assert proc.stdout == transcript.read_text()


def _adapter(cls, max_steps=None, **env):
    return cls(
        name=cls.__name__, model="m", tool_version="v",
        agent_config={}, env_vars=env, max_steps=max_steps,
    )


def test_an_uncapped_command_is_unchanged():
    cmd = _adapter(ClaudeCodeAdapter).get_run_command("/tmp/eval_prompt.txt")
    assert cmd.endswith(
        "--verbose --output-format stream-json 2>&1 | tee /tmp/claude_output.txt"
    )
    assert "step_gate" not in cmd


@pytest.mark.parametrize("cls,cli_fragment,transcript,unit", [
    (ClaudeCodeAdapter, "claude -p", "/tmp/claude_output.txt", "claude_code"),
    (CodexAdapter, "codex exec", "/tmp/codex_output.txt", "codex"),
    (OpenCodeAdapter, "opencode run", "/tmp/opencode_output.txt", "opencode"),
])
def test_a_capped_command_routes_the_cli_through_the_gate(
    cls, cli_fragment, transcript, unit
):
    cmd = _adapter(cls, max_steps=50).get_run_command("/tmp/eval_prompt.txt")

    assert cli_fragment in cmd
    assert f"echo $! > {base_mod.AGENT_PID_PATH}" in cmd
    assert cmd.index(f"tee {transcript}") < cmd.index("step_gate.py")
    assert f"step_gate.py 50 {unit} {base_mod.AGENT_PID_PATH}" in cmd


def test_a_resumed_claude_command_is_capped_at_what_is_left():
    adapter = _adapter(ClaudeCodeAdapter, max_steps=50)
    cmd = adapter.get_resume_command(
        "/tmp/eval_resume_prompt.txt", "sess-1", max_steps=34
    )
    assert "--resume" in cmd
    assert "step_gate.py 34 claude_code" in cmd


def test_a_resumed_command_falls_back_to_the_whole_cap():
    adapter = _adapter(ClaudeCodeAdapter, max_steps=50)
    cmd = adapter.get_resume_command("/tmp/eval_resume_prompt.txt", "sess-1")
    assert "step_gate.py 50 claude_code" in cmd


def test_a_resumed_command_of_an_uncapped_run_stays_uncapped():
    adapter = _adapter(ClaudeCodeAdapter)
    assert adapter._remaining_steps(17) is None
    cmd = adapter.get_resume_command(
        "/tmp/eval_resume_prompt.txt", "sess-1", max_steps=None
    )
    assert "step_gate" not in cmd


def test_an_adapter_without_a_counting_rule_is_not_capped():
    class _Unknown(base_mod.EvalBaselineAdapter):
        def get_install_script(self):
            return ""

        def get_run_command(self, prompt_path):
            return self._piped_cli("mytool run", "/tmp/out.txt")

    adapter = _Unknown(
        name="mytool", model="m", tool_version="v",
        agent_config={}, env_vars={}, max_steps=50,
    )
    assert adapter.step_gate_unit() is None
    assert adapter.get_run_command("/tmp/p") == "mytool run 2>&1 | tee /tmp/out.txt"


def _capped_stream(responses: int, limit: int) -> str:
    lines = [json.dumps({"type": "system", "subtype": "init",
                         "session_id": "sess-1"})]
    for i in range(responses):
        lines.append(json.dumps({
            "type": "assistant", "session_id": "sess-1",
            "message": {"id": f"msg_{i}", "content": [
                {"type": "tool_use", "name": "Bash", "input": {}}]}}))
    lines.append(json.dumps({
        "type": "assistant", "session_id": "sess-1",
        "message": {"id": "msg_cut", "content": [
            {"type": "thinking", "thinking": "cut off mid-"}]}}))
    lines.append(json.dumps({
        "type": step_gate.STEP_CAP_EVENT_TYPE, "unit": "claude_code",
        "limit": limit, "count": limit}))
    return "\n".join(lines)


class _ScriptedClaudeAdapter(ClaudeCodeAdapter):
    def __init__(self, outputs: List[Tuple[int, str]], certs: bool):
        super().__init__(
            name="claude_code", model="claude-opus-5", tool_version="2.1.77",
            agent_config={}, env_vars={}, max_steps=50,
        )
        self._outputs = list(outputs)
        self._certs = certs
        self.calls = 0
        self.cmds: List[str] = []

    def setup_container(self, container, task, prompt_text):
        return True

    def _run_cli(self, container, cmd, timeout):
        if "claude -p" not in cmd:
            return 0, b""
        self.calls += 1
        self.cmds.append(cmd)
        exit_code, text = self._outputs[self.calls - 1]
        return exit_code, text.encode("utf-8")

    def _parse_output(self, container, raw_output):
        certs = {"S1": mock.MagicMock()} if self._certs else {}
        return certs, {"input_tokens": 0, "output_tokens": 0,
                       "total_cost_usd": 0.0}

    def _extract_cost(self, container):
        return None


def _drive(adapter, timeout=None):
    task = EvalTask(
        instance_id="repo__proj.abc.test_x.def.lv1", image_name="img",
        surviving_specs=[], f2p_tests=[], p2p_tests=[], repo_settings={},
    )
    with mock.patch(
        "eval_pipeline.adapters.base.exec_run_with_timeout",
        side_effect=lambda c, cmd, timeout=None: adapter._run_cli(c, cmd, timeout),
    ), mock.patch.object(adapter, "_write_file_to_container"):
        return adapter.run(mock.MagicMock(short_id="abc"), task, "prompt", timeout)


def test_a_capped_run_keeps_the_certificate_the_agent_had_written():
    adapter = _ScriptedClaudeAdapter(
        [(step_gate.STEP_CAP_EXIT_CODE, _capped_stream(50, 50))], certs=True,
    )
    result = _drive(adapter)

    assert set(result.certificates) == {"S1"}
    assert adapter.calls == 1
    assert result.completed_normally is False
    assert result.attempts[0]["step_cap"]["limit"] == 50


def test_a_capped_run_with_no_certificate_is_not_retried():
    adapter = _ScriptedClaudeAdapter(
        [(step_gate.STEP_CAP_EXIT_CODE, _capped_stream(50, 50))] * 3,
        certs=False,
    )
    adapter.RETRY_WAIT_SECONDS = 0
    result = _drive(adapter)

    assert adapter.calls == 1
    assert result.certificates == {}
    assert result.attempts[0]["step_cap"] is not None


def _aborted_stream(responses: int, tag: str) -> str:
    lines = [json.dumps({"type": "system", "subtype": "init",
                         "session_id": "sess-1"})]
    for i in range(responses):
        lines.append(json.dumps({
            "type": "assistant", "session_id": "sess-1",
            "message": {"id": f"msg_{tag}_{i}", "content": [
                {"type": "tool_use", "name": "Bash", "input": {}}]}}))
    lines.append(json.dumps({
        "type": "assistant", "session_id": "sess-1",
        "message": {"id": f"msg_{tag}_cut", "content": [
            {"type": "thinking", "thinking": "cut off mid-"}]}}))
    lines.append(json.dumps({
        "type": "result", "session_id": "sess-1", "result": ""}))
    return "\n".join(lines)


def _gate_budget(cmd: str) -> int:
    return int(cmd.split("step_gate.py ")[1].split()[0])


def test_a_resume_is_given_the_budget_its_chain_has_not_spent():
    adapter = _ScriptedClaudeAdapter(
        [(0, _aborted_stream(16, "a")),
         (0, _aborted_stream(10, "b")),
         (0, _aborted_stream(4, "c"))],
        certs=False,
    )
    adapter.RETRY_WAIT_SECONDS = 0
    result = _drive(adapter)

    assert adapter.calls == 3
    assert "--resume" not in adapter.cmds[0]
    assert all("--resume" in c for c in adapter.cmds[1:])
    assert [_gate_budget(c) for c in adapter.cmds] == [50, 34, 24]
    assert result.cost["successful_llm_calls"] == 30


def test_a_fresh_retry_gets_the_whole_budget_back():
    adapter = _ScriptedClaudeAdapter(
        [(0, _aborted_stream(16, "a")),
         (0, _aborted_stream(10, "b")),
         (0, _aborted_stream(4, "c"))],
        certs=False,
    )
    adapter.RETRY_WAIT_SECONDS = 0
    adapter._extract_session_id = lambda raw_output: None
    _drive(adapter)

    assert [_gate_budget(c) for c in adapter.cmds] == [50, 50, 50]
    assert all("--resume" not in c for c in adapter.cmds)


def test_a_chain_that_overran_its_budget_is_not_resumed_again():
    adapter = _ScriptedClaudeAdapter(
        [(0, _aborted_stream(12, "a")), (0, _aborted_stream(4, "b"))],
        certs=False,
    )
    adapter.max_steps = 10
    adapter.RETRY_WAIT_SECONDS = 0
    _drive(adapter)

    assert adapter.calls == 1


class _ScriptedOpenCodeAdapter(OpenCodeAdapter):
    def __init__(self, outputs: List[Tuple[int, str]]):
        super().__init__(
            name="opencode", model="claude-opus-5", tool_version="1.18.25",
            agent_config={}, env_vars={}, max_steps=50,
        )
        self._outputs = list(outputs)
        self.calls = 0

    def setup_container(self, container, task, prompt_text):
        return True

    def _run_cli(self, container, cmd, timeout):
        if "opencode run" not in cmd:
            return 0, b""
        self.calls += 1
        exit_code, text = self._outputs[self.calls - 1]
        return exit_code, text.encode("utf-8")

    def _parse_output(self, container, raw_output):
        return {}, {"input_tokens": 0, "output_tokens": 0,
                    "total_cost_usd": 0.0}

    def _extract_cost(self, container):
        return None


def test_a_capped_run_is_not_retried_even_when_it_reads_as_a_dropped_stream():
    stream = "\n".join([
        json.dumps({"type": "step_finish", "sessionID": "s",
                    "part": {"reason": "tool-calls"}}),
        json.dumps({"type": "step_finish", "sessionID": "s",
                    "part": {"reason": "tool-calls"}}),
        json.dumps({"type": step_gate.STEP_CAP_EVENT_TYPE, "unit": "opencode",
                    "limit": 2, "count": 2}),
    ])
    adapter = _ScriptedOpenCodeAdapter(
        [(step_gate.STEP_CAP_EXIT_CODE, stream)] * 3
    )
    adapter.RETRY_WAIT_SECONDS = 0
    result = _drive(adapter)

    assert adapter.calls == 1
    assert [e.error_type for e in result.llm_errors] == ["stream_abort"]


def test_an_uncapped_run_records_no_cap():
    stream = "\n".join([
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}),
        json.dumps({"type": "assistant", "session_id": "s", "message": {
            "id": "msg_0", "content": [{"type": "tool_use"}]}}),
        json.dumps({"type": "result", "session_id": "s", "num_turns": 2,
                    "result": "done"}),
    ])
    adapter = _ScriptedClaudeAdapter([(0, stream)], certs=True)
    result = _drive(adapter)

    assert result.completed_normally is True
    assert result.attempts[0]["step_cap"] is None


def test_no_wall_clock_is_not_read_as_a_timeout():
    stream = "\n".join([
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}),
        json.dumps({"type": "result", "session_id": "s", "num_turns": 1,
                    "result": "done"}),
    ])
    adapter = _ScriptedClaudeAdapter([(0, stream)], certs=True)
    result = _drive(adapter, timeout=None)

    assert result.completed_normally is True


def _prompt_env():
    import jinja2
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            str(Path(__file__).resolve().parents[1] / "prompts")
        ),
        undefined=jinja2.StrictUndefined,
    )


def _render(harness_form, max_agent_steps):
    from eval_pipeline.models import SurvivingSpec
    from eval_pipeline.stages.stage2_baseline_execution import _render_prompt

    spec = SurvivingSpec(
        spec={"spec_id": "S1", "category": "behavioral", "text": "t",
              "sources": [], "code_mapping": {}},
        mutant_patch="",
        case_contract="case is a stimulus dict",
    )
    task = EvalTask(
        instance_id="repo__proj.abc.test_x.def.lv1", image_name="img",
        surviving_specs=[spec], f2p_tests=[], p2p_tests=[], repo_settings={},
    )
    return _render_prompt(
        _prompt_env(), task, harness_form=harness_form,
        max_agent_steps=max_agent_steps,
    )


@pytest.mark.parametrize("harness_form", [
    "form1b_form6", "hypofuzz", "saved_inputs", "factory_oracle",
])
def test_a_capped_run_tells_the_agent_its_budget(harness_form):
    prompt = _render(harness_form, 50)

    assert "## Step budget" in prompt
    assert "You have 50 steps" in prompt
    assert "/tmp/eval_output.json` holds at that moment" in prompt


@pytest.mark.parametrize("harness_form", [
    "form1b_form6", "hypofuzz", "saved_inputs", "factory_oracle",
])
def test_an_uncapped_run_says_nothing_about_a_budget(harness_form):
    assert "Step budget" not in _render(harness_form, None)


def test_the_prompt_follows_the_cap_that_is_actually_enforced():
    class _Uncountable(base_mod.EvalBaselineAdapter):
        def get_install_script(self):
            return ""

        def get_run_command(self, prompt_path):
            return self._piped_cli("mytool run", "/tmp/out.txt")

    countable = ClaudeCodeAdapter(
        name="claude_code", model="m", tool_version="v",
        agent_config={}, env_vars={}, max_steps=50,
    )
    uncountable = _Uncountable(
        name="mytool", model="m", tool_version="v",
        agent_config={}, env_vars={}, max_steps=50,
    )

    assert countable.enforced_max_steps() == 50
    assert uncountable.enforced_max_steps() is None
    assert "Step budget" not in _render(
        "factory_oracle", uncountable.enforced_max_steps()
    )


def _load_shim(monkeypatch, tmp_path, name, upstream="http://upstream.invalid/v1",
               max_calls=0):
    import importlib.util

    monkeypatch.setenv("UPSTREAM_BASE_URL", upstream)
    monkeypatch.setenv("LISTEN_PORT", "0")
    monkeypatch.setenv("TOOL_FILTER_AUDIT", str(tmp_path / "audit.log"))
    monkeypatch.setenv("MAX_MODEL_CALLS", str(max_calls))
    from eval_pipeline.adapters import codex as codex_mod
    spec = importlib.util.spec_from_file_location(name, codex_mod.TOOL_FILTER_SRC)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    return shim


def test_the_audit_segment_prefix_agrees_across_the_container_boundary(
    monkeypatch, tmp_path
):
    from eval_pipeline.adapters import codex as codex_mod
    shim = _load_shim(monkeypatch, tmp_path, "_htf_prefix")
    assert (
        shim.AUDIT_SEGMENT_PREFIX
        == step_gate.AUDIT_SEGMENT_PREFIX
        == codex_mod.TOOL_FILTER_SEGMENT_PREFIX
    )


def _audit_line(ok, requests=None, refused=0):
    requests = ok if requests is None else requests
    return (
        f"stripped 1 hosted tool(s) from POST /v1/responses "
        f"(total {requests} in {requests}/{requests} requests; "
        f"ok={ok} failed={requests - ok}; refused={refused})"
    )


def test_the_wire_count_is_read_from_the_last_shim_segment():
    text = "\n".join([
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u (stripping: [])",
        _audit_line(79), _audit_line(80),
        "step_cap_reached limit=80 (total 80 in 80/80 requests; ok=80 failed=0; refused=0)",
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u (stripping: []; budget: 80 model calls)",
    ])
    assert step_gate._wire_count_shim(text) is None
    assert step_gate._wire_count_shim(text + "\n" + _audit_line(3)) == 3
    assert step_gate._wire_count_shim(
        text + "\n" + _audit_line(3) + "\nupstream error for POST /v1/responses: x"
    ) == 3


def _fake_codex_cli(tmp_path: Path, audit: Path, calls: int, idle: bool) -> Path:
    script = tmp_path / "fake_codex.sh"
    tail = "while true; do sleep 1; done" if idle else "exit 0"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        for i in $(seq 1 {calls}); do
            echo "stripped 1 hosted tool(s) from POST /v1/responses (total $i in $i/$i requests; ok=$i failed=0; refused=0)" >> {audit}
            printf '{{"type":"item.completed","item":{{"id":"item_%s","type":"reasoning","text":"x"}}}}\\n' "$i"
            sleep 0.05
        done
        {tail}
    """))
    script.chmod(0o755)
    return script


def _run_wire_gated_pipeline(tmp_path: Path, max_steps: int, calls: int, idle: bool):
    audit = tmp_path / "audit.log"
    audit.write_text(
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u "
        f"(stripping: []; budget: {max_steps} model calls)\n"
    )
    cli = _fake_codex_cli(tmp_path, audit, calls, idle)
    transcript = tmp_path / "transcript.txt"
    pid_file = tmp_path / "agent.pid"
    cmd = (
        f"( {cli} 2>&1 & echo $! > {pid_file}; wait $! ) "
        f"| tee {transcript} "
        f"| {sys.executable} {GATE} {max_steps} codex_wire {pid_file} {audit}"
    )
    started = time.time()
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=90,
    )
    return proc, transcript, pid_file, time.time() - started


def test_the_wire_gate_stops_a_codex_that_would_run_forever(tmp_path):
    proc, transcript, pid_file, elapsed = _run_wire_gated_pipeline(
        tmp_path, max_steps=4, calls=20, idle=True,
    )

    assert proc.returncode == step_gate.STEP_CAP_EXIT_CODE
    assert elapsed < 30

    import os
    pid = int(pid_file.read_text().strip())
    with pytest.raises(OSError):
        os.kill(pid, 0)

    marker = [
        json.loads(line) for line in proc.stdout.splitlines()
        if line.startswith("{") and "step_cap_reached" in line
    ]
    assert len(marker) == 1
    assert marker[0]["type"] == "step_cap_reached"
    assert marker[0]["unit"] == "codex_wire"
    assert marker[0]["limit"] == 4
    assert 4 <= marker[0]["count"] <= 20
    assert "step_cap_reached" not in transcript.read_text()


def test_a_codex_that_exits_on_the_refusal_is_still_recorded_as_capped(tmp_path):
    proc, transcript, _, _ = _run_wire_gated_pipeline(
        tmp_path, max_steps=5, calls=5, idle=False,
    )

    assert proc.returncode == step_gate.STEP_CAP_EXIT_CODE
    marker = [
        json.loads(line) for line in proc.stdout.splitlines()
        if line.startswith("{") and "step_cap_reached" in line
    ]
    assert len(marker) == 1
    assert marker[0]["count"] == 5
    assert "step_cap_reached" not in transcript.read_text()


def test_a_codex_that_finishes_under_the_wire_cap_is_untouched(tmp_path):
    proc, transcript, _, _ = _run_wire_gated_pipeline(
        tmp_path, max_steps=50, calls=3, idle=False,
    )

    assert proc.returncode == 0
    assert "step_cap_reached" not in proc.stdout
    assert proc.stdout == transcript.read_text()


def test_a_wire_unit_refuses_to_run_without_its_log():
    proc = subprocess.run(
        [sys.executable, GATE, "5", "codex_wire", "/tmp/nonexistent.pid"],
        input="", capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2
    assert "audit log" in proc.stderr


def test_a_capped_codex_command_restarts_the_shim_with_the_budget():
    adapter = _adapter(
        CodexAdapter, max_steps=50, OPENAI_BASE_URL="https://host/openai/v1",
    )
    cmd = adapter.get_run_command("/tmp/eval_prompt.txt")

    assert 'pkill -f "[h]osted_tool_filter.py"' in cmd
    assert "export MAX_MODEL_CALLS=50; " in cmd
    assert cmd.index("MAX_MODEL_CALLS=50") < cmd.index("start_tool_filter.sh")
    assert cmd.index("start_tool_filter.sh") < cmd.index("codex exec")
    assert (
        f"step_gate.py 50 codex_wire {base_mod.AGENT_PID_PATH} "
        f"/tmp/hosted_tool_filter.log" in cmd
    )


def test_an_uncapped_codex_command_runs_the_shim_without_a_budget():
    adapter = _adapter(CodexAdapter, OPENAI_BASE_URL="https://host/openai/v1")
    cmd = adapter.get_run_command("/tmp/eval_prompt.txt")

    assert "unset MAX_MODEL_CALLS; " in cmd
    assert "MAX_MODEL_CALLS=" not in cmd
    assert "step_gate" not in cmd


def test_a_codex_without_a_shim_falls_back_to_the_item_estimator():
    adapter = _adapter(CodexAdapter, max_steps=50)
    assert adapter.step_gate_unit() == "codex"
    assert adapter.step_gate_source() is None
    cmd = adapter.get_run_command("/tmp/eval_prompt.txt")
    assert f"step_gate.py 50 codex {base_mod.AGENT_PID_PATH}" in cmd
    assert "hosted_tool_filter.log" not in cmd


def test_every_unit_an_adapter_can_declare_is_known_to_the_gate():
    wired = _adapter(CodexAdapter, max_steps=50, OPENAI_BASE_URL="https://h/v1")
    assert wired.step_gate_unit() == "codex_wire"
    assert wired.step_gate_unit() in step_gate.WIRE_UNITS
    for cls in (ClaudeCodeAdapter, CodexAdapter, OpenCodeAdapter):
        assert _adapter(cls, max_steps=50).step_gate_unit() in step_gate.KNOWN_UNITS


def test_a_stream_is_whole_only_once_its_terminal_event_has_passed(
    monkeypatch, tmp_path
):
    shim = _load_shim(monkeypatch, tmp_path, "_htf_watch")

    watch = shim.StreamWatch()
    watch.feed(b'event: response.created\ndata: {"type":"response.created"}\n\n')
    assert not watch.whole
    watch.feed(b'event: response.completed\ndata: {"type":"respo')
    assert not watch.whole
    watch.feed(b'nse.completed","response":{"id":"r"}}\n\n')
    assert watch.whole

    watch = shim.StreamWatch()
    watch.feed(b'data: {"type":"response.output_text.delta","delta":"see \\"type\\":\\"response.completed\\""}\n\n')
    assert not watch.whole

    watch = shim.StreamWatch()
    watch.feed(b'data: {"type": "response.completed", "response": {}}\n\n')
    assert watch.whole

    watch = shim.StreamWatch()
    watch.feed(b'data: {"type":"response.failed","response":{}}\n\n')
    assert watch.outcome == "failed" and not watch.whole

    watch = shim.StreamWatch()
    watch.feed(b"data: [DONE]\n\n")
    assert watch.whole


def _fake_upstream(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        seen: List[bytes] = []

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            Upstream.seen.append(body)
            if b'"drop"' in body:
                payload = b'event: response.created\ndata: {"type":"response.created"}\n\n'
            else:
                payload = (
                    b'event: response.created\ndata: {"type":"response.created"}\n\n'
                    b'event: response.completed\ndata: {"type":"response.completed","response":{"output":[]}}\n\n'
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.daemon_threads = True
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Upstream


def _serve(shim):
    import threading
    server = shim.ThreadingHTTPServer(("127.0.0.1", 0), shim.Handler)
    server.daemon_threads = True
    shim.open_segment(server.server_address[1])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _settle(shim, **expected):
    deadline = time.time() + 5
    while time.time() < deadline:
        if all(shim._stats[key] == value for key, value in expected.items()):
            return
        time.sleep(0.01)
    assert {k: shim._stats[k] for k in expected} == expected


def _call(port, body: bytes):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/responses", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_shim_refuses_the_call_after_the_budget(monkeypatch, tmp_path):
    upstream, Upstream = _fake_upstream(tmp_path)
    shim = _load_shim(
        monkeypatch, tmp_path, "_htf_cap",
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        max_calls=2,
    )
    server = _serve(shim)
    port = server.server_address[1]
    body = json.dumps({"tools": [{"type": "web_search"}], "input": "x"}).encode()
    try:
        status, _ = _call(port, json.dumps({"drop": True}).encode())
        assert status == 200
        _settle(shim, failed=1, ok=0)

        status, out = _call(port, body)
        assert status == 200 and b"response.completed" in out
        status, out = _call(port, body)
        assert status == 200 and b"response.completed" in out
        _settle(shim, ok=2)

        status, out = _call(port, body)
        assert status == 403
        assert json.loads(out)["error"]["code"] == "step_cap_reached"
        assert len(Upstream.seen) == 3
        assert shim._stats["refused"] == 1
        import urllib.request
        with pytest.raises(Exception):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10)
    finally:
        server.shutdown()
        upstream.shutdown()

    audit = (tmp_path / "audit.log").read_text().splitlines()
    assert audit[0].startswith(shim.AUDIT_SEGMENT_PREFIX)
    assert "budget: 2 model calls" in audit[0]
    assert any(line.startswith("step_cap_reached limit=2") for line in audit)
    assert any(line.startswith("refused POST /v1/responses") for line in audit)
    assert step_gate._wire_count_shim("\n".join(audit)) == 2
    assert all(b"web_search" not in seen for seen in Upstream.seen)


def test_the_shim_without_a_budget_never_refuses(monkeypatch, tmp_path):
    upstream, Upstream = _fake_upstream(tmp_path)
    shim = _load_shim(
        monkeypatch, tmp_path, "_htf_nocap",
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        max_calls=0,
    )
    server = _serve(shim)
    port = server.server_address[1]
    try:
        for _ in range(4):
            status, _ = _call(port, b'{"input":"x"}')
            assert status == 200
        _settle(shim, ok=4, refused=0)
    finally:
        server.shutdown()
        upstream.shutdown()
    audit = (tmp_path / "audit.log").read_text()
    assert "step_cap_reached" not in audit
    assert "budget" not in audit.splitlines()[0]


def test_the_adapter_reads_the_last_segment_of_the_audit_log():
    adapter = _adapter(CodexAdapter, max_steps=50, OPENAI_BASE_URL="https://h/v1")
    text = "\n".join([
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u (stripping: [])",
        _audit_line(80), 
        "step_cap_reached limit=80 (total 80 in 80/80 requests; ok=80 failed=0; refused=0)",
        "refused POST /v1/responses: step cap limit=80 reached (total 80 in 80/80 requests; ok=80 failed=0; refused=1)",
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u (stripping: []; budget: 80 model calls)",
        _audit_line(12, requests=13),
        "upstream error for POST /v1/responses: timeout",
    ])
    last = adapter._last_counted_audit_line(text)
    assert last == _audit_line(12, requests=13)

    container = mock.MagicMock()
    with mock.patch(
        "eval_pipeline.adapters.tool_filter.exec_run_with_timeout",
        return_value=(0, text.encode("utf-8")),
    ):
        counters = adapter._log_tool_filter_audit(container)
    assert counters == {
        "api_requests": 13, "successful_llm_calls": 12, "refused_llm_calls": 0,
    }


def test_the_adapter_reports_calls_the_shim_turned_away():
    adapter = _adapter(CodexAdapter, max_steps=2, OPENAI_BASE_URL="https://h/v1")
    text = "\n".join([
        "hosted_tool_filter listening on 127.0.0.1:8123 -> http://u (stripping: []; budget: 2 model calls)",
        _audit_line(2),
        "step_cap_reached limit=2 (total 2 in 2/2 requests; ok=2 failed=0; refused=0)",
        "refused POST /v1/responses: step cap limit=2 reached (total 2 in 2/2 requests; ok=2 failed=0; refused=1)",
    ])
    with mock.patch(
        "eval_pipeline.adapters.tool_filter.exec_run_with_timeout",
        return_value=(0, text.encode("utf-8")),
    ):
        counters = adapter._log_tool_filter_audit(mock.MagicMock())
    assert counters == {
        "api_requests": 2, "successful_llm_calls": 2, "refused_llm_calls": 1,
    }


def test_the_shims_refusal_is_not_read_as_an_llm_error():
    adapter = _adapter(CodexAdapter, max_steps=5, OPENAI_BASE_URL="https://h/v1")
    raw = "\n".join([
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hi"}}',
        "step_gate: cannot read /tmp/hosted_tool_filter.log after 10s; the cap cannot fire until it can",
        '{"type":"error","message":"Reconnecting... 1/5 (unexpected status 403 Forbidden: '
        "step cap reached: this run's budget of 5 model calls is spent, url: http://127.0.0.1:8123/v1/responses)\"}",
        "step_gate: codex_wire reached 5/5 model calls; stopping the agent",
        '{"type": "step_cap_reached", "unit": "codex_wire", "limit": 5, "count": 5}',
    ])
    assert adapter._detect_llm_errors(raw) == []

    genuine = raw.replace("step cap reached: this run's budget of 5 model calls is spent",
                          "invalid api key")
    assert [e.error_type for e in adapter._detect_llm_errors(genuine)] == [
        "auth_error", "stream_disconnect",
    ]


def test_the_shim_counts_the_messages_wire_and_ends_on_message_stop(
    monkeypatch, tmp_path
):
    shim = _load_shim(monkeypatch, tmp_path, "_htf_messages")

    assert shim.is_model_invocation("POST", "/v1/messages")
    assert shim.is_model_invocation("POST", "/anthropic/v1/messages?beta=true")
    assert not shim.is_model_invocation("POST", "/v1/messages/count_tokens")
    assert not shim.is_model_invocation("GET", "/v1/messages")

    watch = shim.StreamWatch()
    watch.feed(b'event: message_start\ndata: {"type":"message_start","message":{}}\n\n')
    watch.feed(b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n')
    assert not watch.whole
    watch.feed(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    assert watch.whole and watch.outcome == "message_stop"


def test_the_audit_line_says_how_many_tools_the_request_offered(
    monkeypatch, tmp_path
):
    shim = _load_shim(monkeypatch, tmp_path, "_htf_tools")

    body, stripped, tools = shim.filter_body(json.dumps({
        "tools": [{"type": "function", "name": "bash"},
                  {"type": "web_search"},
                  {"type": "function", "name": "read"}],
    }).encode())
    assert (stripped, tools) == (1, 2)
    assert b"web_search" not in body

    body, stripped, tools = shim.filter_body(b'{"messages":[]}')
    assert (stripped, tools) == (0, 0)

    shim._record_outcome("POST", "/v1/chat/completions", 0, ok=True, tools=9)
    shim._record_outcome("POST", "/v1/chat/completions", 0, ok=True, tools=0)
    lines = (tmp_path / "audit.log").read_text().splitlines()
    assert "(tools=9; " in lines[-2]
    assert "(tools=0; " in lines[-1]
