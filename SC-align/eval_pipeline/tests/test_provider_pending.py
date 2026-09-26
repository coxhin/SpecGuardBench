import json
import sys
from pathlib import Path
from unittest import mock

from . import test_attempt_recording as _adapters
from . import test_concurrency as _stage2

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.adapters.base import SUCCESSFUL_LLM_CALLS
from eval_pipeline.config import BaselineConfig, EvalPipelineConfig
from eval_pipeline.egress import ApiOnlyEgress
from eval_pipeline.models import BaselineResult, EvalTask, SurvivingSpec
from eval_pipeline.stages.stage5_metrics import _aggregate_agent_steps
from eval_pipeline.trajectory import session_chains

_claude_stream = _adapters._claude_stream
_run_scripted = _adapters._run_scripted
_run_scripted_codex = _adapters._run_scripted_codex


def _aborted(session="sess-A"):
    return _claude_stream(session, turns=5, out_tokens=100, aborted=True)


def _drive_resume(adapter, task, resume_from):
    commands = []

    def _exec(container, cmd, timeout=None):
        if "claude -p" in cmd or "codex exec" in cmd:
            commands.append(cmd)
        return adapter._run_cli(container, cmd, timeout)

    with mock.patch(
        "eval_pipeline.adapters.base.exec_run_with_timeout", side_effect=_exec,
    ), mock.patch.object(adapter, "_write_file_to_container"), mock.patch.object(
        adapter, "_log_tool_filter_audit", create=True
    ):
        result = adapter.run(
            mock.MagicMock(short_id="abc"), task, "prompt", 600,
            resume_from=resume_from,
        )
    return result, commands


def test_exhausted_provider_retries_mark_the_result_pending():
    result = _run_scripted([(0, _aborted())] * 3, certs_after=99)

    assert result.certificates == {}
    assert result.completed_normally is False
    pending = result.provider_pending
    assert pending is not None
    assert pending["session_id"] == "sess-A"
    assert pending["error_types"] == ["stream_abort"]
    assert pending["container_id"] is None
    assert [a["attempt"] for a in result.attempts] == [1, 2, 3]
    assert pending["chain_steps"] == sum(
        a["steps"][SUCCESSFUL_LLM_CALLS] for a in result.attempts
    )
    assert pending["wall_time_seconds"] == result.wall_time_seconds


def test_a_run_that_gives_up_without_a_provider_error_is_an_outcome():
    clean = _claude_stream("sess-A", turns=5, out_tokens=100)
    result = _run_scripted([(0, clean)], certs_after=99)

    assert result.certificates == {}
    assert result.provider_pending is None


def test_a_certified_run_is_never_pending_whatever_came_before():
    result = _run_scripted(
        [(0, _aborted()), (0, _claude_stream("sess-A", turns=30, out_tokens=2000))],
        certs_after=2,
    )

    assert result.certificates
    assert result.provider_pending is None


def test_a_tool_that_cannot_resume_is_still_marked_pending():
    failed = json.dumps({
        "type": "error",
        "message": "Selected model is at capacity, try a different model",
    })
    result = _run_scripted_codex([(0, failed)] * 3, certs_after=99)

    assert result.provider_pending is not None
    assert result.provider_pending["error_types"] == ["capacity"]
    assert all(a["resumed"] is False for a in result.attempts)


def test_provider_pending_survives_a_json_round_trip():
    result = _run_scripted([(0, _aborted())] * 3, certs_after=99)
    restored = BaselineResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.provider_pending == result.provider_pending

    result.provider_pending = None
    assert "provider_pending" not in result.to_dict()
    assert BaselineResult.from_dict(result.to_dict()).provider_pending is None


def test_resume_from_continues_the_chain_across_invocations():
    first = _run_scripted([(0, _aborted())] * 3, certs_after=99)

    adapter = _adapters._ScriptedClaudeAdapter(
        [(0, _claude_stream("sess-A", turns=30, out_tokens=2000))], certs_after=1
    )
    adapter.RETRY_WAIT_SECONDS = 0
    result, commands = _drive_resume(
        adapter, _adapters._make_task("repo__proj.abc.test_x.def.lv1"),
        {**first.provider_pending, "attempts": first.attempts},
    )

    assert result.certificates
    assert result.provider_pending is None
    assert "--resume sess-A" in commands[0]
    assert [a["attempt"] for a in result.attempts] == [1, 2, 3, 4]
    assert result.attempts[3]["resumed"] is True
    assert result.attempts[3]["resume_of_session"] == "sess-A"
    assert len(session_chains(result.attempts)) == 1
    assert result.cost["output_tokens"] == 3 * 100 + 2000
    assert result.wall_time_seconds >= first.wall_time_seconds


def test_a_second_loss_carries_the_whole_chain_forward():
    first = _run_scripted([(0, _aborted())] * 3, certs_after=99)

    adapter = _adapters._ScriptedClaudeAdapter([(0, _aborted())] * 3, certs_after=99)
    adapter.RETRY_WAIT_SECONDS = 0
    result, _ = _drive_resume(
        adapter, _adapters._make_task("repo__proj.abc.test_x.def.lv1"),
        {**first.provider_pending, "attempts": first.attempts},
    )

    assert result.provider_pending is not None
    assert [a["attempt"] for a in result.attempts] == [1, 2, 3, 4, 5, 6]
    assert result.provider_pending["chain_steps"] == sum(
        a["steps"][SUCCESSFUL_LLM_CALLS] for a in result.attempts
    )
    assert result.provider_pending["wall_time_seconds"] > first.wall_time_seconds


def _executor(tmp_path, skip_existing=False, docker_runner=None):
    return _stage2.TestBaselineExecutorConcurrency()._make_executor(
        _stage2._make_config(tmp_path, n_concurrent=1, skip_existing=skip_existing),
        docker_runner=docker_runner,
    )


def _pending_result(
    iid="inst1", container_id="cid-1", baseline="claude_code", model="opus",
    **overrides,
) -> BaselineResult:
    result = _stage2._make_baseline_result(iid, baseline=baseline, model=model)
    result.completed_normally = False
    result.attempts = [{"attempt": 1, "session_id": "sess-A", "resumed": False,
                        "resume_of_session": None, "exit_code": 0,
                        "raw_output": "...", "steps": {}}]
    result.provider_pending = {
        "session_id": "sess-A", "chain_steps": 7, "error_types": ["server_error"],
        "wall_time_seconds": 30.0, "container_id": container_id,
        "container_name": "fb-eval-inst1", "spec_id": None,
        **overrides,
    }
    return result


def _write(executor, task, bcfg, result) -> Path:
    path = executor._result_path(task, bcfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return path


def _index(tmp_path) -> dict:
    return json.loads(
        (tmp_path / "baseline_output" / "provider_pending.json").read_text(encoding="utf-8")
    )


def test_skip_existing_resumes_a_result_lost_to_the_provider(tmp_path):
    executor = _executor(tmp_path, skip_existing=True)
    task = _stage2._make_task("inst1")
    bcfg = _stage2._make_baseline_cfg()
    path = _write(executor, task, bcfg, _pending_result())

    seen = {}

    def run_single(t, b):
        seen["pending"] = dict(executor._pending)
        return _stage2._make_baseline_result("inst1", certificates={"S1": _stage2._cert()})

    executor._run_single = mock.MagicMock(side_effect=run_single)
    results = executor.run_all([task], [bcfg])

    executor._run_single.assert_called_once()
    assert seen["pending"][str(path)].provider_pending["session_id"] == "sess-A"
    assert results[0].certificates
    assert _index(tmp_path)["n_pending"] == 0


def test_without_skip_existing_a_pending_result_is_simply_rerun(tmp_path):
    executor = _executor(tmp_path, skip_existing=False)
    task = _stage2._make_task("inst1")
    bcfg = _stage2._make_baseline_cfg()
    _write(executor, task, bcfg, _pending_result())

    executor._run_single = mock.MagicMock(
        return_value=_stage2._make_baseline_result("inst1")
    )
    executor.run_all([task], [bcfg])

    executor._run_single.assert_called_once()
    assert executor._pending == {}


def test_the_index_lists_every_result_still_waiting_across_baselines(tmp_path):
    executor = _executor(tmp_path)
    claude = _stage2._make_baseline_cfg("claude_code", "opus")
    codex = _stage2._make_baseline_cfg("codex", "gpt")
    _write(executor, _stage2._make_task("inst1"), claude,
           _pending_result("inst1", spec_id="S3"))
    _write(executor, _stage2._make_task("inst2"), claude,
           _stage2._make_baseline_result("inst2", certificates={"S1": _stage2._cert()}))
    _write(executor, _stage2._make_task("inst1"), codex,
           _pending_result("inst1", container_id="cid-2", baseline="codex", model="gpt",
                           error_types=["rate_limit"]))

    executor._write_provider_pending()

    index = _index(tmp_path)
    assert index["n_pending"] == 2
    rows = {(e["baseline"], e["instance_id"]): e for e in index["pending"]}
    assert rows[("claude_code", "inst1")]["spec_id"] == "S3"
    assert rows[("claude_code", "inst1")]["container_id"] == "cid-1"
    assert rows[("claude_code", "inst1")]["session_id"] == "sess-A"
    assert rows[("claude_code", "inst1")]["attempts"] == 1
    assert rows[("codex", "inst1")]["error_types"] == ["rate_limit"]
    assert rows[("codex", "inst1")]["result_path"].startswith("baseline_output/codex_gpt/")


def test_an_empty_index_is_written_when_nothing_is_waiting(tmp_path):
    executor = _executor(tmp_path)
    executor._run_single = mock.MagicMock(
        return_value=_stage2._make_baseline_result("inst1")
    )
    executor.run_all([_stage2._make_task("inst1")], [_stage2._make_baseline_cfg()])
    assert _index(tmp_path) == {**_index(tmp_path), "n_pending": 0, "pending": []}


def _spec_task() -> EvalTask:
    return EvalTask(
        instance_id="example.instance.lv1",
        image_name="example-image",
        surviving_specs=[SurvivingSpec(
            spec={"spec_id": "S1", "category": "behavioral", "text": "t"},
            mutant_patch="S1.mutant.patch",
        )],
        f2p_tests=[], p2p_tests=[], repo_settings={},
        mask_patch="mask.patch", gold_patch="gold.patch",
    )


def _spec_executor(tmp_path, docker_runner):
    from eval_pipeline.stages.stage2_baseline_execution import BaselineExecutor

    config = EvalPipelineConfig(output_dir=tmp_path, harness_form="factory_oracle")
    loader = mock.MagicMock()
    loader.get_agent_env_vars.return_value = {}
    return BaselineExecutor(config, loader, docker_runner)


_BCFG = BaselineConfig(
    name="claude_code", model="opus", agent_config="claude_code", tool_version="1.0"
)


def _container(cid="cid-1"):
    container = mock.MagicMock(id=cid, short_id=cid[:4])
    container.name = f"fb-eval-{cid}"
    return container


def _drive_single(executor, task, adapter):
    with mock.patch(
        "eval_pipeline.stages.stage2_baseline_execution._create_adapter",
        return_value=adapter,
    ), mock.patch(
        "eval_pipeline.stages.stage2_baseline_execution.prepare_task_code_state",
        return_value=(True, None),
    ) as prepare:
        result = executor._run_single(task, _BCFG)
    return result, prepare


def test_run_single_keeps_the_container_of_a_run_lost_to_the_provider(tmp_path):
    docker_runner = mock.MagicMock()
    container = _container()
    docker_runner.setup_instance.return_value = container
    executor = _spec_executor(tmp_path, docker_runner)
    task = _spec_task()

    lost = _stage2._make_baseline_result(task.instance_id)
    lost.provider_pending = {
        "session_id": "sess-A", "chain_steps": 7, "error_types": ["server_error"],
        "wall_time_seconds": 30.0, "container_id": None, "container_name": None,
        "spec_id": None,
    }
    adapter = mock.MagicMock()
    adapter.run.return_value = lost
    adapter.enforced_max_steps.return_value = None

    result, _ = _drive_single(executor, task, adapter)

    docker_runner.cleanup.assert_not_called()
    assert result.provider_pending["container_id"] == "cid-1"
    assert result.provider_pending["container_name"] == "fb-eval-cid-1"
    assert result.provider_pending["spec_id"] == "S1"
    assert executor._kept == [container]
    assert [e.error_type for e in result.llm_errors] == []


def test_run_single_resumes_inside_the_kept_container(tmp_path):
    docker_runner = mock.MagicMock()
    container = _container()
    docker_runner.get_container.return_value = container
    executor = _spec_executor(tmp_path, docker_runner)
    executor._egress = mock.MagicMock()
    task = _spec_task()

    pending = _pending_result(task.instance_id, spec_id="S1")
    executor._pending[str(executor._result_path(task, _BCFG))] = pending
    adapter = mock.MagicMock()
    adapter.run.return_value = _stage2._make_baseline_result(
        task.instance_id, certificates={"S1": _stage2._cert()}
    )
    adapter.enforced_max_steps.return_value = None

    result, prepare = _drive_single(executor, task, adapter)

    docker_runner.get_container.assert_called_once_with("cid-1")
    docker_runner.setup_instance.assert_not_called()
    prepare.assert_not_called()
    executor._egress.restrict.assert_called_once_with(container)
    resume_from = adapter.run.call_args.kwargs["resume_from"]
    assert resume_from["session_id"] == "sess-A"
    assert resume_from["chain_steps"] == 7
    assert resume_from["attempts"] == pending.attempts
    assert result.certificates
    docker_runner.cleanup.assert_called_once_with(container)
    assert executor._kept == []
    assert executor._pending == {}


def test_a_kept_container_that_is_gone_starts_fresh(tmp_path):
    docker_runner = mock.MagicMock()
    docker_runner.get_container.return_value = None
    fresh = _container("cid-9")
    docker_runner.setup_instance.return_value = fresh
    executor = _spec_executor(tmp_path, docker_runner)
    task = _spec_task()
    executor._pending[str(executor._result_path(task, _BCFG))] = _pending_result(
        task.instance_id
    )
    adapter = mock.MagicMock()
    adapter.run.return_value = _stage2._make_baseline_result(
        task.instance_id, certificates={"S1": _stage2._cert()}
    )
    adapter.enforced_max_steps.return_value = None

    _, prepare = _drive_single(executor, task, adapter)

    docker_runner.setup_instance.assert_called_once()
    prepare.assert_called_once()
    assert adapter.run.call_args.kwargs["resume_from"] is None
    docker_runner.cleanup.assert_called_once_with(fresh)


def test_kept_containers_leave_the_network_before_it_is_removed(tmp_path):
    executor = _executor(tmp_path)
    container = _container()
    egress = mock.MagicMock()
    order = []
    egress.release.side_effect = lambda c: order.append(("release", c.id))
    egress.stop.side_effect = lambda **kw: order.append(("stop", None))
    executor._start_egress = mock.MagicMock(return_value=egress)

    def run_single(task, bcfg):
        with executor._kept_lock:
            executor._kept.append(container)
        return _pending_result(task.instance_id)

    executor._run_single = mock.MagicMock(side_effect=run_single)
    executor.run_all([_stage2._make_task("inst1")], [_stage2._make_baseline_cfg()])

    assert order == [("release", "cid-1"), ("stop", None)]
    assert _index(tmp_path)["n_pending"] == 1
    assert _index(tmp_path)["pending"][0]["container_id"] == "cid-1"


def test_release_returns_a_kept_container_to_the_bridge():
    client = mock.MagicMock()
    egress = ApiOnlyEgress(["api.example.com"], run_id="r1", client=client)
    proxy = client.containers.create.return_value
    proxy.logs.return_value = b"listening on 0.0.0.0:3128; allowed hosts: api.example.com\n"
    with mock.patch("eval_pipeline.egress.ApiOnlyEgress._ship_proxy_script"):
        egress.start()
    container = mock.MagicMock(id="abc123", short_id="abc")
    egress.restrict(container)
    assert egress.is_restricted(container) is True

    egress.release(container)

    client.networks.create.return_value.disconnect.assert_called_once_with(
        container, force=True
    )
    client.networks.get.return_value.connect.assert_called_with(container)
    assert egress.is_restricted(container) is False


def test_runs_lost_to_the_provider_are_counted():
    lost = _stage2._make_baseline_result("inst1")
    lost.provider_pending = {"session_id": "s", "error_types": ["server_error"]}
    counts = _aggregate_agent_steps([
        lost,
        _stage2._make_baseline_result("inst2"),
        _stage2._make_baseline_result("inst3"),
    ])["counts"]

    assert counts["runs_lost_to_provider"] == 1
    assert counts["runs"] == 3
