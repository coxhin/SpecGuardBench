import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Dict
from unittest import mock

import pytest


sys.modules.setdefault("mutation_pipeline", ModuleType("mutation_pipeline"))
_mock_mutation_models = ModuleType("mutation_pipeline.models")
_mock_mutation_models.AtomicSpec = type("AtomicSpec", (), {})
_mock_mutation_models.PipelineInput = type(
    "PipelineInput",
    (),
    {"__init__": lambda self, **kw: None},
)
sys.modules.setdefault("mutation_pipeline.models", _mock_mutation_models)
_mock_mutation_docker = ModuleType("mutation_pipeline.docker_runner")
_mock_mutation_docker.MutationDockerRunner = mock.MagicMock()
sys.modules.setdefault("mutation_pipeline.docker_runner", _mock_mutation_docker)

_mock_featurebench = ModuleType("featurebench")
_mock_featurebench_infer = ModuleType("featurebench.infer")
_mock_featurebench_infer_config = ModuleType("featurebench.infer.config")
_mock_featurebench_infer_config.InferConfigLoader = type("InferConfigLoader", (), {})
_mock_featurebench_infer_config.DatasetLoader = type("DatasetLoader", (), {})
sys.modules.setdefault("featurebench", _mock_featurebench)
sys.modules.setdefault("featurebench.infer", _mock_featurebench_infer)
sys.modules.setdefault("featurebench.infer.config", _mock_featurebench_infer_config)

_mock_harness = ModuleType("featurebench.harness")
_mock_harness_container = ModuleType("featurebench.harness.container")
_mock_harness_container.copy_to_container = mock.MagicMock()
_mock_harness_container.exec_run_with_timeout = mock.MagicMock(return_value=(0, b""))
_mock_harness_constants = ModuleType("featurebench.harness.constants")
_mock_harness_constants.UTF8 = "utf-8"
sys.modules.setdefault("featurebench.harness", _mock_harness)
sys.modules.setdefault("featurebench.harness.container", _mock_harness_container)
sys.modules.setdefault("featurebench.harness.constants", _mock_harness_constants)

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.models import (
    TARGET_STATE_CONTROL,
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    BaselineResult,
    EvalTask,
    ProofExecResult,
    SpecCertificate,
    SurvivingSpec,
)
from eval_pipeline.config import BaselineConfig, EvalPipelineConfig


def _make_task(iid: str, spec_ids=("S1",), probed_by=None) -> EvalTask:
    probed_by = probed_by or {}
    return EvalTask(
        instance_id=iid,
        image_name="img",
        surviving_specs=[
            SurvivingSpec(
                spec={"spec_id": sid, "category": "behavioral"},
                mutant_patch=f"mutant for {sid}",
                preservation_probes=[
                    {"spec_id": p, "verdict": "preserved"}
                    for p in probed_by.get(sid, ())
                ],
            )
            for sid in spec_ids
        ],
        f2p_tests=[],
        p2p_tests=[],
        repo_settings={},
        mask_patch="mask",
        gold_patch="gold",
    )


def _make_baseline_result(
    iid: str,
    baseline: str = "claude_code",
    model: str = "opus",
    certificates: Dict[str, SpecCertificate] | None = None,
) -> BaselineResult:
    return BaselineResult(
        instance_id=iid,
        baseline=baseline,
        model=model,
        tool_version="1.0",
        raw_output="output",
        certificates=certificates or {},
        cost={"input_tokens": 100, "output_tokens": 50, "total_cost_usd": 0.01},
        wall_time_seconds=10.0,
    )


def _cert(sid="S1", code="some code") -> SpecCertificate:
    return SpecCertificate(description=sid, code=code)


def test_spec_certificate_round_trips_saved_inputs():
    cert = SpecCertificate(
        description="replay",
        code="def test_x(): pass",
        saved_inputs=[{"id": "case_0", "args": [1], "kwargs": {}}],
        inputs_file="baseline_saved_inputs/b_m/i/S1.jsonl",
    )

    restored = SpecCertificate.from_dict(cert.to_dict())

    assert restored.saved_inputs == [{"id": "case_0", "args": [1], "kwargs": {}}]
    assert restored.inputs_file == "baseline_saved_inputs/b_m/i/S1.jsonl"


def _make_config(tmp_path: Path, n_concurrent: int = 2, skip_existing: bool = False):
    return EvalPipelineConfig(
        output_dir=tmp_path,
        n_concurrent=n_concurrent,
        skip_existing=skip_existing,
    )


def _make_baseline_cfg(name="claude_code", model="opus"):
    return BaselineConfig(
        name=name, model=model, agent_config=name, tool_version="1.0"
    )


class TestBaselineExecutorConcurrency:
    def _make_executor(self, config, config_loader=None, docker_runner=None):
        from eval_pipeline.stages.stage2_baseline_execution import BaselineExecutor

        if config_loader is None:
            config_loader = mock.MagicMock()
            config_loader.get_agent_env_vars.return_value = {}
        if docker_runner is None:
            docker_runner = mock.MagicMock()
        return BaselineExecutor(config, config_loader, docker_runner)

    def test_all_items_executed(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        tasks = [_make_task("inst1"), _make_task("inst2")]
        baselines = [_make_baseline_cfg()]

        executor._run_single = mock.MagicMock(side_effect=[
            _make_baseline_result("inst1"),
            _make_baseline_result("inst2"),
        ])
        results = executor.run_all(tasks, baselines)
        assert len(results) == 2
        assert executor._run_single.call_count == 2

    def test_multiple_baselines(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=4))
        tasks = [_make_task("inst1"), _make_task("inst2")]
        baselines = [_make_baseline_cfg("claude_code"), _make_baseline_cfg("codex", "gpt")]

        executor._run_single = mock.MagicMock(
            side_effect=lambda t, b: _make_baseline_result(t.instance_id, b.name, b.model)
        )
        results = executor.run_all(tasks, baselines)
        assert len(results) == 4

    def test_actual_parallelism(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=3))
        tasks = [_make_task(f"inst{i}") for i in range(3)]

        active_lock = threading.Lock()
        active = [0]
        peak = [0]

        def slow_run(task, bcfg):
            with active_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.1)
            with active_lock:
                active[0] -= 1
            return _make_baseline_result(task.instance_id)

        executor._run_single = mock.MagicMock(side_effect=slow_run)
        results = executor.run_all(tasks, [_make_baseline_cfg()])
        assert len(results) == 3
        assert peak[0] >= 2

    def test_n_concurrent_serial(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=1))
        tasks = [_make_task(f"inst{i}") for i in range(3)]

        active_lock = threading.Lock()
        active = [0]
        peak = [0]

        def counting_run(task, bcfg):
            with active_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with active_lock:
                active[0] -= 1
            return _make_baseline_result(task.instance_id)

        executor._run_single = mock.MagicMock(side_effect=counting_run)
        executor.run_all(tasks, [_make_baseline_cfg()])
        assert peak[0] == 1

    def test_skip_existing_loads_cached(self, tmp_path):
        executor = self._make_executor(
            _make_config(tmp_path, n_concurrent=2, skip_existing=True)
        )
        task = _make_task("inst1")
        bcfg = _make_baseline_cfg()
        cached = _make_baseline_result("inst1", certificates={"S1": _cert()})
        path = executor._result_path(task, bcfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached.to_dict()), encoding="utf-8")

        executor._run_single = mock.MagicMock()
        results = executor.run_all([task], [bcfg])
        assert len(results) == 1
        executor._run_single.assert_not_called()

    def test_skip_existing_reruns_a_result_that_never_ran_the_agent(self, tmp_path):
        executor = self._make_executor(
            _make_config(tmp_path, n_concurrent=2, skip_existing=True)
        )
        task = _make_task("inst1")
        bcfg = _make_baseline_cfg()
        never_ran = _make_baseline_result("inst1")
        never_ran.raw_output = "ERROR: Installation failed"
        never_ran.completed_normally = False
        never_ran.attempts = []
        path = executor._result_path(task, bcfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(never_ran.to_dict()), encoding="utf-8")

        fresh = _make_baseline_result("inst1", certificates={"S1": _cert()})
        executor._run_single = mock.MagicMock(return_value=fresh)
        results = executor.run_all([task], [bcfg])

        executor._run_single.assert_called_once()
        assert results[0].certificates == {"S1": _cert()}

    def test_skip_existing_keeps_a_run_the_agent_started_and_lost(self, tmp_path):
        executor = self._make_executor(
            _make_config(tmp_path, n_concurrent=2, skip_existing=True)
        )
        task = _make_task("inst1")
        bcfg = _make_baseline_cfg()
        lost = _make_baseline_result("inst1")
        lost.raw_output = "ERROR: step cap reached"
        lost.completed_normally = False
        lost.attempts = [{"attempt": 1, "exit_code": 1, "raw_output": "..."}]
        path = executor._result_path(task, bcfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lost.to_dict()), encoding="utf-8")

        executor._run_single = mock.MagicMock()
        executor.run_all([task], [bcfg])
        executor._run_single.assert_not_called()

    def test_factory_oracle_validates_cached_certificate_set(self, tmp_path):
        config = _make_config(tmp_path, skip_existing=True)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1",))
        bcfg = _make_baseline_cfg()
        cached = _make_baseline_result(
            "inst1", certificates={"S1": _cert(), "S2": _cert()}
        )
        path = executor._result_path(task, bcfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached.to_dict()), encoding="utf-8")

        executor._run_single = mock.MagicMock()
        results = executor.run_all([task], [bcfg])

        assert len(results) == 1
        assert results[0].certificates == {}
        assert results[0].llm_errors[-1].error_type == "invalid_certificate_set"
        executor._run_single.assert_not_called()

    def test_results_saved_to_disk(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        task = _make_task("inst1")
        bcfg = _make_baseline_cfg()
        executor._run_single = mock.MagicMock(
            return_value=_make_baseline_result("inst1")
        )
        executor.run_all([task], [bcfg])
        path = executor._result_path(task, bcfg)
        assert path.exists()
        assert json.loads(path.read_text())["instance_id"] == "inst1"

    def test_failure_does_not_block_others(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        tasks = [_make_task("ok1"), _make_task("fail"), _make_task("ok2")]
        bcfg = _make_baseline_cfg()

        def maybe_fail(task, _bcfg):
            if task.instance_id == "fail":
                raise RuntimeError("boom")
            return _make_baseline_result(task.instance_id)

        executor._run_single = mock.MagicMock(side_effect=maybe_fail)
        results = executor.run_all(tasks, [bcfg])
        assert len(results) == 2
        assert {r.instance_id for r in results} == {"ok1", "ok2"}

    def test_empty_inputs(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path))
        assert executor.run_all([], [_make_baseline_cfg()]) == []
        assert executor.run_all([_make_task("inst1")], []) == []

    def test_n_concurrent_zero_clamped(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=0))
        executor._run_single = mock.MagicMock(return_value=_make_baseline_result("i"))
        results = executor.run_all([_make_task("i")], [_make_baseline_cfg()])
        assert len(results) == 1

    def test_saved_inputs_expands_to_one_launch_per_atomic_spec(self, tmp_path):
        config = _make_config(tmp_path, n_concurrent=2)
        config.harness_form = "saved_inputs"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1", "S2"))

        def run_single(single_task, bcfg):
            sid = single_task.surviving_specs[0].spec_id
            return _make_baseline_result(
                single_task.instance_id,
                bcfg.name,
                bcfg.model,
                certificates={sid: _cert(sid, "code")},
            )

        executor._run_single = mock.MagicMock(side_effect=run_single)

        results = executor.run_all([task], [_make_baseline_cfg()])

        assert len(results) == 2
        launched_specs = [
            call.args[0].surviving_specs[0].spec_id
            for call in executor._run_single.call_args_list
        ]
        assert set(launched_specs) == {"S1", "S2"}
        assert (tmp_path / "baseline_output" / "claude_code_opus" / "inst1_S1.json").exists()
        assert (tmp_path / "baseline_output" / "claude_code_opus" / "inst1_S2.json").exists()

    def test_factory_oracle_expands_to_one_launch_per_atomic_spec(self, tmp_path):
        config = _make_config(tmp_path, n_concurrent=2)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1", "S2"))

        def run_single(single_task, bcfg):
            sid = single_task.surviving_specs[0].spec_id
            return _make_baseline_result(
                single_task.instance_id,
                bcfg.name,
                bcfg.model,
                certificates={sid: _cert(sid, "code")},
            )

        executor._run_single = mock.MagicMock(side_effect=run_single)

        results = executor.run_all([task], [_make_baseline_cfg()])

        assert len(results) == 2
        launched_specs = {
            call.args[0].surviving_specs[0].spec_id
            for call in executor._run_single.call_args_list
        }
        assert launched_specs == {"S1", "S2"}

    def test_factory_oracle_accepts_exact_expected_certificate(self, tmp_path):
        config = _make_config(tmp_path)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1",))
        result = _make_baseline_result(
            "inst1", certificates={"S1": _cert("S1", "code")}
        )

        finalized = executor._finalize_factory_oracle_result(task, result)

        assert set(finalized.certificates) == {"S1"}
        assert finalized.completed_normally is True
        assert finalized.llm_errors == []

    @pytest.mark.parametrize(
        "certificates",
        [
            {},
            {"S2": _cert("S2", "code")},
            {
                "S1": _cert("S1", "code"),
                "S2": _cert("S2", "code"),
            },
        ],
    )
    def test_factory_oracle_rejects_invalid_certificate_set(
        self, tmp_path, certificates
    ):
        config = _make_config(tmp_path)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1",))
        result = _make_baseline_result("inst1", certificates=certificates)

        finalized = executor._finalize_factory_oracle_result(task, result)

        assert finalized.certificates == {}
        assert finalized.completed_normally is False
        assert finalized.llm_errors[-1].error_type == "invalid_certificate_set"
        assert "'S1'" in finalized.llm_errors[-1].sample_message

    def test_factory_oracle_prompt_requires_one_spec_and_has_no_spec_loop(
        self, tmp_path
    ):
        from eval_pipeline.stages.stage2_baseline_execution import _render_prompt

        config = _make_config(tmp_path)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config)
        task = _make_task("inst1", spec_ids=("S1",))
        task.surviving_specs[0].spec["text"] = "The function satisfies S1."
        task.surviving_specs[0].case_contract = "case is a stimulus dict"

        prompt = _render_prompt(
            executor.prompt_env,
            task,
            harness_form="factory_oracle",
        )

        assert "## Specification S1" in prompt
        assert '"S1": {' in prompt
        assert "case is a stimulus dict" in prompt
        assert 'target(*case["args"], **case["kwargs"])' in prompt
        assert "scope of this specification" in prompt
        assert "relevant delegated or transitively called" in prompt
        assert "Do not spend effort covering unrelated functions" in prompt
        assert "use monkeypatching" in prompt
        assert "A violation can manifest in many ways" in prompt
        assert "do not assume that the target always returns normally" in prompt
        assert "Keep exception handling narrowly scoped" in prompt
        source, _, _ = executor.prompt_env.loader.get_source(
            executor.prompt_env, "baseline_factory_oracle_task.j2"
        )
        assert "{% for spec in specs %}" not in source
        with pytest.raises(ValueError, match="exactly one surviving spec"):
            _render_prompt(
                executor.prompt_env,
                _make_task("inst1", spec_ids=("S1", "S2")),
                harness_form="factory_oracle",
            )

    def test_saved_inputs_sidecar_is_extracted(self, tmp_path, monkeypatch):
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        executor = self._make_executor(config)

        from eval_pipeline.stages import stage2_baseline_execution

        monkeypatch.setattr(
            stage2_baseline_execution,
            "exec_run_with_timeout",
            lambda *_args, **_kwargs: (
                0,
                b'{"args":[1],"kwargs":{}}\n{"args":[2],"kwargs":{}}\n',
            ),
        )
        task = _make_task("inst1", spec_ids=("S1",))
        bcfg = _make_baseline_cfg()
        cert = SpecCertificate(
            description="sidecar",
            code="import os\nPATH = os.environ['AGENT_SAVED_INPUTS_PATH']\n",
            inputs_file="/tmp/eval_saved_inputs/S1.jsonl",
        )
        result = _make_baseline_result("inst1", certificates={"S1": cert})

        finalized = executor._finalize_saved_inputs_result(
            mock.MagicMock(), task, bcfg, result
        )

        out_cert = finalized.certificates["S1"]
        assert out_cert.inputs_file == "baseline_saved_inputs/claude_code_opus/inst1/S1.jsonl"
        assert out_cert.saved_inputs == []
        assert (tmp_path / out_cert.inputs_file).read_text(encoding="utf-8").count("\n") == 2


class TestProofExecutorConcurrency:
    def _make_docker_runner(self, triggered_map=None):
        runner = mock.MagicMock()
        runner.setup_instance.return_value = mock.MagicMock(name="container")
        runner.apply_patch.return_value = True
        runner.run_hypothesis_test.return_value = (True, "fired")
        runner.reset_testbed.return_value = None
        runner.cleanup.return_value = None
        return runner

    def _make_executor(self, config, docker_runner=None):
        from eval_pipeline.stages.stage3_proof_execution import ProofExecutor

        if docker_runner is None:
            docker_runner = self._make_docker_runner()
        return ProofExecutor(config, docker_runner)

    def _patch_prepare(self, monkeypatch):
        from eval_pipeline.stages import stage3_proof_execution

        monkeypatch.setattr(
            stage3_proof_execution,
            "prepare_task_code_state",
            lambda **kw: (True, ""),
        )

    def test_runs_cert_twice_per_spec(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        config = _make_config(tmp_path, n_concurrent=2)
        executor = self._make_executor(config)

        task = _make_task("inst1", spec_ids=("S1", "S2"))
        br = _make_baseline_result(
            "inst1", certificates={"S1": _cert("S1", "c1"), "S2": _cert("S2", "c2")}
        )
        results = executor.run_all([task], [br])
        assert len(results) == 4
        target_states = {(r.spec_id, r.target_state) for r in results}
        assert target_states == {
            ("S1", TARGET_STATE_GOLD), ("S1", TARGET_STATE_MUTANT),
            ("S2", TARGET_STATE_GOLD), ("S2", TARGET_STATE_MUTANT),
        }

    def test_skip_existing_per_state(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        config = _make_config(tmp_path, n_concurrent=2, skip_existing=True)
        executor = self._make_executor(config)

        task = _make_task("inst1")
        br = _make_baseline_result("inst1", certificates={"S1": _cert()})

        cached = ProofExecResult(
            instance_id="inst1", baseline="claude_code", model="opus",
            spec_id="S1", target_state=TARGET_STATE_GOLD,
            triggered=False, exit_code=0, proof_output="cached",
            execution_time_seconds=0.1,
        )
        gold_path = executor._result_path(br, "S1", TARGET_STATE_GOLD)
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(json.dumps(cached.to_dict()), encoding="utf-8")

        results = executor.run_all([task], [br])
        states = {r.target_state for r in results}
        assert states == {TARGET_STATE_GOLD, TARGET_STATE_MUTANT}

    def test_skips_certs_without_matching_spec(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))

        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1", certificates={"S1": _cert("S1"), "S2": _cert("S2")}
        )
        results = executor.run_all([task], [br])
        spec_ids = {r.spec_id for r in results}
        assert spec_ids == {"S1"}
        assert len(results) == 2

    def test_missing_task_skipped(self, tmp_path):
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        br = _make_baseline_result(
            "inst_missing", certificates={"S1": _cert()}
        )
        results = executor.run_all([], [br])
        assert results == []

    def test_saved_inputs_mode_replays_serialized_inputs(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        docker_runner.run_saved_inputs_replay.return_value = (
            False, "passed", 0, {"branch_coverage_pct": 100.0, "files": {}}
        )
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        config.saved_inputs_replay_timeout_seconds = 7
        executor = self._make_executor(config, docker_runner=docker_runner)

        saved_inputs = [{"id": "case_0", "args": [3], "kwargs": {}}]
        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="replay double",
                    code=(
                        "import os\n"
                        "def test_replay():\n"
                        "    assert os.environ['AGENT_SAVED_INPUTS_PATH']\n"
                    ),
                    saved_inputs=saved_inputs,
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert docker_runner.run_saved_inputs_replay.call_count == 2
        first_call = docker_runner.run_saved_inputs_replay.call_args_list[0].args
        assert first_call[2] == saved_inputs
        assert first_call[3] == 7
        assert {r.kind for r in results} == {"passed"}

    def test_saved_inputs_mode_replays_jsonl_sidecar(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        docker_runner.run_saved_inputs_replay.return_value = (
            False, "passed", 0, {"branch_coverage_pct": 100.0, "files": {}}
        )
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        config.saved_inputs_replay_timeout_seconds = 7
        executor = self._make_executor(config, docker_runner=docker_runner)

        sidecar = tmp_path / "baseline_saved_inputs" / "claude_code_opus" / "inst1" / "S1.jsonl"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text('{"args":[3],"kwargs":{}}\n', encoding="utf-8")
        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="replay double",
                    code=(
                        "import os\n"
                        "def test_replay():\n"
                        "    assert os.environ['AGENT_SAVED_INPUTS_PATH']\n"
                    ),
                    inputs_file="baseline_saved_inputs/claude_code_opus/inst1/S1.jsonl",
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert docker_runner.run_saved_inputs_replay.call_count == 2
        first_call = docker_runner.run_saved_inputs_replay.call_args_list[0]
        assert first_call.args[2] == []
        assert first_call.args[3] == 7
        assert first_call.kwargs["saved_inputs_file"] == sidecar
        assert {r.kind for r in results} == {"passed"}

    def test_saved_inputs_mode_rejects_empty_payload(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        executor = self._make_executor(config, docker_runner=docker_runner)

        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="missing corpus",
                    code="import os\nPATH = os.environ['AGENT_SAVED_INPUTS_PATH']\n",
                    saved_inputs=[],
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert {r.kind for r in results} == {"infra_error"}
        assert docker_runner.run_saved_inputs_replay.call_count == 0

    def test_saved_inputs_mode_rejects_hypothesis_head(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        executor = self._make_executor(config, docker_runner=docker_runner)

        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="bad fuzz head",
                    code=(
                        "import os\n"
                        "from hypothesis import given, strategies as st\n"
                        "@given(x=st.integers())\n"
                        "def test_replay(x):\n"
                        "    assert os.environ['AGENT_SAVED_INPUTS_PATH']\n"
                    ),
                    saved_inputs=[{"x": 1}],
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert {r.kind for r in results} == {"infra_error"}
        assert all("must not import" in (r.error or "") for r in results)
        assert docker_runner.run_saved_inputs_replay.call_count == 0

    def test_saved_inputs_mode_rejects_non_json_payload(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "saved_inputs"
        executor = self._make_executor(config, docker_runner=docker_runner)

        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="bad payload",
                    code=(
                        "import os\n"
                        "def test_replay():\n"
                        "    assert os.environ['AGENT_SAVED_INPUTS_PATH']\n"
                    ),
                    saved_inputs=[{"not_json": object()}],
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert {r.kind for r in results} == {"infra_error"}
        assert all("not JSON-serializable" in (r.error or "") for r in results)
        assert docker_runner.run_saved_inputs_replay.call_count == 0

    def test_factory_oracle_runs_generated_and_witness_cases(
        self, tmp_path, monkeypatch
    ):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        docker_runner.run_factory_oracle.return_value = (
            False, "passed", 0, {"files": {}}
        )
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        config.saved_inputs_replay_timeout_seconds = 9
        executor = self._make_executor(config, docker_runner=docker_runner)

        task = _make_task("inst1", spec_ids=("S1",))
        task.surviving_specs[0].case_contract = (
            "case has exactly args and kwargs"
        )
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="oracle",
                    code=(
                        "def build_cases():\n"
                        "    yield {'args': (1,), 'kwargs': {}}\n"
                        "def check_case(case):\n"
                        "    assert case['args'][0] >= 0\n"
                    ),
                )
            },
        )

        results = executor.run_all([task], [br])

        assert len(results) == 2
        assert {result.target_state for result in results} == {
            TARGET_STATE_GOLD,
            TARGET_STATE_MUTANT,
        }
        calls = docker_runner.run_factory_oracle.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.args[1].startswith("def build_cases")

    def test_only_gold_runs_the_whole_case_list(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        docker_runner.run_factory_oracle.return_value = (
            False, "passed", 0, {"files": {}}
        )
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config, docker_runner=docker_runner)

        task = _make_task("inst1", spec_ids=("S1", "S2"), probed_by={"S2": ("S1",)})
        cert_code = (
            "def build_cases():\n"
            "    yield {'args': (1,), 'kwargs': {}}\n"
            "def check_case(case):\n"
            "    assert case['args'][0] >= 0\n"
        )
        br = _make_baseline_result(
            "inst1",
            certificates={
                sid: SpecCertificate(description="oracle", code=cert_code)
                for sid in ("S1", "S2")
            },
        )

        results = executor.run_all([task], [br])

        by_state = {}
        for result, call in zip(results, docker_runner.run_factory_oracle.call_args_list):
            by_state.setdefault(result.target_state, []).append(
                call.kwargs.get("run_all_cases")
            )
        assert TARGET_STATE_CONTROL in by_state, "no control arm ran"
        assert by_state[TARGET_STATE_GOLD] == [True] * len(by_state[TARGET_STATE_GOLD])
        assert by_state[TARGET_STATE_MUTANT] == [False] * len(by_state[TARGET_STATE_MUTANT])
        assert by_state[TARGET_STATE_CONTROL] == [False] * len(by_state[TARGET_STATE_CONTROL])

    def test_factory_oracle_runs_the_reference_arm_exhaustively(
        self, tmp_path, monkeypatch
    ):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        docker_runner.run_factory_oracle.return_value = (
            False, "passed", 0, {"files": {}}
        )
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        config.stage2_reference_state = "mutant"
        executor = self._make_executor(config, docker_runner=docker_runner)
        task = _make_task("inst1", spec_ids=("S1", "S2"), probed_by={"S2": ("S1",)})
        cert_code = "def build_cases():\n    yield {'args': (1,), 'kwargs': {}}\ndef check_case(case):\n    pass\n"
        br = _make_baseline_result(
            "inst1",
            certificates={
                sid: SpecCertificate(description="oracle", code=cert_code)
                for sid in ("S1", "S2")
            },
        )

        results = executor.run_all([task], [br])

        by_state = {}
        for result, call in zip(results, docker_runner.run_factory_oracle.call_args_list):
            by_state.setdefault(result.target_state, []).append(
                call.kwargs.get("run_all_cases")
            )
        assert by_state[TARGET_STATE_MUTANT] == [True] * len(by_state[TARGET_STATE_MUTANT])
        assert by_state[TARGET_STATE_GOLD] == [False] * len(by_state[TARGET_STATE_GOLD])
        assert by_state[TARGET_STATE_CONTROL] == [False] * len(by_state[TARGET_STATE_CONTROL])
        assert all("mutant_lines" in r.coverage for r in results if r.coverage is not None)

    def test_factory_oracle_preserves_test_exception_kind(
        self, tmp_path, monkeypatch
    ):
        self._patch_prepare(monkeypatch)
        docker_runner = self._make_docker_runner()
        marker = "__FACTORY_ORACLE_DRIVER_RESULT__"
        docker_runner.run_factory_oracle.side_effect = [
            (False, f'{marker}{json.dumps({"kind": "passed"})}', 0, None),
            (
                True,
                f'{marker}{json.dumps({"kind": "test_exception"})}',
                1,
                None,
            ),
        ]
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        executor = self._make_executor(config, docker_runner=docker_runner)
        task = _make_task("inst1", spec_ids=("S1",))
        br = _make_baseline_result(
            "inst1",
            certificates={
                "S1": SpecCertificate(
                    description="oracle",
                    code=(
                        "def build_cases():\n"
                        "    yield {'args': (), 'kwargs': {}}\n"
                        "def check_case(case):\n"
                        "    assert True\n"
                    ),
                )
            },
        )

        results = executor.run_all([task], [br])
        by_state = {result.target_state: result for result in results}

        assert by_state[TARGET_STATE_MUTANT].kind == "test_exception"

    def test_failure_does_not_block_others(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))

        task_ok = _make_task("ok")
        task_fail = _make_task("fail")
        brs = [
            _make_baseline_result("ok", certificates={"S1": _cert()}),
            _make_baseline_result("fail", certificates={"S1": _cert()}),
        ]

        original_run = executor._run_for_baseline_result

        def maybe_fail(task, br):
            if br.instance_id == "fail":
                raise RuntimeError("simulated container crash")
            return original_run(task, br)

        executor._run_for_baseline_result = maybe_fail
        results = executor.run_all([task_ok, task_fail], brs)
        assert {r.instance_id for r in results} == {"ok"}

    def test_empty_inputs(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        assert executor.run_all([_make_task("i")], []) == []
        assert executor.run_all([], [_make_baseline_result("i", certificates={"S1": _cert()})]) == []

    def test_empty_certificates(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=2))
        results = executor.run_all(
            [_make_task("inst1")], [_make_baseline_result("inst1", certificates={})]
        )
        assert results == []

    def test_actual_parallelism(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=3))

        tasks = [_make_task(f"inst{i}") for i in range(3)]
        brs = [
            _make_baseline_result(f"inst{i}", certificates={"S1": _cert()})
            for i in range(3)
        ]

        active_lock = threading.Lock()
        active = [0]
        peak = [0]
        original_run = executor._run_for_baseline_result

        def tracked_run(task, br):
            with active_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.1)
            result = original_run(task, br)
            with active_lock:
                active[0] -= 1
            return result

        executor._run_for_baseline_result = tracked_run
        executor.run_all(tasks, brs)
        assert peak[0] >= 2

    def test_n_concurrent_serial(self, tmp_path, monkeypatch):
        self._patch_prepare(monkeypatch)
        executor = self._make_executor(_make_config(tmp_path, n_concurrent=1))

        tasks = [_make_task(f"inst{i}") for i in range(3)]
        brs = [
            _make_baseline_result(f"inst{i}", certificates={"S1": _cert()})
            for i in range(3)
        ]

        active_lock = threading.Lock()
        active = [0]
        peak = [0]
        original_run = executor._run_for_baseline_result

        def tracked_run(task, br):
            with active_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            result = original_run(task, br)
            with active_lock:
                active[0] -= 1
            return result

        executor._run_for_baseline_result = tracked_run
        executor.run_all(tasks, brs)
        assert peak[0] == 1


class TestNConcurrentCLIOverride:
    def _write_metadata(self, run_dir: Path, n_concurrent: int) -> None:
        (run_dir / "run_metadata.json").write_text(
            json.dumps({
                "n_concurrent": n_concurrent,
                "harness_form": "factory_oracle",
                "skip_existing": False,
                "baselines": [{
                    "name": "claude_code",
                    "model": "claude-opus-4-8",
                    "agent_config": "claude_code",
                    "tool_version": "2.1.77",
                }],
            }),
            encoding="utf-8",
        )

    def _args(self, **overrides) -> argparse.Namespace:
        base = {
            "stages": None, "skip_existing": False, "n_concurrent": None,
            "timeout": None, "max_agent_steps": None, "harness_form": None,
            "baseline": None, "model": None, "stage2_reference_state": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_resume_without_flag_keeps_metadata_value(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path, n_concurrent=1)
        config = _load_resume_config(tmp_path, self._args(), None)
        assert config.n_concurrent == 1

    def test_resume_flag_overrides_metadata_value(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path, n_concurrent=1)
        config = _load_resume_config(tmp_path, self._args(n_concurrent=4), None)
        assert config.n_concurrent == 4

    def test_resume_flag_is_none_checked_not_truthiness_checked(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path, n_concurrent=8)
        config = _load_resume_config(tmp_path, self._args(n_concurrent=0), None)
        assert config.n_concurrent == 0


class TestResumeBaselineSelection:
    def _write_metadata(self, run_dir: Path, harness_form: str = "factory_oracle") -> None:
        (run_dir / "run_metadata.json").write_text(
            json.dumps({
                "n_concurrent": 4,
                "harness_form": harness_form,
                "skip_existing": False,
                "baselines": [{
                    "name": "codex",
                    "model": "gpt-5.6-sol",
                    "agent_config": "codex",
                    "tool_version": "0.117.0",
                }],
            }),
            encoding="utf-8",
        )

    def _args(self, **overrides) -> argparse.Namespace:
        base = {
            "stages": None, "skip_existing": False, "n_concurrent": None,
            "timeout": None, "max_agent_steps": None, "harness_form": None,
            "baseline": None, "model": None, "stage2_reference_state": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_resume_without_baseline_flag_keeps_the_runs_own_baseline(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path)
        config = _load_resume_config(tmp_path, self._args(), None)

        assert [(b.name, b.model) for b in config.baselines] == [
            ("codex", "gpt-5.6-sol")
        ]

    def test_resume_rejects_a_different_harness_form(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path, harness_form="factory_oracle")

        with pytest.raises(SystemExit):
            _load_resume_config(tmp_path, self._args(harness_form="hypofuzz"), None)

    def test_resume_accepts_the_same_harness_form(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path, harness_form="factory_oracle")
        config = _load_resume_config(
            tmp_path, self._args(harness_form="factory_oracle"), None
        )

        assert config.harness_form == "factory_oracle"

    def test_resume_timeout_override_reaches_the_config(self, tmp_path):
        from eval_pipeline.run import _load_resume_config

        self._write_metadata(tmp_path)
        config = _load_resume_config(tmp_path, self._args(timeout=3600), None)

        assert config.timeout == 3600


class TestContainerIsolation:
    def _run(self, tmp_path, monkeypatch, isolation):
        from eval_pipeline.stages.stage3_proof_execution import ProofExecutor

        monkeypatch.setattr(
            "eval_pipeline.stages.stage3_proof_execution.prepare_task_code_state",
            lambda **kw: (True, None),
        )
        docker_runner = mock.MagicMock()
        docker_runner.setup_instance.side_effect = (
            lambda *a, **k: mock.MagicMock()
        )
        docker_runner.run_factory_oracle.return_value = (False, "", 0, None)

        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        config.container_isolation = isolation
        executor = ProofExecutor(config, docker_runner)

        task = _make_task("inst1", spec_ids=("S1", "S2"))
        br = _make_baseline_result(
            "inst1",
            certificates={
                sid: SpecCertificate(
                    description="oracle",
                    code=(
                        "def build_cases():\n"
                        "    yield {'args': (), 'kwargs': {}}\n"
                        "def check_case(case):\n"
                        "    assert True\n"
                    ),
                )
                for sid in ("S1", "S2")
            },
        )
        executor.run_all([task], [br])
        return docker_runner

    def test_instance_level_reuses_one_container(self, tmp_path, monkeypatch):
        runner = self._run(tmp_path, monkeypatch, "instance")
        assert runner.setup_instance.call_count == 1
        assert runner.reset_testbed.call_count == 4

    def test_spec_level_uses_one_container_per_certificate(
        self, tmp_path, monkeypatch
    ):
        runner = self._run(tmp_path, monkeypatch, "spec")
        assert runner.setup_instance.call_count == 2
        assert runner.reset_testbed.call_count == 0

    def test_arm_level_uses_a_fresh_container_per_code_state(
        self, tmp_path, monkeypatch
    ):
        runner = self._run(tmp_path, monkeypatch, "arm")
        assert runner.setup_instance.call_count == 4
        assert runner.cleanup.call_count == 4
        assert runner.reset_testbed.call_count == 0


class TestNegativeControlArm:
    def test_control_arm_runs_each_certificate_against_a_foreign_mutant(
        self, tmp_path, monkeypatch
    ):
        from eval_pipeline.stages.stage3_proof_execution import ProofExecutor

        monkeypatch.setattr(
            "eval_pipeline.stages.stage3_proof_execution.prepare_task_code_state",
            lambda **kw: (True, None),
        )
        docker_runner = mock.MagicMock()
        docker_runner.run_factory_oracle.return_value = (False, "", 0, None)
        config = _make_config(tmp_path, n_concurrent=1)
        config.harness_form = "factory_oracle"
        config.container_isolation = "instance"
        executor = ProofExecutor(config, docker_runner)

        task = _make_task(
            "inst1",
            spec_ids=("S1", "S2"),
            probed_by={"S1": ["S2"], "S2": ["S1"]},
        )
        br = _make_baseline_result(
            "inst1",
            certificates={
                sid: SpecCertificate(
                    description="oracle",
                    code=(
                        "def build_cases():\n"
                        "    yield {'args': (), 'kwargs': {}}\n"
                        "def check_case(case):\n"
                        "    assert True\n"
                    ),
                )
                for sid in ("S1", "S2")
            },
        )
        results = executor.run_all([task], [br])

        controls = [r for r in results if r.target_state == "control"]
        assert {(r.spec_id, r.control_spec_id) for r in controls} == {
            ("S1", "S2"), ("S2", "S1"),
        }
        assert len(results) == 6


class TestNegativeControlSelection:
    @staticmethod
    def _spec(spec_id, probed, scores=None):
        from eval_pipeline.models import SurvivingSpec

        scores = scores or {}
        probes = []
        for p in probed:
            probe = {"spec_id": p, "verdict": "preserved"}
            if p in scores:
                probe["shared_functions"], probe["shared_lines"] = scores[p]
            probes.append(probe)
        return SurvivingSpec(
            spec={"spec_id": spec_id},
            mutant_patch="diff",
            preservation_probes=probes,
        )

    def test_inverts_the_probe_record(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [
            self._spec("S1", []),
            self._spec("S2", ["S1", "S3"]),
            self._spec("S3", []),
            self._spec("S4", ["S1"]),
        ]
        assert [s.spec_id for s in select_control_mutants("S1", specs)] == ["S2", "S4"]
        assert [s.spec_id for s in select_control_mutants("S3", specs)] == ["S2"]

    def test_asymmetry_survives_the_inversion(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [self._spec("S1", ["S2"]), self._spec("S2", [])]
        assert [s.spec_id for s in select_control_mutants("S2", specs)] == ["S1"]
        assert select_control_mutants("S1", specs) == []

    def test_no_probe_record_means_no_control_arm(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [self._spec("S1", []), self._spec("S2", [])]
        assert select_control_mutants("S1", specs) == []

    def test_fan_out_is_not_truncated_without_a_cap(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [self._spec("S1", [])] + [
            self._spec(f"S{i}", ["S1"]) for i in range(2, 8)
        ]
        assert len(select_control_mutants("S1", specs)) == 6

    def test_the_cap_keeps_the_most_entangled_controls(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [
            self._spec("S1", []),
            self._spec("S2", ["S1"], {"S1": (0, 0)}),
            self._spec("S3", ["S1"], {"S1": (0, 4)}),
            self._spec("S4", ["S1"], {"S1": (2, 30)}),
            self._spec("S5", ["S1"], {"S1": (1, 12)}),
        ]
        ordered = [s.spec_id for s in select_control_mutants("S1", specs)]
        assert ordered == ["S4", "S5", "S3", "S2"]
        assert [
            s.spec_id for s in select_control_mutants("S1", specs, control_arm_k=2)
        ] == ["S4", "S5"]

    def test_the_cap_does_not_apply_to_unscored_records(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [self._spec("S1", [])] + [
            self._spec(f"S{i}", ["S1"]) for i in range(2, 10)
        ]
        assert len(select_control_mutants("S1", specs, control_arm_k=5)) == 8

    def test_the_cap_is_inert_below_its_bound(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [
            self._spec("S1", []),
            self._spec("S2", ["S1"], {"S1": (1, 5)}),
        ]
        assert [
            s.spec_id for s in select_control_mutants("S1", specs, control_arm_k=5)
        ] == ["S2"]

    def test_is_deterministic(self):
        from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

        specs = [
            self._spec("S3", ["S1"]),
            self._spec("S1", []),
            self._spec("S4", ["S1"]),
            self._spec("S2", ["S1"]),
        ]
        expected = ["S2", "S3", "S4"]
        assert [s.spec_id for s in select_control_mutants("S1", specs)] == expected
        assert [
            s.spec_id for s in select_control_mutants("S1", list(reversed(specs)))
        ] == expected
