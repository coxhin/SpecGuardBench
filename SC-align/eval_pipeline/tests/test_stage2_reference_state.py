import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from . import test_concurrency as _stubs

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.config import BaselineConfig, EvalPipelineConfig
from eval_pipeline.models import (
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    EvalTask,
    SurvivingSpec,
)


def _make_task(spec_id: str = "S1", mutant_patch: str = "S1.mutant.patch") -> EvalTask:
    return EvalTask(
        instance_id="example.instance.lv1",
        image_name="example-image",
        surviving_specs=[
            SurvivingSpec(
                spec={"spec_id": spec_id, "category": "behavioral", "text": "t"},
                mutant_patch=mutant_patch,
            )
        ],
        f2p_tests=[],
        p2p_tests=[],
        repo_settings={},
        mask_patch="mask.patch",
        gold_patch="gold.patch",
    )


def _make_executor(tmp_path: Path, reference_state: str):
    from eval_pipeline.stages.stage2_baseline_execution import BaselineExecutor

    config = EvalPipelineConfig(
        output_dir=tmp_path,
        harness_form="factory_oracle",
        stage2_reference_state=reference_state,
    )
    config_loader = mock.MagicMock()
    config_loader.get_agent_env_vars.return_value = {}
    return BaselineExecutor(config, config_loader, mock.MagicMock())


def _capture_code_state_call(tmp_path: Path, reference_state: str):
    executor = _make_executor(tmp_path, reference_state)
    task = _make_task()
    bcfg = BaselineConfig(
        name="codex", model="gpt-5.6-sol", agent_config="codex", tool_version="1.0"
    )

    prepare = mock.MagicMock(return_value=(False, "stop here"))
    with mock.patch(
        "eval_pipeline.stages.stage2_baseline_execution.prepare_task_code_state",
        prepare,
    ):
        executor._run_single(task, bcfg)

    assert prepare.call_count == 1
    return prepare.call_args.kwargs, task


def test_gold_reference_shows_clean_implementation(tmp_path):
    kwargs, _task = _capture_code_state_call(tmp_path, "gold")

    assert kwargs["target_state"] == TARGET_STATE_GOLD
    assert kwargs["surviving_spec"] is None


def test_mutant_reference_shows_the_certified_specs_own_mutant(tmp_path):
    kwargs, task = _capture_code_state_call(tmp_path, "mutant")

    assert kwargs["target_state"] == TARGET_STATE_MUTANT
    assert kwargs["surviving_spec"] is task.surviving_specs[0]
    assert kwargs["surviving_spec"].mutant_patch == "S1.mutant.patch"


def test_mutant_reference_hides_the_gold_tree_from_the_agent(tmp_path):
    kwargs, _task = _capture_code_state_call(tmp_path, "mutant")

    assert kwargs["remove_pristine_copy"] is True


def test_gold_reference_keeps_the_gold_tree(tmp_path):
    kwargs, _task = _capture_code_state_call(tmp_path, "gold")

    assert kwargs["remove_pristine_copy"] is False


def test_default_is_gold_so_normal_runs_are_unaffected():
    assert EvalPipelineConfig().stage2_reference_state == "gold"


@pytest.mark.parametrize("harness_form", ["form1b_form6", "hypofuzz"])
def test_mutant_reference_refused_under_per_instance_forms(harness_form):
    from eval_pipeline.run import _require_per_spec_form_for_mutant_reference

    config = EvalPipelineConfig(
        harness_form=harness_form, stage2_reference_state="mutant"
    )
    with pytest.raises(SystemExit) as excinfo:
        _require_per_spec_form_for_mutant_reference(config)
    assert excinfo.value.code == 1


@pytest.mark.parametrize("harness_form", ["saved_inputs", "factory_oracle"])
def test_mutant_reference_allowed_under_per_spec_forms(harness_form):
    from eval_pipeline.run import _require_per_spec_form_for_mutant_reference

    config = EvalPipelineConfig(
        harness_form=harness_form, stage2_reference_state="mutant"
    )
    _require_per_spec_form_for_mutant_reference(config)


@pytest.mark.parametrize("harness_form", ["form1b_form6", "factory_oracle"])
def test_gold_reference_never_refused(harness_form):
    from eval_pipeline.run import _require_per_spec_form_for_mutant_reference

    config = EvalPipelineConfig(harness_form=harness_form)
    _require_per_spec_form_for_mutant_reference(config)


def _render_factory_oracle_prompt(tmp_path: Path, reference_state: str) -> str:
    from eval_pipeline.stages.stage2_baseline_execution import _render_prompt

    executor = _make_executor(tmp_path, reference_state)
    return _render_prompt(
        executor.prompt_env, _make_task(), harness_form="factory_oracle",
        reference_state=reference_state,
    )


def test_gold_reference_prompt_asks_for_a_clean_run_on_the_implementation(tmp_path):
    prompt = _render_factory_oracle_prompt(tmp_path, "gold")
    assert "run without error against the provided implementation" in prompt
    assert "Write `/tmp/eval_output.json` as soon as" in prompt


def test_mutant_reference_prompt_drops_the_clean_run_clause(tmp_path):
    prompt = _render_factory_oracle_prompt(tmp_path, "mutant")
    assert "run without error against the provided implementation" not in prompt
    assert "Write `/tmp/eval_output.json` as soon as" in prompt
    assert "before you start\nimproving coverage" in prompt or "before you start improving coverage" in prompt


def test_executor_renders_the_prompt_for_its_reference_state(tmp_path):
    from eval_pipeline.stages import stage2_baseline_execution as s2

    executor = _make_executor(tmp_path, "mutant")
    task = _make_task()
    bcfg = BaselineConfig(
        name="codex", model="gpt-5.6-sol", agent_config="codex", tool_version="1.0"
    )
    render = mock.MagicMock(return_value="prompt")
    with mock.patch.object(s2, "_render_prompt", render), mock.patch.object(
        s2, "prepare_task_code_state", return_value=(False, "stop here")
    ):
        executor._run_single(task, bcfg)
    assert render.call_args.kwargs["reference_state"] == "mutant"
