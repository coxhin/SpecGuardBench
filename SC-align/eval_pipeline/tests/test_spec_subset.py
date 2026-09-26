import sys
from pathlib import Path
from unittest import mock

import pytest

from . import test_concurrency as _stubs

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.config import EvalPipelineConfig
from eval_pipeline.models import EvalTask, SurvivingSpec


def _task(iid: str, spec_ids) -> EvalTask:
    return EvalTask(
        instance_id=iid,
        image_name="img",
        surviving_specs=[
            SurvivingSpec(spec={"spec_id": s, "category": "behavioral", "text": "t"},
                          mutant_patch=f"{s}.patch")
            for s in spec_ids
        ],
        f2p_tests=[], p2p_tests=[], repo_settings={},
        mask_patch="m", gold_patch="g",
    )


def _executor(tmp_path: Path, harness_form: str, subset):
    from eval_pipeline.stages.stage2_baseline_execution import BaselineExecutor

    config = EvalPipelineConfig(
        output_dir=tmp_path, harness_form=harness_form, spec_subset=subset,
    )
    loader = mock.MagicMock()
    loader.get_agent_env_vars.return_value = {}
    return BaselineExecutor(config, loader, mock.MagicMock())


def test_subset_launches_only_listed_specs(tmp_path):
    tasks = [_task("a.lv1", ["S1", "S2", "S3"]), _task("b.lv1", ["S1", "S2"])]
    ex = _executor(tmp_path, "factory_oracle", [("a.lv1", "S2"), ("b.lv1", "S1")])
    launched = ex._expand_tasks_for_harness(tasks)
    assert [(t.instance_id, t.surviving_specs[0].spec_id) for t in launched] == [
        ("a.lv1", "S2"), ("b.lv1", "S1"),
    ]
    assert [len(t.surviving_specs) for t in tasks] == [3, 2]


def test_no_subset_launches_everything(tmp_path):
    tasks = [_task("a.lv1", ["S1", "S2"])]
    ex = _executor(tmp_path, "factory_oracle", None)
    assert len(ex._expand_tasks_for_harness(tasks)) == 2


def test_unmatched_entries_are_skipped_not_fatal(tmp_path, caplog):
    tasks = [_task("a.lv1", ["S1"])]
    ex = _executor(tmp_path, "factory_oracle", [("a.lv1", "S1"), ("a.lv1", "S9")])
    launched = ex._expand_tasks_for_harness(tasks)
    assert len(launched) == 1
    assert "a.lv1/S9" in caplog.text


def test_subset_matching_nothing_is_an_error(tmp_path):
    tasks = [_task("a.lv1", ["S1"])]
    ex = _executor(tmp_path, "factory_oracle", [("zzz.lv1", "S1")])
    with pytest.raises(ValueError):
        ex._expand_tasks_for_harness(tasks)


def test_subset_file_parsing(tmp_path):
    from eval_pipeline.run import _read_spec_subset_or_exit

    f = tmp_path / "s.tsv"
    f.write_text("# header\n\na.lv1\tS1\nb.lv1 S2   # trailing\na.lv1\tS1\n")
    assert _read_spec_subset_or_exit(f) == [("a.lv1", "S1"), ("b.lv1", "S2")]

    bad = tmp_path / "bad.tsv"
    bad.write_text("a.lv1\n")
    with pytest.raises(SystemExit):
        _read_spec_subset_or_exit(bad)

    empty = tmp_path / "empty.tsv"
    empty.write_text("# nothing\n")
    with pytest.raises(SystemExit):
        _read_spec_subset_or_exit(empty)


def test_subset_needs_per_spec_form():
    from eval_pipeline.run import _require_per_spec_form_for_spec_subset

    ok = EvalPipelineConfig(harness_form="factory_oracle", spec_subset=[("a", "S1")])
    _require_per_spec_form_for_spec_subset(ok)
    bad = EvalPipelineConfig(harness_form="form1b_form6", spec_subset=[("a", "S1")])
    with pytest.raises(SystemExit):
        _require_per_spec_form_for_spec_subset(bad)


from eval_pipeline.models import (
    BaselineResult, DifferentialResult, SpecCertOutcome, KIND_PASSED, KIND_ASSERTION,
)
from eval_pipeline.stages.stage5_metrics import (
    MetricsReporter, _scope_to_spec_subset, _spec_to_cert_rate,
)


def _outcome(sid: str, has_cert: bool = True, fired_on_mutant: bool = True) -> SpecCertOutcome:
    if not has_cert:
        return SpecCertOutcome(
            spec_id=sid, has_certificate=False,
            gold_triggered=None, mutant_triggered=None,
        )
    return SpecCertOutcome(
        spec_id=sid, has_certificate=True,
        gold_triggered=False, mutant_triggered=fired_on_mutant,
        gold_kind=KIND_PASSED,
        mutant_kind=KIND_ASSERTION if fired_on_mutant else KIND_PASSED,
    )


def _dr(iid: str, per_spec) -> DifferentialResult:
    return DifferentialResult(instance_id=iid, baseline="b", model="m", per_spec=per_spec)


def _br(iid: str, cert_ids) -> BaselineResult:
    return BaselineResult(
        instance_id=iid, baseline="b", model="m", tool_version="", raw_output="",
        certificates={sid: {"code": "x"} for sid in cert_ids},
        cost={}, wall_time_seconds=0.0,
    )


def test_scope_without_subset_is_the_identity():
    drs = [_dr("a.lv1", {"S1": _outcome("S1"), "S2": _outcome("S2", has_cert=False)})]
    assert _scope_to_spec_subset(drs, None) is drs
    assert _scope_to_spec_subset(drs, []) is drs


def test_scope_keeps_only_sampled_specs_and_drops_empty_instances():
    drs = [
        _dr("a.lv1", {"S1": _outcome("S1"), "S2": _outcome("S2", has_cert=False),
                      "S3": _outcome("S3", has_cert=False)}),
        _dr("b.lv1", {"S1": _outcome("S1", has_cert=False)}),
    ]
    scoped = _scope_to_spec_subset(drs, [("a.lv1", "S1"), ("a.lv1", "S2")])
    assert [(d.instance_id, sorted(d.per_spec)) for d in scoped] == [("a.lv1", ["S1", "S2"])]
    assert scoped[0].per_spec["S2"].has_certificate is False
    assert sorted(drs[0].per_spec) == ["S1", "S2", "S3"]


def test_spec_to_cert_counts_only_offered_specs():
    tasks = [_task("a.lv1", ["S1", "S2", "S3"])]
    brs = [_br("a.lv1", ["S1"])]
    full = _spec_to_cert_rate(tasks, brs)
    assert (full["total_specs"], full["total_certs"]) == (3, 1)
    sampled = _spec_to_cert_rate(tasks, brs, [("a.lv1", "S1"), ("a.lv1", "S2")])
    assert (sampled["total_specs"], sampled["total_certs"]) == (2, 1)
    assert sampled["rate"] == 0.5


def _run_stage5(tmp_path: Path, subset):
    config = EvalPipelineConfig(
        output_dir=tmp_path / ("sub" if subset else "full"),
        harness_form="factory_oracle", spec_subset=subset,
    )
    tasks = [_task("a.lv1", ["S1", "S2", "S3"]), _task("b.lv1", ["S1", "S2"])]
    brs = [_br("a.lv1", ["S1", "S2"]), _br("b.lv1", ["S1"])]
    drs = [
        _dr("a.lv1", {"S1": _outcome("S1"), "S2": _outcome("S2", fired_on_mutant=False),
                      "S3": _outcome("S3", has_cert=False)}),
        _dr("b.lv1", {"S1": _outcome("S1"), "S2": _outcome("S2", has_cert=False)}),
    ]
    return MetricsReporter(config).compute_and_save(tasks, brs, drs, [])


def test_stage5_headline_denominator_is_the_sample(tmp_path):
    full = _run_stage5(tmp_path, None)
    assert full["counts"]["certs"] == 5
    assert full["counts"]["missing"] == 2
    assert full["detection_rate"] == 2 / 5
    assert "spec_subset" not in full

    sampled = _run_stage5(tmp_path, [("a.lv1", "S1"), ("a.lv1", "S2"), ("b.lv1", "S1")])
    assert sampled["counts"]["certs"] == 3
    assert sampled["counts"]["missing"] == 0
    assert sampled["counts"]["detected"] == full["counts"]["detected"] == 2
    assert sampled["detection_rate"] == 2 / 3
    assert sampled["sound_certs"]["counts"]["certs"] == 3
    assert sampled["unsound_certs"]["counts"]["certs"] == 3
    assert sampled["spec_to_cert"]["total_specs"] == 3
    assert sampled["spec_subset"] == {"path": None, "specs_listed": 3, "certs_scored": 3}


def test_stage5_subset_naming_every_spec_changes_nothing(tmp_path):
    full = _run_stage5(tmp_path, None)
    everything = [("a.lv1", "S1"), ("a.lv1", "S2"), ("a.lv1", "S3"),
                  ("b.lv1", "S1"), ("b.lv1", "S2")]
    same = _run_stage5(tmp_path / "all", everything)
    marker = same.pop("spec_subset")
    assert marker["certs_scored"] == 5
    assert same == full
