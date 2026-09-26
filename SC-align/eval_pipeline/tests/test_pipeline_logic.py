import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest import mock


_mock_mutation_models = ModuleType("mutation_pipeline.models")
_mock_mutation_models.AtomicSpec = type("AtomicSpec", (), {})
sys.modules.setdefault("mutation_pipeline", ModuleType("mutation_pipeline"))
sys.modules.setdefault("mutation_pipeline.models", _mock_mutation_models)

_mock_featurebench = ModuleType("featurebench")
_mock_featurebench_infer = ModuleType("featurebench.infer")
_mock_featurebench_infer_config = ModuleType("featurebench.infer.config")
_mock_featurebench_infer_config.InferConfigLoader = type("InferConfigLoader", (), {})
_mock_featurebench_infer_config.DatasetLoader = type("DatasetLoader", (), {})
sys.modules.setdefault("featurebench", _mock_featurebench)
sys.modules.setdefault("featurebench.infer", _mock_featurebench_infer)
sys.modules.setdefault("featurebench.infer.config", _mock_featurebench_infer_config)

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

import jinja2
import pytest

from eval_pipeline.models import (
    EVAL_SCHEMA_VERSION,
    KIND_ASSERTION,
    KIND_PASSED,
    KIND_TEST_EXCEPTION,
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    BaselineResult,
    DifferentialResult,
    EvalTask,
    ProofExecResult,
    SpecCertOutcome,
    SpecCertificate,
    SurvivingSpec,
)
from eval_pipeline.config import EvalPipelineConfig, load_mutation_outputs
from eval_pipeline.stages.stage1_task_assembly import (
    _parse_repo_settings,
    _parse_test_list,
)
from eval_pipeline.stages.stage4_differential_analysis import DifferentialAnalyzer
from eval_pipeline.stages.stage5_metrics import MetricsReporter, _aggregate_outcomes


def _make_task(iid="inst1", surviving_specs=None):
    if surviving_specs is None:
        surviving_specs = [
            SurvivingSpec(
                spec={"spec_id": "S1", "category": "behavioral"},
                mutant_patch="diff for S1",
            )
        ]
    return EvalTask(
        instance_id=iid,
        image_name="featurebench/django:latest",
        surviving_specs=surviving_specs,
        f2p_tests=["tests/test_a.py::test_one"],
        p2p_tests=["tests/test_b.py::test_two"],
        repo_settings={"install": "pip install -e ."},
        mask_patch="mask.diff",
        gold_patch="gold.diff",
        test_patch="test.diff",
    )


class TestEvalTaskSerialization:
    def test_roundtrip_basic(self):
        task = _make_task()
        d = task.to_dict()
        restored = EvalTask.from_dict(d)
        assert restored.instance_id == task.instance_id
        assert restored.schema_version == EVAL_SCHEMA_VERSION
        assert len(restored.surviving_specs) == 1
        assert restored.surviving_specs[0].spec_id == "S1"
        assert restored.surviving_specs[0].mutant_patch == "diff for S1"

    def test_roundtrip_via_jsonl(self):
        task = _make_task()
        line = task.to_jsonl()
        d = json.loads(line)
        restored = EvalTask.from_dict(d)
        assert restored.instance_id == task.instance_id

    def test_from_dict_missing_keys(self):
        task = EvalTask.from_dict({})
        assert task.schema_version == EVAL_SCHEMA_VERSION
        assert task.instance_id == ""
        assert task.surviving_specs == []
        assert task.f2p_tests == []

    def test_all_specs_property(self):
        task = _make_task(surviving_specs=[
            SurvivingSpec(spec={"spec_id": "S1"}, mutant_patch="m1"),
            SurvivingSpec(spec={"spec_id": "S2"}, mutant_patch="m2"),
        ])
        assert task.all_specs == [{"spec_id": "S1"}, {"spec_id": "S2"}]


class TestSpecCertificateSerialization:
    def test_roundtrip(self):
        c = SpecCertificate(description="checks X", code="from hypothesis import given\n...")
        restored = SpecCertificate.from_dict(c.to_dict())
        assert restored.description == c.description
        assert restored.code == c.code

    def test_roundtrip_with_saved_inputs(self):
        c = SpecCertificate(
            description="replays corpus",
            code="import os\nPATH = os.environ['AGENT_SAVED_INPUTS_PATH']\n",
            saved_inputs=[{"args": [1, 2], "kwargs": {}}],
            inputs_file="baseline_saved_inputs/b_m/inst1/S1.jsonl",
        )
        restored = SpecCertificate.from_dict(c.to_dict())
        assert restored.saved_inputs == [{"args": [1, 2], "kwargs": {}}]
        assert restored.inputs_file == "baseline_saved_inputs/b_m/inst1/S1.jsonl"

    def test_from_dict_empty(self):
        restored = SpecCertificate.from_dict({})
        assert restored.description == ""
        assert restored.code == ""


class TestBaselineResultSerialization:
    def _make_result(self, **overrides):
        defaults = dict(
            instance_id="inst1",
            baseline="claude_code",
            model="opus-4",
            tool_version="1.0.30",
            raw_output="some raw output",
            certificates={
                "S1": SpecCertificate(description="d1", code="c1"),
                "S2": SpecCertificate(description="d2", code="c2"),
            },
            cost={"input_tokens": 1000, "output_tokens": 500, "total_cost_usd": 0.05},
            wall_time_seconds=42.5,
        )
        defaults.update(overrides)
        return BaselineResult(**defaults)

    def test_roundtrip(self):
        br = self._make_result()
        restored = BaselineResult.from_dict(br.to_dict())
        assert restored.instance_id == br.instance_id
        assert "S1" in restored.certificates
        assert restored.certificates["S1"].code == "c1"
        assert restored.cost["total_cost_usd"] == 0.05

    def test_empty_certificates(self):
        br = self._make_result(certificates={})
        restored = BaselineResult.from_dict(br.to_dict())
        assert restored.certificates == {}

    def test_from_dict_missing_keys(self):
        restored = BaselineResult.from_dict({})
        assert restored.instance_id == ""
        assert restored.certificates == {}
        assert restored.cost == {}


class TestProofExecResultSerialization:
    def test_roundtrip(self):
        pr = ProofExecResult(
            instance_id="inst1",
            baseline="claude_code",
            model="opus-4",
            spec_id="S1",
            target_state=TARGET_STATE_MUTANT,
            triggered=True,
            exit_code=1,
            proof_output="AssertionError: ...",
            execution_time_seconds=5.2,
            error=None,
        )
        restored = ProofExecResult.from_dict(pr.to_dict())
        assert restored.spec_id == "S1"
        assert restored.target_state == TARGET_STATE_MUTANT
        assert restored.triggered is True

    def test_with_error(self):
        pr = ProofExecResult(
            instance_id="x", baseline="b", model="m", spec_id="S1",
            target_state=TARGET_STATE_GOLD, triggered=False, exit_code=0,
            proof_output="", execution_time_seconds=0.0, error="timeout",
        )
        restored = ProofExecResult.from_dict(pr.to_dict())
        assert restored.error == "timeout"


class TestSpecCertOutcome:
    def test_valid_cert_true_when_passes_gold_fires_mutant(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=False, mutant_triggered=True,
            gold_kind=KIND_PASSED, mutant_kind=KIND_ASSERTION,
        )
        assert o.valid_cert is True
        assert o.gold_firing is False
        assert o.missed is False

    def test_valid_cert_true_when_mutant_has_test_exception(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=False, mutant_triggered=True,
            gold_kind=KIND_PASSED, mutant_kind=KIND_TEST_EXCEPTION,
        )
        assert o.valid_cert is True
        assert o.gold_firing is False
        assert o.missed is False

    def test_test_exception_on_gold_is_gold_firing(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=True, mutant_triggered=True,
            gold_kind=KIND_TEST_EXCEPTION, mutant_kind=KIND_TEST_EXCEPTION,
        )
        assert o.cert_broken is False
        assert o.valid_cert is False
        assert o.gold_firing is True

    def test_valid_cert_false_when_gold_firing(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=True, mutant_triggered=True,
            gold_kind=KIND_ASSERTION, mutant_kind=KIND_ASSERTION,
        )
        assert o.valid_cert is False
        assert o.gold_firing is True

    def test_valid_cert_false_when_missed(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=False, mutant_triggered=False,
            gold_kind=KIND_PASSED, mutant_kind=KIND_PASSED,
        )
        assert o.valid_cert is False
        assert o.missed is True

    def test_valid_cert_none_when_outcomes_missing(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=None, mutant_triggered=True,
            gold_kind=None, mutant_kind=KIND_ASSERTION,
        )
        assert o.valid_cert is None

    def test_valid_cert_none_when_error(self):
        o = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=False, mutant_triggered=True,
            gold_kind=KIND_PASSED, mutant_kind=KIND_ASSERTION,
            mutant_error="boom",
        )
        assert o.valid_cert is None


class TestDifferentialResultSerialization:
    def test_roundtrip(self):
        dr = DifferentialResult(
            instance_id="inst1", baseline="b", model="m",
            per_spec={
                "S1": SpecCertOutcome(
                    spec_id="S1", has_certificate=True,
                    gold_triggered=False, mutant_triggered=True,
                    gold_kind=KIND_PASSED, mutant_kind=KIND_ASSERTION,
                ),
            },
        )
        restored = DifferentialResult.from_dict(dr.to_dict())
        assert restored.instance_id == "inst1"
        assert restored.per_spec["S1"].valid_cert is True


class TestLoadMutationOutputs:
    def test_valid_multiline(self, tmp_path):
        f = tmp_path / "output.jsonl"
        records = [
            {"instance_id": "a", "spec": {"spec_id": "S1"}},
            {"instance_id": "a", "spec": {"spec_id": "S2"}},
            {"instance_id": "b", "spec": {"spec_id": "S3"}},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = load_mutation_outputs(f)
        assert set(result.keys()) == {"a", "b"}
        assert len(result["a"]) == 2

    def test_empty_file(self, tmp_path):
        f = tmp_path / "output.jsonl"
        f.write_text("")
        assert load_mutation_outputs(f) == {}

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "output.jsonl"
        content = (
            '{"instance_id": "a", "ok": true}\n'
            'THIS IS NOT JSON\n'
            '{"instance_id": "b", "ok": true}\n'
            '{bad json\n'
        )
        f.write_text(content)
        result = load_mutation_outputs(f)
        assert {"a", "b"} == set(result.keys())

    def test_nonexistent_file(self, tmp_path):
        assert load_mutation_outputs(tmp_path / "nope.jsonl") == {}


class TestParseTestList:
    def test_none(self):
        assert _parse_test_list(None) == []

    def test_valid_list(self):
        assert _parse_test_list(["a", "b"]) == ["a", "b"]

    def test_json_string_of_list(self):
        assert _parse_test_list(json.dumps(["a"])) == ["a"]

    def test_invalid_json(self):
        assert _parse_test_list("not json [") == []


class TestParseRepoSettings:
    def test_none(self):
        assert _parse_repo_settings(None) == {}

    def test_dict(self):
        d = {"install": "pip install -e ."}
        assert _parse_repo_settings(d) == d

    def test_json_string(self):
        assert _parse_repo_settings(json.dumps({"k": "v"})) == {"k": "v"}


def _make_baseline_result_with_certs(iid, baseline, model, certs_map):
    certs = {sid: SpecCertificate(description=sid, code=code)
             for sid, code in certs_map.items()}
    return BaselineResult(
        instance_id=iid, baseline=baseline, model=model,
        tool_version="1.0", raw_output="", certificates=certs,
        cost={"input_tokens": 10, "output_tokens": 5, "total_cost_usd": 0.001},
        wall_time_seconds=1.0,
    )


def _make_proof(iid, baseline, model, spec_id, target_state, triggered,
                error=None, kind=None):
    if kind is None:
        kind = KIND_ASSERTION if triggered else KIND_PASSED
    return ProofExecResult(
        instance_id=iid, baseline=baseline, model=model,
        spec_id=spec_id, target_state=target_state,
        triggered=triggered, exit_code=1 if triggered else 0,
        kind=kind, proof_output="", execution_time_seconds=0.1, error=error,
    )


class TestDifferentialAnalyzer:
    def _run(self, tasks, brs, prs):
        cfg = mock.MagicMock()
        cfg.output_dir = Path(tempfile.mkdtemp())
        return DifferentialAnalyzer(cfg).analyze(tasks, brs, prs)

    def test_valid_cert(self):
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        prs = [
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_GOLD, triggered=False),
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_MUTANT, triggered=True),
        ]
        results = self._run([task], [br], prs)
        assert len(results) == 1
        outcome = results[0].per_spec["S1"]
        assert outcome.valid_cert is True
        assert outcome.gold_firing is False
        assert outcome.missed is False

    def test_gold_firing(self):
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        prs = [
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_GOLD, triggered=True),
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_MUTANT, triggered=True),
        ]
        results = self._run([task], [br], prs)
        outcome = results[0].per_spec["S1"]
        assert outcome.valid_cert is False
        assert outcome.gold_firing is True

    def test_missed(self):
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        prs = [
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_GOLD, triggered=False),
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_MUTANT, triggered=False),
        ]
        results = self._run([task], [br], prs)
        outcome = results[0].per_spec["S1"]
        assert outcome.valid_cert is False
        assert outcome.missed is True

    def test_missing_proof_results_yields_none_outcome(self):
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        results = self._run([task], [br], [])
        outcome = results[0].per_spec["S1"]
        assert outcome.gold_triggered is None
        assert outcome.mutant_triggered is None
        assert outcome.valid_cert is None

    def test_error_propagates_to_outcome(self):
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        prs = [
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_GOLD, triggered=False),
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_MUTANT, triggered=False, error="boom"),
        ]
        results = self._run([task], [br], prs)
        outcome = results[0].per_spec["S1"]
        assert outcome.mutant_error == "boom"
        assert outcome.valid_cert is None

    def test_merges_per_atomic_baseline_results(self):
        task = _make_task(surviving_specs=[
            SurvivingSpec(spec={"spec_id": "S1", "category": "behavioral"}, mutant_patch="m1"),
            SurvivingSpec(spec={"spec_id": "S2", "category": "behavioral"}, mutant_patch="m2"),
        ])
        br1 = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code1"})
        br2 = _make_baseline_result_with_certs("inst1", "b", "m", {"S2": "code2"})
        prs = [
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_GOLD, triggered=False),
            _make_proof("inst1", "b", "m", "S1", TARGET_STATE_MUTANT, triggered=True),
            _make_proof("inst1", "b", "m", "S2", TARGET_STATE_GOLD, triggered=False),
            _make_proof("inst1", "b", "m", "S2", TARGET_STATE_MUTANT, triggered=False),
        ]

        results = self._run([task], [br1, br2], prs)

        assert len(results) == 1
        assert set(results[0].per_spec) == {"S1", "S2"}
        assert results[0].per_spec["S1"].valid_cert is True
        assert results[0].per_spec["S2"].missed is True


def _kind(triggered: bool) -> str:
    return KIND_ASSERTION if triggered else KIND_PASSED


def _outcome_kw(spec_id: str, has_cert: bool, gold_t: bool, mut_t: bool):
    return SpecCertOutcome(
        spec_id=spec_id, has_certificate=has_cert,
        gold_triggered=gold_t, mutant_triggered=mut_t,
        gold_kind=_kind(gold_t), mutant_kind=_kind(mut_t),
    )


def _keyed(outcomes):
    return [(f"inst::b_m::cert{n}", o) for n, o in enumerate(outcomes)]


class TestAggregateOutcomes:
    def _make(self, gold, mutant):
        return SpecCertOutcome(
            spec_id="X", has_certificate=True,
            gold_triggered=gold, mutant_triggered=mutant,
            gold_kind=_kind(gold), mutant_kind=_kind(mutant),
        )

    def test_all_sound(self):
        outcomes = [self._make(False, True), self._make(False, True)]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["detection_rate"] == 1.0
        assert agg["gold_firing_rate"] == 0.0
        assert agg["cross_firing_rate"] == 0.0
        assert agg["sound_cert_rate"] == 1.0
        assert agg["counts"]["valid"] == 2
        assert agg["counts"]["sound"] == 2

    def test_mixed(self):
        outcomes = [
            self._make(False, True),
            self._make(False, False),
            self._make(True, True),
        ]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["detection_rate"] == 2 / 3
        assert agg["gold_firing_rate"] == 1 / 3
        assert agg["sound_cert_rate"] == 1 / 3

    def test_control_firing_lowers_sound_but_not_detection(self):
        outcomes = [self._make(False, True), self._make(False, True)]
        agg = _aggregate_outcomes(
            _keyed(outcomes), control_firing={"inst::b_m::cert0"},
        )
        assert agg["detection_rate"] == 1.0
        assert agg["cross_firing_rate"] == 0.5
        assert agg["sound_cert_rate"] == 0.5

    def test_empty(self):
        agg = _aggregate_outcomes([])
        assert agg["detection_rate"] == 0
        assert agg["gold_firing_rate"] == 0
        assert agg["cross_firing_rate"] == 0
        assert agg["sound_cert_rate"] == 0


class TestMetricsReporter:
    def _make_reporter(self):
        cfg = mock.MagicMock()
        cfg.output_dir = Path(tempfile.mkdtemp())
        return MetricsReporter(cfg)

    def _make_dr(self, iid, per_spec):
        return DifferentialResult(
            instance_id=iid, baseline="b", model="m", per_spec=per_spec,
        )

    def test_summary_aggregation(self):
        reporter = self._make_reporter()
        dr = self._make_dr("i1", {
            "S1": _outcome_kw("S1", True, False, True),
            "S2": _outcome_kw("S2", True, True, True),
        })
        summary = reporter._compute_summary([dr], {})
        assert summary["counts"]["certs"] == 2
        assert summary["counts"]["valid"] == 1
        assert summary["sound_cert_rate"] == 0.5

    def test_summary_control_firing_reaches_the_headline(self):
        reporter = self._make_reporter()
        dr = self._make_dr("i1", {
            "S1": _outcome_kw("S1", True, False, True),
            "S2": _outcome_kw("S2", True, False, True),
        })
        summary = reporter._compute_summary(
            [dr], {}, control_firing={"i1::b_m::S2"},
        )
        assert summary["counts"]["valid"] == 2
        assert summary["counts"]["sound"] == 1
        assert summary["cross_firing_rate"] == 0.5
        assert summary["sound_cert_rate"] == 0.5

    def test_per_baseline_grouping(self):
        reporter = self._make_reporter()
        dr1 = DifferentialResult(
            instance_id="i1", baseline="b1", model="m",
            per_spec={"S1": _outcome_kw("S1", True, False, True)},
        )
        dr2 = DifferentialResult(
            instance_id="i1", baseline="b2", model="m",
            per_spec={"S1": _outcome_kw("S1", True, True, False)},
        )
        per_baseline = reporter._compute_per_baseline([dr1, dr2], {})
        assert per_baseline["b1_m"]["sound_cert_rate"] == 1.0
        assert per_baseline["b2_m"]["sound_cert_rate"] == 0.0

    def test_per_category_uses_spec_categories(self):
        reporter = self._make_reporter()
        task = _make_task(surviving_specs=[
            SurvivingSpec(spec={"spec_id": "S1", "category": "behavioral"}, mutant_patch="m"),
            SurvivingSpec(spec={"spec_id": "S2", "category": "structural"}, mutant_patch="m"),
        ])
        dr = self._make_dr("inst1", {
            "S1": _outcome_kw("S1", True, False, True),
            "S2": _outcome_kw("S2", True, True, True),
        })
        spec_categories = {s.spec_id: s.spec.get("category", "unknown")
                           for s in task.surviving_specs}
        per_cat = reporter._compute_per_category([dr], spec_categories)
        assert per_cat["behavioral"]["sound_cert_rate"] == 1.0
        assert per_cat["structural"]["sound_cert_rate"] == 0.0

    def test_per_category_sees_control_firings(self):
        reporter = self._make_reporter()
        dr = self._make_dr("inst1", {
            "S1": _outcome_kw("S1", True, False, True),
            "S2": _outcome_kw("S2", True, False, True),
        })
        per_cat = reporter._compute_per_category(
            [dr],
            {"S1": "behavioral", "S2": "behavioral"},
            control_firing={"inst1::b_m::S2"},
        )
        assert per_cat["behavioral"]["cross_firing_rate"] == 0.5
        assert per_cat["behavioral"]["sound_cert_rate"] == 0.5

    def test_compute_and_save_writes_summary(self):
        reporter = self._make_reporter()
        task = _make_task()
        dr = self._make_dr("inst1", {
            "S1": _outcome_kw("S1", True, False, True),
        })
        summary = reporter.compute_and_save([task], [], [dr])
        assert (reporter.config.output_dir / "eval_reports" / "summary.json").exists()
        assert summary["counts"]["valid"] == 1

    def test_saved_inputs_mode_writes_coverage_metrics_without_fuzz_compliance(self):
        cfg = mock.MagicMock()
        cfg.output_dir = Path(tempfile.mkdtemp())
        cfg.harness_form = "saved_inputs"
        reporter = MetricsReporter(cfg)
        task = _make_task()
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        dr = self._make_dr("inst1", {
            "S1": _outcome_kw("S1", True, False, True),
        })
        proof = ProofExecResult(
            instance_id="inst1",
            baseline="b",
            model="m",
            spec_id="S1",
            target_state=TARGET_STATE_GOLD,
            triggered=False,
            exit_code=0,
            kind=KIND_PASSED,
            proof_output="",
            execution_time_seconds=0.1,
            coverage={
                "scoped_function": {
                    "function": "f",
                    "file": "m.py",
                    "branch_coverage_pct": 100.0,
                    "line_coverage_pct": 100.0,
                    "covered_branches": 1,
                    "num_branches": 1,
                    "covered_lines": 1,
                    "num_statements": 1,
                },
                "files": {},
            },
        )

        summary = reporter.compute_and_save([task], [br], [dr], proof_results=[proof])

        assert "metric_A" in summary
        assert "spec_to_cert" in summary
        assert "fuzz_compliance" not in summary
        assert summary["spec_to_cert"]["total_certs"] == 1

    def test_every_stratification_carries_the_certificate_blocks(self):
        reporter = self._make_reporter()
        task = _make_task(surviving_specs=[
            SurvivingSpec(spec={"spec_id": "S1", "category": "behavioral"},
                          mutant_patch="m"),
        ])
        br = _make_baseline_result_with_certs("inst1", "b", "m", {"S1": "code"})
        dr = self._make_dr("inst1", {
            "S1": _outcome_kw("S1", True, False, True),
        })

        reporter.compute_and_save([task], [br], [dr], proof_results=[])

        reports = reporter.config.output_dir / "eval_reports"
        cert_blocks = {"negative_controls", "unsound_certs", "sound_certs"}
        summary = json.loads((reports / "summary.json").read_text())
        per_baseline = json.loads((reports / "per_baseline.json").read_text())
        per_instance = json.loads((reports / "per_instance.json").read_text())
        baseline_entry = next(iter(per_baseline.values()))
        instance_entry = next(iter(per_instance.values()))

        assert cert_blocks <= set(summary)
        assert cert_blocks <= set(baseline_entry)
        assert cert_blocks <= set(instance_entry)

    def test_analysis_blocks_are_scoped_to_their_slice(self):
        from eval_pipeline.stages.stage5_metrics import _analysis_blocks

        dr1 = DifferentialResult(
            instance_id="i1", baseline="b", model="m",
            per_spec={"S1": _outcome_kw("S1", True, False, True)},
        )
        dr2 = DifferentialResult(
            instance_id="i2", baseline="b", model="m",
            per_spec={"S1": _outcome_kw("S1", True, False, True)},
        )

        both = _analysis_blocks([dr1, dr2], [])
        just_one = _analysis_blocks([dr1], [])

        assert both["sound_certs"]["counts"]["certs"] == 2
        assert just_one["sound_certs"]["counts"]["certs"] == 1
        assert both["unsound_certs"]["counts"]["certs"] == 2
        assert just_one["unsound_certs"]["counts"]["certs"] == 1


class TestBaselineTaskTemplate:
    @pytest.fixture
    def prompt_env(self):
        prompts_dir = Path(__file__).parent.parent / "prompts"
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

    def _make_specs(self):
        return [
            {
                "spec_id": "S1", "category": "behavioral",
                "text": "Must normalize values to [0,1]",
                "code_mapping": [
                    {"file": "module/foo.py", "function": "normalize", "lines": "10-20"},
                ],
            },
        ]

    def test_executable_certificate_language_present(self, prompt_env):
        result = prompt_env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=self._make_specs(),
        )
        assert "executable certificate" in result.lower()
        assert "form 1b + form 6" in result.lower() or "form1b" in result.lower() or "Form 1b" in result
        assert "in_function_assertions" not in result

    def test_code_mapping_is_never_rendered(self, prompt_env):
        result = prompt_env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=self._make_specs(),
        )
        assert "**Code mapping:**" not in result
        assert "module/foo.py" not in result
        assert "10-20" not in result

    def test_test_paths_index_is_never_rendered(self, prompt_env):
        result = prompt_env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=self._make_specs(),
        )
        assert "test_paths.json" not in result


class TestEvalPipelineConfig:
    def test_provide_code_mapping_is_gone(self):
        cfg = EvalPipelineConfig()
        assert not hasattr(cfg, "provide_code_mapping")

    def test_provide_test_paths_is_gone(self):
        cfg = EvalPipelineConfig()
        assert not hasattr(cfg, "provide_test_paths")
