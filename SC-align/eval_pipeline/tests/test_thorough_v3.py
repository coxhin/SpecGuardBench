from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Tuple
from unittest import mock

import pytest


_mock_mp = ModuleType("mutation_pipeline")
sys.modules.setdefault("mutation_pipeline", _mock_mp)
_mock_mp_models = ModuleType("mutation_pipeline.models")
_mock_mp_models.AtomicSpec = type("AtomicSpec", (), {})


class _PI:
    def __init__(self, **kw):
        self.__dict__.update(kw)


_mock_mp_models.PipelineInput = _PI
sys.modules.setdefault("mutation_pipeline.models", _mock_mp_models)
sys.modules.setdefault(
    "mutation_pipeline.docker_runner", ModuleType("mutation_pipeline.docker_runner")
)

_mock_fb = ModuleType("featurebench")
sys.modules.setdefault("featurebench", _mock_fb)
_mock_fb_infer = ModuleType("featurebench.infer")
sys.modules.setdefault("featurebench.infer", _mock_fb_infer)
_mock_fb_infer_cfg = ModuleType("featurebench.infer.config")
_mock_fb_infer_cfg.InferConfigLoader = type("InferConfigLoader", (), {})
_mock_fb_infer_cfg.DatasetLoader = type("DatasetLoader", (), {})
sys.modules.setdefault("featurebench.infer.config", _mock_fb_infer_cfg)
_mock_h = ModuleType("featurebench.harness")
sys.modules.setdefault("featurebench.harness", _mock_h)
_mock_hc = ModuleType("featurebench.harness.container")
_mock_hc.copy_to_container = mock.MagicMock()
_mock_hc.exec_run_with_timeout = mock.MagicMock(return_value=(0, b""))
sys.modules.setdefault("featurebench.harness.container", _mock_hc)
_mock_hk = ModuleType("featurebench.harness.constants")
_mock_hk.UTF8 = "utf-8"
sys.modules.setdefault("featurebench.harness.constants", _mock_hk)

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))


from eval_pipeline.adapters.base import EvalBaselineAdapter
from eval_pipeline.config import EvalPipelineConfig
from eval_pipeline.models import (
    KIND_ASSERTION,
    KIND_COLLECTION_ERROR,
    KIND_ERROR,
    KIND_INFRA,
    KIND_INVALID_CASE,
    KIND_INVALID_CERTIFICATE,
    KIND_NO_TESTS,
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
    classify_pytest_output,
)
from eval_pipeline.stages import stage3_proof_execution as s3
from eval_pipeline.stages.stage4_differential_analysis import (
    DifferentialAnalyzer,
)
from eval_pipeline.stages.stage5_metrics import (
    MetricsReporter,
    _aggregate_outcomes,
)


class TestClassifyPytestOutput:
    def test_passed(self):
        out = "===== 1 passed in 0.12s =====\n"
        assert classify_pytest_output(0, out) == KIND_PASSED

    def test_assertion_failure(self):
        out = textwrap.dedent("""
            =================================== FAILURES ===================================
            __________________________________ test_one ____________________________________
                def test_one():
            >       assert 1 == 2
            E       AssertionError: assert 1 == 2

            FAILED test_x.py::test_one - AssertionError: assert 1 == 2
            ============================== 1 failed in 0.04s ===============================
        """)
        assert classify_pytest_output(1, out) == KIND_ASSERTION

    def test_pytest_raises_mismatch_is_assertion(self):
        out = textwrap.dedent("""
            E       Failed: DID NOT RAISE <class 'ValueError'>
            FAILED test_x.py::test_raises - Failed: DID NOT RAISE
        """)
        assert classify_pytest_output(1, out) == KIND_ASSERTION

    def test_collection_error_module_not_found(self):
        out = textwrap.dedent("""
            ==================================== ERRORS ====================================
            ___________________ ERROR collecting test_hypothesis_proof.py ___________________
            ImportError while importing test module '/tmp/test_hypothesis_proof.py'.
            E   ModuleNotFoundError: No module named 'doesnotexist'
            !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
            ============================== 1 error in 0.05s ================================
        """)
        assert classify_pytest_output(2, out) == KIND_COLLECTION_ERROR

    def test_collection_error_syntax(self):
        out = textwrap.dedent("""
            ==================================== ERRORS ====================================
            ___________________ ERROR collecting test_x.py ___________________
            test_x.py:5: in <module>
                def broken( :
                          ^
            E   SyntaxError: invalid syntax
            !!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!
        """)
        assert classify_pytest_output(2, out) == KIND_COLLECTION_ERROR

    def test_no_tests(self):
        out = "no tests ran in 0.01s\n"
        assert classify_pytest_output(5, out) == KIND_NO_TESTS

    def test_other_error(self):
        out = "INTERNALERROR> something exploded\n"
        assert classify_pytest_output(3, out) == KIND_ERROR

    def test_pytest_pretty_inline_failed_marker(self):
        out = textwrap.dedent("""
            collecting ... collected 5 items
            ../tmp/test_hypothesis_proof.py::test_x FAILED  [ 40%] [ 40%]

            =================================== FAILURES ===================================
            ___________________________________ test_x ____________________________________
            >       with pytest.raises(ValueError, match='nope'):
            E       Failed: DID NOT RAISE <class 'ValueError'>

            ┃  File  ┃  Function  ┃  Line  ┃  Error Line  ┃  Error  ┃
            ┃ test_x ┃ test_x     ┃     12 ┃ pytest.raise ┃ Failed  ┃
        """)
        assert "AssertionError" not in out
        assert "\nFAILED " not in out
        assert not out.startswith("FAILED ")
        assert classify_pytest_output(1, out) == KIND_ASSERTION

    def test_unhandled_exception_in_test_body_is_assertion(self):
        out = textwrap.dedent("""
            >       assert ta.validate_python(123) == 123
            E       pydantic_core._pydantic_core.ValidationError: 1 validation error for none
            E         Input should be None [type=none_required, input_value=123, input_type=int]

            ../tmp/test_hypothesis_proof.py::test_no_steps_returns_any FAILED [ 33%]
        """)
        assert "AssertionError" not in out
        assert classify_pytest_output(1, out) == KIND_ASSERTION

    def test_collection_error_at_exit_one_still_collection_error(self):
        out = textwrap.dedent("""
            ==================================== ERRORS ====================================
            ImportError while importing test module '/tmp/test_x.py'.
            E   ModuleNotFoundError: No module named 'broken'
        """)
        assert classify_pytest_output(1, out) == KIND_COLLECTION_ERROR


def _keyed(outcomes):
    return [(f"inst::b_m::cert{n}", o) for n, o in enumerate(outcomes)]


def _outcome(gold_kind=None, mutant_kind=None):
    def trig(kind):
        if kind is None:
            return None
        if kind == KIND_PASSED:
            return False
        if kind == KIND_INFRA:
            return False
        return True

    return SpecCertOutcome(
        spec_id="S1",
        has_certificate=True,
        gold_triggered=trig(gold_kind),
        mutant_triggered=trig(mutant_kind),
        gold_kind=gold_kind,
        mutant_kind=mutant_kind,
    )


class TestSpecCertOutcomeKindAware:
    def test_valid_when_gold_passed_mutant_assertion(self):
        o = _outcome(KIND_PASSED, KIND_ASSERTION)
        assert o.cert_broken is False
        assert o.valid_cert is True
        assert o.gold_firing is False
        assert o.missed is False

    def test_valid_when_gold_passed_mutant_test_exception(self):
        o = _outcome(KIND_PASSED, KIND_TEST_EXCEPTION)
        assert o.cert_broken is False
        assert o.valid_cert is True
        assert o.gold_firing is False
        assert o.missed is False

    def test_broken_when_gold_collection_error(self):
        o = _outcome(KIND_COLLECTION_ERROR, KIND_ASSERTION)
        assert o.cert_broken is True
        assert o.valid_cert is None
        assert o.gold_firing is None
        assert o.missed is None

    def test_broken_when_gold_no_tests(self):
        o = _outcome(KIND_NO_TESTS, KIND_PASSED)
        assert o.cert_broken is True
        assert o.valid_cert is None

    def test_broken_when_gold_infra_error(self):
        o = _outcome(KIND_INFRA, KIND_ASSERTION)
        assert o.cert_broken is True
        assert o.valid_cert is None

    def test_broken_when_gold_invalid_case(self):
        o = _outcome(KIND_INVALID_CASE, KIND_ASSERTION)
        assert o.cert_broken is True
        assert o.valid_cert is None

    def test_broken_when_gold_invalid_certificate(self):
        o = _outcome(KIND_INVALID_CERTIFICATE, KIND_ASSERTION)
        assert o.cert_broken is True
        assert o.valid_cert is None

    def test_gold_firing_when_assertion_fired_on_gold(self):
        o = _outcome(KIND_ASSERTION, KIND_ASSERTION)
        assert o.cert_broken is False
        assert o.gold_firing is True
        assert o.valid_cert is False

    def test_gold_firing_when_test_exception_occurs_on_gold(self):
        o = _outcome(KIND_TEST_EXCEPTION, KIND_TEST_EXCEPTION)
        assert o.cert_broken is False
        assert o.gold_firing is True
        assert o.valid_cert is False

    def test_missed_when_mutant_passes(self):
        o = _outcome(KIND_PASSED, KIND_PASSED)
        assert o.cert_broken is False
        assert o.missed is True
        assert o.valid_cert is False

    def test_missed_when_mutant_collection_error(self):
        o = _outcome(KIND_PASSED, KIND_COLLECTION_ERROR)
        assert o.cert_broken is False
        assert o.missed is True
        assert o.valid_cert is False

    def test_data_missing_yields_none(self):
        o = _outcome(None, None)
        assert o.cert_broken is None
        assert o.valid_cert is None
        assert o.gold_firing is None
        assert o.missed is None


class TestAggregateOutcomesKindAware:
    def test_broken_certs_excluded_from_rates(self):
        outcomes = [
            _outcome(KIND_PASSED, KIND_ASSERTION),
            _outcome(KIND_PASSED, KIND_PASSED),
            _outcome(KIND_COLLECTION_ERROR, KIND_ASSERTION),
            _outcome(KIND_COLLECTION_ERROR, KIND_PASSED),
        ]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["counts"] == {
            "certs": 4, "detected": 1, "gold_firings": 0, "cross_firings": 0,
            "valid": 1, "sound": 1, "scorable": 2, "broken": 2, "missing": 0,
        }
        assert agg["broken_rate"] == 0.5
        assert agg["detection_rate"] == 0.25
        assert agg["sound_cert_rate"] == 0.25
        assert agg["scorable_rate"] == 0.5

    def test_cross_firing_demotes_an_otherwise_sound_cert(self):
        outcomes = [
            _outcome(KIND_PASSED, KIND_ASSERTION),
            _outcome(KIND_PASSED, KIND_ASSERTION),
        ]
        agg = _aggregate_outcomes(
            _keyed(outcomes), control_firing={"inst::b_m::cert1"},
        )
        assert agg["counts"]["valid"] == 2
        assert agg["counts"]["sound"] == 1
        assert agg["counts"]["cross_firings"] == 1
        assert agg["detection_rate"] == 1.0
        assert agg["cross_firing_rate"] == 0.5
        assert agg["sound_cert_rate"] == 0.5

    def test_no_control_arm_reads_as_zero_cross_firing(self):
        agg = _aggregate_outcomes(_keyed([_outcome(KIND_PASSED, KIND_ASSERTION)]))
        assert agg["cross_firing_rate"] == 0.0
        assert agg["sound_cert_rate"] == 1.0

    def test_gold_firing_only_counts_assertion_on_gold(self):
        outcomes = [
            _outcome(KIND_COLLECTION_ERROR, KIND_ASSERTION),
            _outcome(KIND_ASSERTION, KIND_PASSED),
        ]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["counts"]["gold_firings"] == 1
        assert agg["counts"]["broken"] == 1


class _FakeContainer:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.short_id = "fake0"


class FakePytestRunner:
    def __init__(self, gold_src: str, mutant_src: str):
        self.gold_src = gold_src
        self.mutant_src = mutant_src
        self.containers: dict[str, _FakeContainer] = {}
        self._state_for: dict[str, str] = {}

    def setup_instance(self, pipeline_input):
        d = Path(tempfile.mkdtemp(prefix="fakectr_"))
        c = _FakeContainer(d)
        self.containers[c.short_id] = c
        self._state_for[c.short_id] = "base"
        return c

    def reset_testbed(self, container, pipeline_input):
        impl = container.workdir / "impl.py"
        if impl.exists():
            impl.unlink()
        self._state_for[container.short_id] = "base"

    def cleanup(self, container):
        self.reset_testbed(container, None)
        try:
            container.workdir.rmdir()
        except OSError:
            pass

    def _purge_bytecode(self, container):
        cache = container.workdir / "__pycache__"
        if cache.exists():
            for p in cache.iterdir():
                p.unlink()

    def apply_patch(self, container, patch_content: str) -> bool:
        impl = container.workdir / "impl.py"
        if patch_content == "mask":
            impl.write_text("")
            self._state_for[container.short_id] = "mask"
        elif patch_content == "gold":
            impl.write_text(self.gold_src)
            self._state_for[container.short_id] = "gold"
        elif patch_content == "mutant":
            impl.write_text(self.mutant_src)
            self._state_for[container.short_id] = "mutant"
        self._purge_bytecode(container)
        return True

    def run_hypothesis_test(
        self, container, hypothesis_code: str, timeout: int,
        run_all_tests: bool = False,
    ) -> Tuple[bool, str]:
        cert_path = container.workdir / "cert_test.py"
        cert_path.write_text(hypothesis_code)

        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(container.workdir) + os.pathsep + env.get("PYTHONPATH", "")
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        pytest_args = [sys.executable, "-m", "pytest", str(cert_path)]
        if not run_all_tests:
            pytest_args.append("-x")
        pytest_args += ["-v", "--tb=long"]
        proc = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        triggered = proc.returncode != 0
        return triggered, (proc.stdout + "\n" + proc.stderr)


def _fake_prepare(container, docker_runner, task, target_state,
                  surviving_spec=None, restore_test_with_reverse_patch=False):
    docker_runner.apply_patch(container, "mask")
    docker_runner.apply_patch(container, "gold")
    if target_state == TARGET_STATE_MUTANT:
        if surviving_spec is None:
            return False, "missing surviving_spec"
        docker_runner.apply_patch(container, "mutant")
    return True, ""


GOLD_SRC = textwrap.dedent("""
    def add(x, y):
        return x + y

    def double(x):
        return x * 2
""")

MUTANT_SRC = textwrap.dedent("""
    def add(x, y):
        return x + y

    def double(x):
        return x + 2
""")


CERT_DOUBLE_GOOD = textwrap.dedent("""
    from impl import double

    def test_double_doubles_things():
        assert double(3) == 6
        assert double(0) == 0
        assert double(5) == 10
""")

CERT_ADD_NO_SIGNAL = textwrap.dedent("""
    from impl import add

    def test_add_works():
        assert add(1, 2) == 3
""")

CERT_BROKEN = textwrap.dedent("""
    import this_module_does_not_exist_anywhere_in_the_world
    from impl import double

    def test_uses_missing():
        assert double(3) == 6
""")


@pytest.fixture
def fake_runner():
    return FakePytestRunner(GOLD_SRC, MUTANT_SRC)


def _make_task() -> EvalTask:
    return EvalTask(
        instance_id="inst1",
        image_name="img",
        surviving_specs=[
            SurvivingSpec(spec={"spec_id": "S_double", "category": "behavioral",
                                "text": "double(x) returns 2*x", "code_mapping": []},
                          mutant_patch="mutant"),
            SurvivingSpec(spec={"spec_id": "S_add", "category": "behavioral",
                                "text": "add(x,y) returns x+y", "code_mapping": []},
                          mutant_patch="mutant"),
            SurvivingSpec(spec={"spec_id": "S_broken", "category": "behavioral",
                                "text": "irrelevant", "code_mapping": []},
                          mutant_patch="mutant"),
        ],
        f2p_tests=[], p2p_tests=[], repo_settings={},
        mask_patch="mask", gold_patch="gold",
    )


def _make_baseline_result() -> BaselineResult:
    return BaselineResult(
        instance_id="inst1", baseline="claude_code", model="opus",
        tool_version="x", raw_output="",
        certificates={
            "S_double": SpecCertificate(description="catches double", code=CERT_DOUBLE_GOOD),
            "S_add": SpecCertificate(description="weak", code=CERT_ADD_NO_SIGNAL),
            "S_broken": SpecCertificate(description="bad import", code=CERT_BROKEN),
        },
        cost={"input_tokens": 1, "output_tokens": 1, "total_cost_usd": 0.0},
        wall_time_seconds=0.0,
    )


class TestRealPytestEndToEnd:
    def test_full_flow(self, tmp_path, fake_runner, monkeypatch):
        monkeypatch.setattr(s3, "prepare_task_code_state", _fake_prepare)

        cfg = EvalPipelineConfig(
            output_dir=tmp_path, n_concurrent=1, hypothesis_timeout=60,
        )
        executor = s3.ProofExecutor(cfg, fake_runner)
        task = _make_task()
        br = _make_baseline_result()

        proof_results = executor.run_all([task], [br])
        assert len(proof_results) == 6

        by_key = {(p.spec_id, p.target_state): p for p in proof_results}

        p = by_key[("S_double", TARGET_STATE_GOLD)]
        assert p.kind == KIND_PASSED, f"got kind={p.kind}, output:\n{p.proof_output[-400:]}"
        assert p.triggered is False

        p = by_key[("S_double", TARGET_STATE_MUTANT)]
        assert p.kind == KIND_ASSERTION, f"got kind={p.kind}, output:\n{p.proof_output[-400:]}"
        assert p.triggered is True

        assert by_key[("S_add", TARGET_STATE_GOLD)].kind == KIND_PASSED
        assert by_key[("S_add", TARGET_STATE_MUTANT)].kind == KIND_PASSED

        p = by_key[("S_broken", TARGET_STATE_GOLD)]
        assert p.kind == KIND_COLLECTION_ERROR, f"got kind={p.kind}, output:\n{p.proof_output[-400:]}"
        assert p.triggered is True

        analyzer = DifferentialAnalyzer(cfg)
        diffs = analyzer.analyze([task], [br], proof_results)
        assert len(diffs) == 1
        per_spec = diffs[0].per_spec

        assert per_spec["S_double"].valid_cert is True
        assert per_spec["S_double"].cert_broken is False
        assert per_spec["S_double"].gold_firing is False
        assert per_spec["S_double"].missed is False

        assert per_spec["S_add"].cert_broken is False
        assert per_spec["S_add"].missed is True
        assert per_spec["S_add"].valid_cert is False

        assert per_spec["S_broken"].cert_broken is True
        assert per_spec["S_broken"].valid_cert is None
        assert per_spec["S_broken"].gold_firing is None
        assert per_spec["S_broken"].missed is None

        reporter = MetricsReporter(cfg)
        summary = reporter.compute_and_save([task], [br], diffs)
        c = summary["counts"]
        assert c["certs"] == 3
        assert c["broken"] == 1
        assert c["scorable"] == 2
        assert c["valid"] == 1
        assert c["detected"] == 1
        assert c["gold_firings"] == 0
        assert c["missing"] == 0
        assert abs(summary["sound_cert_rate"] - 1 / 3) < 1e-9
        assert abs(summary["detection_rate"] - 1 / 3) < 1e-9
        assert summary["gold_firing_rate"] == 0.0
        assert abs(summary["broken_rate"] - 1 / 3) < 1e-9
        assert abs(summary["scorable_rate"] - 2 / 3) < 1e-9

        reports = cfg.output_dir / "eval_reports"
        for name in ("summary.json", "per_baseline.json", "per_category.json",
                     "per_instance.json"):
            assert (reports / name).exists(), f"missing {name}"
        assert not (reports / "ablation_test_access.json").exists()

        per_instance = json.loads((reports / "per_instance.json").read_text())
        key = "inst1__claude_code_opus"
        assert key in per_instance
        per_spec_section = per_instance[key]["per_spec"]
        assert per_spec_section["S_double"]["valid_cert"] is True
        assert per_spec_section["S_broken"]["cert_broken"] is True

        per_baseline = json.loads((reports / "per_baseline.json").read_text())
        assert "claude_code_opus" in per_baseline
        assert per_baseline["claude_code_opus"]["counts"]["certs"] == 3


class _NullAdapter(EvalBaselineAdapter):
    def get_install_script(self):
        return ""

    def get_run_command(self, prompt_path):
        return ""


class TestExtractCertificatesFromText:
    def setup_method(self):
        self.adapter = _NullAdapter(
            name="x", model="m", tool_version="1",
            agent_config={}, env_vars={},
        )

    def test_clean_json_block(self):
        text = '{"certificates": {"S1": {"description": "d", "code": "c"}}}'
        out = self.adapter._extract_certificates_from_text(text)
        assert out is not None
        certs, cost = out
        assert "S1" in certs
        assert certs["S1"].code == "c"

    def test_embedded_in_chatter(self):
        text = (
            "Here is my output:\n"
            'Some preamble {"unrelated": 1} more text\n'
            'Final answer: {"certificates": {"S1": {"description": "d", "code": "code"}}}\n'
            "Thanks!"
        )
        out = self.adapter._extract_certificates_from_text(text)
        assert out is not None
        certs, _ = out
        assert "S1" in certs

    def test_multiline_code_with_quotes(self):
        cert_obj = {
            "certificates": {
                "S1": {
                    "description": 'has "quotes" and newlines',
                    "code": 'def test():\n    assert True, "ok"\n',
                }
            }
        }
        text = "blah blah " + json.dumps(cert_obj) + " trailing"
        out = self.adapter._extract_certificates_from_text(text)
        assert out is not None
        certs, _ = out
        assert "test():\n" in certs["S1"].code

    def test_null_certificates_field(self):
        text = '{"certificates": null}'
        out = self.adapter._extract_certificates_from_text(text)
        assert out is None

    def test_empty_certificates_dict(self):
        text = '{"certificates": {}}'
        out = self.adapter._extract_certificates_from_text(text)
        assert out is None

    def test_no_certificates_key(self):
        text = '{"other": "thing"}'
        out = self.adapter._extract_certificates_from_text(text)
        assert out is None

    def test_picks_last_certificates_block(self):
        text = (
            '{"certificates": {"S1": {"description": "old", "code": "old"}}}\n'
            'Then I revised:\n'
            '{"certificates": {"S1": {"description": "new", "code": "new"}}}\n'
        )
        out = self.adapter._extract_certificates_from_text(text)
        assert out is not None
        certs, _ = out
        assert certs["S1"].code == "new"

    def test_malformed_json_rejected(self):
        text = '{"certificates": {S1: not valid'
        out = self.adapter._extract_certificates_from_text(text)
        assert out is None


from eval_pipeline.config import MUTATION_OUTPUT_SCHEMA_VERSION
from eval_pipeline.stages.stage1_task_assembly import (
    TaskAssembler, save_eval_tasks, load_eval_tasks,
)


_GOLD_PATCH_INST1 = """\
diff --git a/inst1/m.py b/inst1/m.py
--- a/inst1/m.py
+++ b/inst1/m.py
@@ -1,2 +1,4 @@
 existing
+def f1(x):
+    return x
 last
"""

_GOLD_PATCH_INST2 = """\
diff --git a/inst2/m.py b/inst2/m.py
--- a/inst2/m.py
+++ b/inst2/m.py
@@ -1,2 +1,4 @@
 existing
+def g1(x):
+    return x
 last
"""


class TestStage1WithRealisticInput:
    def _write_mutation_output(self, tmp_path: Path) -> Path:
        rows = [
            {
                "schema_version": MUTATION_OUTPUT_SCHEMA_VERSION,
                "instance_id": "inst1",
                "spec": {"spec_id": "S1", "text": "...", "category": "behavioral",
                         "code_mapping": []},
                "mask_patch": "mp", "gold_patch": _GOLD_PATCH_INST1,
                "mutant_patch": "mut1", "test_patch": "tp",
            },
            {
                "schema_version": MUTATION_OUTPUT_SCHEMA_VERSION,
                "instance_id": "inst1",
                "spec": {"spec_id": "S2", "text": "...", "category": "structural",
                         "code_mapping": []},
                "mask_patch": "mp", "gold_patch": _GOLD_PATCH_INST1,
                "mutant_patch": "mut2", "test_patch": "tp",
            },
            {
                "schema_version": MUTATION_OUTPUT_SCHEMA_VERSION,
                "instance_id": "inst2",
                "spec": {"spec_id": "S1", "text": "...", "category": "behavioral",
                         "code_mapping": []},
                "mask_patch": "mp2", "gold_patch": _GOLD_PATCH_INST2,
                "mutant_patch": "mut3", "test_patch": "tp2",
            },
        ]
        path = tmp_path / "mutation_output.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    def test_assemble_and_roundtrip(self, tmp_path):
        mu_path = self._write_mutation_output(tmp_path)
        cfg = EvalPipelineConfig(
            output_dir=tmp_path, mutation_output=mu_path,
        )
        hf_data = {
            "inst1": {"FAIL_TO_PASS": [], "PASS_TO_PASS": [],
                      "repo_settings": {}, "image_name": "img1"},
            "inst2": {"FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": [],
                      "repo_settings": {}, "image_name": "img2"},
        }
        tasks = TaskAssembler(cfg, hf_data).assemble()

        assert len(tasks) == 2
        by_id = {t.instance_id: t for t in tasks}
        assert len(by_id["inst1"].surviving_specs) == 2
        assert len(by_id["inst2"].surviving_specs) == 1

        save_eval_tasks(tasks, tmp_path)
        loaded = load_eval_tasks(tmp_path)
        assert len(loaded) == 2
        assert {t.instance_id for t in loaded} == {"inst1", "inst2"}

        loaded_by_id = {t.instance_id: t for t in loaded}
        inst1 = loaded_by_id["inst1"]
        assert [s.mutant_patch for s in inst1.surviving_specs] == ["mut1", "mut2"]


import jinja2


class TestPromptEdgeCases:
    @pytest.fixture
    def env(self):
        prompts_dir = _SC_ALIGN / "eval_pipeline" / "prompts"
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

    def test_empty_specs_list(self, env):
        out = env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[],
        )
        assert "executable certificate" in out.lower()

    def test_code_mapping_never_reaches_the_prompt(self, env):
        spec = {"spec_id": "S1", "category": "behavioral", "text": "X must Y"}
        mapped = {**spec, "code_mapping": [
            {"file": "module/foo.py", "function": "normalize", "lines": "10-20"},
        ]}
        out = env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[mapped],
        )
        assert "S1" in out
        assert "module/foo.py" not in out
        assert out == env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[spec],
        )

    def test_test_paths_index_never_rendered(self, env):
        out = env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[{"spec_id": "S1", "category": "behavioral",
                    "text": "X", "code_mapping": []}],
        )
        assert "test_paths.json" not in out

    def test_hard_rules_are_present(self, env):
        out = env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[{"spec_id": "S1", "category": "b", "text": "X", "code_mapping": []}],
        )
        for rule_keyword in (
            "no edits", "no reference implementation",
            "match exception type", "wraps=",
        ):
            assert rule_keyword.lower() in out.lower(), f"Hard rule missing: {rule_keyword}"

    def test_output_format_block_present(self, env):
        out = env.get_template("baseline_task.j2").render(
            max_agent_steps=None,
            specs=[{"spec_id": "S1", "category": "b", "text": "X", "code_mapping": []}],
        )
        assert '"certificates"' in out
        assert "/tmp/eval_output.json" in out


class TestMissingCertsInStage4:
    def _task_with_three_specs(self) -> EvalTask:
        return EvalTask(
            instance_id="inst1",
            image_name="img",
            surviving_specs=[
                SurvivingSpec(
                    spec={"spec_id": "S1", "category": "behavioral",
                          "text": "...", "code_mapping": []},
                    mutant_patch="m1",
                ),
                SurvivingSpec(
                    spec={"spec_id": "S2", "category": "behavioral",
                          "text": "...", "code_mapping": []},
                    mutant_patch="m2",
                ),
                SurvivingSpec(
                    spec={"spec_id": "S3", "category": "behavioral",
                          "text": "...", "code_mapping": []},
                    mutant_patch="m3",
                ),
            ],
            f2p_tests=[], p2p_tests=[], repo_settings={},
            mask_patch="mask", gold_patch="gold",
        )

    def test_stage4_builds_outcomes_for_missing_certs(self, tmp_path):
        task = self._task_with_three_specs()
        br = BaselineResult(
            instance_id="inst1", baseline="b", model="m",
            tool_version="1", raw_output="",
            certificates={
                "S1": SpecCertificate(description="", code="x"),
                "S3": SpecCertificate(description="", code="y"),
            },
            cost={"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0},
            wall_time_seconds=0.0,
        )
        prs = [
            ProofExecResult(
                instance_id="inst1", baseline="b", model="m", spec_id="S1",
                target_state=TARGET_STATE_GOLD, triggered=False, exit_code=0,
                kind=KIND_PASSED, proof_output="", execution_time_seconds=0.0,
            ),
            ProofExecResult(
                instance_id="inst1", baseline="b", model="m", spec_id="S1",
                target_state=TARGET_STATE_MUTANT, triggered=True, exit_code=1,
                kind=KIND_ASSERTION, proof_output="", execution_time_seconds=0.0,
            ),
            ProofExecResult(
                instance_id="inst1", baseline="b", model="m", spec_id="S3",
                target_state=TARGET_STATE_GOLD, triggered=False, exit_code=0,
                kind=KIND_PASSED, proof_output="", execution_time_seconds=0.0,
            ),
            ProofExecResult(
                instance_id="inst1", baseline="b", model="m", spec_id="S3",
                target_state=TARGET_STATE_MUTANT, triggered=False, exit_code=0,
                kind=KIND_PASSED, proof_output="", execution_time_seconds=0.0,
            ),
        ]
        cfg = EvalPipelineConfig(output_dir=tmp_path)
        diffs = DifferentialAnalyzer(cfg).analyze([task], [br], prs)
        assert len(diffs) == 1
        per_spec = diffs[0].per_spec
        assert set(per_spec.keys()) == {"S1", "S2", "S3"}, (
            "Stage 4 must enumerate surviving_specs, not br.certificates"
        )
        assert per_spec["S2"].has_certificate is False
        assert per_spec["S2"].gold_triggered is None
        assert per_spec["S2"].mutant_triggered is None
        assert per_spec["S1"].has_certificate is True
        assert per_spec["S1"].valid_cert is True
        assert per_spec["S3"].has_certificate is True
        assert per_spec["S3"].missed is True


class TestTotalDenominatorRegression:
    def _missing_outcome(self, spec_id: str) -> SpecCertOutcome:
        return SpecCertOutcome(
            spec_id=spec_id, has_certificate=False,
            gold_triggered=None, mutant_triggered=None,
            gold_kind=None, mutant_kind=None,
        )

    def _valid_outcome(self, spec_id: str) -> SpecCertOutcome:
        return SpecCertOutcome(
            spec_id=spec_id, has_certificate=True,
            gold_triggered=False, mutant_triggered=True,
            gold_kind=KIND_PASSED, mutant_kind=KIND_ASSERTION,
        )

    def _broken_outcome(self, spec_id: str) -> SpecCertOutcome:
        return SpecCertOutcome(
            spec_id=spec_id, has_certificate=True,
            gold_triggered=True, mutant_triggered=True,
            gold_kind=KIND_COLLECTION_ERROR, mutant_kind=KIND_ASSERTION,
        )

    def test_missing_certs_hurt_rate(self):
        outcomes = [
            self._valid_outcome("S1"),
            self._missing_outcome("S2"),
            self._missing_outcome("S3"),
            self._missing_outcome("S4"),
            self._missing_outcome("S5"),
        ]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["counts"]["certs"] == 5
        assert agg["counts"]["missing"] == 4
        assert agg["counts"]["valid"] == 1
        assert agg["counts"]["detected"] == 1
        assert agg["counts"]["scorable"] == 1
        assert agg["detection_rate"] == 0.2
        assert agg["sound_cert_rate"] == 0.2
        assert agg["scorable_rate"] == 0.2

    def test_broken_certs_hurt_rate(self):
        outcomes = [self._valid_outcome("S0")] + [
            self._broken_outcome(f"S{i}") for i in range(1, 10)
        ]
        agg = _aggregate_outcomes(_keyed(outcomes))
        assert agg["counts"]["certs"] == 10
        assert agg["counts"]["broken"] == 9
        assert agg["counts"]["valid"] == 1
        assert agg["sound_cert_rate"] == 0.1
        assert agg["broken_rate"] == 0.9

    def test_mutant_only_error_counts_in_denominator_not_numerator(self):
        outcome_with_mutant_error = SpecCertOutcome(
            spec_id="S1", has_certificate=True,
            gold_triggered=False, mutant_triggered=True,
            gold_kind=KIND_PASSED, mutant_kind=KIND_ASSERTION,
            mutant_error="boom",
        )
        assert outcome_with_mutant_error.valid_cert is None
        assert outcome_with_mutant_error.missed is None
        assert outcome_with_mutant_error.cert_broken is False

        agg = _aggregate_outcomes(_keyed([outcome_with_mutant_error]))
        assert agg["counts"]["certs"] == 1
        assert agg["counts"]["detected"] == 0
        assert agg["counts"]["valid"] == 0
        assert agg["counts"]["sound"] == 0
        assert agg["counts"]["broken"] == 0
        assert agg["counts"]["missing"] == 0
        assert agg["counts"]["scorable"] == 0
        assert agg["detection_rate"] == 0.0
        assert agg["sound_cert_rate"] == 0.0
