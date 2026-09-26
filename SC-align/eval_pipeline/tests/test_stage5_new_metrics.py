from typing import Any, Dict, List, Optional

from eval_pipeline.models import EvalTask, ProofExecResult, SurvivingSpec
from eval_pipeline.stages.stage5_metrics import (
    _aggregate_metric_A,
    _aggregate_metric_C_mean,
    _aggregate_metric_C_per_instance,
    _aggregate_metric_D_mean,
    _aggregate_metric_D_per_instance,
    _spec_to_cert_rate,
)


def _spec(
    spec_id: str,
    file: str,
    func: str,
    category: str = "test",
) -> Dict[str, Any]:
    return {
        "spec_id": spec_id,
        "category": category,
        "code_mapping": [{"file": file, "function": func}],
    }


def _surviving(spec: Dict[str, Any]) -> SurvivingSpec:
    return SurvivingSpec(spec=spec, mutant_patch="")


def _task(
    iid: str,
    specs: List[Dict[str, Any]],
    gold_funcs: List[Dict[str, str]],
) -> EvalTask:
    return EvalTask(
        instance_id=iid,
        image_name="dummy",
        surviving_specs=[_surviving(s) for s in specs],
        f2p_tests=[],
        p2p_tests=[],
        repo_settings={},
        gold_patch_functions=gold_funcs,
    )


def _trial(
    instance_id: str,
    baseline: str,
    model: str,
    spec_id: str,
    state: str,
    scoped_function: Optional[Dict[str, Any]] = None,
    extended_scope: Optional[List[Dict[str, str]]] = None,
    scope_resolution: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
) -> ProofExecResult:
    cov = None
    if (
        scoped_function is not None
        or files is not None
        or extended_scope is not None
        or scope_resolution is not None
    ):
        cov = {
            "files": files or {},
        }
        if scoped_function is not None:
            cov["scoped_function"] = scoped_function
        if extended_scope is not None:
            cov["extended_scope"] = extended_scope
        if scope_resolution is not None:
            cov["scope_resolution"] = scope_resolution
    return ProofExecResult(
        instance_id=instance_id,
        baseline=baseline,
        model=model,
        spec_id=spec_id,
        target_state=state,
        triggered=False,
        exit_code=0,
        kind="passed",
        proof_output="",
        execution_time_seconds=0.0,
        coverage=cov,
    )


def _func_block(file, name, summary, executed_lines=None, executed_branches=None,
                missing_branches=None):
    return {
        "summary": summary,
        "executed_lines": executed_lines or [],
        "missing_lines": [],
        "executed_branches": executed_branches or [],
        "missing_branches": missing_branches or [],
    }


def _file_with_functions(file, functions):
    return {
        "executed_lines": [],
        "missing_lines": [],
        "executed_branches": [],
        "missing_branches": [],
        "functions": functions,
        "summary": {},
    }


def test_A_gold_only():
    prs = [
        _trial("iid", "b", "m", "S1", "gold", scoped_function={
            "function": "f1", "file": "a.py",
            "branch_coverage_pct": 100.0, "line_coverage_pct": 100.0,
            "covered_branches": 2, "num_branches": 2,
            "covered_lines": 3, "num_statements": 3,
            "executed_lines": [1, 2, 3], "executed_branches": [[1, 2], [1, 3]],
            "missing_branches": [],
        }),
        _trial("iid", "b", "m", "S1", "mutant", scoped_function={
            "function": "f1", "file": "a.py",
            "branch_coverage_pct": 50.0, "line_coverage_pct": 50.0,
            "covered_branches": 1, "num_branches": 2,
            "covered_lines": 1, "num_statements": 3,
            "executed_lines": [1], "executed_branches": [[1, 2]],
            "missing_branches": [[1, 3]],
        }),
    ]
    r = _aggregate_metric_A(prs)
    assert r["reference_state"] == "gold"
    assert r["edge_depth"]["n"] == 1
    assert r["edge_depth"]["mean"] == 100.0
    assert "iid::b_m::S1" in r["per_spec"]

    r = _aggregate_metric_A(prs, reference_state="mutant")
    assert r["reference_state"] == "mutant"
    assert r["edge_depth"]["n"] == 1
    assert r["edge_depth"]["mean"] == 50.0


def test_A_with_extended_scope():
    files = {
        "/testbed/a.py": _file_with_functions("a.py", {
            "f1": _func_block("a.py", "f1", {
                "num_statements": 2, "covered_lines": 2,
                "num_branches": 2, "covered_branches": 1,
            }, executed_lines=[10, 11], executed_branches=[[10, 11]],
               missing_branches=[[10, 12]]),
            "f2": _func_block("a.py", "f2", {
                "num_statements": 1, "covered_lines": 1,
                "num_branches": 0, "covered_branches": 0,
            }, executed_lines=[20]),
        }),
    }
    prs = [
        _trial("iid", "b", "m", "S1", "gold",
               extended_scope=[{"file": "a.py", "name": "f1"},
                               {"file": "a.py", "name": "f2"}],
               scope_resolution={"complete": True, "unresolved_calls": []},
               files=files),
    ]
    r = _aggregate_metric_A(prs)
    s = r["per_spec"]["iid::b_m::S1"]
    assert s["num_functions_in_scope"] == 2
    assert s["num_statements"] == 3
    assert s["covered_lines"] == 3
    assert s["covered_edges"] == 2
    assert s["num_edges"] == 3
    assert abs(s["edge_coverage_pct"] - (200 / 3)) < 1e-9
    assert s["covered_branches"] == 1
    assert s["num_branches"] == 2
    assert s["branch_coverage_pct"] == 50.0
    assert s["scope_resolution_complete"] is True
    assert s["unresolved_calls"] == []
    assert r["counts"]["scope_complete"] == 1
    assert r["counts"]["scope_incomplete"] == 0
    assert s["line_coverage_pct"] == 100.0


def test_A_excludes_trials_without_coverage():
    prs = [
        _trial("iid", "b", "m", "S1", "gold", scoped_function={
            "function": "f1", "file": "a.py",
            "branch_coverage_pct": 80.0, "line_coverage_pct": 100.0,
            "covered_branches": 4, "num_branches": 5,
            "covered_lines": 3, "num_statements": 3,
            "executed_lines": [1, 2, 3], "executed_branches": [],
            "missing_branches": [],
        }),
        _trial("iid", "b", "m", "S2", "gold"),
    ]
    r = _aggregate_metric_A(prs)
    assert r["counts"]["specs_with_data"] == 1
    assert r["edge_depth"]["n"] == 1


def test_A_branchless_has_edge_pct():
    prs = [
        _trial("iid", "b", "m", "S1", "gold", scoped_function={
            "function": "f1", "file": "a.py",
            "branch_coverage_pct": None, "line_coverage_pct": 100.0,
            "covered_branches": 0, "num_branches": 0,
            "covered_lines": 1, "num_statements": 1,
            "executed_lines": [1], "executed_branches": [],
            "missing_branches": [],
        }),
    ]
    r = _aggregate_metric_A(prs)
    assert r["edge_depth"]["n"] == 1
    assert r["edge_depth"]["mean"] == 100.0
    assert r["line_depth"]["mean"] == 100.0


def test_C_per_instance_declared():
    task = _task(
        "iid",
        specs=[_spec("S1", "a.py", "f1"), _spec("S2", "a.py", "f2")],
        gold_funcs=[
            {"file": "a.py", "name": "f1"},
            {"file": "a.py", "name": "f2"},
            {"file": "a.py", "name": "f3_uncovered"},
        ],
    )
    r = _aggregate_metric_C_per_instance(task)
    d = r["declared"]
    assert d["gold_patch_functions_total"] == 3
    assert d["gold_patch_functions_targeted"] == 2
    assert abs(d["ratio"] - 2 / 3) < 1e-9
    assert d["missing"] == [{"file": "a.py", "name": "f3_uncovered"}]


def test_C_extended_via_proof_results():
    task = _task(
        "iid",
        specs=[_spec("S1", "a.py", "f1")],
        gold_funcs=[
            {"file": "a.py", "name": "f1"},
            {"file": "a.py", "name": "f2"},
            {"file": "a.py", "name": "f3"},
        ],
    )
    prs = [
        _trial("iid", "b", "m", "S1", "gold",
               extended_scope=[
                   {"file": "a.py", "name": "f1"},
                   {"file": "a.py", "name": "f2"},
               ]),
    ]
    r = _aggregate_metric_C_per_instance(task, prs)
    assert r["declared"]["ratio"] == 1 / 3
    assert r["extended"]["ratio"] == 2 / 3
    assert r["extended"]["missing"] == [{"file": "a.py", "name": "f3"}]


def test_C_mean_across_tasks():
    t1 = _task(
        "i1",
        specs=[_spec("S1", "a.py", "f1")],
        gold_funcs=[{"file": "a.py", "name": "f1"}],
    )
    t2 = _task(
        "i2",
        specs=[_spec("S1", "b.py", "g1")],
        gold_funcs=[
            {"file": "b.py", "name": "g1"},
            {"file": "b.py", "name": "g2"},
        ],
    )
    r = _aggregate_metric_C_mean([t1, t2])
    assert r["n_instances"] == 2
    assert abs(r["declared_ratio_mean"] - 0.75) < 1e-9


def test_C_empty_gold_funcs():
    task = _task("iid", specs=[_spec("S1", "a.py", "f1")], gold_funcs=[])
    r = _aggregate_metric_C_per_instance(task)
    assert r["declared"]["gold_patch_functions_total"] == 0
    assert r["declared"]["ratio"] is None


def test_spec_to_cert_rate():
    from eval_pipeline.models import BaselineResult
    tasks = [_task("i1", [_spec("S1", "a.py", "f1"), _spec("S2", "a.py", "f2")],
                   [{"file": "a.py", "name": "f1"}])]
    brs = [BaselineResult(
        instance_id="i1", baseline="b", model="m",
        raw_output="", certificates={"S1": None},
        cost={}, wall_time_seconds=0, llm_errors=[],
        completed_normally=True, tool_version="1",
    )]
    r = _spec_to_cert_rate(tasks, brs)
    assert r["total_specs"] == 2
    assert r["total_certs"] == 1
    assert r["rate"] == 0.5
    assert r["per_instance"]["i1"]["rate"] == 0.5


def test_spec_to_cert_rate_merges_per_atomic_results():
    from eval_pipeline.models import BaselineResult
    tasks = [_task("i1", [_spec("S1", "a.py", "f1"), _spec("S2", "a.py", "f2")],
                   [{"file": "a.py", "name": "f1"}])]
    brs = [
        BaselineResult(
            instance_id="i1", baseline="b", model="m",
            raw_output="", certificates={"S1": None},
            cost={}, wall_time_seconds=0, llm_errors=[],
            completed_normally=True, tool_version="1",
        ),
        BaselineResult(
            instance_id="i1", baseline="b", model="m",
            raw_output="", certificates={"S2": None},
            cost={}, wall_time_seconds=0, llm_errors=[],
            completed_normally=True, tool_version="1",
        ),
    ]
    r = _spec_to_cert_rate(tasks, brs)
    assert r["total_specs"] == 2
    assert r["total_certs"] == 2
    assert r["rate"] == 1.0
    assert r["per_instance"]["i1"]["rate"] == 1.0


def _trial_with_function(file: str, name: str, fblock: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "coverage": {
            "files": {
                f"/testbed/{file}": {
                    "executed_lines": fblock.get("executed_lines", []),
                    "missing_lines": fblock.get("missing_lines", []),
                    "executed_branches": fblock.get("executed_branches", []),
                    "missing_branches": fblock.get("missing_branches", []),
                    "functions": {name: fblock},
                }
            }
        }
    }


def test_D_single_cert():
    from eval_pipeline.coverage_scoping import FunctionInfo, aggregate_gold_patch_coverage
    gold = [FunctionInfo("a.py", "f1")]
    fblock = {
        "executed_lines": [10, 11],
        "missing_lines": [12],
        "executed_branches": [[10, 11]],
        "missing_branches": [[10, 12]],
        "summary": {
            "num_statements": 3, "covered_lines": 2,
            "num_branches": 2, "covered_branches": 1,
        },
    }
    trials = [_trial_with_function("a.py", "f1", fblock)]
    r = aggregate_gold_patch_coverage(trials, gold)
    assert r["covered_edges"] == 1
    assert r["total_edges"] == 2
    assert r["edge_coverage_pct"] == 50.0
    assert r["covered_branches"] == 1
    assert r["total_branches"] == 2
    assert r["branch_coverage_pct"] == 50.0


def test_D_union_across_trials():
    from eval_pipeline.coverage_scoping import FunctionInfo, aggregate_gold_patch_coverage
    gold = [FunctionInfo("a.py", "f1")]
    t1_block = {
        "executed_lines": [10, 11],
        "missing_lines": [12],
        "executed_branches": [[10, 11]],
        "missing_branches": [[10, 12]],
        "summary": {"num_statements": 3, "covered_lines": 2,
                    "num_branches": 2, "covered_branches": 1},
    }
    t2_block = {
        "executed_lines": [10, 12],
        "missing_lines": [11],
        "executed_branches": [[10, 12]],
        "missing_branches": [[10, 11]],
        "summary": {"num_statements": 3, "covered_lines": 2,
                    "num_branches": 2, "covered_branches": 1},
    }
    trials = [
        _trial_with_function("a.py", "f1", t1_block),
        _trial_with_function("a.py", "f1", t2_block),
    ]
    r = aggregate_gold_patch_coverage(trials, gold)
    assert r["covered_edges"] == 2
    assert r["total_edges"] == 2
    assert r["edge_coverage_pct"] == 100.0
    assert r["covered_branches"] == 2
    assert r["total_branches"] == 2
    assert r["branch_coverage_pct"] == 100.0


def _control_trial(spec_id: str, control_spec_id: str, kind: str, iid: str = "iid"):
    tr = _trial(iid, "b", "m", spec_id, "control")
    tr.control_spec_id = control_spec_id
    tr.kind = kind
    return tr


def _gold_trial(spec_id: str, kind: str, iid: str = "iid"):
    tr = _trial(iid, "b", "m", spec_id, "gold")
    tr.kind = kind
    return tr


def _key(spec_id: str, iid: str = "iid") -> str:
    return f"{iid}::b_m::{spec_id}"


def test_negative_controls_report_coverage():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "passed"),
        _control_trial("S1", "S2", "passed"),
        _gold_trial("S9", "passed"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["misattribution_cert_rate"] == 0.0
    assert out["control_coverage"] == 0.5
    assert out["counts"]["certs_total"] == 2
    assert out["counts"]["certs_with_control_arm"] == 1
    assert out["counts"]["certs_without_control_arm"] == 1


def test_negative_controls_silent_run_is_clean():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "passed"),
        _control_trial("S1", "S2", "passed"),
        _control_trial("S1", "S3", "passed"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["misattribution_cert_rate"] == 0.0
    assert out["misattribution_run_rate"] == 0.0
    assert out["cross_spec_firing_certs"] == []


def test_negative_controls_flag_certificate_that_fires_broadly():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "passed"),
        _gold_trial("S4", "passed"),
        _control_trial("S1", "S2", "assertion"),
        _control_trial("S1", "S3", "assertion"),
        _control_trial("S4", "S2", "passed"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["cross_spec_firing_certs"] == [_key("S1")]
    assert out["per_cert"][_key("S1")] == {"runs": 2, "fired": 2}
    assert out["misattribution_cert_rate"] == 0.5
    assert out["misattribution_run_rate"] == 2 / 3


def test_negative_controls_grid_is_reported_per_control_mutant():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "passed"),
        _gold_trial("S2", "passed"),
        _gold_trial("S3", "passed"),
        _control_trial("S1", "S9", "assertion"),
        _control_trial("S2", "S9", "assertion"),
        _control_trial("S3", "S9", "test_exception"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["per_control_mutant"]["iid::S9"] == {"runs": 3, "fired": 3}
    assert out["counts"]["control_firings"] == 3


def test_negative_controls_ignore_gold_and_mutant_arms():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _trial("iid", "b", "m", "S1", "gold"),
        _trial("iid", "b", "m", "S1", "mutant"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["counts"]["control_runs_scored"] == 0


def test_negative_controls_score_certs_that_fired_on_gold_too():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "assertion"),
        _control_trial("S1", "S2", "assertion"),
        _control_trial("S1", "S3", "assertion"),
        _gold_trial("S4", "passed"),
        _control_trial("S4", "S2", "passed"),
        _control_trial("S4", "S3", "passed"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["cross_spec_firing_certs"] == [_key("S1")]
    assert out["misattribution_cert_rate"] == 0.5
    assert out["counts"] == {
        "certs_scored": 2,
        "control_runs_scored": 4,
        "control_firings": 2,
        "certs_with_any_firing": 1,
        "controls_per_cert": {"2": 2},
        "certs_total": 2,
        "certs_with_control_arm": 2,
        "certs_without_control_arm": 0,
    }


def test_negative_controls_do_not_merge_specs_across_instances():
    from eval_pipeline.stages.stage5_metrics import _aggregate_negative_controls

    prs = [
        _gold_trial("S1", "passed", iid="alpha"),
        _gold_trial("S1", "passed", iid="beta"),
        _control_trial("S1", "S9", "assertion", iid="alpha"),
        _control_trial("S1", "S9", "assertion", iid="beta"),
    ]
    out = _aggregate_negative_controls(prs)

    assert out["counts"]["certs_with_any_firing"] == 2
    assert out["cross_spec_firing_certs"] == [_key("S1", "alpha"), _key("S1", "beta")]
    assert out["per_cert"][_key("S1", "alpha")] == {"runs": 1, "fired": 1}
    assert out["per_cert"][_key("S1", "beta")] == {"runs": 1, "fired": 1}
