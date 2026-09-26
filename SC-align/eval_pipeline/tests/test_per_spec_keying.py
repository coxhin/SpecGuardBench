from __future__ import annotations

from typing import Any, Dict, List, Optional

from eval_pipeline.models import ProofExecResult
from eval_pipeline.stages.stage5_metrics import (
    _aggregate_metric_A,
    _aggregate_mutant_lines,
)

BASELINE = "claude_code"
MODEL = "opus"


def _func_block(summary: Dict[str, Any], executed_lines: List[int]) -> Dict[str, Any]:
    return {
        "summary": summary,
        "executed_lines": executed_lines,
        "missing_lines": [],
        "executed_branches": [],
        "missing_branches": [],
    }


def _gold_trial(
    instance_id: str,
    spec_id: str,
    covered_lines: int,
    baseline: str = BASELINE,
    model: str = MODEL,
) -> ProofExecResult:
    files = {
        "/testbed/a.py": {
            "executed_lines": [],
            "missing_lines": [],
            "executed_branches": [],
            "missing_branches": [],
            "summary": {},
            "functions": {
                "f": _func_block(
                    {
                        "num_statements": 2,
                        "covered_lines": covered_lines,
                        "num_branches": 0,
                        "covered_branches": 0,
                    },
                    executed_lines=[10, 11][:covered_lines],
                ),
            },
        },
    }
    return ProofExecResult(
        instance_id=instance_id,
        baseline=baseline,
        model=model,
        spec_id=spec_id,
        target_state="gold",
        triggered=False,
        exit_code=0,
        kind="passed",
        proof_output="",
        execution_time_seconds=1.0,
        coverage={
            "files": files,
            "extended_scope": [{"file": "a.py", "name": "f"}],
            "scope_resolution": {"complete": True, "unresolved_calls": []},
        },
    )


def _mutant_lines_trial(
    instance_id: str,
    spec_id: str,
    state: str,
    hit_rate: float,
    baseline: str = BASELINE,
    model: str = MODEL,
) -> ProofExecResult:
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
        execution_time_seconds=1.0,
        coverage={
            "files": {},
            "mutant_lines": {
                "hit_rate": hit_rate,
                "total": 10,
                "hit": int(hit_rate * 10),
                "detail": {},
            },
        },
    )


def test_metric_A_keeps_same_spec_id_from_different_instances():
    prs = [
        _gold_trial("inst-a", "S1", covered_lines=2),
        _gold_trial("inst-b", "S1", covered_lines=1),
    ]
    r = _aggregate_metric_A(prs)

    assert set(r["per_spec"]) == {
        f"inst-a::{BASELINE}_{MODEL}::S1",
        f"inst-b::{BASELINE}_{MODEL}::S1",
    }
    assert r["per_spec"][f"inst-a::{BASELINE}_{MODEL}::S1"]["line_coverage_pct"] == 100.0
    assert r["per_spec"][f"inst-b::{BASELINE}_{MODEL}::S1"]["line_coverage_pct"] == 50.0


def test_metric_A_keeps_same_spec_id_from_different_baselines():
    prs = [
        _gold_trial("inst-a", "S1", covered_lines=2, baseline="claude_code"),
        _gold_trial("inst-a", "S1", covered_lines=1, baseline="codex", model="gpt"),
    ]
    r = _aggregate_metric_A(prs)

    assert set(r["per_spec"]) == {
        "inst-a::claude_code_opus::S1",
        "inst-a::codex_gpt::S1",
    }


def test_metric_A_specs_total_matches_specs_with_data():
    prs = [
        _gold_trial("inst-a", "S1", covered_lines=2),
        _gold_trial("inst-a", "S2", covered_lines=2),
        _gold_trial("inst-b", "S1", covered_lines=1),
        _gold_trial("inst-b", "S2", covered_lines=1),
    ]
    r = _aggregate_metric_A(prs)

    assert r["counts"]["specs_with_data"] == 4
    assert r["counts"]["specs_total"] == 4


def test_metric_A_skipped_entries_are_also_fully_keyed():
    def _no_coverage(instance_id: str) -> ProofExecResult:
        pr = _gold_trial(instance_id, "S1", covered_lines=2)
        pr.coverage = None
        return pr

    r = _aggregate_metric_A([_no_coverage("inst-a"), _no_coverage("inst-b")])

    assert set(r["per_spec"]) == {
        f"inst-a::{BASELINE}_{MODEL}::S1",
        f"inst-b::{BASELINE}_{MODEL}::S1",
    }
    assert all(v == {"skipped": True} for v in r["per_spec"].values())


def test_mutant_lines_keeps_same_spec_id_from_different_instances():
    prs = [
        _mutant_lines_trial("inst-a", "S1", "gold", 1.0),
        _mutant_lines_trial("inst-a", "S1", "mutant", 0.9),
        _mutant_lines_trial("inst-b", "S1", "gold", 0.2),
        _mutant_lines_trial("inst-b", "S1", "mutant", 0.1),
    ]
    agg = _aggregate_mutant_lines(prs)

    assert set(agg["per_spec"]) == {
        f"inst-a::{BASELINE}_{MODEL}::S1",
        f"inst-b::{BASELINE}_{MODEL}::S1",
    }
    assert agg["per_spec"][f"inst-a::{BASELINE}_{MODEL}::S1"] == {
        "gold": 1.0, "mutant": 0.9,
    }
    assert agg["per_spec"][f"inst-b::{BASELINE}_{MODEL}::S1"] == {
        "gold": 0.2, "mutant": 0.1,
    }


def test_mutant_lines_keeps_same_spec_id_from_different_baselines():
    prs = [
        _mutant_lines_trial("inst-a", "S1", "gold", 1.0, baseline="claude_code"),
        _mutant_lines_trial(
            "inst-a", "S1", "gold", 0.3, baseline="codex", model="gpt"
        ),
    ]
    agg = _aggregate_mutant_lines(prs)

    assert agg["per_spec"]["inst-a::claude_code_opus::S1"]["gold"] == 1.0
    assert agg["per_spec"]["inst-a::codex_gpt::S1"]["gold"] == 0.3


def test_mutant_lines_state_means_were_never_affected():
    prs = [
        _mutant_lines_trial("inst-a", "S1", "gold", 1.0),
        _mutant_lines_trial("inst-b", "S1", "gold", 0.0),
    ]
    agg = _aggregate_mutant_lines(prs)

    assert agg["hit_rate_mean_gold"] == 0.5
    assert agg["counts"]["trials_with_mutant_lines_block"] == 2
