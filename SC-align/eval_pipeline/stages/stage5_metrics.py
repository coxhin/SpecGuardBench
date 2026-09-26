"""Stage 5: metrics and reporting."""

import json
import logging
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Set, Tuple

from ..adapters.base import SUCCESSFUL_LLM_CALLS
from ..coverage_scoping import (
    FunctionInfo,
    aggregate_gold_patch_coverage,
    function_edge_stats,
    scope_to_function,
    spec_breadth,
)
from ..models import (
    SEMANTIC_FAILURE_KINDS,
    TARGET_STATE_CONTROL,
    TARGET_STATE_MUTANT,
    BaselineResult,
    DifferentialResult,
    EvalTask,
    ProofExecResult,
    SpecCertOutcome,
    KIND_PASSED,
    TARGET_STATE_GOLD,
)

logger = logging.getLogger(__name__)


def _format_cert_key(
    instance_id: str, baseline: str, model: str, spec_id: str
) -> str:
    """The one spelling of a certificate's identity."""
    return f"{instance_id}::{baseline}_{model}::{spec_id}"


def _outcome_cert_key(dr: DifferentialResult, spec_id: str) -> str:
    """Cert key for one entry of a differential result."""
    return _format_cert_key(dr.instance_id, dr.baseline, dr.model, spec_id)


def _outcome_keys(
    dr: DifferentialResult,
) -> Iterable[Tuple[str, SpecCertOutcome]]:
    """One ``(cert_key, outcome)`` pair per spec in a differential result."""
    for spec_id, outcome in dr.per_spec.items():
        yield _outcome_cert_key(dr, spec_id), outcome


def _scope_to_spec_subset(
    differential_results: List[DifferentialResult],
    spec_subset: Optional[List[Tuple[str, str]]],
) -> List[DifferentialResult]:
    """Drop the outcomes of specs the run never launched."""
    if not spec_subset:
        return differential_results
    wanted = {tuple(pair) for pair in spec_subset}
    scoped: List[DifferentialResult] = []
    for dr in differential_results:
        kept = {
            sid: outcome for sid, outcome in dr.per_spec.items()
            if (dr.instance_id, sid) in wanted
        }
        if kept:
            scoped.append(replace(dr, per_spec=kept))
    return scoped


def _aggregate_outcomes(
    outcomes: Iterable[Tuple[str, SpecCertOutcome]],
    control_firing: AbstractSet[str] = frozenset(),
) -> Dict[str, Any]:
    """Compute headline counts + rates from cert outcomes keyed by cert identity."""
    total = 0
    detected = 0
    gold_firings = 0
    cross_firings = 0
    valid = 0
    sound = 0
    scorable = 0
    broken = 0
    missing = 0

    for key, o in outcomes:
        total += 1
        fired_on_control = key in control_firing
        if fired_on_control:
            cross_firings += 1
        if o.has_certificate is False:
            missing += 1
            continue
        if o.cert_broken is True:
            broken += 1
            continue
        if o.gold_firing is True:
            gold_firings += 1
        if o.missed is False:
            detected += 1
        valid_v = o.valid_cert
        if valid_v is not None:
            scorable += 1
            if valid_v:
                valid += 1
                if not fired_on_control:
                    sound += 1

    return {
        "detection_rate": detected / total if total else 0,
        "gold_firing_rate": gold_firings / total if total else 0,
        "cross_firing_rate": cross_firings / total if total else 0,
        "sound_cert_rate": sound / total if total else 0,
        "broken_rate": broken / total if total else 0,
        "scorable_rate": scorable / total if total else 0,
        "counts": {
            "certs": total,
            "detected": detected,
            "gold_firings": gold_firings,
            "cross_firings": cross_firings,
            "valid": valid,
            "sound": sound,
            "scorable": scorable,
            "broken": broken,
            "missing": missing,
        },
    }


def _group_baseline_results(
    baseline_results: List[BaselineResult],
) -> Dict[str, List[BaselineResult]]:
    """Group baseline_results by (baseline_model) key matching per_baseline keys."""
    groups: Dict[str, List[BaselineResult]] = defaultdict(list)
    for br in baseline_results:
        groups[f"{br.baseline}_{br.model}"].append(br)
    return groups


def _quantiles(xs: List[float]) -> Dict[str, Any]:
    """Distribution block for a list of counts."""
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None,
                "p25": None, "p75": None, "n": 0}
    s = sorted(xs)

    def _pct(p: float) -> float:
        if len(s) == 1:
            return float(s[0])
        pos = p / 100.0 * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))

    return {
        "mean": sum(s) / len(s),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "p25": _pct(25),
        "p75": _pct(75),
        "n": len(s),
    }


def _aggregate_agent_steps(
    baseline_results: List[BaselineResult],
) -> Dict[str, Any]:
    """How much work each baseline run took, in model calls and agent steps."""
    calls: List[float] = []
    requests: List[float] = []
    attempts: List[float] = []
    step_values: Dict[str, List[float]] = defaultdict(list)
    n_with_calls = 0
    n_capped = 0
    n_lost_to_provider = 0

    for br in baseline_results:
        if br.provider_pending is not None:
            n_lost_to_provider += 1
        cost = br.cost or {}
        got = cost.get(SUCCESSFUL_LLM_CALLS)
        if isinstance(got, (int, float)):
            calls.append(float(got))
            n_with_calls += 1
        req = cost.get("api_requests")
        if isinstance(req, (int, float)):
            requests.append(float(req))

        records = br.attempts or []
        attempts.append(float(len(records)))
        if any(record.get("step_cap") for record in records):
            n_capped += 1
        totals: Dict[str, float] = defaultdict(float)
        for record in records:
            for key, value in (record.get("steps") or {}).items():
                if isinstance(value, (int, float)):
                    totals[key] += float(value)
        for key, value in totals.items():
            step_values[key].append(value)

    block: Dict[str, Any] = {
        "successful_llm_calls": _quantiles(calls),
        "api_requests": _quantiles(requests),
        "attempts": _quantiles(attempts),
        "steps": {k: _quantiles(v) for k, v in sorted(step_values.items())},
        "counts": {
            "runs": len(baseline_results),
            "runs_with_call_count": n_with_calls,
            "successful_llm_calls_total": int(sum(calls)) if calls else 0,
            "api_requests_total": int(sum(requests)) if requests else 0,
            "requests_without_completion": (
                int(sum(requests) - sum(calls)) if requests and calls else None
            ),
            "runs_stopped_by_step_cap": n_capped,
            "runs_lost_to_provider": n_lost_to_provider,
        },
    }
    return block


def _aggregate_fuzz_compliance(
    baseline_results: List[BaselineResult],
) -> Dict[str, Any]:
    """Compute fuzz-compliance rates across a list of baseline results."""
    total = 0
    first_try_ok = 0
    final_ok = 0
    for br in baseline_results:
        fc = br.fuzz_compliance or {}
        for entry in fc.values():
            total += 1
            if entry.get("first_try_compliant"):
                first_try_ok += 1
            if entry.get("final_compliant"):
                final_ok += 1
    return {
        "fuzz_compliance_first_try_rate": first_try_ok / total if total else 0,
        "fuzz_compliance_final_rate": final_ok / total if total else 0,
        "counts": {
            "certs_audited": total,
            "first_try_compliant": first_try_ok,
            "final_compliant": final_ok,
        },
    }


def _unsound_certs(
    differential_results: List[DifferentialResult],
    negative_controls: Dict[str, Any],
) -> Dict[str, Any]:
    """Union the two ways a certificate can accuse code it should have cleared."""
    firing_on_controls = set(negative_controls.get("cross_spec_firing_certs", []))

    total = 0
    gold_only = 0
    control_only = 0
    both = 0
    for dr in differential_results:
        for key, outcome in _outcome_keys(dr):
            total += 1
            on_gold = outcome.gold_firing is True
            on_control = key in firing_on_controls
            if on_gold and on_control:
                both += 1
            elif on_gold:
                gold_only += 1
            elif on_control:
                control_only += 1

    unsound = gold_only + control_only + both
    return {
        "unsound_cert_rate": unsound / total if total else 0,
        "counts": {
            "certs": total,
            "unsound": unsound,
            "gold_only": gold_only,
            "control_only": control_only,
            "both": both,
        },
    }


def _sound_certs(
    differential_results: List[DifferentialResult],
    negative_controls: Dict[str, Any],
) -> Dict[str, Any]:
    """The exclusion breakdown behind the ``sound_cert_rate`` headline."""
    firing = set(negative_controls.get("cross_spec_firing_certs", []))
    with_control_arm = set(negative_controls.get("per_cert", {}))

    total = 0
    sound = 0
    not_valid = 0
    control_firing = 0
    sound_untested = 0
    for dr in differential_results:
        for key, outcome in _outcome_keys(dr):
            total += 1
            if outcome.valid_cert is not True:
                not_valid += 1
            elif key in firing:
                control_firing += 1
            else:
                sound += 1
                if key not in with_control_arm:
                    sound_untested += 1

    return {
        "sound_cert_rate": sound / total if total else 0,
        "counts": {
            "certs": total,
            "sound": sound,
            "not_valid": not_valid,
            "control_firing": control_firing,
            "sound_untested_on_controls": sound_untested,
        },
    }


def _cert_key(pr: ProofExecResult) -> str:
    """Identity of one certificate across its arms."""
    return _format_cert_key(pr.instance_id, pr.baseline, pr.model, pr.spec_id)


def _control_mutant_key(pr: ProofExecResult) -> str:
    """Identity of the foreign mutant applied for one control run."""
    return f"{pr.instance_id}::{pr.control_spec_id or '?'}"


def _aggregate_negative_controls(
    proof_results: List[ProofExecResult],
) -> Dict[str, Any]:
    """Summarize the negative-control arm."""
    all_certs: set = set()
    certs_with_an_arm: set = set()
    for pr in proof_results:
        if pr.target_state == TARGET_STATE_GOLD:
            all_certs.add(_cert_key(pr))
        elif pr.target_state == TARGET_STATE_CONTROL:
            certs_with_an_arm.add(_cert_key(pr))

    scored_runs = 0
    fired = 0
    by_cert: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"runs": 0, "fired": 0}
    )
    by_control_mutant: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"runs": 0, "fired": 0}
    )

    for pr in proof_results:
        if pr.target_state != TARGET_STATE_CONTROL:
            continue
        cert = _cert_key(pr)
        scored_runs += 1
        by_cert[cert]["runs"] += 1
        control_id = _control_mutant_key(pr)
        by_control_mutant[control_id]["runs"] += 1
        if pr.kind in SEMANTIC_FAILURE_KINDS:
            fired += 1
            by_cert[cert]["fired"] += 1
            by_control_mutant[control_id]["fired"] += 1

    firing_certs = sorted(k for k, c in by_cert.items() if c["fired"] > 0)
    controls_per_cert: Dict[str, int] = {}
    for counts in by_cert.values():
        key = str(counts["runs"])
        controls_per_cert[key] = controls_per_cert.get(key, 0) + 1
    controls_per_cert = dict(sorted(controls_per_cert.items(), key=lambda kv: int(kv[0])))

    return {
        "misattribution_cert_rate": (
            len(firing_certs) / len(by_cert) if by_cert else 0
        ),
        "misattribution_run_rate": fired / scored_runs if scored_runs else 0,
        "control_coverage": (
            len(certs_with_an_arm) / len(all_certs) if all_certs else 0
        ),
        "cross_spec_firing_certs": firing_certs,
        "per_cert": {k: dict(v) for k, v in by_cert.items()},
        "per_control_mutant": {k: dict(v) for k, v in by_control_mutant.items()},
        "counts": {
            "certs_scored": len(by_cert),
            "control_runs_scored": scored_runs,
            "control_firings": fired,
            "certs_with_any_firing": len(firing_certs),
            "controls_per_cert": controls_per_cert,
            "certs_total": len(all_certs),
            "certs_with_control_arm": len(certs_with_an_arm),
            "certs_without_control_arm": len(all_certs - certs_with_an_arm),
        },
    }


def _aggregate_mutant_lines(
    proof_results: List[ProofExecResult],
) -> Dict[str, Any]:
    """Aggregate mutant-line hit-rate per trial by target_state and spec."""
    hit_by_state: Dict[str, List[float]] = defaultdict(list)
    hit_by_cert: Dict[str, Dict[str, Optional[float]]] = defaultdict(
        lambda: {"gold": None, "mutant": None}
    )

    n_total = 0
    n_with_block = 0
    for pr in proof_results:
        n_total += 1
        cov = pr.coverage
        if cov is None:
            continue
        ml = cov.get("mutant_lines")
        if not ml:
            continue
        n_with_block += 1
        rate = ml.get("hit_rate")
        if rate is None:
            continue
        hit_by_state[pr.target_state].append(float(rate))
        hit_by_cert[_cert_key(pr)][pr.target_state] = float(rate)

    def _mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "hit_rate_mean": _mean(
            [v for vs in hit_by_state.values() for v in vs]
        ),
        "hit_rate_mean_gold": _mean(hit_by_state.get("gold", [])),
        "hit_rate_mean_mutant": _mean(hit_by_state.get("mutant", [])),
        "per_spec": dict(hit_by_cert),
        "counts": {
            "trials_total": n_total,
            "trials_with_mutant_lines_block": n_with_block,
        },
    }


def _compute_per_spec_scope_coverage(
    pr: ProofExecResult,
) -> Optional[Dict[str, Any]]:
    """Compute coverage over the extended scope for one trial."""
    cov = pr.coverage
    if not cov:
        return None
    ext_scope = cov.get("extended_scope")
    if not ext_scope:
        sf = cov.get("scoped_function")
        if sf is None:
            return None
        edge_stats = function_edge_stats(sf)
        edge_pct = sf.get("edge_coverage_pct", edge_stats["edge_coverage_pct"])
        return {
            "edge_coverage_pct": edge_pct,
            "branch_coverage_pct": sf.get("branch_coverage_pct"),
            "line_coverage_pct": sf.get("line_coverage_pct"),
            "covered_edges": sf.get(
                "covered_edges", edge_stats["covered_edges"]
            ),
            "num_edges": sf.get("num_edges", edge_stats["num_edges"]),
            "covered_branches": sf.get("covered_branches", 0),
            "num_branches": sf.get("num_branches", 0),
            "covered_lines": sf.get("covered_lines", 0),
            "num_statements": sf.get("num_statements", 0),
            "num_functions_in_scope": 1,
            "scope": [{"file": sf.get("file"), "function": sf.get("function")}],
            "scope_resolution_complete": None,
            "unresolved_calls": [],
        }

    total_edges = 0
    covered_edge_set: Set[Tuple[str, str, Any, Any]] = set()
    total_branches = 0
    covered_branch_set: Set[Tuple[str, int, int]] = set()
    total_lines = 0
    covered_line_set: Set[Tuple[str, int]] = set()

    for entry in ext_scope:
        f = entry.get("file", "")
        n = entry.get("name", "")
        if not f or not n:
            continue
        scoped = scope_to_function(cov, f, n)
        if scoped is None or scoped.get("source") == "not_found":
            continue
        total_edges += int(scoped.get("num_edges") or 0)
        total_branches += int(scoped.get("num_branches") or 0)
        total_lines += int(scoped.get("num_statements") or 0)
        for edge in scoped.get("executed_edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                covered_edge_set.add((f, n, edge[0], edge[1]))
        for arc in scoped.get("executed_branches", []):
            if isinstance(arc, (list, tuple)) and len(arc) >= 2:
                covered_branch_set.add((f, arc[0], arc[1]))
        for ln in scoped.get("executed_lines", []):
            if isinstance(ln, int):
                covered_line_set.add((f, ln))

    ce = len(covered_edge_set)
    cb = len(covered_branch_set)
    cl = len(covered_line_set)
    resolution = cov.get("scope_resolution") or {}
    return {
        "edge_coverage_pct": (ce / total_edges * 100) if total_edges else None,
        "branch_coverage_pct": (cb / total_branches * 100) if total_branches else None,
        "line_coverage_pct": (cl / total_lines * 100) if total_lines else None,
        "covered_edges": ce,
        "num_edges": total_edges,
        "covered_branches": cb,
        "num_branches": total_branches,
        "covered_lines": cl,
        "num_statements": total_lines,
        "num_functions_in_scope": len(ext_scope),
        "scope": ext_scope,
        "scope_resolution_complete": resolution.get("complete"),
        "unresolved_calls": resolution.get("unresolved_calls", []),
    }


def _aggregate_metric_A(
    proof_results: List[ProofExecResult],
    reference_state: str = "gold",
) -> Dict[str, Any]:
    """Metric A: per-spec edge depth over the extended scope, on the reference arm.
    """
    edge_values: List[float] = []
    line_values: List[float] = []
    per_spec: Dict[str, Dict[str, Any]] = {}
    n_with_data = 0
    n_scope_complete = 0
    n_scope_incomplete = 0

    for pr in proof_results:
        if pr.target_state != reference_state:
            continue
        result = _compute_per_spec_scope_coverage(pr)
        if result is None:
            per_spec[_cert_key(pr)] = {"skipped": True}
            continue
        n_with_data += 1
        epct = result["edge_coverage_pct"]
        lpct = result["line_coverage_pct"]
        per_spec[_cert_key(pr)] = {
            "edge_coverage_pct": epct,
            "line_coverage_pct": lpct,
            "covered_edges": result["covered_edges"],
            "num_edges": result["num_edges"],
            "branch_coverage_pct": result.get("branch_coverage_pct"),
            "covered_branches": result["covered_branches"],
            "num_branches": result["num_branches"],
            "covered_lines": result["covered_lines"],
            "num_statements": result["num_statements"],
            "num_functions_in_scope": result["num_functions_in_scope"],
            "scope": result.get("scope"),
            "scope_resolution_complete": result.get(
                "scope_resolution_complete"
            ),
            "unresolved_calls": result.get("unresolved_calls", []),
        }
        if result.get("scope_resolution_complete") is True:
            n_scope_complete += 1
        elif result.get("scope_resolution_complete") is False:
            n_scope_incomplete += 1
        if epct is not None:
            edge_values.append(float(epct))
        if lpct is not None:
            line_values.append(float(lpct))

    def _percentile(xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        if len(xs) == 1:
            return xs[0]
        s = sorted(xs)
        pos = max(0.0, min(100.0, p)) / 100.0 * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    def _median(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        return statistics.median(xs)

    def _hist(xs: List[float], n_bins: int = 10) -> List[int]:
        if not xs:
            return []
        bins = [0] * n_bins
        bin_width = 100.0 / n_bins
        for x in xs:
            idx = min(n_bins - 1, int(x // bin_width))
            bins[max(0, idx)] += 1
        return bins

    def _block(xs: List[float]) -> Dict[str, Any]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else None,
            "median": _median(xs),
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
            "p25": _percentile(xs, 25),
            "p75": _percentile(xs, 75),
            "n": len(xs),
            "distribution_10bin": _hist(xs),
        }

    return {
        "reference_state": reference_state,
        "edge_depth": _block(edge_values),
        "line_depth": _block(line_values),
        "per_spec": per_spec,
        "counts": {
            "specs_with_data": n_with_data,
            "specs_total": len(per_spec),
            "scope_complete": n_scope_complete,
            "scope_incomplete": n_scope_incomplete,
        },
    }


def _aggregate_metric_C_per_instance(
    task: EvalTask,
    proof_results: Optional[List[ProofExecResult]] = None,
) -> Dict[str, Any]:
    """Per-instance share of gold-patch functions the specs name or reach by calls."""
    gold = [
        FunctionInfo(f["file"], f["name"])
        for f in (task.gold_patch_functions or [])
        if f.get("file") and f.get("name")
    ]
    targeted_declared: List[Tuple[str, str]] = []
    for sp in task.surviving_specs:
        cm = sp.spec.get("code_mapping") or []
        for entry in cm:
            f = entry.get("file")
            n = entry.get("function")
            if f and n:
                targeted_declared.append((f, n))
    c_declared = spec_breadth(gold, targeted_declared)

    targeted_extended: Set[Tuple[str, str]] = set(targeted_declared)
    if proof_results:
        for pr in proof_results:
            cov = pr.coverage or {}
            for entry in cov.get("extended_scope", []):
                f = entry.get("file", "")
                n = entry.get("name", "")
                if f and n:
                    targeted_extended.add((f, n))
    c_extended = spec_breadth(gold, list(targeted_extended))

    return {
        "declared": c_declared,
        "extended": c_extended,
    }


def _aggregate_metric_C_mean(
    tasks: List[EvalTask],
    proof_results: Optional[List[ProofExecResult]] = None,
) -> Dict[str, Any]:
    """Mean breadth ratio across instances (declared + extended)."""
    proof_by_instance: Dict[str, List[ProofExecResult]] = defaultdict(list)
    for pr in (proof_results or []):
        proof_by_instance[pr.instance_id].append(pr)

    declared_ratios: List[float] = []
    extended_ratios: List[float] = []
    for t in tasks:
        r = _aggregate_metric_C_per_instance(
            t, proof_by_instance.get(t.instance_id),
        )
        d = r.get("declared", {}).get("ratio")
        if d is not None:
            declared_ratios.append(d)
        e = r.get("extended", {}).get("ratio")
        if e is not None:
            extended_ratios.append(e)

    def _m(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "declared_ratio_mean": _m(declared_ratios),
        "extended_ratio_mean": _m(extended_ratios),
        "n_instances": len(declared_ratios),
    }


def _spec_to_cert_rate(
    tasks: List[EvalTask],
    baseline_results: List[BaselineResult],
    spec_subset: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """How many surviving specs did the baseline LLM actually write certs for?"""
    wanted = {tuple(pair) for pair in spec_subset} if spec_subset else None
    certs_by_group: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for br in baseline_results:
        key = (br.instance_id, br.baseline, br.model)
        certs_by_group[key].update(br.certificates.keys())

    task_by_iid: Dict[str, EvalTask] = {t.instance_id: t for t in tasks}
    use_qualified_keys = len(certs_by_group) > 1

    total_specs = 0
    total_certs = 0
    per_instance: Dict[str, Dict[str, int]] = {}
    if certs_by_group:
        group_items = sorted(certs_by_group.items())
    else:
        group_items = [((t.instance_id, "", ""), set()) for t in tasks]

    for (iid, baseline, model), cert_ids in group_items:
        task = task_by_iid.get(iid)
        if task is None:
            continue
        surviving_ids = {s.spec_id for s in task.surviving_specs}
        if wanted is not None:
            surviving_ids = {sid for sid in surviving_ids if (iid, sid) in wanted}
        n_specs = len(surviving_ids)
        n_certs = len(surviving_ids & cert_ids)
        total_specs += n_specs
        total_certs += n_certs
        entry_key = iid
        if use_qualified_keys:
            entry_key = f"{iid}__{baseline}_{model}"
        per_instance[entry_key] = {
            "surviving_specs": n_specs,
            "certs_emitted": n_certs,
            "rate": (n_certs / n_specs) if n_specs else None,
        }
    return {
        "total_specs": total_specs,
        "total_certs": total_certs,
        "rate": (total_certs / total_specs) if total_specs else None,
        "per_instance": per_instance,
    }


def _aggregate_metric_D_per_instance(
    task: EvalTask,
    proof_results_for_instance: List[ProofExecResult],
) -> Dict[str, Any]:
    """Per-instance D, split by code_state."""
    gold = [
        FunctionInfo(f["file"], f["name"])
        for f in (task.gold_patch_functions or [])
        if f.get("file") and f.get("name")
    ]
    if not gold:
        return {"gold": None, "mutant": None, "n_gold_functions": 0}

    out: Dict[str, Any] = {"n_gold_functions": len(gold)}
    for state in ("gold", "mutant"):
        trials = [
            {"coverage": pr.coverage}
            for pr in proof_results_for_instance
            if pr.target_state == state and pr.coverage is not None
        ]
        if not trials:
            out[state] = None
            continue
        out[state] = aggregate_gold_patch_coverage(trials, gold)
    return out


def _aggregate_metric_D_mean(
    tasks: List[EvalTask],
    proof_results: List[ProofExecResult],
    reference_state: str = "gold",
) -> Dict[str, Any]:
    """Mean of per-instance D's edge_coverage_pct, by code_state."""
    by_instance: Dict[str, List[ProofExecResult]] = defaultdict(list)
    for pr in proof_results:
        by_instance[pr.instance_id].append(pr)

    gold_epcts: List[float] = []
    mutant_epcts: List[float] = []
    for task in tasks:
        prs = by_instance.get(task.instance_id, [])
        d = _aggregate_metric_D_per_instance(task, prs)
        if isinstance(d.get("gold"), dict):
            v = d["gold"].get("edge_coverage_pct")
            if v is not None:
                gold_epcts.append(float(v))
        if isinstance(d.get("mutant"), dict):
            v = d["mutant"].get("edge_coverage_pct")
            if v is not None:
                mutant_epcts.append(float(v))

    def _m(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "exhaustive_side": reference_state,
        "edge_coverage_pct_mean_gold": _m(gold_epcts),
        "edge_coverage_pct_mean_mutant": _m(mutant_epcts),
        "edge_coverage_pct_delta_mean": (
            (_m(gold_epcts) - _m(mutant_epcts))
            if (gold_epcts and mutant_epcts) else None
        ),
        "counts": {
            "instances_with_gold": len(gold_epcts),
            "instances_with_mutant": len(mutant_epcts),
        },
    }


def _analysis_blocks(
    diff_results: List[DifferentialResult],
    proof_results: List[ProofExecResult],
) -> Dict[str, Any]:
    """The per-certificate analysis blocks, built in one place."""
    negative_controls = _aggregate_negative_controls(proof_results)
    return {
        "negative_controls": negative_controls,
        "unsound_certs": _unsound_certs(diff_results, negative_controls),
        "sound_certs": _sound_certs(diff_results, negative_controls),
    }


class MetricsReporter:
    """Computes and saves evaluation metrics."""

    def __init__(self, config):
        self.config = config

    def compute_and_save(
        self,
        tasks: List[EvalTask],
        baseline_results: List[BaselineResult],
        differential_results: List[DifferentialResult],
        proof_results: Optional[List[ProofExecResult]] = None,
    ) -> Dict[str, Any]:
        """Compute all metrics and save reports."""
        proof_results = proof_results or []
        reports_dir = self.config.output_dir / "eval_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        spec_subset = getattr(self.config, "spec_subset", None)
        if not isinstance(spec_subset, (list, tuple)):
            spec_subset = None
        differential_results = _scope_to_spec_subset(differential_results, spec_subset)

        spec_categories: Dict[str, str] = {}
        for task in tasks:
            for spec in task.all_specs:
                sid = spec.get("spec_id", "")
                if sid:
                    spec_categories[sid] = spec.get("category", "unknown")

        cost_by_key: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for br in baseline_results:
            key = (br.instance_id, br.baseline, br.model)
            cost_by_key.setdefault(key, {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0})
            for k in ("input_tokens", "output_tokens", "total_cost_usd"):
                cost_by_key[key][k] += br.cost.get(k, 0)

        blocks = _analysis_blocks(differential_results, proof_results)
        control_firing = set(
            blocks["negative_controls"].get("cross_spec_firing_certs", [])
        )

        summary = self._compute_summary(
            differential_results, cost_by_key, control_firing
        )
        harness_form = getattr(self.config, "harness_form", "form1b_form6")
        reference_state = getattr(self.config, "stage2_reference_state", "gold")
        is_hypofuzz = harness_form == "hypofuzz"
        has_coverage_metrics = harness_form in {
            "hypofuzz", "saved_inputs", "factory_oracle"
        }
        summary["agent_steps"] = _aggregate_agent_steps(baseline_results)
        if is_hypofuzz:
            summary["fuzz_compliance"] = _aggregate_fuzz_compliance(baseline_results)
        if has_coverage_metrics:
            summary["metric_A"] = _aggregate_metric_A(proof_results, reference_state)
            summary["metric_C"] = _aggregate_metric_C_mean(tasks, proof_results)
            summary["metric_D"] = _aggregate_metric_D_mean(
                tasks, proof_results, reference_state
            )
            summary["spec_to_cert"] = _spec_to_cert_rate(
                tasks, baseline_results, spec_subset,
            )
            summary["mutant_lines"] = _aggregate_mutant_lines(proof_results)
        summary.update(blocks)
        if spec_subset is not None:
            path = getattr(self.config, "spec_subset_path", None)
            summary["spec_subset"] = {
                "path": str(path) if path else None,
                "specs_listed": len(spec_subset),
                "certs_scored": summary["counts"]["certs"],
            }
        self._save_json(reports_dir / "summary.json", summary)

        per_baseline = self._compute_per_baseline(
            differential_results, cost_by_key, proof_results, control_firing
        )
        for group_key, group_brs in _group_baseline_results(baseline_results).items():
            if group_key in per_baseline:
                per_baseline[group_key]["agent_steps"] = _aggregate_agent_steps(group_brs)
        if is_hypofuzz:
            for group_key, group_brs in _group_baseline_results(baseline_results).items():
                if group_key in per_baseline:
                    per_baseline[group_key]["fuzz_compliance"] = (
                        _aggregate_fuzz_compliance(group_brs)
                    )
        if has_coverage_metrics:
            cov_by_baseline: Dict[str, List[ProofExecResult]] = defaultdict(list)
            for pr in proof_results:
                cov_by_baseline[f"{pr.baseline}_{pr.model}"].append(pr)
            for group_key, group_prs in cov_by_baseline.items():
                if group_key in per_baseline:
                    per_baseline[group_key]["metric_A"] = (
                        _aggregate_metric_A(group_prs, reference_state)
                    )
                    per_baseline[group_key]["metric_C"] = (
                        _aggregate_metric_C_mean(tasks, group_prs)
                    )
                    per_baseline[group_key]["metric_D"] = (
                        _aggregate_metric_D_mean(tasks, group_prs, reference_state)
                    )
                    per_baseline[group_key]["mutant_lines"] = (
                        _aggregate_mutant_lines(group_prs)
                    )
        self._save_json(reports_dir / "per_baseline.json", per_baseline)

        per_category = self._compute_per_category(
            differential_results, spec_categories, control_firing
        )
        self._save_json(reports_dir / "per_category.json", per_category)

        per_instance = self._compute_per_instance(
            differential_results, cost_by_key, control_firing
        )
        if has_coverage_metrics:
            prs_by_instance_and_baseline: Dict[
                Tuple[str, str, str], List[ProofExecResult]
            ] = defaultdict(list)
            for pr in proof_results:
                key = (pr.instance_id, pr.baseline, pr.model)
                prs_by_instance_and_baseline[key].append(pr)
            tasks_by_iid: Dict[str, EvalTask] = {t.instance_id: t for t in tasks}
            for dr in differential_results:
                key_str = f"{dr.instance_id}__{dr.baseline}_{dr.model}"
                entry = per_instance.get(key_str)
                if entry is None:
                    continue
                task = tasks_by_iid.get(dr.instance_id)
                if task is None:
                    continue
                prs = prs_by_instance_and_baseline.get(
                    (dr.instance_id, dr.baseline, dr.model), []
                )
                entry["metric_A"] = _aggregate_metric_A(prs, reference_state)
                entry["metric_C"] = _aggregate_metric_C_per_instance(task, prs)
                entry["metric_D"] = _aggregate_metric_D_per_instance(task, prs)

        prs_by_key: Dict[Tuple[str, str, str], List[ProofExecResult]] = defaultdict(list)
        for pr in proof_results:
            prs_by_key[(pr.instance_id, pr.baseline, pr.model)].append(pr)
        for dr in differential_results:
            entry = per_instance.get(f"{dr.instance_id}__{dr.baseline}_{dr.model}")
            if entry is None:
                continue
            entry.update(_analysis_blocks(
                [dr], prs_by_key.get((dr.instance_id, dr.baseline, dr.model), []),
            ))

        brs_by_key: Dict[Tuple[str, str, str], List[BaselineResult]] = defaultdict(list)
        for br in baseline_results:
            brs_by_key[(br.instance_id, br.baseline, br.model)].append(br)
        for (iid, baseline, model), group_brs in brs_by_key.items():
            entry = per_instance.get(f"{iid}__{baseline}_{model}")
            if entry is not None:
                entry["agent_steps"] = _aggregate_agent_steps(group_brs)
        self._save_json(reports_dir / "per_instance.json", per_instance)

        logger.info("Stage 5: reports saved to %s", reports_dir)
        return summary

    def _compute_summary(
        self,
        diff_results: List[DifferentialResult],
        cost_by_key: Dict[Tuple[str, str, str], Dict[str, float]],
        control_firing: AbstractSet[str] = frozenset(),
    ) -> Dict[str, Any]:
        all_outcomes: List[Tuple[str, SpecCertOutcome]] = []
        total_cost: Dict[str, float] = {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}
        seen_keys = set()

        for dr in diff_results:
            key = (dr.instance_id, dr.baseline, dr.model)
            if key in cost_by_key and key not in seen_keys:
                seen_keys.add(key)
                for k in total_cost:
                    total_cost[k] += cost_by_key[key].get(k, 0)
            all_outcomes.extend(_outcome_keys(dr))

        summary = _aggregate_outcomes(all_outcomes, control_firing)
        summary["cost"] = total_cost
        return summary

    def _compute_per_baseline(
        self,
        diff_results: List[DifferentialResult],
        cost_by_key: Dict[Tuple[str, str, str], Dict[str, float]],
        proof_results: Optional[List[ProofExecResult]] = None,
        control_firing: AbstractSet[str] = frozenset(),
    ) -> Dict[str, Any]:
        """Group metrics by (baseline, model)."""
        groups: Dict[str, List[DifferentialResult]] = defaultdict(list)
        for dr in diff_results:
            groups[f"{dr.baseline}_{dr.model}"].append(dr)

        result: Dict[str, Any] = {}
        for group_key, drs in groups.items():
            group_cost: Dict[str, float] = defaultdict(float)
            for dr in drs:
                key = (dr.instance_id, dr.baseline, dr.model)
                if key in cost_by_key:
                    for k in ("input_tokens", "output_tokens", "total_cost_usd"):
                        group_cost[k] += cost_by_key[key].get(k, 0)

            outcomes: List[Tuple[str, SpecCertOutcome]] = []
            for dr in drs:
                outcomes.extend(_outcome_keys(dr))

            agg = _aggregate_outcomes(outcomes, control_firing)
            agg["cost"] = dict(group_cost)
            if proof_results is not None:
                group_prs = [
                    pr for pr in proof_results
                    if f"{pr.baseline}_{pr.model}" == group_key
                ]
                agg.update(_analysis_blocks(drs, group_prs))
            result[group_key] = agg
        return result

    def _compute_per_category(
        self,
        diff_results: List[DifferentialResult],
        spec_categories: Dict[str, str],
        control_firing: AbstractSet[str] = frozenset(),
    ) -> Dict[str, Any]:
        """Group cert quality by spec category."""
        by_cat: Dict[str, List[Tuple[str, SpecCertOutcome]]] = defaultdict(list)
        for dr in diff_results:
            for sid, outcome in dr.per_spec.items():
                cat = spec_categories.get(sid, "unknown")
                by_cat[cat].append((_outcome_cert_key(dr, sid), outcome))
        return {
            cat: _aggregate_outcomes(items, control_firing)
            for cat, items in by_cat.items()
        }

    def _compute_per_instance(
        self,
        diff_results: List[DifferentialResult],
        cost_by_key: Dict[Tuple[str, str, str], Dict[str, float]],
        control_firing: AbstractSet[str] = frozenset(),
    ) -> Dict[str, Any]:
        """Per-instance breakdown for debugging."""
        result: Dict[str, Any] = {}
        for dr in diff_results:
            key_str = f"{dr.instance_id}__{dr.baseline}_{dr.model}"
            agg = _aggregate_outcomes(_outcome_keys(dr), control_firing)
            cost_key = (dr.instance_id, dr.baseline, dr.model)
            agg["cost"] = cost_by_key.get(cost_key, {})
            agg["per_spec"] = {sid: o.to_dict() for sid, o in dr.per_spec.items()}
            result[key_str] = agg
        return result

    def _save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
