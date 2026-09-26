from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mutation_pipeline.control_filter import (
    apply_control_filter,
    controlled_spec_ids,
    filter_uncontrolled_specs,
)
from mutation_pipeline.models import (
    AtomicSpec,
    MutationResult,
    Proof,
    SpecCategory,
)


def make_result(
    spec_id: str,
    probes: Sequence[str],
    instance_id: str = "inst-1",
) -> MutationResult:
    return MutationResult(
        schema_version=3,
        instance_id=instance_id,
        spec=AtomicSpec(
            spec_id=spec_id,
            text=f"spec {spec_id}",
            category=SpecCategory.BEHAVIORAL,
            sources=[],
            code_mapping=[],
        ),
        mask_patch="",
        gold_patch="",
        mutant_patch=f"--- patch for {spec_id}",
        test_patch="",
        proof=Proof(witness_script=f"# witness {spec_id}"),
        metadata={
            "preservation_probes": [
                {"spec_id": pid, "verdict": "preserved"} for pid in probes
            ]
        },
    )


def kept_ids(results: List[MutationResult]) -> List[str]:
    kept, _ = filter_uncontrolled_specs(results)
    return [r.spec.spec_id for r in kept]


def dropped_ids(results: List[MutationResult]) -> List[str]:
    _, dropped = filter_uncontrolled_specs(results)
    return [r.spec.spec_id for r in dropped]


def test_mutual_probes_are_both_kept():
    results = [make_result("S1", ["S2"]), make_result("S2", ["S1"])]
    assert kept_ids(results) == ["S1", "S2"]
    assert dropped_ids(results) == []


def test_spec_no_mutant_probes_is_dropped():
    results = [
        make_result("S1", ["S2"]),
        make_result("S2", ["S1"]),
        make_result("S3", ["S1", "S2"]),
    ]
    assert kept_ids(results) == ["S1", "S2"]
    assert dropped_ids(results) == ["S3"]


def test_probing_yourself_is_not_a_control():
    results = [make_result("S1", ["S1"])]
    assert kept_ids(results) == []


def test_probe_of_a_spec_that_did_not_survive_is_not_a_control():
    results = [make_result("S1", ["S9"]), make_result("S2", ["S9"])]
    assert kept_ids(results) == []


def test_probes_without_a_spec_id_are_ignored():
    results = [
        make_result("S1", ["S2"]),
        make_result("S2", []),
    ]
    results[1].metadata["preservation_probes"] = [{"verdict": "inconclusive"}]
    assert kept_ids(results) == []


def test_diverged_probe_still_counts_as_a_control_pairing():
    results = [make_result("S1", ["S2"]), make_result("S2", ["S1"])]
    results[0].metadata["preservation_probes"] = [
        {"spec_id": "S2", "verdict": "diverged"}
    ]
    assert kept_ids(results) == ["S1", "S2"]


def test_a_row_that_probes_nobody_is_kept_if_someone_probes_it():
    results = [
        make_result("S1", []),
        make_result("S2", ["S3"]),
        make_result("S3", ["S2", "S1"]),
    ]
    results[0].metadata = {}
    assert kept_ids(results) == ["S1", "S2", "S3"]


def test_filter_iterates_after_a_drop_removes_a_control():
    results = [
        make_result("S1", ["S2"]),
        make_result("S2", ["S3"]),
        make_result("S3", []),
    ]
    assert dropped_ids(results) == ["S1", "S2", "S3"]
    assert kept_ids(results) == []


def test_cycle_survives_the_fixpoint():
    results = [
        make_result("S1", ["S3"]),
        make_result("S2", ["S1"]),
        make_result("S3", ["S2"]),
    ]
    assert kept_ids(results) == ["S1", "S2", "S3"]


def test_cluster_survives_while_a_dangling_chain_is_cut():
    results = [
        make_result("S1", ["S2"]),
        make_result("S2", ["S1"]),
        make_result("S3", ["S1", "S2"]),
    ]
    assert kept_ids(results) == ["S1", "S2"]
    assert dropped_ids(results) == ["S3"]


def test_control_pairing_does_not_cross_instances():
    results = [
        make_result("S1", ["S2"], instance_id="inst-a"),
        make_result("S2", ["S1"], instance_id="inst-b"),
    ]
    assert kept_ids(results) == []


def test_instances_are_filtered_independently():
    results = [
        make_result("S1", ["S2"], instance_id="inst-a"),
        make_result("S2", ["S1"], instance_id="inst-a"),
        make_result("S1", [], instance_id="inst-b"),
    ]
    kept, dropped = filter_uncontrolled_specs(results)
    assert [(r.instance_id, r.spec.spec_id) for r in kept] == [
        ("inst-a", "S1"),
        ("inst-a", "S2"),
    ]
    assert [(r.instance_id, r.spec.spec_id) for r in dropped] == [("inst-b", "S1")]


def test_input_order_is_preserved():
    results = [
        make_result("S3", ["S1"]),
        make_result("S1", ["S3"]),
        make_result("S2", ["S1"]),
    ]
    assert kept_ids(results) == ["S3", "S1"]


def test_controlled_spec_ids_reports_the_fixpoint_set():
    results = [make_result("S1", ["S2"]), make_result("S2", ["S1"]), make_result("S3", [])]
    assert controlled_spec_ids(results) == {"S1", "S2"}


def test_empty_input():
    assert filter_uncontrolled_specs([]) == ([], [])


def test_apply_control_filter_drops_uncontrolled_specs():
    results = [make_result("S1", ["S2"]), make_result("S2", ["S1"]), make_result("S3", [])]
    kept = apply_control_filter(results, preservation_probe_k=2)
    assert [r.spec.spec_id for r in kept] == ["S1", "S2"]


def test_apply_control_filter_is_a_noop_when_step5_is_disabled():
    results = [make_result("S1", []), make_result("S2", [])]
    kept = apply_control_filter(results, preservation_probe_k=0)
    assert [r.spec.spec_id for r in kept] == ["S1", "S2"]


def test_every_kept_spec_gets_a_control_from_the_eval_side_selector():
    from eval_pipeline.models import SurvivingSpec
    from eval_pipeline.stages.stage3_proof_execution import select_control_mutants

    results = [
        make_result("S1", ["S2"]),
        make_result("S2", ["S1", "S3"]),
        make_result("S3", ["S2"]),
        make_result("S4", ["S1", "S2"]),
    ]
    kept, dropped = filter_uncontrolled_specs(results)

    surviving = [
        SurvivingSpec(
            spec=r.spec.to_dict(),
            mutant_patch=r.mutant_patch,
            preservation_probes=r.metadata["preservation_probes"],
        )
        for r in kept
    ]
    assert surviving, "expected the cluster to survive"
    for spec in surviving:
        assert select_control_mutants(spec.spec_id, surviving), (
            f"{spec.spec_id} was kept but has no control mutant"
        )

    all_specs = [
        SurvivingSpec(
            spec=r.spec.to_dict(),
            mutant_patch=r.mutant_patch,
            preservation_probes=r.metadata["preservation_probes"],
        )
        for r in results
    ]
    for r in dropped:
        assert not select_control_mutants(r.spec.spec_id, all_specs)
