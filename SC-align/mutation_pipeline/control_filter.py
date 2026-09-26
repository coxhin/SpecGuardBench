"""Final filter: drop surviving specs that no mutant can control."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Set, Tuple

from .models import MutationResult

logger = logging.getLogger(__name__)


def _probed_spec_ids(result: MutationResult) -> Set[str]:
    """Specs whose witness was run against this result's mutant at Step 5."""
    probes = (result.metadata or {}).get("preservation_probes") or []
    return {
        str(probe.get("spec_id"))
        for probe in probes
        if isinstance(probe, dict) and probe.get("spec_id")
    }


def controlled_spec_ids(results: Sequence[MutationResult]) -> Set[str]:
    """Ids of the specs in one instance that keep a control under the fixpoint."""
    probed_by: Dict[str, Set[str]] = {}
    for result in results:
        spec_id = result.spec.spec_id
        probed_by.setdefault(spec_id, set()).update(_probed_spec_ids(result))

    kept = set(probed_by)
    while True:
        controlled = {
            spec_id
            for spec_id in kept
            if any(
                spec_id in probed_by[other]
                for other in kept
                if other != spec_id
            )
        }
        if controlled == kept:
            return kept
        kept = controlled


def filter_uncontrolled_specs(
    results: Sequence[MutationResult],
) -> Tuple[List[MutationResult], List[MutationResult]]:
    """Split results into the ones eval can score and the ones it cannot."""
    by_instance: Dict[str, List[MutationResult]] = {}
    for result in results:
        by_instance.setdefault(result.instance_id, []).append(result)

    keep_ids = {
        instance_id: controlled_spec_ids(rows)
        for instance_id, rows in by_instance.items()
    }

    kept: List[MutationResult] = []
    dropped: List[MutationResult] = []
    for result in results:
        target = (
            kept
            if result.spec.spec_id in keep_ids[result.instance_id]
            else dropped
        )
        target.append(result)
    return kept, dropped


def apply_control_filter(
    results: Sequence[MutationResult],
    preservation_probe_k: int,
) -> List[MutationResult]:
    """Run the filter for a run and log what it removed."""
    if preservation_probe_k <= 0:
        logger.warning(
            "Step 5 was disabled (preservation_probe_k=0), so no spec has a "
            "negative control; skipping the control filter and writing all "
            "%d results. Eval will score these with no control arm.",
            len(results),
        )
        return list(results)

    kept, dropped = filter_uncontrolled_specs(results)
    for result in dropped:
        logger.info(
            "[%s][%s] Dropped from output.jsonl: no surviving mutant probes "
            "this spec, so eval would have no negative control for its "
            "certificate",
            result.instance_id,
            result.spec.spec_id,
        )
    if dropped:
        logger.info(
            "Control filter: %d of %d surviving specs had no negative control "
            "and were dropped",
            len(dropped),
            len(results),
        )
    return kept
