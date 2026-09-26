"""Step 5: Cross-spec preservation probe (observational)."""

import hashlib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from ..docker_runner import MutationDockerRunner
from ..models import PipelineInput, PreservationProbe, StepResult
from .step4_witness_execution import _extract_observation, _wrap_with_markers

logger = logging.getLogger(__name__)


VERDICT_PRESERVED = "preserved"
VERDICT_DIVERGED = "diverged"
VERDICT_MUTANT_ERROR = "mutant_error"
VERDICT_INCONCLUSIVE = "inconclusive"

_ERROR_TAIL_CHARS = 2000


_LINE_RANGE_RE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
_INT_RE = re.compile(r"\d+")


def _parse_line_ranges(text: str) -> List[Tuple[int, int]]:
    """Pull line ranges out of a ``CodeMapping.lines`` string."""
    text = text or ""
    ranges = [(int(a), int(b)) for a, b in _LINE_RANGE_RE.findall(text)]
    if ranges:
        return [(a, b) if a <= b else (b, a) for a, b in ranges]
    return [(int(n), int(n)) for n in _INT_RE.findall(text)]


def _spans_by_file(spec) -> Dict[str, List[Tuple[int, int]]]:
    """Merge a spec's mapped line ranges into disjoint spans per file."""
    by_file: Dict[str, List[Tuple[int, int]]] = {}
    for cm in spec.code_mapping or []:
        for lo, hi in _parse_line_ranges(getattr(cm, "lines", "")):
            by_file.setdefault(cm.file or "", []).append((lo, hi))
    for path, spans in by_file.items():
        merged: List[Tuple[int, int]] = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        by_file[path] = merged
    return by_file


def _function_keys(spec) -> set:
    return {
        (cm.file or "", cm.function)
        for cm in spec.code_mapping or []
        if (cm.function or "").strip()
    }


def _files(spec) -> set:
    return {cm.file for cm in spec.code_mapping or [] if (cm.file or "").strip()}


def _shared_line_count(a, b) -> int:
    """Total lines the two specs' mapped regions have in common."""
    a_spans, b_spans = _spans_by_file(a), _spans_by_file(b)
    total = 0
    for path, spans in a_spans.items():
        for lo, hi in spans:
            for other_lo, other_hi in b_spans.get(path, ()):
                overlap = min(hi, other_hi) - max(lo, other_lo) + 1
                if overlap > 0:
                    total += overlap
    return total


def rank_probe_candidates(spec_id: str, specs: List) -> List[Tuple[str, int, int]]:
    """Every code-grounded candidate for this spec, most entangled first."""
    target = next((s for s in specs if s.spec_id == spec_id), None)
    if target is None:
        return []

    target_functions = _function_keys(target)
    target_files = _files(target)

    ranked: List[Tuple[int, int, str]] = []
    for candidate in specs:
        if candidate.spec_id == spec_id:
            continue
        shared_functions = len(target_functions & _function_keys(candidate))
        shared_lines = _shared_line_count(target, candidate)
        if shared_functions or shared_lines or (target_files & _files(candidate)):
            ranked.append((-shared_functions, -shared_lines, candidate.spec_id))

    ranked.sort()
    return [
        (candidate_id, -neg_functions, -neg_lines)
        for neg_functions, neg_lines, candidate_id in ranked
    ]


def select_probe_specs(spec_id: str, specs: List, k: int) -> List[str]:
    """Pick up to ``k`` other specs whose witnesses probe this mutation."""
    if k <= 0:
        return []
    return [
        candidate_id
        for candidate_id, _, _ in rank_probe_candidates(spec_id, specs)[:k]
    ]


class PreservationProber:
    """Runs other specs' witnesses against this spec's mutant."""

    def __init__(self, docker_runner: MutationDockerRunner):
        self.docker_runner = docker_runner
        self._gold_observations: Dict[Tuple[str, str], str] = {}

    @staticmethod
    def _cache_key(instance_id: str, witness_script: str) -> Tuple[str, str]:
        digest = hashlib.sha256(witness_script.encode("utf-8")).hexdigest()
        return (instance_id, digest)

    def record_gold_observation(
        self, instance_id: str, witness_script: str, observation: str
    ) -> None:
        """Seed the gold cache with an observation obtained elsewhere."""
        if not observation:
            return
        self._gold_observations[self._cache_key(instance_id, witness_script)] = observation

    def probe_one(
        self,
        pipeline_input: PipelineInput,
        spec_id: str,
        mutations: List[dict],
        foreign_spec_id: str,
        witness_script: str,
        timeout: int,
    ) -> PreservationProbe:
        """Run one foreign witness against one mutant, in its own container."""
        iid = pipeline_input.instance_id
        probe = PreservationProbe(
            spec_id=foreign_spec_id, verdict=VERDICT_INCONCLUSIVE
        )

        gold_observation = self._gold_observations.get(
            self._cache_key(iid, witness_script)
        )
        if gold_observation is None:
            probe.error = (
                "No gold reading for this witness: it never passed Step 4, so "
                "there is nothing to compare the mutant reading against"
            )
            return probe
        probe.gold_observation = gold_observation

        try:
            wrapped = _wrap_with_markers(witness_script)
        except SyntaxError as e:
            probe.error = f"Could not AST-wrap witness: {e}"
            return probe

        t0 = time.time()
        container = None
        try:
            container = self.docker_runner.setup_instance(pipeline_input)
            if not self.docker_runner.apply_mutation(container, mutations):
                probe.error = (
                    "Could not apply the mutation for preservation probing"
                )
                return probe

            failed, output = self.docker_runner.run_witness_script(
                container, wrapped, timeout,
                env={"WITNESS_ALLOW_BIND_MISMATCH": "1"},
            )
            observation = _extract_observation(output)
            if failed:
                probe.verdict = VERDICT_MUTANT_ERROR
                probe.mutant_observation = observation
                probe.error = output[-_ERROR_TAIL_CHARS:]
            elif not observation:
                probe.error = "Mutant run produced no markered observation"
            else:
                probe.mutant_observation = observation
                probe.verdict = (
                    VERDICT_DIVERGED if observation != gold_observation
                    else VERDICT_PRESERVED
                )
        except Exception as e:
            logger.error(
                f"[{iid}][{spec_id}] Step 5 probe of {foreign_spec_id} "
                f"failed: {e}", exc_info=True
            )
            probe.error = str(e)
        finally:
            if container is not None:
                self.docker_runner.cleanup(container)

        logger.info(
            f"[{iid}][{spec_id}] Step 5: probed {foreign_spec_id} in "
            f"{time.time() - t0:.1f}s — {probe.verdict}"
        )
        return probe


def probes_from_result(step5_result: Optional[StepResult]) -> List[PreservationProbe]:
    """Rebuild probe objects from a Step 5 result, tolerating a missing step."""
    if step5_result is None or not isinstance(step5_result.data, dict):
        return []
    return [
        PreservationProbe.from_dict(d)
        for d in step5_result.data.get("probes", [])
        if isinstance(d, dict)
    ]
