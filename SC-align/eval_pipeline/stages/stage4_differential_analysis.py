"""Stage 4: per-spec certificate quality."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..models import (
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    BaselineResult,
    DifferentialResult,
    EvalTask,
    ProofExecResult,
    SpecCertOutcome,
)

logger = logging.getLogger(__name__)


class DifferentialAnalyzer:
    """Aggregates cert execution outcomes into per-spec quality verdicts."""

    def __init__(self, config):
        self.config = config

    def analyze(
        self,
        tasks: List[EvalTask],
        baseline_results: List[BaselineResult],
        proof_results: List[ProofExecResult],
    ) -> List[DifferentialResult]:
        """Run per-spec aggregation across all results."""
        proof_map: Dict[Tuple[str, str, str, str, str], ProofExecResult] = {}
        for pr in proof_results:
            key = (pr.instance_id, pr.baseline, pr.model, pr.spec_id, pr.target_state)
            proof_map[key] = pr

        grouped: Dict[Tuple[str, str, str], BaselineResult] = {}
        for br in baseline_results:
            key = (br.instance_id, br.baseline, br.model)
            if key in grouped:
                logger.info(
                    "Merging duplicate baseline result for %s", key
                )
                self._merge_baseline_result(grouped[key], br)
                continue
            grouped[key] = br

        tasks_by_iid: Dict[str, EvalTask] = {t.instance_id: t for t in tasks}

        all_results: List[DifferentialResult] = []
        for (iid, baseline, model), br in grouped.items():
            per_spec: Dict[str, SpecCertOutcome] = {}
            task = tasks_by_iid.get(iid)
            if task is None:
                surviving_ids = list(br.certificates.keys())
            else:
                surviving_ids = [s.spec_id for s in task.surviving_specs]

            for spec_id in surviving_ids:
                certificate = br.certificates.get(spec_id)
                has_cert = bool(certificate and certificate.code.strip())

                gold_pr = proof_map.get(
                    (iid, baseline, model, spec_id, TARGET_STATE_GOLD)
                )
                mutant_pr = proof_map.get(
                    (iid, baseline, model, spec_id, TARGET_STATE_MUTANT)
                )

                if has_cert:
                    outcome = SpecCertOutcome(
                        spec_id=spec_id,
                        has_certificate=True,
                        gold_triggered=gold_pr.triggered if gold_pr else None,
                        mutant_triggered=mutant_pr.triggered if mutant_pr else None,
                        gold_kind=gold_pr.kind if gold_pr else None,
                        mutant_kind=mutant_pr.kind if mutant_pr else None,
                        gold_error=gold_pr.error if gold_pr else None,
                        mutant_error=mutant_pr.error if mutant_pr else None,
                    )
                else:
                    outcome = SpecCertOutcome(
                        spec_id=spec_id,
                        has_certificate=False,
                        gold_triggered=None,
                        mutant_triggered=None,
                        gold_kind=None,
                        mutant_kind=None,
                        gold_error=None,
                        mutant_error=None,
                    )
                per_spec[spec_id] = outcome

            diff = DifferentialResult(
                instance_id=iid,
                baseline=baseline,
                model=model,
                per_spec=per_spec,
            )
            all_results.append(diff)
            self._save_result(diff)

        self._save_combined(all_results)
        logger.info(
            "Stage 4: aggregated %d (instance, baseline, model) groups",
            len(all_results),
        )
        return all_results

    def _merge_baseline_result(
        self,
        target: BaselineResult,
        incoming: BaselineResult,
    ) -> None:
        """Merge per-atomic-spec certs for Stage 4 analysis."""
        for spec_id, cert in incoming.certificates.items():
            if spec_id in target.certificates:
                logger.warning(
                    "Duplicate certificate for (%s, %s, %s, %s); keeping first",
                    target.instance_id, target.baseline, target.model, spec_id,
                )
                continue
            target.certificates[spec_id] = cert
        if incoming.fuzz_compliance:
            if target.fuzz_compliance is None:
                target.fuzz_compliance = {}
            target.fuzz_compliance.update(incoming.fuzz_compliance)

    def _save_result(self, result: DifferentialResult) -> None:
        subdir = (
            self.config.output_dir
            / "differential"
            / f"{result.baseline}_{result.model}"
        )
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{result.instance_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    def _save_combined(self, results: List[DifferentialResult]) -> None:
        combined_dir = self.config.output_dir / "differential"
        combined_dir.mkdir(parents=True, exist_ok=True)
        path = combined_dir / "all_results.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.to_jsonl() + "\n")
        logger.info("Stage 4: saved combined results to %s", path)


def load_differential_results(output_dir: Path) -> List[DifferentialResult]:
    """Load all differential analysis results from disk."""
    results: List[DifferentialResult] = []
    combined_path = output_dir / "differential" / "all_results.jsonl"
    if not combined_path.exists():
        return results
    with open(combined_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(DifferentialResult.from_dict(json.loads(line)))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to parse differential result line: %s", e)
    logger.info("Loaded %d differential results from %s", len(results), combined_path)
    return results
