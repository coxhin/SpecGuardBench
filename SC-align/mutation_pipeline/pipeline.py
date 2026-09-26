"""Main pipeline orchestrator."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2

from featurebench.infer.config import InferConfigLoader

from .config import MutationPipelineConfig, create_llm_caller
from .control_filter import apply_control_filter
from .docker_runner import MutationDockerRunner
from .models import (
    AtomicSpec,
    CodeMapping,
    FilterSurvival,
    MutationResult,
    PipelineInput,
    PreservationProbe,
    Proof,
    SpecCategory,
    SpecSource,
    StepResult,
)
from .steps.step1_spec_decomposition import SpecDecomposer
from .steps.step2_mutation_generation import MutationGenerator
from .steps.step3_test_verification import TestVerifier
from .steps.step4_witness_execution import WitnessExecutor
from .steps.step5_preservation_probe import (
    VERDICT_DIVERGED,
    PreservationProber,
    rank_probe_candidates,
)
from .steps.step6_llm_verification import LLMVerifier

logger = logging.getLogger(__name__)


@dataclass
class _VerifiedSpec:
    """What one spec carries out of the verification phase into the probe one."""
    spec: AtomicSpec
    attempt: int
    step2_result: StepResult
    step2_time: float
    mutations: List[dict]
    mutant_patch: str
    proof: Proof
    mutation_strategy: str
    gold_stray_output: str
    mutant_stray_output: str
    filter_survival: FilterSurvival
    step_timings: Dict[str, float]


class PreservationRejected(Exception):
    """Step 6b judged the mutation to break a spec it had to preserve."""


class SpecInfeasible(Exception):
    """Step 6 judged the *spec* unmutatable, not the sample."""


def _minimal_edit(find: str, replace: str) -> Tuple[str, str]:
    """Strip the common prefix and suffix shared by a find/replace pair."""
    start = 0
    limit = min(len(find), len(replace))
    while start < limit and find[start] == replace[start]:
        start += 1

    end = 0
    limit = min(len(find) - start, len(replace) - start)
    while end < limit and find[-1 - end] == replace[-1 - end]:
        end += 1

    return find[start: len(find) - end], replace[start: len(replace) - end]


def _mutation_signature(mutations: List[dict]) -> Tuple:
    """Anchor-independent identity of a mutation set."""
    return tuple(
        (m.get("file", ""),) + _minimal_edit(m.get("find", ""), m.get("replace", ""))
        for m in mutations
    )


def _spec_functions(spec: AtomicSpec) -> set:
    return {cm.function for cm in spec.code_mapping if cm.function}


def _merge_specs(group: List[AtomicSpec]) -> AtomicSpec:
    """Fold a redundant group into one spec, keeping every constituent's text."""
    primary = group[0]
    categories = {s.category for s in group}
    for candidate in (
        SpecCategory.BEHAVIORAL,
        SpecCategory.STRUCTURAL,
        SpecCategory.IMPLEMENTATION_CONSTRAINT,
    ):
        if candidate in categories:
            category = candidate
            break
    else:
        category = primary.category

    sources: List[SpecSource] = []
    for spec in group:
        for source in spec.sources:
            if source not in sources:
                sources.append(source)

    code_mapping: List[CodeMapping] = []
    seen_mappings = set()
    for spec in group:
        for cm in spec.code_mapping:
            key = (cm.file, cm.lines, cm.function)
            if key not in seen_mappings:
                seen_mappings.add(key)
                code_mapping.append(cm)

    constituents: List[str] = []
    for spec in group:
        constituents.append(spec.spec_id)
        constituents.extend(
            c for c in (spec.merged_from or []) if c != spec.spec_id
        )

    group_ids = {spec.spec_id for spec in group}
    independence_check: Dict[str, Any] = {}
    for spec in group:
        for other_id, evidence in (spec.independence_check or {}).items():
            if other_id not in group_ids:
                independence_check[other_id] = evidence

    return AtomicSpec(
        spec_id=primary.spec_id,
        text="\n\n".join(spec.text for spec in group),
        category=category,
        sources=sources,
        code_mapping=code_mapping,
        merged_from=constituents,
        independence_check=independence_check or None,
    )


def _dedupe_redundant_specs(
    specs: List[AtomicSpec],
    step2_by_id: Dict[str, Tuple[StepResult, float]],
) -> Tuple[List[AtomicSpec], List[List[str]]]:
    """Merge specs whose Step 2 mutations make the identical source edit."""
    by_signature: Dict[Tuple, List[AtomicSpec]] = {}
    order: List[Tuple] = []

    for spec in specs:
        entry = step2_by_id.get(spec.spec_id)
        step2_result = entry[0] if entry else None
        if not step2_result or not step2_result.success or not step2_result.data:
            continue
        mutations = step2_result.data.get("mutations") or []
        if not mutations:
            continue
        signature = _mutation_signature(mutations)
        if signature not in by_signature:
            by_signature[signature] = []
            order.append(signature)
        by_signature[signature].append(spec)

    absorbed: Dict[str, AtomicSpec] = {}
    dropped: set = set()
    merged_groups: List[List[str]] = []

    for signature in order:
        group = by_signature[signature]
        if len(group) < 2:
            continue
        shared = set.intersection(*(_spec_functions(spec) for spec in group))
        if not shared:
            continue
        merged = _merge_specs(group)
        absorbed[group[0].spec_id] = merged
        dropped.update(spec.spec_id for spec in group[1:])
        merged_groups.append([spec.spec_id for spec in group])

    if not merged_groups:
        return specs, []

    rewritten = [
        absorbed.get(spec.spec_id, spec)
        for spec in specs
        if spec.spec_id not in dropped
    ]
    return rewritten, merged_groups


def symmetrise_probe_selection(
    probe_ids_by_spec: Dict[str, List[str]],
    witness_by_spec: Dict[str, str],
) -> int:
    """Add the reverse of every chosen probe edge, in place."""
    added = 0
    for spec_id in list(probe_ids_by_spec):
        if spec_id not in witness_by_spec:
            continue
        for probed_id in list(probe_ids_by_spec[spec_id]):
            reverse = probe_ids_by_spec.get(probed_id)
            if reverse is None or spec_id in reverse:
                continue
            reverse.append(spec_id)
            added += 1
    return added


class MutationPipeline:
    """Orchestrates the full mutation pipeline."""

    def __init__(
        self,
        config: MutationPipelineConfig,
        config_loader: InferConfigLoader,
    ):
        if not config.llm_b_name:
            raise ValueError(
                "llm_b_name is required — Step 6 verification cannot be skipped. "
                "Pass --llm-b with an [llm.*] config name."
            )

        self.config = config
        self.config_loader = config_loader

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prompts_dir = Path(__file__).parent / "prompts"
        self.prompt_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        self.llm_a = create_llm_caller(config_loader, config.llm_a_name)

        self.llm_b = create_llm_caller(config_loader, config.llm_b_name)

        self.docker_runner = MutationDockerRunner(
            proxy_port=config.proxy_port,
            gpu_ids=config.gpu_ids,
        )

        self.spec_decomposer = SpecDecomposer(self.llm_a, self.prompt_env)
        self.mutation_generator = MutationGenerator(self.llm_a, self.prompt_env)
        self.test_verifier = TestVerifier(self.docker_runner)
        self.witness_executor = WitnessExecutor(self.docker_runner)
        self.preservation_prober = PreservationProber(self.docker_runner)
        self.llm_verifier = LLMVerifier(
            self.llm_b, self.prompt_env, votes=config.step6_votes
        )

        self._stats: Dict[str, int] = {
            "total_instances": 0,
            "total_specs": 0,
            "step1_success": 0,
            "step2_success": 0,
            "specs_merged": 0,
            "step3_pass": 0,
            "step4_diverged": 0,
            "step5_probe_diverged": 0,
            "step6_approved": 0,
            "specs_infeasible": 0,
            "specs_overreaching": 0,
            "final_surviving": 0,
            "dropped_no_control": 0,
            "eval_ready": 0,
        }

    def run_all(
        self,
        inputs: List[PipelineInput],
        existing_results: Optional[List[MutationResult]] = None,
        completed_instances: int = 0,
    ) -> List[MutationResult]:
        """Run the pipeline on all instances."""
        existing_results = list(existing_results or [])
        all_results: List[MutationResult] = list(existing_results)
        self._stats["total_instances"] = len(inputs) + completed_instances
        self._stats["final_surviving"] = len(existing_results)

        if completed_instances:
            logger.info(
                f"Resuming with {completed_instances} completed instances "
                f"and {len(existing_results)} existing surviving mutations"
            )

        logger.info(f"Starting mutation pipeline on {len(inputs)} instances")
        pipeline_start = time.time()

        for i, pipeline_input in enumerate(inputs, 1):
            logger.info(
                f"[{i}/{len(inputs)}] Processing {pipeline_input.instance_id}"
            )
            instance_start = time.time()
            usage_before = self._usage_snapshot()

            try:
                instance_results = self.run_instance(pipeline_input)
                all_results.extend(instance_results)
                self._save_instance_results(pipeline_input.instance_id, instance_results)
            except Exception as e:
                logger.error(
                    f"[{pipeline_input.instance_id}] Instance failed: {e}",
                    exc_info=True,
                )
            finally:
                self._save_instance_usage(pipeline_input.instance_id, usage_before)
                instance_elapsed = time.time() - instance_start
                logger.info(
                    f"[{i}/{len(inputs)}] Instance {pipeline_input.instance_id} "
                    f"completed in {instance_elapsed:.1f}s"
                )

        pipeline_elapsed = time.time() - pipeline_start
        logger.info(f"All instances processed in {pipeline_elapsed:.1f}s")

        eval_ready = apply_control_filter(
            all_results, self.config.preservation_probe_k
        )
        self._stats["dropped_no_control"] = len(all_results) - len(eval_ready)
        self._stats["eval_ready"] = len(eval_ready)

        self.save_final_output(eval_ready)

        self._log_statistics()

        return eval_ready

    def run_instance(self, pipeline_input: PipelineInput) -> List[MutationResult]:
        """Process a single instance through all steps for all its specs."""
        iid = pipeline_input.instance_id
        results: List[MutationResult] = []

        t0 = time.time()
        step1_result = self.spec_decomposer.decompose(pipeline_input)
        step1_elapsed = time.time() - t0
        self._save_step_artifact(iid, "_all", "step1_spec_decomposition", step1_result)

        if not step1_result.success:
            logger.warning(f"[{iid}] Step 1 failed ({step1_elapsed:.1f}s): {step1_result.error}")
            return results

        specs: List[AtomicSpec] = step1_result.data
        self._stats["step1_success"] += 1
        self._stats["total_specs"] += len(specs)

        logger.info(f"[{iid}] Step 1: {len(specs)} specs decomposed in {step1_elapsed:.1f}s")

        results = self._run_pipelined(specs, pipeline_input)

        logger.info(f"[{iid}] Final: {len(results)}/{len(specs)} specs survived")
        return results

    def _run_pipelined(
        self,
        specs: List[AtomicSpec],
        pipeline_input: PipelineInput,
    ) -> List[MutationResult]:
        """Verification phase, barrier, probe phase."""
        iid = pipeline_input.instance_id
        pipeline_start = time.time()
        specs_total = len(specs)

        verified = self._run_verification_phase(specs, pipeline_input)

        kept = [spec for spec in specs if spec.spec_id in verified]
        step2_by_id = {
            sid: (v.step2_result, v.step2_time) for sid, v in verified.items()
        }
        kept, merged_groups = _dedupe_redundant_specs(kept, step2_by_id)
        for group in merged_groups:
            logger.info(
                f"[{iid}] Merged redundant specs {'+'.join(group)} into "
                f"{group[0]}: identical Step 2 mutation, so they are not "
                f"independently violable"
            )
            self._stats["specs_merged"] += len(group) - 1
            for absorbed in group[1:]:
                verified.pop(absorbed, None)
        for spec in kept:
            verified[spec.spec_id].spec = spec

        witness_by_spec: Dict[str, str] = {
            sid: v.proof.witness_script for sid, v in verified.items()
        }

        results = self._run_probe_phase(
            kept, pipeline_input, verified, witness_by_spec
        )

        logger.info(
            f"[{iid}] Pipeline complete: {len(verified)}/{specs_total} specs "
            f"cleared verification, {len(results)} survived all filters in "
            f"{time.time() - pipeline_start:.1f}s wall time"
        )
        return results

    def _run_verification_phase(
        self,
        specs: List[AtomicSpec],
        pipeline_input: PipelineInput,
    ) -> Dict[str, "_VerifiedSpec"]:
        """Run Steps 2, 3, 4 and 6a for every spec, one task each."""
        iid = pipeline_input.instance_id
        max_attempts = self.config.step2_max_attempts
        workers = max(1, min(self.config.container_pool_size, len(specs)))
        logger.info(
            f"[{iid}] Verification phase: {len(specs)} specs, {workers} in parallel"
        )

        verified: Dict[str, _VerifiedSpec] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._verify_spec, spec, pipeline_input, specs): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                done += 1
                try:
                    result, counts = future.result()
                except Exception as e:
                    logger.error(
                        f"[{iid}][{spec.spec_id}] Verification task crashed: {e}",
                        exc_info=True,
                    )
                    continue
                for key, value in counts.items():
                    self._stats[key] = self._stats.get(key, 0) + value
                if result is not None:
                    verified[spec.spec_id] = result
                logger.info(
                    f"[{iid}][{spec.spec_id}] Verification "
                    f"{'passed' if result is not None else 'failed'} "
                    f"after {result.attempt if result is not None else max_attempts} "
                    f"attempt(s) ({done}/{len(specs)} done)"
                )
        return verified

    def _verify_spec(
        self,
        spec: AtomicSpec,
        pipeline_input: PipelineInput,
        all_specs: List[AtomicSpec],
    ) -> Tuple[Optional["_VerifiedSpec"], Dict[str, int]]:
        """One spec through Step 2 → 3 → 4 → 6a, retrying the whole chain."""
        iid = pipeline_input.instance_id
        sid = spec.spec_id
        counts: Dict[str, int] = {}
        max_attempts = self.config.step2_max_attempts

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                logger.info(
                    f"[{iid}][{sid}] Retrying step 2 "
                    f"(attempt {attempt}/{max_attempts}) after downstream rejection"
                )
            step2_result, step2_time = self._run_step2_single(
                spec, pipeline_input, all_specs
            )
            self._save_step_artifact(
                iid, sid,
                self._attempt_name("step2_mutation_generation", attempt),
                step2_result,
            )
            if step2_result.success:
                counts["step2_success"] = counts.get("step2_success", 0) + 1

            try:
                verified = self._verify_attempt(
                    spec, pipeline_input, step2_result, step2_time,
                    attempt, max_attempts, counts,
                )
            except SpecInfeasible as e:
                counts["specs_infeasible"] = counts.get("specs_infeasible", 0) + 1
                logger.info(
                    f"[{iid}][{sid}] Spec judged infeasible at Step 6a "
                    f"(attempt {attempt}/{max_attempts}); dropped without "
                    f"further retries: {e}"
                )
                return None, counts
            if verified is not None:
                return verified, counts

        logger.info(
            f"[{iid}][{sid}] Exhausted {max_attempts} step-2 attempts; spec dropped"
        )
        return None, counts

    def _verify_attempt(
        self,
        spec: AtomicSpec,
        pipeline_input: PipelineInput,
        step2_result: StepResult,
        step2_time: float,
        attempt: int,
        max_attempts: int,
        counts: Dict[str, int],
    ) -> Optional["_VerifiedSpec"]:
        """One attempt at Steps 3, 4 and 6a."""
        iid = pipeline_input.instance_id
        sid = spec.spec_id
        filter_survival = FilterSurvival()
        step_timings: Dict[str, float] = {"step2_mutation_generation": step2_time}
        attempt_tag = f"[attempt {attempt}/{max_attempts}]"

        if not step2_result.success:
            logger.info(
                f"[{iid}][{sid}] {attempt_tag} Filtered at Step 2: {step2_result.error}"
            )
            return None

        mutation_data = step2_result.data
        mutations = mutation_data["mutations"]
        witness_script = mutation_data["witness_script"]
        mutation_strategy = mutation_data["mutation_strategy"]
        proof = Proof(
            witness_script=witness_script,
            divergence_type=mutation_data.get("divergence_type", ""),
            trigger_condition=mutation_data.get("trigger_condition", ""),
            case_contract=mutation_data.get("case_contract", ""),
            witness_factory=mutation_data.get("witness_factory", ""),
            witness_observer=mutation_data.get("witness_observer", ""),
        )

        container = None
        try:
            container = self.docker_runner.setup_instance(pipeline_input)

            t0 = time.time()
            step3_result = self.test_verifier.verify(
                pipeline_input, mutations, self.config.timeout,
                container=container,
            )
            step_timings["step3_test_verification"] = time.time() - t0
            self._save_step_artifact(
                iid, sid,
                self._attempt_name("step3_test_verification", attempt),
                step3_result,
            )
            filter_survival.tests_passed = step3_result.success

            if not step3_result.success:
                logger.info(
                    f"[{iid}][{sid}] {attempt_tag} Filtered at Step 3 "
                    f"({step_timings['step3_test_verification']:.1f}s): tests failed"
                )
                self._log_step_timings(iid, sid, step_timings)
                return None

            logger.info(
                f"[{iid}][{sid}] {attempt_tag} Step 3 completed in "
                f"{step_timings['step3_test_verification']:.1f}s"
            )
            counts["step3_pass"] = counts.get("step3_pass", 0) + 1
            mutant_patch = step3_result.data["mutant_patch"]

            t0 = time.time()
            step4_result = self.witness_executor.execute(
                pipeline_input, witness_script, self.config.hypothesis_timeout,
                container=container,
            )
            step_timings["step4_witness_execution"] = time.time() - t0
            self._save_step_artifact(
                iid, sid,
                self._attempt_name("step4_witness_execution", attempt),
                step4_result,
            )
            filter_survival.witness_diverged = step4_result.success

            if not step4_result.success:
                logger.info(
                    f"[{iid}][{sid}] {attempt_tag} Filtered at Step 4 "
                    f"({step_timings['step4_witness_execution']:.1f}s): "
                    f"{step4_result.error}"
                )
                self._log_step_timings(iid, sid, step_timings)
                return None

            logger.info(
                f"[{iid}][{sid}] {attempt_tag} Step 4 completed in "
                f"{step_timings['step4_witness_execution']:.1f}s"
            )
            counts["step4_diverged"] = counts.get("step4_diverged", 0) + 1
        finally:
            if container is not None:
                self.docker_runner.cleanup(container)

        proof.actual_gold_output = step4_result.data["gold_output"]
        proof.actual_mutant_output = step4_result.data["mutant_output"]
        gold_stray_output = step4_result.data.get("gold_stray_output", "")
        mutant_stray_output = step4_result.data.get("mutant_stray_output", "")

        self.preservation_prober.record_gold_observation(
            iid, witness_script, proof.actual_gold_output
        )

        t0 = time.time()
        step6a_result = self.llm_verifier.verify_target(
            spec, mutant_patch, proof, pipeline_input,
            mutation_strategy=mutation_strategy,
            gold_stray_output=gold_stray_output,
            mutant_stray_output=mutant_stray_output,
        )
        step_timings["step6a_target_verification"] = time.time() - t0
        self._save_step_artifact(
            iid, sid,
            self._attempt_name("step6a_target_verification", attempt),
            step6a_result,
        )

        if not step6a_result.success:
            category = (step6a_result.data or {}).get("category")
            logger.info(
                f"[{iid}][{sid}] {attempt_tag} Filtered at Step 6a "
                f"({step_timings['step6a_target_verification']:.1f}s): "
                f"verifier rejected ({category})"
            )
            self._log_step_timings(iid, sid, step_timings)
            if category == "non_behavioral":
                raise SpecInfeasible(step6a_result.error or "non_behavioral")
            return None

        logger.info(
            f"[{iid}][{sid}] {attempt_tag} Step 6a completed in "
            f"{step_timings['step6a_target_verification']:.1f}s"
        )
        return _VerifiedSpec(
            spec=spec,
            attempt=attempt,
            step2_result=step2_result,
            step2_time=step2_time,
            mutations=mutations,
            mutant_patch=mutant_patch,
            proof=proof,
            mutation_strategy=mutation_strategy,
            gold_stray_output=gold_stray_output,
            mutant_stray_output=mutant_stray_output,
            filter_survival=filter_survival,
            step_timings=step_timings,
        )

    def _run_probe_phase(
        self,
        specs: List[AtomicSpec],
        pipeline_input: PipelineInput,
        verified: Dict[str, "_VerifiedSpec"],
        witness_by_spec: Dict[str, str],
    ) -> List[MutationResult]:
        """Probe every mutant, then judge preservation, then assemble results."""
        iid = pipeline_input.instance_id
        ranked_by_spec: Dict[str, List[Tuple[str, int, int]]] = {}
        probe_ids_by_spec: Dict[str, List[str]] = {}
        for spec in specs:
            ranked = rank_probe_candidates(
                spec.spec_id,
                [
                    s for s in specs
                    if s.spec_id == spec.spec_id or s.spec_id in witness_by_spec
                ],
            )
            ranked_by_spec[spec.spec_id] = ranked
            k = self.config.preservation_probe_k
            probe_ids_by_spec[spec.spec_id] = (
                [candidate_id for candidate_id, _, _ in ranked[:k]] if k > 0 else []
            )

        score_by_pair: Dict[Tuple[str, str], Tuple[int, int]] = {
            (sid, candidate_id): (shared_functions, shared_lines)
            for sid, ranked in ranked_by_spec.items()
            for candidate_id, shared_functions, shared_lines in ranked
        }

        closure_added = symmetrise_probe_selection(
            probe_ids_by_spec, witness_by_spec
        )

        for spec in specs:
            if not probe_ids_by_spec[spec.spec_id]:
                logger.info(
                    f"[{iid}][{spec.spec_id}] Step 5: no spec shares code with "
                    f"this one — no preservation probe, and eval will have no "
                    f"negative control backed by this mutant"
                )

        tasks = [
            (spec.spec_id, probed_id)
            for spec in specs
            for probed_id in probe_ids_by_spec[spec.spec_id]
        ]
        workers = max(1, min(self.config.container_pool_size, len(tasks) or 1))
        logger.info(
            f"[{iid}] Probe phase: {len(tasks)} probes over {len(specs)} specs "
            f"({closure_added} added by symmetric closure), "
            f"{workers} in parallel"
        )

        probes_by_spec: Dict[str, List[PreservationProbe]] = {
            spec.spec_id: [] for spec in specs
        }
        if tasks:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.preservation_prober.probe_one,
                        pipeline_input,
                        sid,
                        verified[sid].mutations,
                        probed_id,
                        witness_by_spec[probed_id],
                        self.config.hypothesis_timeout,
                    ): (sid, probed_id)
                    for sid, probed_id in tasks
                }
                for future in as_completed(futures):
                    sid, probed_id = futures[future]
                    try:
                        probes_by_spec[sid].append(future.result())
                    except Exception as e:
                        logger.warning(
                            f"[{iid}][{sid}] Step 5 probe of {probed_id} "
                            f"crashed: {e}", exc_info=True
                        )
            logger.info(f"[{iid}] Probe phase ran in {time.time() - t0:.1f}s")

        for sid, probes in probes_by_spec.items():
            for probe in probes:
                shared_functions, shared_lines = score_by_pair.get(
                    (sid, probe.spec_id), (0, 0)
                )
                probe.shared_functions = shared_functions
                probe.shared_lines = shared_lines
            probes.sort(
                key=lambda p: (-p.shared_functions, -p.shared_lines, p.spec_id)
            )

        for spec in specs:
            sid = spec.spec_id
            probes = probes_by_spec[sid]
            self._save_step_artifact(
                iid, sid,
                self._attempt_name(
                    "step5_preservation_probe", verified[sid].attempt
                ),
                StepResult(
                    success=True,
                    data={
                        "probes": [p.to_dict() for p in probes],
                        "diverged_spec_ids": [
                            p.spec_id for p in probes
                            if p.verdict == VERDICT_DIVERGED
                        ],
                    },
                    step_name="preservation_probe",
                ),
            )
            if any(p.verdict == VERDICT_DIVERGED for p in probes):
                self._stats["step5_probe_diverged"] += 1

        by_spec: Dict[str, MutationResult] = {}
        judge_workers = max(1, min(self.config.step6b_concurrency, len(specs) or 1))
        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            futures = {
                executor.submit(
                    self._judge_preservation,
                    pipeline_input, specs, verified[spec.spec_id],
                    probes_by_spec[spec.spec_id],
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result, counts = future.result()
                except Exception as e:
                    logger.error(
                        f"[{iid}][{spec.spec_id}] Step 6b task crashed: {e}",
                        exc_info=True,
                    )
                    continue
                for key, value in counts.items():
                    self._stats[key] = self._stats.get(key, 0) + value
                if result is not None:
                    by_spec[spec.spec_id] = result

        return [by_spec[s.spec_id] for s in specs if s.spec_id in by_spec]

    def _judge_preservation(
        self,
        pipeline_input: PipelineInput,
        specs: List[AtomicSpec],
        verified: "_VerifiedSpec",
        probes: List[PreservationProbe],
    ) -> Tuple[Optional[MutationResult], Dict[str, int]]:
        """Run Step 6b over one spec's probes and build its result row."""
        iid = pipeline_input.instance_id
        spec = verified.spec
        sid = spec.spec_id
        counts: Dict[str, int] = {}
        other_specs = [s for s in specs if s.spec_id != sid]

        t0 = time.time()
        broken_by = self._run_preservation_audit(
            spec, verified.mutant_patch, verified.proof, pipeline_input,
            other_specs, probes, verified.mutation_strategy,
            verified.gold_stray_output, verified.mutant_stray_output,
            verified.attempt,
        )
        verified.step_timings["step6b_preservation_audit"] = time.time() - t0
        verified.filter_survival.verifier_approved = broken_by is None
        self._log_step_timings(iid, sid, verified.step_timings)

        if broken_by is not None:
            counts["specs_overreaching"] = counts.get("specs_overreaching", 0) + 1
            logger.info(
                f"[{iid}][{sid}] Mutation breaks {broken_by}; dropped without "
                f"further retries"
            )
            logger.info(
                f"[{iid}][{sid}] Spec completed (survived=False, "
                f"attempt={verified.attempt}, probes={len(probes)})"
            )
            return None, counts

        logger.info(
            f"[{iid}][{sid}] Step 6b completed in "
            f"{verified.step_timings['step6b_preservation_audit']:.1f}s"
        )
        logger.info(
            f"[{iid}][{sid}] Spec completed (survived=True, "
            f"attempt={verified.attempt}, probes={len(probes)})"
        )
        counts["step6_approved"] = counts.get("step6_approved", 0) + 1
        counts["final_surviving"] = counts.get("final_surviving", 0) + 1

        return MutationResult(
            schema_version=3,
            instance_id=iid,
            spec=spec,
            mask_patch=pipeline_input.mask_patch or "",
            gold_patch=pipeline_input.gold_patch,
            mutant_patch=verified.mutant_patch,
            test_patch=pipeline_input.test_patch or "",
            proof=verified.proof,
            metadata={
                "mutations": verified.mutations,
                "mutation_strategy": verified.mutation_strategy,
                "filter_survival": verified.filter_survival.to_dict(),
                "step2_attempt": verified.attempt,
                "preservation_probes": [p.to_dict() for p in probes],
            },
        ), counts

    def _run_step2_single(
        self,
        spec: AtomicSpec,
        pipeline_input: PipelineInput,
        all_specs: List[AtomicSpec],
    ) -> Tuple[StepResult, float]:
        """Run Step 2 for a single spec."""
        t0 = time.time()
        other_specs = [s for s in all_specs if s.spec_id != spec.spec_id]
        step2_result = self.mutation_generator.generate(spec, pipeline_input, other_specs)
        elapsed = time.time() - t0
        return step2_result, elapsed

    @staticmethod
    def _attempt_name(step_name: str, attempt: int) -> str:
        """Artifact name for an attempt; attempts after the first get a suffix."""
        return step_name if attempt == 1 else f"{step_name}_attempt{attempt}"

    def _run_preservation_audit(
        self,
        spec: AtomicSpec,
        mutant_patch: str,
        proof: Proof,
        pipeline_input: PipelineInput,
        other_specs: List[AtomicSpec],
        preservation_probes: List[PreservationProbe],
        mutation_strategy: str,
        gold_stray_output: str,
        mutant_stray_output: str,
        attempt: int,
    ) -> Optional[str]:
        """Ask Step 6b one question per call; return the spec that broke."""
        iid = pipeline_input.instance_id
        sid = spec.spec_id

        def audit(probe: Optional[PreservationProbe]) -> StepResult:
            return self.llm_verifier.verify_preservation(
                spec, mutant_patch, proof, pipeline_input, other_specs,
                probe=probe,
                mutation_strategy=mutation_strategy,
                gold_stray_output=gold_stray_output,
                mutant_stray_output=mutant_stray_output,
            )

        planned: List[Tuple[str, Optional[PreservationProbe]]] = [
            (probe.spec_id, probe) for probe in preservation_probes
        ]
        planned.append(("patch-audit", None))

        workers = max(1, min(self.config.step6b_concurrency, len(planned)))
        results_by_id: Dict[str, StepResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(audit, probe): probed_id
                for probed_id, probe in planned
            }
            for future in as_completed(futures):
                probed_id = futures[future]
                try:
                    results_by_id[probed_id] = future.result()
                except Exception as e:
                    logger.error(
                        f"[{iid}][{sid}] Step 6b[{probed_id}] task crashed: {e}",
                        exc_info=True,
                    )
                    results_by_id[probed_id] = StepResult(
                        success=False,
                        error=f"Step 6b task crashed: {e}",
                        data={"verdict": "reject", "category": "infra_error",
                              "reasoning": str(e)},
                        step_name="llm_verification",
                    )

        calls: List[Tuple[str, StepResult]] = [
            (probed_id, results_by_id[probed_id])
            for probed_id, _ in planned
            if probed_id in results_by_id
        ]

        broken_by: Optional[str] = None
        for probed_id, result in calls:
            self._save_step_artifact(
                iid, sid,
                self._attempt_name(f"step6b_preservation_{probed_id}", attempt),
                result,
            )
            if result.success:
                continue
            category = (result.data or {}).get("category")
            if category != "wrong_spec":
                logger.warning(
                    f"[{iid}][{sid}] Step 6b[{probed_id}] did not return a "
                    f"verdict ({category}); treating as no evidence: "
                    f"{result.error}"
                )
                continue
            logger.info(
                f"[{iid}][{sid}] Step 6b: mutation breaks {probed_id} — "
                f"{result.error}"
            )
            if broken_by is None:
                broken_by = probed_id
        return broken_by

    def _log_step_timings(
        self, iid: str, sid: str, step_timings: Dict[str, float]
    ) -> None:
        """Log a timing summary for all steps executed on a spec."""
        total = sum(step_timings.values())
        parts = ", ".join(f"{k}: {v:.1f}s" for k, v in step_timings.items())
        logger.info(f"[{iid}][{sid}] Step timings (total {total:.1f}s): {parts}")

    def _save_instance_results(
        self, instance_id: str, results: List[MutationResult]
    ) -> None:
        """Save per-instance results to output directory."""
        instance_dir = self.output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        results_path = instance_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _usage_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Both callers' running token accounts, cut at the same point."""
        return {
            role: getattr(self, role).snapshot_usage()
            for role in ("llm_a", "llm_b")
        }

    def _save_instance_usage(
        self, instance_id: str, before: Dict[str, Dict[str, int]]
    ) -> None:
        """Record what one instance alone cost, as a delta against the totals."""
        after = self._usage_snapshot()
        usage = {
            role: {k: v - before[role][k] for k, v in after[role].items()}
            for role in after
        }

        usage_path = self.output_dir / instance_id / "llm_usage.json"
        prior = self._read_instance_usage(usage_path)
        if prior is not None:
            usage = {
                role: {k: v + prior.get(role, {}).get(k, 0) for k, v in counts.items()}
                for role, counts in usage.items()
            }

        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_path, "w", encoding="utf-8") as f:
            json.dump(
                {"instance_id": instance_id, **usage}, f, ensure_ascii=False, indent=2
            )

    @staticmethod
    def _read_instance_usage(
        usage_path: Path,
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """One instance's token record, or None if absent or unreadable."""
        if not usage_path.exists():
            return None
        try:
            with open(usage_path, encoding="utf-8") as f:
                record = json.load(f)
            return {"llm_a": record["llm_a"], "llm_b": record["llm_b"]}
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            logger.warning(f"Skipping malformed usage record {usage_path}: {e}")
            return None

    def _collect_instance_usage(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Every instance's token record, read back off disk, keyed by instance."""
        per_instance: Dict[str, Dict[str, Dict[str, int]]] = {}
        for usage_path in sorted(self.output_dir.glob("*/llm_usage.json")):
            record = self._read_instance_usage(usage_path)
            if record is not None:
                per_instance[usage_path.parent.name] = record
        return per_instance

    def _sum_usage(
        self, role: str, per_instance: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Dict[str, int]:
        """Total one role's spend across instances."""
        totals = dict.fromkeys(getattr(self, role).snapshot_usage(), 0)
        for record in per_instance.values():
            for key, value in record.get(role, {}).items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def _save_step_artifact(
        self,
        instance_id: str,
        spec_id: str,
        step_name: str,
        step_result: StepResult,
    ) -> None:
        """Save intermediate step output for debugging."""
        instance_dir = self.output_dir / instance_id / "artifacts"
        instance_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = instance_dir / f"{spec_id}_{step_name}.json"

        artifact: Dict[str, Any] = {
            "success": step_result.success,
            "error": step_result.error,
            "step_name": step_result.step_name,
        }

        try:
            if step_result.data is not None:
                artifact["data"] = self._make_serializable(step_result.data)
        except Exception:
            artifact["data"] = "<non-serializable>"

        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Recursively convert dataclass objects to dicts for JSON serialization."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: MutationPipeline._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MutationPipeline._make_serializable(item) for item in obj]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)

    def save_final_output(self, results: List[MutationResult]) -> None:
        """Save all surviving results as JSONL."""
        output_path = self.output_dir / "output.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.to_jsonl() + "\n")

        stats_path = self.output_dir / "pipeline_stats.json"
        per_instance = self._collect_instance_usage()
        stats = {
            **self._stats,
            "llm_usage": {
                **{
                    role: self._sum_usage(role, per_instance)
                    for role in ("llm_a", "llm_b")
                },
                "per_instance": per_instance,
            },
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llm_a_name": self.config.llm_a_name,
                "llm_b_name": self.config.llm_b_name,
                "timeout": self.config.timeout,
                "hypothesis_timeout": self.config.hypothesis_timeout,
            },
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Final output: {len(results)} results saved to {output_path}")

    def _log_statistics(self) -> None:
        """Log pipeline filter statistics."""
        s = self._stats
        logger.info("=" * 60)
        logger.info("Pipeline Statistics")
        logger.info("=" * 60)
        logger.info(f"  Instances processed:     {s['total_instances']}")
        logger.info(f"  Total specs decomposed:  {s['total_specs']}")
        logger.info(f"  Step 1 (decomposition):  {s['step1_success']} instances succeeded")
        logger.info(f"  Step 2 (mutation gen):    {s['step2_success']} specs generated")
        logger.info(f"  Redundant specs merged:   {s['specs_merged']} folded away")
        logger.info(f"  Step 3 (test verify):     {s['step3_pass']} passed tests")
        logger.info(f"  Step 4 (witness diverge): {s['step4_diverged']} diverged on witness")
        logger.info(
            f"  Step 5 (preservation):    {s['step5_probe_diverged']} moved another "
            f"spec's witness (not a filter)"
        )
        logger.info(
            f"  Step 6a (targeted spec):  {s['specs_infeasible']} specs judged "
            f"unmutatable (dropped, no retry)"
        )
        logger.info(
            f"  Step 6b (preservation):   {s['specs_overreaching']} mutations broke "
            f"another spec (dropped, no retry)"
        )
        logger.info(f"  Step 6 (both halves):     {s['step6_approved']} approved")
        logger.info(f"  Final surviving:          {s['final_surviving']}")
        logger.info(
            f"  No negative control:      {s['dropped_no_control']} dropped from output.jsonl"
        )
        logger.info(f"  Eval-ready (output.jsonl):{s['eval_ready']}")
        logger.info("=" * 60)
