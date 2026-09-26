"""Main pipeline orchestrator."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvalPipelineConfig, load_hf_metadata
from .models import BaselineResult, DifferentialResult, EvalTask, ProofExecResult
from .stages.stage1_task_assembly import TaskAssembler, load_eval_tasks, save_eval_tasks
from .stages.stage2_baseline_execution import BaselineExecutor, load_baseline_results
from .stages.stage3_proof_execution import ProofExecutor, load_proof_results
from .stages.stage4_differential_analysis import DifferentialAnalyzer, load_differential_results
from .stages.stage5_metrics import MetricsReporter

logger = logging.getLogger(__name__)


class EvalPipeline:
    """Orchestrates the full evaluation pipeline."""

    def __init__(
        self,
        config: EvalPipelineConfig,
        config_loader,
    ):
        self.config = config
        self.config_loader = config_loader
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docker_runner = None

    def _get_docker_runner(self):
        if self.docker_runner is None:
            from mutation_pipeline.docker_runner import MutationDockerRunner

            self.docker_runner = MutationDockerRunner(
                proxy_port=self.config.proxy_port,
                gpu_ids=self.config.gpu_ids,
            )
        return self.docker_runner

    def run(self, stages: Optional[List[int]] = None) -> None:
        """Run the pipeline (all stages or a subset)."""
        stages = stages or self.config.stages or [1, 2, 3, 4, 5]
        pipeline_start = time.time()

        tasks: List[EvalTask] = []
        baseline_results: List[BaselineResult] = []
        proof_results: List[ProofExecResult] = []
        diff_results: List[DifferentialResult] = []

        if 1 in stages:
            logger.info("=" * 60)
            logger.info("Stage 1: Task Assembly")
            logger.info("=" * 60)
            t0 = time.time()
            hf_data = load_hf_metadata(self.config_loader, self.config)
            assembler = TaskAssembler(self.config, hf_data)
            tasks = assembler.assemble()
            save_eval_tasks(tasks, self.output_dir)
            logger.info(f"Stage 1 completed in {time.time() - t0:.1f}s")
        else:
            tasks = load_eval_tasks(self.output_dir)

        if 2 in stages:
            logger.info("=" * 60)
            logger.info("Stage 2: Baseline Execution")
            logger.info("=" * 60)
            t0 = time.time()
            docker_runner = self._get_docker_runner()
            executor = BaselineExecutor(
                self.config, self.config_loader, docker_runner
            )
            baseline_results = executor.run_all(tasks, self.config.baselines)
            logger.info(f"Stage 2 completed in {time.time() - t0:.1f}s")
        else:
            baseline_results = load_baseline_results(self.output_dir)

        if 3 in stages:
            logger.info("=" * 60)
            logger.info("Stage 3: Proof Execution")
            logger.info("=" * 60)
            t0 = time.time()
            proof_executor = ProofExecutor(self.config, self._get_docker_runner())
            proof_results = proof_executor.run_all(tasks, baseline_results)
            logger.info(f"Stage 3 completed in {time.time() - t0:.1f}s")
        else:
            proof_results = load_proof_results(self.output_dir)

        if 4 in stages:
            logger.info("=" * 60)
            logger.info("Stage 4: Differential Analysis")
            logger.info("=" * 60)
            t0 = time.time()
            analyzer = DifferentialAnalyzer(self.config)
            diff_results = analyzer.analyze(tasks, baseline_results, proof_results)
            logger.info(f"Stage 4 completed in {time.time() - t0:.1f}s")
        else:
            diff_results = load_differential_results(self.output_dir)

        if 5 in stages:
            logger.info("=" * 60)
            logger.info("Stage 5: Metrics & Reporting")
            logger.info("=" * 60)
            t0 = time.time()
            reporter = MetricsReporter(self.config)
            summary = reporter.compute_and_save(
                tasks, baseline_results, diff_results,
                proof_results=proof_results,
            )
            logger.info(f"Stage 5 completed in {time.time() - t0:.1f}s")
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        elapsed = time.time() - pipeline_start
        stats = {
            "stages_run": stages,
            "total_tasks": len(tasks),
            "total_baseline_results": len(baseline_results),
            "total_proof_results": len(proof_results),
            "total_differential_results": len(diff_results),
            "wall_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Pipeline completed in {elapsed:.1f}s")
