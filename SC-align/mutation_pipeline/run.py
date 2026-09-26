#!/usr/bin/env python3
"""CLI entry point for the Spec-Code Misalignment Mutation Pipeline."""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SC_ALIGN = Path(__file__).resolve().parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from featurebench.infer.config import InferConfigLoader
from run_naming import run_dir_name

from .config import MutationPipelineConfig, load_pipeline_inputs
from .control_filter import apply_control_filter
from .models import MutationResult
from .pipeline import MutationPipeline


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    timeout_provided = any(arg == "--timeout" or arg.startswith("--timeout=") for arg in argv)
    hypothesis_timeout_provided = any(
        arg == "--hypothesis-timeout" or arg.startswith("--hypothesis-timeout=")
        for arg in argv
    )
    step2_max_attempts_provided = any(
        arg == "--step2-max-attempts" or arg.startswith("--step2-max-attempts=")
        for arg in argv
    )
    container_pool_provided = any(
        arg == "--container-pool" or arg.startswith("--container-pool=")
        for arg in argv
    )
    preservation_probe_k_provided = any(
        arg == "--preservation-probe-k" or arg.startswith("--preservation-probe-k=")
        for arg in argv
    )
    step6_votes_provided = any(
        arg == "--step6-votes" or arg.startswith("--step6-votes=")
        for arg in argv
    )
    step6b_concurrency_provided = any(
        arg == "--step6b-concurrency" or arg.startswith("--step6b-concurrency=")
        for arg in argv
    )

    parser = argparse.ArgumentParser(
        description="Spec-Code Misalignment Mutation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--llm-a",
        type=str,
        default=None,
        help="LLM-A config name from [llm.*] in config_mutation.toml (e.g., 'claude-opus-5')",
    )
    parser.add_argument(
        "--llm-b",
        type=str,
        default=None,
        help="LLM-B config name for step 6 verification (required for a fresh run)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="SC-align/mutation_output/runs",
        help="Base output directory for runs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Short note appended to the run directory name, e.g. 'full_set' "
            "-> runs/2026-08-31__14-30-00__full_set"
        ),
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to process",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Docker test timeout (seconds)",
    )
    parser.add_argument(
        "--hypothesis-timeout",
        type=int,
        default=120,
        help="Hypothesis test timeout (seconds)",
    )
    parser.add_argument(
        "--step2-max-attempts",
        type=int,
        default=3,
        help=(
            "Maximum total attempts (initial + retries) at Step 2 mutation "
            "generation per spec. Each downstream filter rejection (Step 2 "
            "form / Step 3 tests / Step 4 witness) loops back to a fresh LLM "
            "call. Default: 3."
        ),
    )
    parser.add_argument(
        "--container-pool",
        type=int,
        default=8,
        help=(
            "Concurrent Docker containers per instance. Bounds both phases: "
            "the verification phase runs one task per spec and the probe phase "
            "one task per probe, and neither exceeds this many containers at "
            "once. Sized by the host, not by the instance. Default: 8."
        ),
    )
    parser.add_argument(
        "--preservation-probe-k",
        type=int,
        default=3,
        help=(
            "How many other specs' witnesses each spec chooses at Step 5 to "
            "check that the mutation preserved the specs it does not target. "
            "Bounds choices, not probes received: the reverse of every chosen "
            "edge is added too, so a widely-chosen mutant runs more than k. "
            "Readings are handed to Step 6, never rejected on directly. "
            "0 disables the step. Default: 3."
        ),
    )
    parser.add_argument(
        "--step6-votes",
        type=int,
        default=3,
        help=(
            "Independent LLM-B ballots behind one Step 6 verdict, 6a and 6b "
            "alike; the majority decides. Every ballot is kept in the artifact "
            "so agreement can be measured afterwards. Keep it odd. 1 restores "
            "the old single-call behaviour. Default: 3."
        ),
    )
    parser.add_argument(
        "--step6b-concurrency",
        type=int,
        default=12,
        help=(
            "Concurrent Step 6b preservation audits, across specs and across "
            "one spec's probes. Not bounded by --container-pool: this phase is "
            "LLM-only and uses no container. Default: 12."
        ),
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=None,
        help="Proxy port for network access in containers",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1')",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config_mutation.toml",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LiberCoders/FeatureBench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="full",
        help="HuggingFace dataset split",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a previous run directory "
            "(e.g., SC-align/mutation_output/runs/2026-04-12__10-30-00). "
            "Most args are loaded from run_metadata.json; --timeout, "
            "--hypothesis-timeout, --proxy-port, --gpu-ids, --container-pool, "
            "--step2-max-attempts, --preservation-probe-k, --step6-votes and "
            "--step6b-concurrency can be overridden."
        ),
    )

    args = parser.parse_args()
    args._timeout_provided = timeout_provided
    args._hypothesis_timeout_provided = hypothesis_timeout_provided
    args._step2_max_attempts_provided = step2_max_attempts_provided
    args._container_pool_provided = container_pool_provided
    args._preservation_probe_k_provided = preservation_probe_k_provided
    args._step6_votes_provided = step6_votes_provided
    args._step6b_concurrency_provided = step6b_concurrency_provided
    return args


def setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)

    file_handler = logging.FileHandler(output_dir / "pipeline.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_fmt)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for lib in ["httpx", "openai", "urllib3", "docker", "httpcore"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def _save_run_metadata(output_dir: Path, config: MutationPipelineConfig, instance_ids: list) -> None:
    """Save run metadata to run_metadata.json."""
    metadata = {
        "llm_a_name": config.llm_a_name,
        "llm_b_name": config.llm_b_name,
        "output_dir": str(output_dir),
        "n_concurrent": config.n_concurrent,
        "instance_ids": instance_ids,
        "timeout": config.timeout,
        "hypothesis_timeout": config.hypothesis_timeout,
        "proxy_port": config.proxy_port,
        "gpu_ids": config.gpu_ids,
        "dataset": config.dataset,
        "split": config.split,
        "container_pool_size": config.container_pool_size,
        "step2_max_attempts": config.step2_max_attempts,
        "preservation_probe_k": config.preservation_probe_k,
        "step6_votes": config.step6_votes,
        "step6b_concurrency": config.step6b_concurrency,
        "start_time": datetime.now().isoformat(),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _update_end_time(output_dir: Path) -> None:
    """Update end_time in run_metadata.json."""
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["end_time"] = datetime.now().isoformat()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def _load_resume_config(resume_dir: Path, args: argparse.Namespace) -> MutationPipelineConfig:
    """Load pipeline config from a previous run's run_metadata.json."""
    metadata_path = resume_dir / "run_metadata.json"
    if not metadata_path.exists():
        print(f"Error: run_metadata.json not found in {resume_dir}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata.get("llm_b_name"):
        print(
            f"Error: {metadata_path} has no llm_b_name — that run skipped Step 6 "
            "verification, which is no longer supported. Start a fresh run with "
            "--llm-b instead of resuming it.",
            file=sys.stderr,
        )
        sys.exit(1)

    return MutationPipelineConfig(
        llm_a_name=metadata["llm_a_name"],
        llm_b_name=metadata["llm_b_name"],
        output_dir=resume_dir,
        n_concurrent=metadata.get("n_concurrent", 1),
        instance_ids=metadata.get("instance_ids"),
        timeout=args.timeout if getattr(args, "_timeout_provided", False) else metadata.get("timeout", 600),
        hypothesis_timeout=(
            args.hypothesis_timeout
            if getattr(args, "_hypothesis_timeout_provided", False)
            else metadata.get("hypothesis_timeout", 120)
        ),
        proxy_port=args.proxy_port if args.proxy_port is not None else metadata.get("proxy_port"),
        gpu_ids=args.gpu_ids if args.gpu_ids is not None else metadata.get("gpu_ids"),
        dataset=metadata.get("dataset", "LiberCoders/FeatureBench"),
        split=metadata.get("split", "full"),
        container_pool_size=args.container_pool if getattr(
            args, "_container_pool_provided", False
        ) else metadata.get("container_pool_size", 8),
        step2_max_attempts=args.step2_max_attempts if getattr(
            args, "_step2_max_attempts_provided", False
        ) else metadata.get("step2_max_attempts", 3),
        preservation_probe_k=args.preservation_probe_k if getattr(
            args, "_preservation_probe_k_provided", False
        ) else metadata.get("preservation_probe_k", 3),
        step6_votes=args.step6_votes if getattr(
            args, "_step6_votes_provided", False
        ) else metadata.get("step6_votes", 1),
        step6b_concurrency=args.step6b_concurrency if getattr(
            args, "_step6b_concurrency_provided", False
        ) else metadata.get("step6b_concurrency", 12),
    )


def _write_output_jsonl(output_path: Path, results: list[MutationResult]) -> None:
    """Rewrite output.jsonl from a list of per-spec results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.to_jsonl() + "\n")


def _legacy_output_path_for_run(run_dir: Path) -> Path:
    """Map a timestamped run directory to the base output.jsonl path."""
    base_output_dir = run_dir.parent
    legacy_output_dir = base_output_dir.parent if base_output_dir.name == "runs" else base_output_dir
    return legacy_output_dir / "output.jsonl"


def _sync_legacy_output(run_dir: Path) -> Path:
    """Copy the combined output.jsonl to the base output directory."""
    source = run_dir / "output.jsonl"
    legacy_output_path = _legacy_output_path_for_run(run_dir)
    legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, legacy_output_path)
    return legacy_output_path


def _load_completed_instance_results(run_dir: Path) -> dict[str, list[MutationResult]]:
    """Load per-instance results.json files to support resume without reruns."""
    logger = logging.getLogger(__name__)
    completed: dict[str, list[MutationResult]] = {}

    for results_path in sorted(run_dir.glob("*/results.json")):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping malformed resume artifact {results_path}: {e}")
            continue

        if not isinstance(payload, list):
            logger.warning(f"Skipping malformed resume artifact {results_path}: expected a JSON list")
            continue

        instance_id = results_path.parent.name
        completed[instance_id] = [
            MutationResult.from_dict(item)
            for item in payload
            if isinstance(item, dict)
        ]

    return completed


def main() -> None:
    args = parse_args()

    resume_dir = Path(args.resume) if args.resume else None

    if resume_dir:
        pipeline_config = _load_resume_config(resume_dir, args)
        run_dir = resume_dir
    else:
        if not args.llm_a:
            print("Error: --llm-a is required (unless using --resume)", file=sys.stderr)
            sys.exit(1)
        if not args.llm_b:
            print("Error: --llm-b is required (unless using --resume)", file=sys.stderr)
            sys.exit(1)

        run_dir = Path(args.output_dir) / run_dir_name(args.tag)

        pipeline_config = MutationPipelineConfig(
            llm_a_name=args.llm_a,
            llm_b_name=args.llm_b,
            output_dir=run_dir,
            instance_ids=args.instance_ids,
            timeout=args.timeout,
            hypothesis_timeout=args.hypothesis_timeout,
            proxy_port=args.proxy_port,
            gpu_ids=args.gpu_ids,
            dataset=args.dataset,
            split=args.split,
            container_pool_size=args.container_pool,
            step2_max_attempts=args.step2_max_attempts,
            preservation_probe_k=args.preservation_probe_k,
            step6_votes=args.step6_votes,
            step6b_concurrency=args.step6b_concurrency,
        )

    setup_logging(run_dir)
    logger = logging.getLogger(__name__)

    if resume_dir:
        logger.info(f"Resuming mutation pipeline from {resume_dir}")
    else:
        logger.info(f"Starting new mutation pipeline run at {run_dir}")

    config_path = Path(args.config_path).expanduser() if args.config_path else None
    config_loader = InferConfigLoader(config_path)

    logger.info("Loading pipeline inputs...")
    inputs = load_pipeline_inputs(config_loader, pipeline_config)
    logger.info(f"Loaded {len(inputs)} instances")

    if not inputs:
        logger.error("No instances to process. Check --instance-ids, --split, and data files.")
        sys.exit(1)

    existing_results: list[MutationResult] = []
    completed_instance_ids: set[str] = set()

    if resume_dir:
        completed_results = _load_completed_instance_results(run_dir)
        completed_instance_ids = set(completed_results)
        existing_results = [
            result
            for instance_id in sorted(completed_results)
            for result in completed_results[instance_id]
        ]

        if completed_instance_ids:
            pending_inputs = [
                pipeline_input
                for pipeline_input in inputs
                if pipeline_input.instance_id not in completed_instance_ids
            ]
            logger.info(
                f"Resume found {len(completed_instance_ids)} completed instances; "
                f"{len(pending_inputs)} remaining to process"
            )
            inputs = pending_inputs
        else:
            logger.info("Resume found no completed instances to reuse")

        if not inputs:
            eval_ready = apply_control_filter(
                existing_results, pipeline_config.preservation_probe_k
            )
            _write_output_jsonl(run_dir / "output.jsonl", eval_ready)
            legacy_output_path = _sync_legacy_output(run_dir)
            _update_end_time(run_dir)
            logger.info(
                f"Resume complete: no remaining instances to process; "
                f"synced combined output to {legacy_output_path}"
            )
            return

    if not resume_dir:
        _save_run_metadata(run_dir, pipeline_config, [inp.instance_id for inp in inputs])

    pipeline = MutationPipeline(pipeline_config, config_loader)
    results = pipeline.run_all(
        inputs,
        existing_results=existing_results,
        completed_instances=len(completed_instance_ids),
    )

    legacy_output_path = _sync_legacy_output(run_dir)

    _update_end_time(run_dir)

    logger.info(f"Pipeline complete: {len(results)} mutations survived all filters")
    logger.info(f"Synced combined output to {legacy_output_path}")


if __name__ == "__main__":
    main()
