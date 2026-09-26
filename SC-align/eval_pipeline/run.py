#!/usr/bin/env python3
"""CLI entry point for the Evaluation Pipeline."""

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SC_ALIGN = Path(__file__).resolve().parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from featurebench.infer.config import InferConfigLoader
from eval_pipeline.config import (
    BaselineConfig,
    load_eval_config_from_toml,
    EvalPipelineConfig,
)
from eval_pipeline.pipeline import EvalPipeline
from run_naming import run_dir_name

_LATEST_RUN_FILE = "latest_run.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Pipeline for Spec-Code Alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help=(
            "Which baseline to evaluate, by its [eval_pipeline.baseline.<name>] "
            "name in config_eval.toml. Required for a fresh run; on --resume it "
            "defaults to the resumed run's baseline, and naming a different one "
            "adds that baseline to the same run directory."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Override the baseline's model for this run. Also becomes the model "
            "env var handed to the agent CLI, so results cannot be labelled with "
            "a model other than the one that ran."
        ),
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help=(
            "Short note appended to the run directory name, e.g. "
            "'lite_set' -> runs/2026-08-31__14-30-00__lite_set"
        ),
    )
    parser.add_argument(
        "--config-path", type=str, default=None,
        help="Path to config_eval.toml",
    )
    parser.add_argument(
        "--mutation-output", type=str, default=None,
        help=(
            "Path to the mutation output.jsonl to evaluate, e.g. "
            "SC-align/mutation_output/runs/<run>/output.jsonl. Left unset it "
            "falls back to the SC-align/mutation_output/output.jsonl alias, "
            "which every finishing mutation run overwrites and so names no "
            "particular run -- point it at a run's own file"
        ),
    )
    parser.add_argument(
        "--harness-form", type=str, default=None,
        choices=["form1b_form6", "hypofuzz", "saved_inputs", "factory_oracle"],
        help="Which harness form the baseline writes certificates in",
    )
    parser.add_argument(
        "--stage2-reference-state", type=str, default=None,
        choices=["gold", "mutant"],
        help=(
            "Which implementation Stage 2 shows the baseline while it writes "
            "its certificate. 'mutant' shows the certified spec's own mutant, "
            "turning the run into a probe of whether the certificate encodes "
            "the specification or the code on disk; scoring is unchanged, so "
            "gold_firing_rate becomes the measurement. Requires "
            "--harness-form saved_inputs or factory_oracle"
        ),
    )
    parser.add_argument(
        "--egress", type=str, default=None, choices=["open", "api_only"],
        help=(
            "What the baseline container may reach while the agent runs. "
            "'api_only' installs the tool on the open network, then moves the "
            "container onto an internal network whose only exit is a proxy "
            "allowing the agent environment's *_BASE_URL hosts"
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help=(
            "Override the per-baseline Docker run timeout (seconds). "
            "0 removes the wall clock, which is also the default when neither "
            "this nor config_eval.toml sets one"
        ),
    )
    parser.add_argument(
        "--max-agent-steps", type=int, default=None,
        help=(
            "Stop each baseline invocation after this many model calls. "
            "0 removes the cap. Counted the same way for every baseline "
            "(see resources/step_gate.py)"
        ),
    )
    parser.add_argument(
        "--stages", type=int, nargs="+", default=None,
        help="Stages to run (e.g., --stages 1 2 3). Default: all",
    )
    parser.add_argument(
        "--instance-ids", type=str, nargs="+", default=None,
        help="Specific instance IDs to process",
    )
    parser.add_argument(
        "--spec-subset", type=str, default=None,
        help=(
            "Path to a file naming which specifications Stage 2 launches a "
            "baseline for: one 'instance_id<TAB>spec_id' per line, '#' "
            "comments allowed. The whole mutation output is still loaded so "
            "each sampled certificate keeps its full control arm. Requires "
            "a per-spec harness form (saved_inputs or factory_oracle)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Override the base output directory. The run still lands in "
            "<base>/runs/<timestamp>[__<tag>]/ underneath it."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip cases that already have artifacts from a previous run",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=None,
        help="Override the number of work items executed concurrently",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Resume from a previous run directory "
            "(e.g., SC-align/eval_output/runs/2026-04-12__10-30-00), or "
            "'latest' for the most recent run under the base output directory. "
            "Most args are loaded from run_metadata.json; --stages, "
            "--skip-existing, --n-concurrent, --timeout, --max-agent-steps "
            "and --baseline (with --model) can be overridden. --harness-form "
            "and --stage2-reference-state cannot: the existing certificates "
            "were written under the original ones."
        ),
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    fh = logging.FileHandler(output_dir / "pipeline.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root.addHandler(console)
    root.addHandler(fh)

    for lib in ["httpx", "openai", "urllib3", "docker", "httpcore"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def _baseline_to_dict(b: BaselineConfig) -> Dict[str, object]:
    """Serialize a baseline for run_metadata.json."""
    return {
        "name": b.name,
        "model": b.model,
        "agent_config": b.agent_config,
        "adapter": b.adapter,
        "tool_version": b.tool_version,
        "price_input_per_mtok": b.price_input_per_mtok,
        "price_output_per_mtok": b.price_output_per_mtok,
        "price_cache_read_per_mtok": b.price_cache_read_per_mtok,
        "price_cache_write_per_mtok": b.price_cache_write_per_mtok,
    }


def _apply_budget_overrides(config: EvalPipelineConfig, args) -> None:
    """Let the command line set or remove each per-invocation budget."""
    if args.timeout is not None:
        config.timeout = args.timeout or None
    if args.max_agent_steps is not None:
        config.max_agent_steps = args.max_agent_steps or None


def _save_run_metadata(output_dir: Path, config: EvalPipelineConfig) -> None:
    """Save run metadata to run_metadata.json."""
    metadata = {
        "output_dir": str(output_dir),
        "mutation_output": str(config.mutation_output),
        "n_concurrent": config.n_concurrent,
        "timeout": config.timeout,
        "max_agent_steps": config.max_agent_steps,
        "hypothesis_timeout": config.hypothesis_timeout,
        "restore_test_with_reverse_patch": config.restore_test_with_reverse_patch,
        "instance_ids": config.instance_ids,
        "baselines": [_baseline_to_dict(b) for b in config.baselines],
        "proxy_port": config.proxy_port,
        "gpu_ids": config.gpu_ids,
        "dataset": config.dataset,
        "split": config.split,
        "stages": config.stages,
        "skip_existing": config.skip_existing,
        "harness_form": config.harness_form,
        "hypofuzz_timeout_seconds": config.hypofuzz_timeout_seconds,
        "control_arm_k": config.control_arm_k,
        "saved_inputs_replay_timeout_seconds": (
            config.saved_inputs_replay_timeout_seconds
        ),
        "stage2_reference_state": config.stage2_reference_state,
        "egress": config.egress,
        "egress_allow_hosts": config.egress_allow_hosts,
        "spec_subset_path": (
            str(config.spec_subset_path) if config.spec_subset_path else None
        ),
        "spec_subset": (
            [list(pair) for pair in config.spec_subset]
            if config.spec_subset is not None else None
        ),
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


def _require_mutation_output(config: EvalPipelineConfig) -> None:
    """Refuse to start Stage 1 without a readable mutation output."""
    if config.stages is not None and 1 not in config.stages:
        return

    path = Path(config.mutation_output)
    if not path.exists():
        print(
            f"Error: mutation output not found: {path}\n"
            "Pass --mutation-output, or set MUTATION_OUTPUT in "
            "SC-align/scripts/run_eval.sh. To list candidates:\n"
            "  ls -t SC-align/mutation_output/runs/*/output.jsonl | head",
            file=sys.stderr,
        )
        sys.exit(1)
    if path.stat().st_size == 0:
        print(
            f"Error: mutation output is empty: {path}\n"
            "That run produced no surviving mutations, so there is nothing to "
            "evaluate. Pick a different run.",
            file=sys.stderr,
        )
        sys.exit(1)


PER_SPEC_HARNESS_FORMS = {"saved_inputs", "factory_oracle"}


def _read_spec_subset_or_exit(path: Path) -> List[Tuple[str, str]]:
    """Parse a ``--spec-subset`` file into (instance_id, spec_id) pairs."""
    if not path.exists():
        print(f"Error: --spec-subset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            print(
                f"Error: {path}:{lineno}: expected 'instance_id spec_id', "
                f"got {raw!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        pair = (parts[0], parts[1])
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    if not pairs:
        print(f"Error: --spec-subset file names no specs: {path}", file=sys.stderr)
        sys.exit(1)
    return pairs


def _require_per_spec_form_for_spec_subset(config: EvalPipelineConfig) -> None:
    """A spec subset only makes sense where one launch serves one spec."""
    if config.spec_subset is None:
        return
    if config.harness_form in PER_SPEC_HARNESS_FORMS:
        return
    print(
        "Error: --spec-subset requires a per-specification harness form, "
        f"but this run is {config.harness_form}. Use --harness-form "
        "saved_inputs or factory_oracle.",
        file=sys.stderr,
    )
    sys.exit(1)


def _require_per_spec_form_for_mutant_reference(config: EvalPipelineConfig) -> None:
    """Refuse a mutant reference state under a harness form that batches specs."""
    if config.stage2_reference_state != "mutant":
        return
    if config.harness_form in {"saved_inputs", "factory_oracle"}:
        return
    print(
        "Error: --stage2-reference-state mutant requires a per-specification "
        f"harness form, but this run is {config.harness_form}. Use "
        "--harness-form saved_inputs or factory_oracle.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_latest_run_dir(base_output_dir: Path) -> Optional[Path]:
    """Find the most recent run directory holding Stage 1 output."""
    if (base_output_dir / "eval_tasks.jsonl").exists():
        return base_output_dir

    pointer_path = base_output_dir / _LATEST_RUN_FILE
    if pointer_path.exists():
        try:
            pointed_dir = Path(pointer_path.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            pointed_dir = None
        if pointed_dir and (pointed_dir / "eval_tasks.jsonl").exists():
            return pointed_dir

    runs_dir = base_output_dir / "runs"
    if runs_dir.exists():
        for candidate in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
            if (candidate / "eval_tasks.jsonl").exists():
                return candidate

    return None


def _merge_resumed_baselines(run_dir: Path, config: EvalPipelineConfig) -> None:
    """Add this invocation's baseline to the resumed run's recorded list."""
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    recorded = metadata.get("baselines", [])
    seen = {(b.get("name"), b.get("model")) for b in recorded}
    for b in config.baselines:
        if (b.name, b.model) not in seen:
            recorded.append(_baseline_to_dict(b))
    metadata["baselines"] = recorded

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _legacy_output_dir_for_run(run_dir: Path) -> Path:
    """Map a timestamped run directory back to the base eval output dir."""
    parent = run_dir.parent
    if parent.name == "runs":
        return parent.parent
    return run_dir


def _save_latest_run_pointer(run_dir: Path) -> Path:
    """Record the most recent run dir for helper scripts that expect a base output dir.
    """
    legacy_output_dir = _legacy_output_dir_for_run(run_dir)
    legacy_output_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = legacy_output_dir / _LATEST_RUN_FILE
    pointer_path.write_text(str(run_dir), encoding="utf-8")
    return pointer_path


def _load_baseline_catalog_or_exit(
    config_loader: InferConfigLoader,
) -> Dict[str, "BaselineConfig"]:
    """Read the baseline catalog, failing loudly if config_eval.toml declares none."""
    catalog = load_eval_config_from_toml(config_loader).baseline_catalog
    if not catalog:
        print(
            "Error: config_eval.toml declares no baselines. Expected at least one "
            "[eval_pipeline.baseline.<name>] table.",
            file=sys.stderr,
        )
        sys.exit(1)
    return catalog


def _select_baseline(
    catalog: Dict[str, "BaselineConfig"],
    name: str,
    model_override: Optional[str],
) -> "BaselineConfig":
    """Pick one baseline out of the catalog, applying the per-run model override."""
    if name not in catalog:
        known = ", ".join(sorted(catalog)) or "(none)"
        print(
            f"Error: no baseline named '{name}' in config_eval.toml. Known: {known}",
            file=sys.stderr,
        )
        sys.exit(1)

    selected = replace(catalog[name])
    if model_override:
        selected.model = model_override
    if not selected.model:
        print(
            f"Error: baseline '{name}' has no model. Set one in its "
            "[eval_pipeline.baseline] table or pass --model.",
            file=sys.stderr,
        )
        sys.exit(1)
    return selected


def _load_resume_config(
    resume_dir: Path,
    args: argparse.Namespace,
    config_loader: InferConfigLoader,
) -> EvalPipelineConfig:
    """Load pipeline config from a previous run's run_metadata.json."""
    metadata_path = resume_dir / "run_metadata.json"
    if not metadata_path.exists():
        print(f"Error: run_metadata.json not found in {resume_dir}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    baselines = [
        BaselineConfig(**b)
        for b in metadata.get("baselines", [])
    ]

    config = EvalPipelineConfig(
        output_dir=resume_dir,
        mutation_output=Path(metadata.get("mutation_output", "SC-align/mutation_output/output.jsonl")),
        n_concurrent=metadata.get("n_concurrent", 4),
        timeout=metadata.get("timeout"),
        max_agent_steps=metadata.get("max_agent_steps"),
        hypothesis_timeout=metadata.get("hypothesis_timeout", 120),
        restore_test_with_reverse_patch=metadata.get("restore_test_with_reverse_patch", False),
        instance_ids=metadata.get("instance_ids"),
        baselines=baselines,
        proxy_port=metadata.get("proxy_port"),
        gpu_ids=metadata.get("gpu_ids"),
        dataset=metadata.get("dataset", "LiberCoders/FeatureBench"),
        split=metadata.get("split", "full"),
        stages=metadata.get("stages"),
        skip_existing=metadata.get("skip_existing", False),
        harness_form=metadata.get("harness_form", "form1b_form6"),
        hypofuzz_timeout_seconds=metadata.get("hypofuzz_timeout_seconds", 60),
        control_arm_k=metadata.get("control_arm_k"),
        saved_inputs_replay_timeout_seconds=metadata.get(
            "saved_inputs_replay_timeout_seconds", 120
        ),
        stage2_reference_state=metadata.get("stage2_reference_state", "gold"),
        egress=metadata.get("egress", "open"),
        egress_allow_hosts=list(metadata.get("egress_allow_hosts", [])),
        spec_subset=(
            [tuple(pair) for pair in metadata["spec_subset"]]
            if metadata.get("spec_subset") is not None else None
        ),
        spec_subset_path=(
            Path(metadata["spec_subset_path"])
            if metadata.get("spec_subset_path") else None
        ),
    )

    if args.stages:
        config.stages = args.stages
    if args.skip_existing:
        config.skip_existing = True
    if args.n_concurrent is not None:
        config.n_concurrent = args.n_concurrent
    _apply_budget_overrides(config, args)

    if args.harness_form and args.harness_form != config.harness_form:
        print(
            f"Error: cannot resume {resume_dir} with --harness-form "
            f"{args.harness_form}; it ran as {config.harness_form} and its "
            "artifacts can only be read back under that form.",
            file=sys.stderr,
        )
        sys.exit(1)

    if (
        args.stage2_reference_state
        and args.stage2_reference_state != config.stage2_reference_state
    ):
        print(
            f"Error: cannot resume {resume_dir} with --stage2-reference-state "
            f"{args.stage2_reference_state}; its certificates were written "
            f"against the {config.stage2_reference_state} implementation.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.baseline:
        catalog = _load_baseline_catalog_or_exit(config_loader)
        config.baselines = [_select_baseline(catalog, args.baseline, args.model)]

    return config


def main() -> None:
    args = parse_args()

    config_path = Path(args.config_path).expanduser() if args.config_path else None
    config_loader = InferConfigLoader(config_path)

    if args.resume == "latest":
        base = Path(args.output_dir) if args.output_dir else Path(
            load_eval_config_from_toml(config_loader).output_dir
        )
        resume_dir = _resolve_latest_run_dir(base)
        if resume_dir is None:
            print(
                f"Error: no run with Stage 1 output found under {base}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        resume_dir = Path(args.resume) if args.resume else None

    if resume_dir:
        config = _load_resume_config(resume_dir, args, config_loader)
        run_dir = resume_dir
    else:
        if not args.baseline:
            print(
                "Error: --baseline is required. A run evaluates one baseline; "
                "pass its [eval_pipeline.baseline.<name>] name from config_eval.toml.",
                file=sys.stderr,
            )
            sys.exit(1)

        config = load_eval_config_from_toml(config_loader)
        config.baselines = [
            _select_baseline(
                _load_baseline_catalog_or_exit(config_loader),
                args.baseline,
                args.model,
            )
        ]

        if args.mutation_output:
            config.mutation_output = Path(args.mutation_output)
        if args.stages:
            config.stages = args.stages
        if args.instance_ids:
            config.instance_ids = args.instance_ids
        if args.skip_existing:
            config.skip_existing = True
        if args.n_concurrent is not None:
            config.n_concurrent = args.n_concurrent
        if args.harness_form:
            config.harness_form = args.harness_form
        if args.stage2_reference_state:
            config.stage2_reference_state = args.stage2_reference_state
        if args.egress:
            config.egress = args.egress
        if args.spec_subset:
            config.spec_subset_path = Path(args.spec_subset)
            config.spec_subset = _read_spec_subset_or_exit(config.spec_subset_path)
        _apply_budget_overrides(config, args)

        _require_mutation_output(config)
        _require_per_spec_form_for_mutant_reference(config)
        _require_per_spec_form_for_spec_subset(config)

        base_output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
        run_dir = base_output_dir / "runs" / run_dir_name(args.tag)
        config.output_dir = run_dir

    setup_logging(run_dir)
    logger = logging.getLogger(__name__)

    baseline_label = ", ".join(f"{b.name}/{b.model}" for b in config.baselines)
    if resume_dir:
        logger.info(f"Resuming evaluation pipeline from {resume_dir}")
        logger.info(f"Baseline for this invocation: {baseline_label}")
    else:
        logger.info(f"Starting new evaluation pipeline run at {run_dir}")
        logger.info(f"Baseline: {baseline_label} (harness_form={config.harness_form})")

    pointer_path = _save_latest_run_pointer(run_dir)
    logger.info(f"Recorded latest eval run at {pointer_path}")

    if resume_dir:
        _merge_resumed_baselines(run_dir, config)
    else:
        _save_run_metadata(run_dir, config)

    pipeline = EvalPipeline(config, config_loader)
    pipeline.run(config.stages)

    _update_end_time(run_dir)


if __name__ == "__main__":
    main()
