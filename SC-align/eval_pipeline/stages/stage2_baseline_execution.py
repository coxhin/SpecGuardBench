"""Stage 2: baseline execution."""

import json
import logging
import io
import re
import shlex
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2

from ..adapters.base import EvalBaselineAdapter
from ..config import AGENT_MODEL_ENV_VAR, BaselineConfig, EvalPipelineConfig
from ..egress import ApiOnlyEgress, hosts_from_env
from ..fuzz_compliance import Violation, validate_fuzz_compliance
from ..models import (
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    BaselineResult,
    EvalTask,
    LLMCallError,
    SpecCertificate,
)
from ..task_state import prepare_task_code_state
from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

logger = logging.getLogger(__name__)


_FORM1B_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "form1b_form6_harness_guide.md"
)
_HYPOFUZZ_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "hypofuzz_harness_guide.md"
)
_SAVED_INPUTS_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "saved_inputs_harness_guide.md"
)
_FACTORY_ORACLE_GUIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "factory_oracle_harness_guide.md"
)
_HARNESS_GUIDE_PATH = _FORM1B_GUIDE_PATH
_SAVED_INPUTS_CONTAINER_DIR = "/tmp/eval_saved_inputs"


def _artifact_safe_name(value: str) -> str:
    """Make instance/spec ids safe for artifact file names."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown")
    return safe[:180] or "unknown"


def _validate_saved_inputs_jsonl(text: str) -> Optional[str]:
    """Return an error string if saved-input JSONL is invalid."""
    count = 0
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            return f"line {lineno} is not valid JSON: {e.msg}"
        if isinstance(record, dict) and "why" in record:
            return "saved-input records must not contain a 'why' field"
        count += 1
    if count == 0:
        return "saved-input JSONL is empty"
    return None


def _read_container_file_text(container, container_path: str) -> str:
    """Read a text file from a Docker container."""
    try:
        bits, _stat = container.get_archive(container_path)
        archive = b"".join(bits)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if not members:
                raise FileNotFoundError(container_path)
            extracted = tar.extractfile(members[0])
            if extracted is None:
                raise FileNotFoundError(container_path)
            return extracted.read().decode(UTF8, errors="replace")
    except Exception:
        exit_code, output = exec_run_with_timeout(
            container,
            f"cat {shlex.quote(container_path)}",
            timeout=30,
        )
        if exit_code != 0:
            raise RuntimeError(output.decode(UTF8, errors="replace")[:500])
        return output.decode(UTF8, errors="replace")


def _single_spec_task(task: EvalTask, spec_id: str) -> EvalTask:
    """Clone ``task`` with only one surviving spec for per-atomic launches."""
    matching = [s for s in task.surviving_specs if s.spec_id == spec_id]
    if not matching:
        raise ValueError(f"spec_id {spec_id!r} not found in {task.instance_id}")
    return EvalTask(
        instance_id=task.instance_id,
        image_name=task.image_name,
        surviving_specs=matching,
        f2p_tests=task.f2p_tests,
        p2p_tests=task.p2p_tests,
        repo_settings=task.repo_settings,
        mask_patch=task.mask_patch,
        gold_patch=task.gold_patch,
        test_patch=task.test_patch,
        gold_patch_functions=task.gold_patch_functions,
        schema_version=task.schema_version,
    )


def _create_adapter(
    baseline_cfg: BaselineConfig,
    agent_env_vars: Dict[str, str],
    max_steps: Optional[int] = None,
) -> EvalBaselineAdapter:
    """Factory: instantiate the right adapter for a baseline config."""
    from ..adapters.claude_code import ClaudeCodeAdapter
    from ..adapters.codex import CodexAdapter
    from ..adapters.gemini_cli import GeminiCLIAdapter
    from ..adapters.opencode import OpenCodeAdapter
    from ..adapters.openhands import OpenHandsAdapter

    adapter_map = {
        "claude_code": ClaudeCodeAdapter,
        "codex": CodexAdapter,
        "gemini_cli": GeminiCLIAdapter,
        "opencode": OpenCodeAdapter,
        "openhands": OpenHandsAdapter,
    }
    cls = adapter_map.get(baseline_cfg.adapter)
    if cls is None:
        raise ValueError(
            f"Unknown adapter '{baseline_cfg.adapter}' for baseline "
            f"'{baseline_cfg.name}'; known: {', '.join(sorted(adapter_map))}"
        )
    return cls(
        name=baseline_cfg.name,
        model=baseline_cfg.model,
        tool_version=baseline_cfg.tool_version,
        agent_config={
            "price_input_per_mtok": baseline_cfg.price_input_per_mtok,
            "price_output_per_mtok": baseline_cfg.price_output_per_mtok,
            "price_cache_read_per_mtok": baseline_cfg.price_cache_read_per_mtok,
            "price_cache_write_per_mtok": baseline_cfg.price_cache_write_per_mtok,
        },
        env_vars=agent_env_vars,
        max_steps=max_steps,
    )


def _never_ran(result: BaselineResult) -> bool:
    """True for a result the adapter wrote without ever invoking the agent."""
    return not result.attempts and result.raw_output.startswith("ERROR:")


def _render_prompt(
    prompt_env: jinja2.Environment,
    task: EvalTask,
    harness_form: str = "form1b_form6",
    previous_violations: Optional[List[Dict[str, Any]]] = None,
    max_agent_steps: Optional[int] = None,
    reference_state: str = "gold",
) -> str:
    """Render the baseline task prompt from the Jinja2 template."""
    if harness_form == "hypofuzz":
        template = prompt_env.get_template("baseline_hypofuzz_task.j2")
    elif harness_form == "saved_inputs":
        template = prompt_env.get_template("baseline_saved_inputs_task.j2")
    elif harness_form == "factory_oracle":
        if len(task.surviving_specs) != 1:
            raise ValueError(
                "factory_oracle prompt requires exactly one surviving spec; "
                f"got {len(task.surviving_specs)} for {task.instance_id}"
            )
        template = prompt_env.get_template("baseline_factory_oracle_task.j2")
        surviving = task.surviving_specs[0]
        return template.render(
            spec={
                **surviving.spec,
                "case_contract": surviving.case_contract,
            },
            previous_violations=previous_violations or [],
            max_agent_steps=max_agent_steps,
            reference_is_mutant=reference_state == "mutant",
        )
    else:
        template = prompt_env.get_template("baseline_task.j2")
    return template.render(
        specs=[
            {
                **surviving.spec,
                "case_contract": surviving.case_contract,
            }
            for surviving in task.surviving_specs
        ],
        previous_violations=previous_violations or [],
        max_agent_steps=max_agent_steps,
    )


class BaselineExecutor:
    """Runs baselines on eval tasks (one invocation per instance × baseline)."""

    def __init__(
        self,
        config: EvalPipelineConfig,
        config_loader,
        docker_runner,
    ):
        self.config = config
        self.config_loader = config_loader
        self.docker_runner = docker_runner
        self._egress: Optional[ApiOnlyEgress] = None
        self._pending: Dict[str, BaselineResult] = {}
        self._kept: List[Any] = []
        self._kept_lock = threading.Lock()

        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.prompt_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        if self.config.harness_form == "hypofuzz":
            guide_path = _HYPOFUZZ_GUIDE_PATH
            self._harness_guide_container_path = "/testbed/hypofuzz_harness_guide.md"
        elif self.config.harness_form == "saved_inputs":
            guide_path = _SAVED_INPUTS_GUIDE_PATH
            self._harness_guide_container_path = "/testbed/saved_inputs_harness_guide.md"
        elif self.config.harness_form == "factory_oracle":
            guide_path = _FACTORY_ORACLE_GUIDE_PATH
            self._harness_guide_container_path = "/testbed/factory_oracle_harness_guide.md"
        else:
            guide_path = _FORM1B_GUIDE_PATH
            self._harness_guide_container_path = "/testbed/form1b_form6_harness_guide.md"

        if not guide_path.exists():
            logger.warning(
                "Harness guide not found at %s — baselines will not have local "
                "access to the reference document.",
                guide_path,
            )
            self._harness_guide_text = ""
        else:
            self._harness_guide_text = guide_path.read_text(encoding="utf-8")

    def run_all(
        self,
        tasks: List[EvalTask],
        baselines: List[BaselineConfig],
    ) -> List[BaselineResult]:
        """Run all baselines on all tasks concurrently."""
        all_results: List[BaselineResult] = []

        effective_tasks = self._expand_tasks_for_harness(tasks)
        work_items = [(task, bcfg) for task in effective_tasks for bcfg in baselines]
        n_workers = max(1, self.config.n_concurrent)
        logger.info(
            f"Stage 2: {len(work_items)} work items "
            f"({len(effective_tasks)} effective tasks x {len(baselines)} baselines), "
            f"n_concurrent={n_workers}"
        )

        to_run: List[tuple] = []
        for task, bcfg in work_items:
            result_path = self._result_path(task, bcfg)
            if self.config.skip_existing and result_path.exists():
                result = BaselineResult.from_dict(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                if _never_ran(result):
                    logger.info(
                        f"  Re-running (cached result never ran the agent: "
                        f"{result.raw_output[:80]!r}): {result_path}"
                    )
                    to_run.append((task, bcfg))
                    continue
                if result.provider_pending is not None:
                    pending = result.provider_pending
                    logger.info(
                        f"  Resuming (lost to provider: "
                        f"{', '.join(pending.get('error_types') or [])}; "
                        f"container {pending.get('container_name') or '<none>'}): "
                        f"{result_path}"
                    )
                    self._pending[str(result_path)] = result
                    to_run.append((task, bcfg))
                    continue
                logger.info(f"  Skipping (result exists): {result_path}")
                if self.config.harness_form == "factory_oracle":
                    result = self._finalize_factory_oracle_result(task, result)
                all_results.append(result)
            else:
                to_run.append((task, bcfg))

        if not to_run:
            logger.info(f"Stage 2: all {len(all_results)} results cached, nothing to run")
            self._write_provider_pending()
            return all_results

        logger.info(f"Stage 2: {len(to_run)} items to run, {len(all_results)} cached")

        total = len(to_run)
        counter_lock = threading.Lock()
        started_count = [0]
        completed_count = [0]

        def _run_item(task_bcfg):
            task, bcfg = task_bcfg
            with counter_lock:
                started_count[0] += 1
                idx = started_count[0]
            logger.info(
                f"[started {idx}/{total}] "
                f"{self._task_label(task)} with {bcfg.name}/{bcfg.model}"
            )
            t0 = time.time()
            result = self._run_single(task, bcfg)
            self._save_result(result, self._result_path(task, bcfg))
            elapsed = time.time() - t0
            with counter_lock:
                completed_count[0] += 1
                done = completed_count[0]
            logger.info(
                f"[done {done}/{total}] "
                f"{self._task_label(task)} with {bcfg.name}/{bcfg.model} "
                f"({elapsed:.1f}s)"
            )
            return result

        self._egress = self._start_egress(baselines)
        try:
            self._run_pool(to_run, all_results, n_workers, _run_item, counter_lock,
                           completed_count, total)
        finally:
            with self._kept_lock:
                kept = list(self._kept)
            if self._egress is not None:
                for container in kept:
                    self._egress.release(container)
                self._egress.stop(log_path=self.config.output_dir / "egress_proxy.log")
                self._egress = None
            self._write_provider_pending()

        logger.info(f"Stage 2: completed {len(all_results)} baseline runs")
        return all_results

    def _start_egress(self, baselines: List[BaselineConfig]) -> Optional[ApiOnlyEgress]:
        """The run's egress sidecar under api_only, else None."""
        if self.config.egress != "api_only":
            return None
        hosts = set(self.config.egress_allow_hosts)
        for bcfg in baselines:
            hosts |= hosts_from_env(
                self.config_loader.get_agent_env_vars(bcfg.agent_config)
            )
        egress = ApiOnlyEgress(
            hosts,
            run_id=f"{os.getpid()}-{int(time.time())}",
            proxy_image=self.config.egress_proxy_image,
        )
        egress.start()
        return egress

    @staticmethod
    def _run_pool(to_run, all_results, n_workers, _run_item, counter_lock,
                  completed_count, total) -> None:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_item, item): item for item in to_run
            }
            for future in as_completed(futures):
                task, bcfg = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    with counter_lock:
                        completed_count[0] += 1
                        done = completed_count[0]
                    logger.error(
                        f"[failed {done}/{total}] "
                        f"{task.instance_id} with {bcfg.name}/{bcfg.model}: {e}",
                        exc_info=True,
                    )

    def _expand_tasks_for_harness(self, tasks: List[EvalTask]) -> List[EvalTask]:
        """Per-spec forms launch one baseline per atomic spec."""
        if self.config.harness_form not in {"saved_inputs", "factory_oracle"}:
            return tasks
        subset = getattr(self.config, "spec_subset", None)
        wanted = set(map(tuple, subset)) if subset is not None else None
        matched: set = set()
        expanded: List[EvalTask] = []
        for task in tasks:
            for surviving_spec in task.surviving_specs:
                key = (task.instance_id, surviving_spec.spec_id)
                if wanted is not None and key not in wanted:
                    continue
                matched.add(key)
                expanded.append(_single_spec_task(task, surviving_spec.spec_id))
        if wanted is not None:
            missing = sorted(wanted - matched)
            for iid, sid in missing:
                logger.warning(
                    "spec subset names %s/%s but the mutation output has no "
                    "such spec (or the instance was filtered out); skipping",
                    iid, sid,
                )
            logger.info(
                "Stage 2: spec subset launches %d of %d specs (%d listed, %d unmatched)",
                len(expanded),
                sum(len(t.surviving_specs) for t in tasks),
                len(wanted), len(missing),
            )
            if not expanded:
                raise ValueError("spec subset matched no spec in the mutation output")
        return expanded

    def _task_label(self, task: EvalTask) -> str:
        if len(task.surviving_specs) == 1:
            return f"{task.instance_id}/{task.surviving_specs[0].spec_id}"
        return task.instance_id

    def _run_single(
        self, task: EvalTask, bcfg: BaselineConfig
    ) -> BaselineResult:
        """Run one baseline on one task, in the configured reference code state."""
        agent_env = self.config_loader.get_agent_env_vars(bcfg.agent_config)
        model_env_var = AGENT_MODEL_ENV_VAR.get(bcfg.adapter)
        if model_env_var and bcfg.model:
            agent_env = {**agent_env, model_env_var: bcfg.model}
        adapter = _create_adapter(
            bcfg, agent_env, max_steps=self.config.max_agent_steps
        )
        adapter.egress = self._egress

        prompt_text = _render_prompt(
            self.prompt_env, task,
            harness_form=self.config.harness_form,
            max_agent_steps=adapter.enforced_max_steps(),
            reference_state=self.config.stage2_reference_state,
        )

        from mutation_pipeline.models import PipelineInput

        fake_input = PipelineInput(
            instance_id=task.instance_id,
            level=1,
            task_statement="",
            interface_descriptions=[],
            gold_patch="",
            image_name=task.image_name,
        )

        pending = self._pending.pop(str(self._result_path(task, bcfg)), None)
        resume_from: Optional[Dict[str, Any]] = None

        container = None
        kept = False
        try:
            if pending is not None:
                container = self._adopt_kept_container(task, bcfg, pending)
                if container is not None:
                    resume_from = {
                        **(pending.provider_pending or {}),
                        "attempts": pending.attempts,
                    }
            if container is None:
                container = self.docker_runner.setup_instance(fake_input)
                reference_is_mutant = self.config.stage2_reference_state == "mutant"
                reference_spec = task.surviving_specs[0] if reference_is_mutant else None
                ok, err = prepare_task_code_state(
                    container=container,
                    docker_runner=self.docker_runner,
                    task=task,
                    target_state=(
                        TARGET_STATE_MUTANT if reference_is_mutant else TARGET_STATE_GOLD
                    ),
                    surviving_spec=reference_spec,
                    restore_test_with_reverse_patch=self.config.restore_test_with_reverse_patch,
                    remove_pristine_copy=reference_is_mutant,
                )
                if not ok:
                    logger.warning(
                        "[%s][%s/%s] task state setup failed: %s",
                        task.instance_id,
                        bcfg.name,
                        bcfg.model,
                        err,
                    )
                    return adapter._error_result(task, f"Task state setup failed: {err}", 0.0)

                if self._harness_guide_text:
                    adapter._write_file_to_container(
                        container,
                        self._harness_guide_text,
                        self._harness_guide_container_path,
                    )

            result = adapter.run(
                container, task, prompt_text, self.config.timeout,
                resume_from=resume_from,
            )

            if result.provider_pending is not None:
                result.provider_pending.update({
                    "container_id": container.id,
                    "container_name": getattr(container, "name", None),
                    "spec_id": (
                        task.surviving_specs[0].spec_id
                        if len(task.surviving_specs) == 1 else None
                    ),
                })
                kept = True
                with self._kept_lock:
                    self._kept.append(container)
                logger.warning(
                    "[%s][%s/%s] lost to provider; keeping container %s for resume",
                    self._task_label(task), bcfg.name, bcfg.model,
                    result.provider_pending["container_name"] or container.short_id,
                )

            if self.config.harness_form == "saved_inputs":
                result = self._finalize_saved_inputs_result(
                    container, task, bcfg, result,
                )
            elif self.config.harness_form == "factory_oracle":
                result = self._finalize_factory_oracle_result(task, result)

            if self.config.harness_form == "hypofuzz":
                result = self._lint_and_retry(
                    adapter, container, task, result,
                )
            return result
        finally:
            if container and not kept:
                self.docker_runner.cleanup(container)

    def _adopt_kept_container(
        self, task: EvalTask, bcfg: BaselineConfig, pending: BaselineResult
    ):
        """The container ``pending`` was kept in, ready to run in; or None."""
        record = pending.provider_pending or {}
        container_id = record.get("container_id")
        label = f"[{self._task_label(task)}][{bcfg.name}/{bcfg.model}]"
        if not container_id:
            logger.warning("%s pending result names no container; starting fresh", label)
            return None
        container = self.docker_runner.get_container(container_id)
        if container is None:
            logger.warning(
                "%s kept container %s is gone; starting fresh (session %s cannot "
                "be continued)", label, record.get("container_name") or container_id,
                record.get("session_id") or "<none>",
            )
            return None
        if self._egress is not None:
            self._egress.restrict(container)
        logger.info(
            "%s resuming in kept container %s (session %s, %s model calls spent)",
            label, getattr(container, "name", None) or container.short_id,
            record.get("session_id") or "<none>", record.get("chain_steps") or 0,
        )
        return container

    def _write_provider_pending(self) -> None:
        """Index every result on disk that is still waiting on the provider."""
        entries: List[Dict[str, Any]] = []
        baseline_dir = self.config.output_dir / "baseline_output"
        if baseline_dir.exists():
            for subdir in sorted(baseline_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for json_file in sorted(subdir.glob("*.json")):
                    try:
                        data = json.loads(json_file.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as e:
                        logger.warning("could not read %s: %s", json_file, e)
                        continue
                    record = data.get("provider_pending")
                    if not record:
                        continue
                    entries.append({
                        "instance_id": data.get("instance_id"),
                        "spec_id": record.get("spec_id"),
                        "baseline": data.get("baseline"),
                        "model": data.get("model"),
                        "result_path": str(json_file.relative_to(self.config.output_dir)),
                        "error_types": record.get("error_types") or [],
                        "session_id": record.get("session_id"),
                        "chain_steps": record.get("chain_steps"),
                        "attempts": len(data.get("attempts") or []),
                        "container_id": record.get("container_id"),
                        "container_name": record.get("container_name"),
                    })
        out_path = baseline_dir / "provider_pending.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "n_pending": len(entries),
                "pending": entries,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if entries:
            logger.warning(
                "Stage 2: %d result(s) lost to the provider and left to resume; "
                "see %s. Their containers are kept; rerun with --skip-existing "
                "to continue them.", len(entries), out_path,
            )
        else:
            logger.info("Stage 2: nothing left to resume (%s)", out_path)

    def _lint_and_retry(
        self,
        adapter: EvalBaselineAdapter,
        container,
        task: EvalTask,
        result: BaselineResult,
    ) -> BaselineResult:
        """In hypofuzz mode, run the AST linter on every cert and retry once."""
        compliance: Dict[str, Any] = {}
        first_violations_all: List[Dict[str, Any]] = []

        for spec_id, cert in result.certificates.items():
            viols = validate_fuzz_compliance(cert.code)
            compliance[spec_id] = {
                "first_try_compliant": not viols,
                "first_violations": [v.to_dict() for v in viols],
                "final_compliant": not viols,
                "final_violations": [v.to_dict() for v in viols],
            }
            if viols:
                for v in viols:
                    logger.warning(
                        "[%s][%s/%s] fuzz-compliance violation in cert %s: "
                        "%s — %s",
                        task.instance_id, adapter.name, adapter.model,
                        spec_id, v.rule_id, v.message,
                    )
                for v in viols:
                    d = v.to_dict()
                    d["spec_id"] = spec_id
                    first_violations_all.append(d)

        if not first_violations_all or not result.certificates:
            result.fuzz_compliance = compliance
            return result

        logger.info(
            "[%s][%s/%s] %d fuzz-compliance violations on first try; "
            "running one retry pass",
            task.instance_id, adapter.name, adapter.model,
            len(first_violations_all),
        )

        retry_prompt = _render_prompt(
            self.prompt_env, task,
            harness_form="hypofuzz",
            previous_violations=first_violations_all,
            max_agent_steps=adapter.enforced_max_steps(),
        )

        retry_result = adapter.run(
            container, task, retry_prompt, self.config.timeout,
        )

        merged_certs = dict(retry_result.certificates)
        for spec_id, cert in merged_certs.items():
            viols = validate_fuzz_compliance(cert.code)
            entry = compliance.get(spec_id, {
                "first_try_compliant": True,
                "first_violations": [],
            })
            entry["final_compliant"] = not viols
            entry["final_violations"] = [v.to_dict() for v in viols]
            compliance[spec_id] = entry

        for spec_id in list(compliance.keys()):
            if spec_id not in merged_certs:
                compliance[spec_id]["final_compliant"] = False
                compliance[spec_id]["final_violations"] = [{
                    "rule_id": "R1_has_given",
                    "test_func": None,
                    "message": "retry pass did not emit a certificate for this spec",
                }]

        result.certificates = merged_certs
        result.raw_output = (
            result.raw_output
            + "\n\n---RETRY PASS---\n"
            + retry_result.raw_output
        )
        offset = len(result.attempts)
        for record in retry_result.attempts:
            record = dict(record)
            record["attempt"] += offset
            record["lint_retry_pass"] = True
            result.attempts.append(record)
        for k in ("input_tokens", "output_tokens", "total_cost_usd"):
            result.cost[k] = result.cost.get(k, 0) + retry_result.cost.get(k, 0)
        result.wall_time_seconds += retry_result.wall_time_seconds
        result.llm_errors.extend(retry_result.llm_errors)
        result.completed_normally = (
            result.completed_normally and retry_result.completed_normally
        )
        result.fuzz_compliance = compliance
        return result

    def _finalize_saved_inputs_result(
        self,
        container,
        task: EvalTask,
        bcfg: BaselineConfig,
        result: BaselineResult,
    ) -> BaselineResult:
        """Normalize saved-input certificates after a per-atomic-spec launch."""
        expected_ids = {s.spec_id for s in task.surviving_specs}
        result.certificates = {
            sid: cert for sid, cert in result.certificates.items()
            if sid in expected_ids
        }

        for spec_id, cert in list(result.certificates.items()):
            if cert.inputs_file:
                self._extract_saved_inputs_file(container, task, bcfg, spec_id, cert)
        return result

    def _finalize_factory_oracle_result(
        self,
        task: EvalTask,
        result: BaselineResult,
    ) -> BaselineResult:
        """Enforce one correctly keyed certificate per atomic invocation."""
        if len(task.surviving_specs) != 1:
            raise ValueError(
                "factory_oracle result finalization requires exactly one "
                f"surviving spec; got {len(task.surviving_specs)} for "
                f"{task.instance_id}"
            )

        expected_spec_id = task.surviving_specs[0].spec_id
        actual_spec_ids = set(result.certificates)
        if actual_spec_ids == {expected_spec_id}:
            return result
        if not actual_spec_ids and result.provider_pending is not None:
            return result

        actual_text = ", ".join(sorted(actual_spec_ids)) or "<none>"
        message = (
            "factory_oracle must return exactly one certificate keyed by "
            f"{expected_spec_id!r}; received: {actual_text}"
        )
        logger.warning("[%s] %s", task.instance_id, message)
        result.certificates = {}
        result.llm_errors.append(LLMCallError(
            error_type="invalid_certificate_set",
            count=1,
            sample_message=message,
        ))
        result.completed_normally = False
        return result

    def _extract_saved_inputs_file(
        self,
        container,
        task: EvalTask,
        bcfg: BaselineConfig,
        spec_id: str,
        cert: SpecCertificate,
    ) -> None:
        raw_path = cert.inputs_file.strip()
        if not raw_path.startswith(_SAVED_INPUTS_CONTAINER_DIR + "/"):
            logger.warning(
                "[%s][%s/%s][%s] ignoring inputs_file outside %s: %s",
                task.instance_id, bcfg.name, bcfg.model, spec_id,
                _SAVED_INPUTS_CONTAINER_DIR, raw_path,
            )
            return

        try:
            text = _read_container_file_text(container, raw_path)
        except Exception as e:
            logger.warning(
                "[%s][%s/%s][%s] failed to read saved-input file %s: %s",
                task.instance_id, bcfg.name, bcfg.model, spec_id, raw_path,
                str(e)[:500],
            )
            return

        validation_error = _validate_saved_inputs_jsonl(text)
        if validation_error:
            logger.warning(
                "[%s][%s/%s][%s] invalid saved-input JSONL %s: %s",
                task.instance_id, bcfg.name, bcfg.model, spec_id, raw_path,
                validation_error,
            )
            return

        rel_path = (
            Path("baseline_saved_inputs")
            / f"{bcfg.name}_{bcfg.model}"
            / _artifact_safe_name(task.instance_id)
            / f"{_artifact_safe_name(spec_id)}.jsonl"
        )
        out_path = self.config.output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        cert.inputs_file = rel_path.as_posix()
        cert.saved_inputs = []

    def _result_path(self, task: EvalTask, bcfg: BaselineConfig) -> Path:
        subdir = self.config.output_dir / "baseline_output" / f"{bcfg.name}_{bcfg.model}"
        subdir.mkdir(parents=True, exist_ok=True)
        stem = _artifact_safe_name(task.instance_id)
        if self.config.harness_form in {"saved_inputs", "factory_oracle"} and len(task.surviving_specs) == 1:
            stem = f"{stem}_{_artifact_safe_name(task.surviving_specs[0].spec_id)}"
        return subdir / f"{stem}.json"

    def _save_result(self, result: BaselineResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def load_baseline_results(output_dir: Path) -> List[BaselineResult]:
    """Load all baseline results from disk."""
    results = []
    baseline_dir = output_dir / "baseline_output"
    if not baseline_dir.exists():
        return results
    for subdir in sorted(baseline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                results.append(BaselineResult.from_dict(json.loads(f.read())))
    logger.info(f"Loaded {len(results)} baseline results from {baseline_dir}")
    return results
