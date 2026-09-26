"""Stage 3: certificate execution."""

import ast
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..models import (
    KIND_ASSERTION,
    KIND_COLLECTION_ERROR,
    KIND_INFRA,
    KIND_INVALID_CASE,
    KIND_INVALID_CERTIFICATE,
    KIND_PASSED,
    KIND_TEST_EXCEPTION,
    KIND_TIMEOUT,
    TARGET_STATE_CONTROL,
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    BaselineResult,
    EvalTask,
    ProofExecResult,
    SpecCertificate,
    SurvivingSpec,
    classify_hypofuzz_output,
    classify_pytest_output,
)
from ..coverage_scoping import (
    FunctionInfo,
    scope_to_function,
)
from ..patch_utils import mutant_line_hit_rate, parse_mutant_lines
from ..task_state import prepare_task_code_state

logger = logging.getLogger(__name__)

_FACTORY_ORACLE_RESULT_MARKER = "__FACTORY_ORACLE_DRIVER_RESULT__"
_FACTORY_ORACLE_KINDS = {
    KIND_PASSED,
    KIND_ASSERTION,
    KIND_TEST_EXCEPTION,
    KIND_COLLECTION_ERROR,
    KIND_INVALID_CASE,
    KIND_INVALID_CERTIFICATE,
    KIND_INFRA,
}

_CALL_CHAIN_SCRIPT = (
    Path(__file__).parent.parent / "call_chain_resolver.py"
).read_text(encoding="utf-8")


def select_control_mutants(
    spec_id: str,
    specs: List[SurvivingSpec],
    control_arm_k: Optional[int] = None,
) -> List[SurvivingSpec]:
    """The mutants that act as negative controls for this spec's certificate."""
    scored: List[Tuple[int, int, str, SurvivingSpec]] = []
    scores_recorded = True
    for candidate in specs:
        if candidate.spec_id == spec_id or spec_id not in candidate.control_spec_ids:
            continue
        probe = next(
            (
                p for p in candidate.preservation_probes
                if str(p.get("spec_id")) == spec_id
            ),
            {},
        )
        if "shared_functions" not in probe:
            scores_recorded = False
        scored.append((
            -int(probe.get("shared_functions") or 0),
            -int(probe.get("shared_lines") or 0),
            candidate.spec_id,
            candidate,
        ))

    scored.sort(key=lambda entry: entry[:3])
    controls = [candidate for *_, candidate in scored]

    if control_arm_k and control_arm_k > 0 and len(controls) > control_arm_k:
        if scores_recorded:
            controls = controls[:control_arm_k]
        else:
            logger.warning(
                "spec %s has %d controls and control_arm_k=%d, but its probe "
                "records carry no entanglement score — scoring the full arm "
                "rather than truncating it in an arbitrary order",
                spec_id, len(controls), control_arm_k,
            )
    return controls


def _factory_oracle_result(output_text: str) -> Optional[Dict[str, Any]]:
    """Parse the structured factory-oracle result from runner output."""
    for line in reversed((output_text or "").splitlines()):
        if not line.startswith(_FACTORY_ORACLE_RESULT_MARKER):
            continue
        payload = line[len(_FACTORY_ORACLE_RESULT_MARKER):]
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8


def _extend_scope_in_container(
    container,
    declared: List[FunctionInfo],
    gold_funcs: List[FunctionInfo],
    testbed_root: str = "/testbed",
) -> Dict[str, Any]:
    """Resolve the call-chain scope inside the source container."""
    payload = json.dumps({
        "declared": [fi.as_dict() for fi in declared],
        "gold_funcs": [fi.as_dict() for fi in gold_funcs],
        "testbed_root": testbed_root,
    })
    write_cmd = f"cat > /tmp/_call_chain_input.json << 'PAYLOAD_EOF'\n{payload}\nPAYLOAD_EOF"
    exec_run_with_timeout(container, write_cmd, timeout=10)
    exec_run_with_timeout(
        container,
        "cat > /tmp/_call_chain.py << 'SCRIPT_EOF'\n"
        + _CALL_CHAIN_SCRIPT
        + "\nSCRIPT_EOF",
        timeout=10,
    )
    exit_code, output = exec_run_with_timeout(
        container,
        "python /tmp/_call_chain.py < /tmp/_call_chain_input.json",
        timeout=30,
    )
    if exit_code != 0:
        raise RuntimeError(
            f"In-container call-chain script failed (exit={exit_code}): "
            f"{output[:500]}"
        )
    result = json.loads(output)
    if not isinstance(result, dict) or not isinstance(result.get("scope"), list):
        raise RuntimeError("in-container call-chain resolver returned invalid JSON")
    return result


_FOLD_LINES_SCRIPT = r"""
import json, sys
sys.path.insert(0, "/testbed")
from eval_patch_utils import fold_lines_to_statements
req = json.load(sys.stdin)
out = {}
for rel_path, lines in req["mutant_lines"].items():
    try:
        with open(req["testbed_root"].rstrip("/") + "/" + rel_path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        out[rel_path] = {"error": str(e)}
        continue
    folded = fold_lines_to_statements(source, lines)
    out[rel_path] = {str(k): v for k, v in folded.items()}
print(json.dumps(out))
"""


def _fold_mutant_lines_in_container(
    container,
    mutant_lines: Dict[str, List[int]],
    testbed_root: str = "/testbed",
) -> Dict[str, Dict[int, Optional[int]]]:
    """Fold mutant lines to statement first lines, against the arm's source."""
    from .. import patch_utils

    helper_src = Path(patch_utils.__file__).read_text(encoding="utf-8")
    exec_run_with_timeout(
        container,
        f"cat > {testbed_root}/eval_patch_utils.py << 'HELPER_EOF'\n"
        + helper_src + "\nHELPER_EOF",
        timeout=10,
    )
    exec_run_with_timeout(
        container,
        "cat > /tmp/_fold_lines.py << 'SCRIPT_EOF'\n" + _FOLD_LINES_SCRIPT + "\nSCRIPT_EOF",
        timeout=10,
    )
    payload = json.dumps({"mutant_lines": mutant_lines, "testbed_root": testbed_root})
    exec_run_with_timeout(
        container,
        f"cat > /tmp/_fold_lines_input.json << 'PAYLOAD_EOF'\n{payload}\nPAYLOAD_EOF",
        timeout=10,
    )
    exit_code, output = exec_run_with_timeout(
        container,
        f"python /tmp/_fold_lines.py < /tmp/_fold_lines_input.json; rm -f {testbed_root}/eval_patch_utils.py",
        timeout=30,
    )
    if exit_code != 0:
        raise RuntimeError(
            f"In-container line folding failed (exit={exit_code}): {output[:500]}"
        )
    raw = json.loads(output)
    anchors: Dict[str, Dict[int, Optional[int]]] = {}
    for rel_path, folded in raw.items():
        if "error" in folded:
            raise RuntimeError(f"cannot fold {rel_path}: {folded['error']}")
        anchors[rel_path] = {int(k): v for k, v in folded.items()}
    return anchors


def _saved_inputs_payload_is_nonempty(payload: Any) -> bool:
    """Return True when a saved-input cert actually carries replay data."""
    if payload is None:
        return False
    if isinstance(payload, (list, tuple, dict, str, bytes)):
        return len(payload) > 0
    return True


def _saved_inputs_payload_error(payload: Any) -> Optional[str]:
    """Validate that a saved-input payload can be replayed by the runner."""
    if not _saved_inputs_payload_is_nonempty(payload):
        return "saved_inputs certificate has empty saved_inputs payload"
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as e:
        return f"saved_inputs payload is not JSON-serializable: {e}"
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        if isinstance(record, dict) and "why" in record:
            return "saved-input records must not contain a 'why' field"
    return None


def _saved_inputs_file_error(path: Path) -> Optional[str]:
    """Validate a local saved-input JSONL sidecar without loading it all."""
    if not path.exists():
        return f"saved-input JSONL file does not exist: {path}"
    if not path.is_file():
        return f"saved-input JSONL path is not a file: {path}"
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
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
    except OSError as e:
        return f"failed to read saved-input JSONL file: {e}"
    if count == 0:
        return "saved-input JSONL file is empty"
    return None


def _resolve_saved_inputs_file(output_dir: Path, stored_path: str) -> Optional[Path]:
    """Resolve a stored sidecar path and ensure it stays under output_dir."""
    if not stored_path:
        return None
    root = output_dir.resolve()
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _callable_short_name(expr: ast.AST) -> str:
    """Best-effort short callable name for decorators/calls."""
    target = expr.func if isinstance(expr, ast.Call) else expr
    while isinstance(target, ast.Attribute):
        target = target.attr
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, str):
        return target
    return ""


def _saved_inputs_code_error(cert_code: str) -> Optional[str]:
    """Reject final saved-input certs that still contain a fuzzing head."""
    try:
        tree = ast.parse(cert_code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in {"hypothesis", "hypofuzz"}:
                    return (
                        "saved_inputs certificate must not import "
                        f"{alias.name}; replay saved inputs instead"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in {"hypothesis", "hypofuzz"}:
                return (
                    "saved_inputs certificate must not import "
                    f"from {node.module}; replay saved inputs instead"
                )
        elif isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                short = _callable_short_name(decorator)
                if short in {"given", "example", "settings"}:
                    return (
                        "saved_inputs certificate must not use Hypothesis "
                        f"decorator @{short}; replay saved inputs instead"
                    )
    return None


_FORBIDDEN_CASE_KEYS = frozenset({
    "expected", "expected_result", "expected_calls", "should_raise",
})

_ARGUMENT_PAYLOAD_KEYS = frozenset({"args", "kwargs"})


def _case_level_nodes(root: ast.AST) -> Iterator[ast.AST]:
    """``ast.walk`` over ``root``, minus every case dict's argument payload."""
    stack: List[ast.AST] = [root]
    while stack:
        node = stack.pop()
        yield node
        skip: Set[int] = set()
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in _ARGUMENT_PAYLOAD_KEYS
                ):
                    skip.add(id(value))
        stack.extend(
            child for child in ast.iter_child_nodes(node) if id(child) not in skip
        )


def _factory_oracle_code_error(cert_code: str) -> Optional[str]:
    """Validate the frozen factory/oracle module contract."""
    try:
        tree = ast.parse(cert_code)
    except SyntaxError:
        return None
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in ("build_cases", "check_case"):
        if required not in functions:
            return f"factory_oracle certificate must define {required}()"
    for node in _case_level_nodes(functions["build_cases"]):
        if isinstance(node, ast.Assert):
            return "build_cases() must not contain assertions"
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and key.value in _FORBIDDEN_CASE_KEYS
                ):
                    return (
                        "build_cases() contains forbidden oracle key "
                        f"{key.value!r}"
                    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "getsource", "getsourcefile", "getsourcelines", "findsource",
        }:
            return "factory_oracle certificate must not inspect source text"
        if isinstance(node, ast.Name) and node.id in {
            "getsource", "getsourcefile", "getsourcelines", "findsource",
        }:
            return "factory_oracle certificate must not inspect source text"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.split(".", 1)[0] in {"hypothesis", "hypofuzz"} for name in names):
                return "factory_oracle certificate must not use Hypothesis or Hypofuzz"
    return None


def _reference_arm(config) -> str:
    """The arm whose code state Stage 2 showed the baseline."""
    reference = getattr(config, "stage2_reference_state", "gold")
    return TARGET_STATE_MUTANT if reference == "mutant" else TARGET_STATE_GOLD


class ProofExecutor:
    """Executes per-spec certificates on the gold, mutant, and control code states."""

    def __init__(self, config, docker_runner):
        self.config = config
        self.docker_runner = docker_runner

    def run_all(
        self,
        tasks: List[EvalTask],
        baseline_results: List[BaselineResult],
    ) -> List[ProofExecResult]:
        """Execute every certificate on the gold, mutant, and control code states."""
        task_map: Dict[str, EvalTask] = {t.instance_id: t for t in tasks}

        work_items: List[Tuple[EvalTask, BaselineResult]] = []
        for br in baseline_results:
            task = task_map.get(br.instance_id)
            if task is None:
                logger.warning(
                    "No task for %s, skipping %d certificates",
                    br.instance_id,
                    len(br.certificates),
                )
                continue
            if not br.certificates:
                continue
            work_items.append((task, br))

        n_workers = max(1, self.config.n_concurrent)
        logger.info(
            "Stage 3: %d baseline results to execute (n_concurrent=%d)",
            len(work_items),
            n_workers,
        )

        all_results: List[ProofExecResult] = []
        total = len(work_items)
        counter_lock = threading.Lock()
        completed_count = [0]

        def _run_tracked(task: EvalTask, br: BaselineResult) -> List[ProofExecResult]:
            n_specs = len(br.certificates)
            logger.info(
                "[started] %s with %s/%s (%d certs)",
                br.instance_id, br.baseline, br.model, n_specs,
            )
            t0 = time.time()
            results = self._run_for_baseline_result(task, br)
            elapsed = time.time() - t0
            with counter_lock:
                completed_count[0] += 1
                done = completed_count[0]
            logger.info(
                "[done %d/%d] %s with %s/%s — %d cert outcomes in %.1fs",
                done, total, br.instance_id, br.baseline, br.model,
                len(results), elapsed,
            )
            return results

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_tracked, task, br): (task, br)
                for task, br in work_items
            }
            for future in as_completed(futures):
                task, br = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    with counter_lock:
                        completed_count[0] += 1
                        done = completed_count[0]
                    logger.error(
                        "[failed %d/%d] %s with %s/%s: %s",
                        done, total, br.instance_id, br.baseline, br.model, e,
                        exc_info=True,
                    )

        logger.info("Stage 3: executed %d cert outcomes", len(all_results))
        return all_results

    def _run_for_baseline_result(
        self,
        task: EvalTask,
        br: BaselineResult,
    ) -> List[ProofExecResult]:
        """Run one baseline result's certs on the gold, mutant, and control states."""
        from mutation_pipeline.models import PipelineInput

        results: List[ProofExecResult] = []

        spec_by_id: Dict[str, SurvivingSpec] = {
            s.spec_id: s for s in task.surviving_specs
        }

        fake_input = PipelineInput(
            instance_id=task.instance_id,
            level=1,
            task_statement="",
            interface_descriptions=[],
            gold_patch="",
            image_name=task.image_name,
        )

        isolation = getattr(self.config, "container_isolation", "arm")
        container = None
        try:
            if isolation == "instance":
                container = self.docker_runner.setup_instance(fake_input)

            for spec_id, certificate in br.certificates.items():
                surviving_spec = spec_by_id.get(spec_id)
                if surviving_spec is None:
                    logger.warning(
                        "[%s][%s/%s] cert %s has no matching surviving spec; skipping",
                        br.instance_id, br.baseline, br.model, spec_id,
                    )
                    continue

                if isolation == "spec":
                    container = self.docker_runner.setup_instance(fake_input)

                controls = select_control_mutants(
                    spec_id,
                    list(spec_by_id.values()),
                    getattr(self.config, "control_arm_k", None),
                )
                if not controls:
                    logger.info(
                        "[%s][%s/%s] spec %s has no control mutant — no other "
                        "spec's mutation probed it, so overfitting cannot be "
                        "measured here",
                        br.instance_id, br.baseline, br.model, spec_id,
                    )
                arms: List[Tuple[str, Optional[SurvivingSpec]]] = [
                    (TARGET_STATE_GOLD, None),
                    (TARGET_STATE_MUTANT, None),
                ]
                arms.extend((TARGET_STATE_CONTROL, control) for control in controls)

                try:
                    for target_state, control_spec in arms:
                        state_key = (
                            f"{target_state}_{control_spec.spec_id}"
                            if control_spec is not None
                            else target_state
                        )
                        result_path = self._result_path(br, spec_id, state_key)
                        if self.config.skip_existing and result_path.exists():
                            logger.info("  Skipping (result exists): %s", result_path)
                            results.append(ProofExecResult.from_dict(
                                json.loads(result_path.read_text(encoding="utf-8"))
                            ))
                            continue

                        if isolation == "arm":
                            container = self.docker_runner.setup_instance(fake_input)

                        try:
                            result = self._execute_single(
                                container=container,
                                task=task,
                                br=br,
                                spec_id=spec_id,
                                certificate=certificate,
                                target_state=target_state,
                                surviving_spec=surviving_spec,
                                control_spec=control_spec,
                            )
                            results.append(result)
                            self._save_result(result, result_path)

                            if isolation == "instance":
                                self.docker_runner.reset_testbed(
                                    container, fake_input
                                )
                        finally:
                            if isolation == "arm" and container:
                                self.docker_runner.cleanup(container)
                                container = None
                finally:
                    if isolation == "spec" and container:
                        self.docker_runner.cleanup(container)
                        container = None

        except Exception as e:
            logger.error(
                "[%s][%s/%s] container-level failure: %s",
                br.instance_id, br.baseline, br.model, e,
                exc_info=True,
            )
        finally:
            if container:
                self.docker_runner.cleanup(container)

        return results

    def _mutant_lines_block(
        self,
        container,
        surviving_spec: SurvivingSpec,
        target_state: str,
        coverage: Dict[str, Any],
        label: str,
    ) -> Dict[str, Any]:
        """The ``mutant_lines`` block for one arm's coverage."""
        side = "new" if target_state == TARGET_STATE_MUTANT else "old"
        mutant_lines = parse_mutant_lines(surviving_spec.mutant_patch, side=side)
        anchors = None
        fold_error: Optional[str] = None
        if mutant_lines:
            try:
                anchors = _fold_mutant_lines_in_container(container, mutant_lines)
            except Exception as e:
                fold_error = str(e)[:300]
                logger.warning("[%s] mutant-line folding failed: %s", label, fold_error)
        block = mutant_line_hit_rate(coverage, mutant_lines, anchors=anchors)
        block["side"] = side
        block["folded"] = anchors is not None
        if fold_error:
            block["fold_error"] = fold_error
        return block

    def _execute_single(
        self,
        container,
        task: EvalTask,
        br: BaselineResult,
        spec_id: str,
        certificate: SpecCertificate,
        target_state: str,
        surviving_spec: SurvivingSpec,
        control_spec: Optional[SurvivingSpec] = None,
    ) -> ProofExecResult:
        """Run one certificate against one code state."""
        t0 = time.time()
        patch_spec = control_spec if control_spec is not None else surviving_spec

        try:
            mutant_state = target_state in (
                TARGET_STATE_MUTANT, TARGET_STATE_CONTROL
            )
            ok, err = prepare_task_code_state(
                container=container,
                docker_runner=self.docker_runner,
                task=task,
                target_state=TARGET_STATE_MUTANT if mutant_state else TARGET_STATE_GOLD,
                surviving_spec=patch_spec if mutant_state else None,
                restore_test_with_reverse_patch=self.config.restore_test_with_reverse_patch,
            )
            if not ok:
                return ProofExecResult(
                    instance_id=task.instance_id,
                    baseline=br.baseline,
                    model=br.model,
                    spec_id=spec_id,
                    target_state=target_state,
                    triggered=False,
                    exit_code=0,
                    kind=KIND_INFRA,
                    proof_output="",
                    error=f"Failed to prepare {target_state} code state: {err}",
                    execution_time_seconds=time.time() - t0,
                )

            if not certificate.code.strip():
                return ProofExecResult(
                    instance_id=task.instance_id,
                    baseline=br.baseline,
                    model=br.model,
                    spec_id=spec_id,
                    target_state=target_state,
                    triggered=False,
                    exit_code=0,
                    kind=KIND_INFRA,
                    proof_output="",
                    error="Certificate is empty",
                    execution_time_seconds=time.time() - t0,
                )

            harness_form = getattr(self.config, "harness_form", "form1b_form6")
            saved_inputs_file: Optional[Path] = None
            if harness_form == "saved_inputs":
                if certificate.inputs_file:
                    saved_inputs_file = _resolve_saved_inputs_file(
                        Path(self.config.output_dir), certificate.inputs_file
                    )
                    if saved_inputs_file is None:
                        payload_error = (
                            "saved_inputs inputs_file must resolve under the "
                            "evaluation output directory"
                        )
                    else:
                        payload_error = _saved_inputs_file_error(saved_inputs_file)
                else:
                    payload_error = _saved_inputs_payload_error(certificate.saved_inputs)
                if payload_error:
                    return ProofExecResult(
                        instance_id=task.instance_id,
                        baseline=br.baseline,
                        model=br.model,
                        spec_id=spec_id,
                        target_state=target_state,
                        triggered=False,
                        exit_code=0,
                        kind=KIND_INFRA,
                        proof_output="",
                        error=payload_error,
                        execution_time_seconds=time.time() - t0,
                    )
                code_error = _saved_inputs_code_error(certificate.code)
                if code_error:
                    return ProofExecResult(
                        instance_id=task.instance_id,
                        baseline=br.baseline,
                        model=br.model,
                        spec_id=spec_id,
                        target_state=target_state,
                        triggered=False,
                        exit_code=0,
                        kind=KIND_INFRA,
                        proof_output="",
                        error=code_error,
                        execution_time_seconds=time.time() - t0,
                    )
                if "AGENT_SAVED_INPUTS_PATH" not in certificate.code:
                    return ProofExecResult(
                        instance_id=task.instance_id,
                        baseline=br.baseline,
                        model=br.model,
                        spec_id=spec_id,
                        target_state=target_state,
                        triggered=False,
                        exit_code=0,
                        kind=KIND_INFRA,
                        proof_output="",
                        error=(
                            "saved_inputs certificate must load the replay "
                            "corpus via AGENT_SAVED_INPUTS_PATH"
                        ),
                        execution_time_seconds=time.time() - t0,
                    )
            coverage: Optional[Dict[str, Any]] = None
            exhaustive = target_state == _reference_arm(self.config)
            if harness_form == "hypofuzz":
                code_mapping = surviving_spec.spec.get("code_mapping", [])
                target_funcs = [
                    (cm["file"], cm["function"])
                    for cm in code_mapping
                    if cm.get("file") and cm.get("function")
                ]
                triggered, output_text, exit_code, coverage = (
                    self.docker_runner.run_hypofuzz_campaign(
                        container,
                        certificate.code,
                        self.config.hypofuzz_timeout_seconds,
                        run_all_tests=exhaustive,
                    )
                )
                if coverage is not None:
                    if target_funcs:
                        scoped_funcs = []
                        for tf_file, tf_name in target_funcs:
                            s = scope_to_function(coverage, tf_file, tf_name)
                            if s is not None:
                                scoped_funcs.append(s)
                        if scoped_funcs:
                            coverage["scoped_function"] = scoped_funcs[0]
                            if len(scoped_funcs) > 1:
                                coverage["scoped_function_all"] = scoped_funcs

                    gold_funcs = [
                        FunctionInfo(f["file"], f["name"])
                        for f in (task.gold_patch_functions or [])
                        if f.get("file") and f.get("name")
                    ]
                    if target_funcs and gold_funcs:
                        declared = [
                            FunctionInfo(f, n) for f, n in target_funcs
                        ]
                        try:
                            resolution = _extend_scope_in_container(
                                container, declared, gold_funcs,
                                testbed_root="/testbed",
                            )
                            coverage["extended_scope"] = resolution["scope"]
                            coverage["scope_resolution"] = {
                                "complete": bool(resolution.get("complete")),
                                "unresolved_calls": resolution.get(
                                    "unresolved_calls", []
                                ),
                            }
                        except Exception as e:
                            logger.warning(
                                "extend_scope_in_container failed for "
                                "%s/%s: %s", task.instance_id, spec_id, e
                            )

                    coverage["mutant_lines"] = self._mutant_lines_block(
                        container, surviving_spec, target_state, coverage,
                        f"{task.instance_id}/{spec_id}/{target_state}",
                    )
                kind = classify_hypofuzz_output(exit_code, output_text)
            elif harness_form == "saved_inputs":
                code_mapping = surviving_spec.spec.get("code_mapping", [])
                target_funcs = [
                    (cm["file"], cm["function"])
                    for cm in code_mapping
                    if cm.get("file") and cm.get("function")
                ]
                triggered, output_text, exit_code, coverage = (
                    self.docker_runner.run_saved_inputs_replay(
                        container,
                        certificate.code,
                        certificate.saved_inputs,
                        self.config.saved_inputs_replay_timeout_seconds,
                        saved_inputs_file=saved_inputs_file,
                        run_all_tests=exhaustive,
                    )
                )
                if coverage is not None:
                    if target_funcs:
                        scoped_funcs = []
                        for tf_file, tf_name in target_funcs:
                            s = scope_to_function(coverage, tf_file, tf_name)
                            if s is not None:
                                scoped_funcs.append(s)
                        if scoped_funcs:
                            coverage["scoped_function"] = scoped_funcs[0]
                            if len(scoped_funcs) > 1:
                                coverage["scoped_function_all"] = scoped_funcs

                    gold_funcs = [
                        FunctionInfo(f["file"], f["name"])
                        for f in (task.gold_patch_functions or [])
                        if f.get("file") and f.get("name")
                    ]
                    if target_funcs and gold_funcs:
                        declared = [
                            FunctionInfo(f, n) for f, n in target_funcs
                        ]
                        try:
                            resolution = _extend_scope_in_container(
                                container, declared, gold_funcs,
                                testbed_root="/testbed",
                            )
                            coverage["extended_scope"] = resolution["scope"]
                            coverage["scope_resolution"] = {
                                "complete": bool(resolution.get("complete")),
                                "unresolved_calls": resolution.get(
                                    "unresolved_calls", []
                                ),
                            }
                        except Exception as e:
                            logger.warning(
                                "extend_scope_in_container failed for "
                                "%s/%s: %s", task.instance_id, spec_id, e
                            )

                    coverage["mutant_lines"] = self._mutant_lines_block(
                        container, surviving_spec, target_state, coverage,
                        f"{task.instance_id}/{spec_id}/{target_state}",
                    )
                if (
                    exit_code in (-1, 124)
                    or "Timeout after" in (output_text or "")
                ):
                    kind = KIND_TIMEOUT
                else:
                    kind = classify_pytest_output(exit_code, output_text)
            elif harness_form == "factory_oracle":
                contract_error = _factory_oracle_code_error(certificate.code)
                if contract_error:
                    return ProofExecResult(
                        instance_id=task.instance_id,
                        baseline=br.baseline,
                        model=br.model,
                        spec_id=spec_id,
                        target_state=target_state,
                        triggered=False,
                        exit_code=0,
                        kind=KIND_INFRA,
                        proof_output="",
                        error=contract_error,
                        execution_time_seconds=time.time() - t0,
                    )
                triggered, output_text, exit_code, coverage = (
                    self.docker_runner.run_factory_oracle(
                        container,
                        certificate.code,
                        self.config.saved_inputs_replay_timeout_seconds,
                        run_all_cases=exhaustive,
                    )
                )
                if coverage is not None:
                    coverage["mutant_lines"] = self._mutant_lines_block(
                        container, surviving_spec, target_state, coverage,
                        f"{task.instance_id}/{spec_id}/{target_state}",
                    )
                    code_mapping = surviving_spec.spec.get("code_mapping", [])
                    target_funcs = [
                        (cm["file"], cm["function"])
                        for cm in code_mapping
                        if cm.get("file") and cm.get("function")
                    ]
                    scoped_funcs = []
                    for target_file, target_name in target_funcs:
                        scoped = scope_to_function(
                            coverage, target_file, target_name
                        )
                        if scoped is not None:
                            scoped_funcs.append(scoped)
                    if scoped_funcs:
                        coverage["scoped_function"] = scoped_funcs[0]
                        if len(scoped_funcs) > 1:
                            coverage["scoped_function_all"] = scoped_funcs
                    gold_funcs = [
                        FunctionInfo(f["file"], f["name"])
                        for f in (task.gold_patch_functions or [])
                        if f.get("file") and f.get("name")
                    ]
                    if target_funcs and gold_funcs:
                        declared = [
                            FunctionInfo(target_file, target_name)
                            for target_file, target_name in target_funcs
                        ]
                        try:
                            resolution = _extend_scope_in_container(
                                container, declared, gold_funcs,
                                testbed_root="/testbed",
                            )
                            coverage["extended_scope"] = resolution["scope"]
                            coverage["scope_resolution"] = {
                                "complete": bool(resolution.get("complete")),
                                "unresolved_calls": resolution.get(
                                    "unresolved_calls", []
                                ),
                            }
                        except Exception as exc:
                            logger.warning(
                                "extend_scope_in_container failed for %s/%s: %s",
                                task.instance_id, spec_id, exc,
                            )
                driver_result = _factory_oracle_result(output_text)
                if exit_code in (-1, 124):
                    kind = KIND_TIMEOUT
                else:
                    driver_kind = (
                        driver_result.get("kind")
                        if driver_result is not None
                        else None
                    )
                    if driver_kind in _FACTORY_ORACLE_KINDS:
                        kind = driver_kind
                    elif exit_code == 0:
                        kind = KIND_PASSED
                    elif exit_code == 1:
                        kind = KIND_ASSERTION
                    else:
                        kind = KIND_INFRA
            else:
                triggered, output_text = self.docker_runner.run_hypothesis_test(
                    container,
                    certificate.code,
                    self.config.hypothesis_timeout,
                    run_all_tests=exhaustive,
                )
                exit_code = 1 if triggered else 0
                kind = classify_pytest_output(exit_code, output_text)
            return ProofExecResult(
                instance_id=task.instance_id,
                baseline=br.baseline,
                model=br.model,
                spec_id=spec_id,
                target_state=target_state,
                triggered=triggered,
                exit_code=exit_code,
                kind=kind,
                proof_output=output_text,
                execution_time_seconds=time.time() - t0,
                coverage=coverage,
                control_spec_id=(
                    control_spec.spec_id if control_spec is not None else None
                ),
            )

        except Exception as e:
            return ProofExecResult(
                instance_id=task.instance_id,
                baseline=br.baseline,
                model=br.model,
                spec_id=spec_id,
                target_state=target_state,
                triggered=False,
                exit_code=0,
                kind=KIND_INFRA,
                proof_output="",
                error=str(e),
                execution_time_seconds=time.time() - t0,
            )

    def _result_path(
        self, br: BaselineResult, spec_id: str, target_state: str
    ) -> Path:
        subdir = (
            self.config.output_dir
            / "proof_results"
            / f"{br.baseline}_{br.model}"
        )
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{br.instance_id}_{spec_id}_{target_state}.json"

    def _save_result(self, result: ProofExecResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def load_proof_results(output_dir: Path) -> List[ProofExecResult]:
    """Load all certificate execution results from disk."""
    results: List[ProofExecResult] = []
    proof_dir = output_dir / "proof_results"
    if not proof_dir.exists():
        return results
    for subdir in sorted(proof_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    results.append(ProofExecResult.from_dict(json.loads(f.read())))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load proof result %s: %s", json_file, e)
    logger.info("Loaded %d proof results from %s", len(results), proof_dir)
    return results
