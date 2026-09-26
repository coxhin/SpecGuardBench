"""Step 4: Witness Execution (observation-diff verdict)."""

import ast
import logging
import re
from typing import List, Tuple

from docker.models.containers import Container

from ..docker_runner import MutationDockerRunner
from ..models import OBSERVATION_LINE_PREFIX, PipelineInput, StepResult

logger = logging.getLogger(__name__)


_MARKER_BEGIN = "=== __WITNESS_OBS_BEGIN__ ==="
_MARKER_END = "=== __WITNESS_OBS_END__ ==="

_MARKER_RE = re.compile(
    rf"{re.escape(_MARKER_BEGIN)}\s*\n(.*?)\n?{re.escape(_MARKER_END)}",
    re.DOTALL,
)


def _wrap_with_markers(witness_script: str) -> str:
    """AST-inject ``print(BEGIN)`` / ``print(END)`` markers around each ``test_*``
    function body and append the ``__main__`` block that runs them.
    """
    tree = ast.parse(witness_script)

    def _print_stmt(marker: str) -> ast.stmt:
        return ast.parse(f'print({marker!r}, flush=True)').body[0]

    test_names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_names.append(node.name)
            original_body = node.body
            try_finally = ast.Try(
                body=original_body,
                handlers=[],
                orelse=[],
                finalbody=[_print_stmt(_MARKER_END)],
            )
            node.body = [_print_stmt(_MARKER_BEGIN), try_finally]

    if test_names:
        calls = "\n".join(f"    {name}()" for name in test_names)
        tree.body.append(
            ast.parse(f"if __name__ == '__main__':\n{calls}\n").body[0]
        )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _fenced_segments(run_output: str) -> List[str]:
    """Everything between BEGIN/END marker pairs, one entry per pair."""
    return [m.strip("\n") for m in _MARKER_RE.findall(run_output)]


def _extract_observation(run_output: str) -> str:
    """Concatenate the declared observations inside the BEGIN/END fences."""
    segments = []
    for segment in _fenced_segments(run_output):
        segments.append("\n".join(
            line[len(OBSERVATION_LINE_PREFIX):]
            for line in segment.splitlines()
            if line.startswith(OBSERVATION_LINE_PREFIX)
        ))
    return "\n----\n".join(segments) if any(segments) else ""


def _extract_stray_output(run_output: str) -> str:
    """The fenced lines that are *not* declared observations."""
    lines = []
    for segment in _fenced_segments(run_output):
        lines.extend(
            line for line in segment.splitlines()
            if not line.startswith(OBSERVATION_LINE_PREFIX)
        )
    return "\n".join(lines)


class WitnessExecutor:
    """Runs the witness twice and diffs the observations it declared."""

    def __init__(self, docker_runner: MutationDockerRunner):
        self.docker_runner = docker_runner

    def execute(
        self,
        pipeline_input: PipelineInput,
        witness_script: str,
        timeout: int,
        container: Container,
    ) -> StepResult:
        """Run the witness twice as a plain program and check that the two markered
        observation segments differ.
        """
        try:
            wrapped = _wrap_with_markers(witness_script)
        except SyntaxError as e:
            return StepResult(
                success=False,
                error=f"Could not AST-wrap witness with stdout markers: {e}",
                step_name="witness_execution",
            )

        try:
            mutant_triggered, mutant_out = self.docker_runner.run_witness_script(
                container, wrapped, timeout,
                env={"WITNESS_ALLOW_BIND_MISMATCH": "1"},
            )
            if mutant_triggered:
                return StepResult(
                    success=False,
                    error=(
                        "Witness exited non-zero on mutant; print-form "
                        "witnesses must run cleanly (wrap target calls in "
                        "try/except and print a stable token instead of raising)"
                    ),
                    data={
                        "mutant_output": mutant_out,
                        "gold_output": "",
                        "mutant_observation": _extract_observation(mutant_out),
                        "gold_observation": "",
                    },
                    step_name="witness_execution",
                )

            self.docker_runner.reset_testbed(container, pipeline_input)

            gold_triggered, gold_out = self.docker_runner.run_witness_script(
                container, wrapped, timeout
            )
            if gold_triggered:
                return StepResult(
                    success=False,
                    error=(
                        "Witness exited non-zero on gold; the witness "
                        "must run cleanly against the correct implementation. "
                        "If the trace shows 'case does not match target "
                        "signature', the case contradicts the spec's own "
                        "signature and is broken — build cases the correct "
                        "implementation accepts, and let the mutant be the "
                        "side that diverges."
                    ),
                    data={
                        "mutant_output": mutant_out,
                        "gold_output": gold_out,
                        "mutant_observation": _extract_observation(mutant_out),
                        "gold_observation": _extract_observation(gold_out),
                    },
                    step_name="witness_execution",
                )

            mutant_obs = _extract_observation(mutant_out)
            gold_obs = _extract_observation(gold_out)

            if not mutant_obs or not gold_obs:
                return StepResult(
                    success=False,
                    error=(
                        "Witness produced no declared observation in one or "
                        "both runs (no prefixed line inside the markers — "
                        "likely build_cases returned nothing, the body "
                        "bypassed the wrapper, or observe_case never returned)"
                    ),
                    data={
                        "mutant_output": mutant_out,
                        "gold_output": gold_out,
                        "mutant_observation": mutant_obs,
                        "gold_observation": gold_obs,
                    },
                    step_name="witness_execution",
                )

            diverged = mutant_obs != gold_obs

            logger.info(
                f"[{pipeline_input.instance_id}] Step 4: witness "
                f"{'DIVERGED' if diverged else 'MATCHED (no divergence)'} "
                f"(mutant_obs={len(mutant_obs)} chars, gold_obs={len(gold_obs)} chars)"
            )
            return StepResult(
                success=diverged,
                data={
                    "mutant_output": mutant_obs,
                    "gold_output": gold_obs,
                    "mutant_stray_output": _extract_stray_output(mutant_out),
                    "gold_stray_output": _extract_stray_output(gold_out),
                    "mutant_raw_output": mutant_out,
                    "gold_raw_output": gold_out,
                },
                error=None if diverged else "Witness produced identical observations on mutant and gold",
                step_name="witness_execution",
            )

        except Exception as e:
            logger.error(
                f"[{pipeline_input.instance_id}] Step 4 error: {e}", exc_info=True
            )
            return StepResult(
                success=False,
                error=str(e),
                step_name="witness_execution",
            )
