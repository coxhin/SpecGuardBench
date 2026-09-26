"""Step 3: Existing Test Verification."""

import logging
from typing import Optional

from docker.models.containers import Container

from ..docker_runner import MutationDockerRunner
from ..models import PipelineInput, StepResult

logger = logging.getLogger(__name__)


class TestVerifier:
    """Verifies that a clean mutant passes existing tests."""

    def __init__(self, docker_runner: MutationDockerRunner):
        self.docker_runner = docker_runner

    def verify(
        self,
        pipeline_input: PipelineInput,
        mutations: list,
        timeout: int,
        container: Optional[Container] = None,
    ) -> StepResult:
        """Run existing tests against the clean mutant."""
        owns_container = container is None
        try:
            if owns_container:
                container = self.docker_runner.setup_instance(pipeline_input)

            if not self.docker_runner.apply_mutation(
                container, mutations
            ):
                return StepResult(
                    success=False,
                    error="Failed to apply mutation diff",
                    step_name="test_verification",
                )

            allowed_files = {m["file"] for m in mutations if m.get("file")}
            mutant_patch = self.docker_runner.get_mutation_diff(
                container, allowed_files=allowed_files
            )
            if not mutant_patch.strip():
                return StepResult(
                    success=False,
                    error="Mutation produced an empty mutant_patch",
                    step_name="test_verification",
                )

            self.docker_runner.restore_test_files(container, pipeline_input)

            all_passed, output = self.docker_runner.run_existing_tests(
                container, pipeline_input, timeout
            )

            logger.info(
                f"[{pipeline_input.instance_id}] Step 3: Tests {'PASSED' if all_passed else 'FAILED'}"
            )
            return StepResult(
                success=all_passed,
                data={"output": output, "mutant_patch": mutant_patch},
                step_name="test_verification",
            )

        except Exception as e:
            logger.error(
                f"[{pipeline_input.instance_id}] Step 3 error: {e}", exc_info=True
            )
            return StepResult(
                success=False,
                error=str(e),
                step_name="test_verification",
            )

        finally:
            if owns_container and container is not None:
                self.docker_runner.cleanup(container)
