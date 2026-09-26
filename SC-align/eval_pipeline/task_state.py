"""Helpers to prepare ``/testbed`` into a target code state for one EvalTask."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Tuple

from docker.models.containers import Container

from featurebench.harness.constants import UTF8
from featurebench.harness.container import copy_to_container, exec_run_with_timeout

from .models import TARGET_STATE_GOLD, TARGET_STATE_MUTANT, EvalTask, SurvivingSpec

logger = logging.getLogger(__name__)


def _run_cmd(container: Container, command: str, timeout: int = 60) -> Tuple[bool, str]:
    exit_code, output = exec_run_with_timeout(container, command, timeout=timeout)
    output_text = output.decode(UTF8, errors="replace")
    return exit_code == 0, output_text


def _reinitialize_git(container: Container) -> Tuple[bool, str]:
    git_cmds = [
        "cd /testbed && rm -rf .git",
        "cd /testbed && git init",
        'cd /testbed && git config user.email "fb@bench.com"',
        'cd /testbed && git config user.name "FeatureBench"',
        "cd /testbed && git add -A",
        'cd /testbed && git commit -m "Initial commit for SC-align eval" --allow-empty',
    ]
    for cmd in git_cmds:
        ok, out = _run_cmd(container, cmd, timeout=60)
        if not ok:
            return False, f"git re-init failed: {cmd}\n{out}"
    return True, ""


_LEAK_PATHS = (
    "/testbed/.git",
    "/root/my_repo/.git",
    "/tmp/mutation_patch.diff",
    "/tmp/eval_test_patch.diff",
)


def _remove_git(container: Container) -> Tuple[bool, str]:
    paths = " ".join(_LEAK_PATHS)
    checks = " && ".join(f"test ! -e {p}" for p in _LEAK_PATHS)
    return _run_cmd(container, f"rm -rf {paths} && {checks}", timeout=60)


_PRISTINE_COPY = "/root/my_repo"


def _remove_pristine_copy(container: Container) -> Tuple[bool, str]:
    """Delete the image's gold tree so nothing can diff or import against it."""
    return _run_cmd(
        container,
        f"rm -rf {_PRISTINE_COPY} && test ! -e {_PRISTINE_COPY}",
        timeout=120,
    )


def _reverse_apply_test_patch(container: Container, test_patch: str) -> Tuple[bool, str]:
    if not test_patch or not test_patch.strip():
        return True, ""

    patch_content = test_patch if test_patch.endswith("\n") else test_patch + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding=UTF8) as f:
        f.write(patch_content)
        local_path = f.name

    try:
        container_path = "/tmp/eval_test_patch.diff"
        copy_to_container(container, local_path, container_path)
        cmd = f"cd /testbed && git apply --reverse --whitespace=fix {container_path}"
        return _run_cmd(container, cmd, timeout=120)
    finally:
        os.unlink(local_path)
        _run_cmd(container, f"rm -f {container_path}", timeout=30)


def prepare_task_code_state(
    container: Container,
    docker_runner,
    task: EvalTask,
    target_state: str = TARGET_STATE_GOLD,
    surviving_spec: SurvivingSpec | None = None,
    restore_test_with_reverse_patch: bool = False,
    remove_pristine_copy: bool = False,
) -> Tuple[bool, str]:
    """Prepare ``/testbed`` for evaluating one task in the requested code state."""
    if task.schema_version != 3:
        return False, f"unsupported eval task schema_version={task.schema_version}"

    if target_state not in (TARGET_STATE_GOLD, TARGET_STATE_MUTANT):
        return False, f"unknown target_state={target_state}"

    mask_patch = task.mask_patch or ""
    gold_patch = task.gold_patch or ""
    if not gold_patch.strip():
        return False, "missing gold_patch"

    mutant_patch = ""
    if target_state == TARGET_STATE_MUTANT:
        if surviving_spec is None:
            return False, "mutant target_state requires surviving_spec"
        mutant_patch = surviving_spec.mutant_patch or ""
        if not mutant_patch.strip():
            return False, f"missing mutant_patch for spec {surviving_spec.spec_id}"

    if mask_patch.strip():
        if not docker_runner.apply_patch(container, mask_patch):
            return False, "failed to apply mask patch"

    ok, err = _reinitialize_git(container)
    if not ok:
        return False, err

    if not docker_runner.apply_patch(container, gold_patch):
        return False, "failed to apply gold patch"

    if target_state == TARGET_STATE_MUTANT:
        if not docker_runner.apply_patch(container, mutant_patch):
            return False, "failed to apply mutant patch"

    if restore_test_with_reverse_patch:
        ok, out = _reverse_apply_test_patch(container, task.test_patch or "")
        if not ok:
            spec_id = surviving_spec.spec_id if surviving_spec else "-"
            logger.warning(
                "[%s/%s/%s] reverse test_patch failed (continuing): %s",
                task.instance_id,
                target_state,
                spec_id,
                out[:500],
            )

    ok, out = _remove_git(container)
    if not ok:
        return False, f"failed to remove /testbed/.git: {out}"

    if remove_pristine_copy:
        ok, out = _remove_pristine_copy(container)
        if not ok:
            return False, f"failed to remove {_PRISTINE_COPY}: {out}"

    return True, ""
