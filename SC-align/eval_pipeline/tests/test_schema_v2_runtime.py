from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

from eval_pipeline.config import MUTATION_OUTPUT_SCHEMA_VERSION
from eval_pipeline.models import (
    EVAL_SCHEMA_VERSION,
    TARGET_STATE_GOLD,
    TARGET_STATE_MUTANT,
    EvalTask,
    SurvivingSpec,
)
from eval_pipeline.stages.stage1_task_assembly import TaskAssembler
from eval_pipeline.task_state import prepare_task_code_state


class FakeDockerRunner:
    def __init__(self) -> None:
        self.applied_patches: list[str] = []

    def apply_patch(self, container: Any, patch_content: str) -> bool:
        self.applied_patches.append(patch_content)
        return True


@dataclass
class DummyContainer:
    short_id: str = "dummy"


def _make_hf_item() -> dict[str, Any]:
    return {
        "FAIL_TO_PASS": ["tests/test_one.py"],
        "PASS_TO_PASS": ["tests/test_two.py"],
        "repo_settings": {"test_cmd": "pytest -q"},
        "image_name": "example-image",
        "patch": "mask.patch",
        "test_patch": "test.patch",
    }


_GOLD_PATCH_WITH_NEW_FUNC = """\
diff --git a/pkg/m.py b/pkg/m.py
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -1,2 +1,4 @@
 existing
+def added(x):
+    return x + 1
 last
"""


def _make_mutation_row(spec_id: str, mutant_patch: str = "mutant.patch") -> dict[str, Any]:
    return {
        "schema_version": MUTATION_OUTPUT_SCHEMA_VERSION,
        "instance_id": "example.instance.lv1",
        "spec": {"spec_id": spec_id, "text": "spec", "category": "behavioral", "code_mapping": []},
        "mask_patch": "mask.patch",
        "gold_patch": _GOLD_PATCH_WITH_NEW_FUNC,
        "mutant_patch": mutant_patch,
        "test_patch": "test.patch",
        "metadata": {"mutations": []},
        "proof": {
            "case_contract": "case has exactly args and kwargs",
            "witness_factory": (
                "def build_cases():\n"
                "    yield {'args': (60,), 'kwargs': {}}\n"
            ),
        },
    }


def test_stage1_emits_one_task_per_instance() -> None:
    hf_data = {"example.instance.lv1": _make_hf_item()}
    assembler = TaskAssembler.__new__(TaskAssembler)
    assembler.config = type("Cfg", (), {"mutation_output": Path("unused")})()
    assembler.hf_data = hf_data

    task = assembler._assemble_instance(
        "example.instance.lv1",
        [_make_mutation_row("S1"), _make_mutation_row("S2", mutant_patch="mutant-two.patch")],
        hf_data["example.instance.lv1"],
    )

    assert task.schema_version == EVAL_SCHEMA_VERSION
    assert task.instance_id == "example.instance.lv1"
    assert task.gold_patch == _GOLD_PATCH_WITH_NEW_FUNC
    assert task.mask_patch == "mask.patch"
    assert task.test_patch == "test.patch"
    assert task.gold_patch_functions == [{"file": "pkg/m.py", "name": "added"}]

    assert [s.spec_id for s in task.surviving_specs] == ["S1", "S2"]
    assert task.surviving_specs[0].mutant_patch == "mutant.patch"
    assert task.surviving_specs[1].mutant_patch == "mutant-two.patch"
    assert "args and kwargs" in task.surviving_specs[0].case_contract
    assert "build_cases" in task.surviving_specs[0].witness_factory


def test_stage1_dedupes_repeated_spec_ids() -> None:
    hf_data = {"example.instance.lv1": _make_hf_item()}
    assembler = TaskAssembler.__new__(TaskAssembler)
    assembler.config = type("Cfg", (), {"mutation_output": Path("unused")})()
    assembler.hf_data = hf_data

    task = assembler._assemble_instance(
        "example.instance.lv1",
        [
            _make_mutation_row("S1", mutant_patch="first.patch"),
            _make_mutation_row("S1", mutant_patch="second.patch"),
        ],
        hf_data["example.instance.lv1"],
    )
    assert len(task.surviving_specs) == 1
    assert task.surviving_specs[0].mutant_patch == "first.patch"


def test_stage1_rejects_old_format_rows() -> None:
    hf_data = {"example.instance.lv1": _make_hf_item()}
    assembler = TaskAssembler.__new__(TaskAssembler)
    assembler.config = type("Cfg", (), {"mutation_output": Path("unused")})()
    assembler.hf_data = hf_data

    old_row = {
        "instance_id": "example.instance.lv1",
        "spec": {"spec_id": "S1", "text": "spec", "category": "behavioral", "code_mapping": []},
        "aligned_code": "old-gold.diff",
        "misaligned_code": "old-mutant",
        "mutation_diff": "old-mutant",
        "proof": {"in_function_assertions": [], "harness_assertions": [], "hypothesis_harness": ""},
    }

    with pytest.raises(ValueError, match="unsupported mutation output schema_version=0"):
        assembler._validate_mutation_output_rows("example.instance.lv1", [old_row])


def _make_v3_task(mutant_patch: str = "mutant.patch") -> EvalTask:
    return EvalTask(
        instance_id="example.instance.lv1",
        image_name="example-image",
        surviving_specs=[
            SurvivingSpec(
                spec={"spec_id": "S1", "category": "behavioral"},
                mutant_patch=mutant_patch,
            )
        ],
        f2p_tests=[],
        p2p_tests=[],
        repo_settings={},
        mask_patch="mask.patch",
        gold_patch="gold.patch",
        test_patch="test.patch",
    )


def test_prepare_task_code_state_gold_applies_mask_then_gold() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    container = DummyContainer()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("eval_pipeline.task_state._reinitialize_git", lambda _c: (True, ""))
    patches_when_git_removed: list = []
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_git",
        lambda _c: (patches_when_git_removed.extend(runner.applied_patches), (True, ""))[1],
    )
    monkeypatch.setattr(
        "eval_pipeline.task_state._reverse_apply_test_patch",
        lambda _c, _p: (True, ""),
    )

    try:
        ok, err = prepare_task_code_state(
            container=container,
            docker_runner=runner,
            task=task,
            target_state=TARGET_STATE_GOLD,
            surviving_spec=None,
            restore_test_with_reverse_patch=False,
        )
    finally:
        monkeypatch.undo()

    assert ok is True
    assert err == ""
    assert runner.applied_patches == ["mask.patch", "gold.patch"]
    assert patches_when_git_removed == ["mask.patch", "gold.patch"]


def test_prepare_task_code_state_mutant_applies_all_three() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    container = DummyContainer()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("eval_pipeline.task_state._reinitialize_git", lambda _c: (True, ""))
    patches_when_git_removed: list = []
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_git",
        lambda _c: (patches_when_git_removed.extend(runner.applied_patches), (True, ""))[1],
    )
    monkeypatch.setattr(
        "eval_pipeline.task_state._reverse_apply_test_patch",
        lambda _c, _p: (True, ""),
    )

    try:
        ok, err = prepare_task_code_state(
            container=container,
            docker_runner=runner,
            task=task,
            target_state=TARGET_STATE_MUTANT,
            surviving_spec=task.surviving_specs[0],
            restore_test_with_reverse_patch=False,
        )
    finally:
        monkeypatch.undo()

    assert ok is True
    assert err == ""
    assert runner.applied_patches == ["mask.patch", "gold.patch", "mutant.patch"]
    assert patches_when_git_removed == ["mask.patch", "gold.patch", "mutant.patch"]


def test_prepare_task_code_state_mutant_requires_surviving_spec() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()

    ok, err = prepare_task_code_state(
        container=DummyContainer(),
        docker_runner=runner,
        task=task,
        target_state=TARGET_STATE_MUTANT,
        surviving_spec=None,
    )
    assert ok is False
    assert "surviving_spec" in err


def test_prepare_task_code_state_rejects_empty_mutant_patch() -> None:
    task = _make_v3_task(mutant_patch="")

    runner = FakeDockerRunner()
    ok, err = prepare_task_code_state(
        container=DummyContainer(),
        docker_runner=runner,
        task=task,
        target_state=TARGET_STATE_MUTANT,
        surviving_spec=task.surviving_specs[0],
    )
    assert ok is False
    assert "missing mutant_patch" in err


def test_prepare_task_code_state_rejects_unknown_target_state() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    ok, err = prepare_task_code_state(
        container=DummyContainer(),
        docker_runner=runner,
        task=task,
        target_state="unknown",
    )
    assert ok is False
    assert "unknown target_state" in err


def test_prepare_task_code_state_removes_pristine_copy_last_when_asked() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    container = DummyContainer()
    monkeypatch = pytest.MonkeyPatch()
    order: list = []
    monkeypatch.setattr("eval_pipeline.task_state._reinitialize_git", lambda _c: (True, ""))
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_git",
        lambda _c: (order.append(("git", list(runner.applied_patches))), (True, ""))[1],
    )
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_pristine_copy",
        lambda _c: (order.append(("my_repo", list(runner.applied_patches))), (True, ""))[1],
    )

    try:
        ok, err = prepare_task_code_state(
            container=container,
            docker_runner=runner,
            task=task,
            target_state=TARGET_STATE_MUTANT,
            surviving_spec=task.surviving_specs[0],
            remove_pristine_copy=True,
        )
    finally:
        monkeypatch.undo()

    assert (ok, err) == (True, "")
    assert order == [
        ("git", ["mask.patch", "gold.patch", "mutant.patch"]),
        ("my_repo", ["mask.patch", "gold.patch", "mutant.patch"]),
    ]


def test_prepare_task_code_state_keeps_pristine_copy_by_default() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    monkeypatch = pytest.MonkeyPatch()
    called: list = []
    monkeypatch.setattr("eval_pipeline.task_state._reinitialize_git", lambda _c: (True, ""))
    monkeypatch.setattr("eval_pipeline.task_state._remove_git", lambda _c: (True, ""))
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_pristine_copy",
        lambda _c: (called.append(1), (True, ""))[1],
    )

    try:
        ok, _ = prepare_task_code_state(
            container=DummyContainer(),
            docker_runner=runner,
            task=task,
            target_state=TARGET_STATE_GOLD,
        )
    finally:
        monkeypatch.undo()

    assert ok is True
    assert called == []


def test_prepare_task_code_state_fails_when_pristine_copy_survives() -> None:
    task = _make_v3_task()
    runner = FakeDockerRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("eval_pipeline.task_state._reinitialize_git", lambda _c: (True, ""))
    monkeypatch.setattr("eval_pipeline.task_state._remove_git", lambda _c: (True, ""))
    monkeypatch.setattr(
        "eval_pipeline.task_state._remove_pristine_copy",
        lambda _c: (False, "rm: cannot remove"),
    )

    try:
        ok, err = prepare_task_code_state(
            container=DummyContainer(),
            docker_runner=runner,
            task=task,
            target_state=TARGET_STATE_MUTANT,
            surviving_spec=task.surviving_specs[0],
            remove_pristine_copy=True,
        )
    finally:
        monkeypatch.undo()

    assert ok is False
    assert "/root/my_repo" in err
