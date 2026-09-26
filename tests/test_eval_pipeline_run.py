from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SC_ALIGN = _ROOT / "SC-align"
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

import eval_pipeline.run as eval_run_mod
from eval_pipeline.config import BaselineConfig, EvalPipelineConfig
from run_naming import run_dir_name


def test_save_latest_run_pointer_tracks_timestamped_eval_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "SC-align" / "eval_output" / "runs" / "2026-04-12__10-30-00"

    pointer_path = eval_run_mod._save_latest_run_pointer(run_dir)

    assert pointer_path == tmp_path / "SC-align" / "eval_output" / "latest_run.txt"
    assert pointer_path.read_text(encoding="utf-8") == str(run_dir)


def test_resolve_latest_run_dir_uses_latest_run_pointer(tmp_path: Path) -> None:
    base_output_dir = tmp_path / "SC-align" / "eval_output"
    run_dir = base_output_dir / "runs" / "2026-04-12__10-30-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_tasks.jsonl").write_text("", encoding="utf-8")
    (base_output_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    assert eval_run_mod._resolve_latest_run_dir(base_output_dir) == run_dir


def test_resolve_latest_run_dir_falls_back_to_latest_timestamped_run(tmp_path: Path) -> None:
    base_output_dir = tmp_path / "SC-align" / "eval_output"
    older_run = base_output_dir / "runs" / "2026-04-12__10-30-00"
    newer_run = base_output_dir / "runs" / "2026-04-13__09-15-00"
    older_run.mkdir(parents=True, exist_ok=True)
    newer_run.mkdir(parents=True, exist_ok=True)
    (older_run / "eval_tasks.jsonl").write_text("", encoding="utf-8")
    (newer_run / "eval_tasks.jsonl").write_text("", encoding="utf-8")

    assert eval_run_mod._resolve_latest_run_dir(base_output_dir) == newer_run


def test_resolve_latest_run_dir_returns_none_when_no_stage1_output(tmp_path: Path) -> None:
    base_output_dir = tmp_path / "SC-align" / "eval_output"
    (base_output_dir / "runs" / "2026-04-12__10-30-00").mkdir(parents=True)

    assert eval_run_mod._resolve_latest_run_dir(base_output_dir) is None


def _catalog() -> dict:
    return {
        "codex": BaselineConfig(
            name="codex", model="gpt-5.6-sol",
            agent_config="codex", tool_version="0.117.0",
        ),
        "claude_code": BaselineConfig(
            name="claude_code", model="claude-opus-4-8",
            agent_config="claude_code", tool_version="2.1.77",
        ),
    }


def test_select_baseline_picks_by_name_and_keeps_catalog_model() -> None:
    selected = eval_run_mod._select_baseline(_catalog(), "codex", None)

    assert selected.name == "codex"
    assert selected.model == "gpt-5.6-sol"
    assert selected.tool_version == "0.117.0"


def test_select_baseline_applies_model_override_without_mutating_catalog() -> None:
    catalog = _catalog()

    selected = eval_run_mod._select_baseline(catalog, "codex", "gpt-5.6-luna")

    assert selected.model == "gpt-5.6-luna"
    assert catalog["codex"].model == "gpt-5.6-sol"


def test_select_baseline_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        eval_run_mod._select_baseline(_catalog(), "openhands", None)


def _config_with_mutation_output(path: Path, stages=None) -> EvalPipelineConfig:
    return EvalPipelineConfig(mutation_output=path, stages=stages)


def test_require_mutation_output_rejects_missing_file(tmp_path: Path) -> None:
    config = _config_with_mutation_output(tmp_path / "nope" / "output.jsonl")

    with pytest.raises(SystemExit):
        eval_run_mod._require_mutation_output(config)


def test_require_mutation_output_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "output.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        eval_run_mod._require_mutation_output(_config_with_mutation_output(empty))


def test_require_mutation_output_accepts_a_populated_file(tmp_path: Path) -> None:
    populated = tmp_path / "output.jsonl"
    populated.write_text('{"instance_id": "x"}\n', encoding="utf-8")

    eval_run_mod._require_mutation_output(_config_with_mutation_output(populated))


def test_require_mutation_output_skipped_when_stage1_not_run(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "output.jsonl"

    eval_run_mod._require_mutation_output(
        _config_with_mutation_output(missing, stages=[3, 4, 5])
    )


def test_run_dir_name_appends_tag_after_timestamp() -> None:
    tagged = run_dir_name("codex56sol_sklearn")
    untagged = run_dir_name(None)

    assert tagged.endswith("__codex56sol_sklearn")
    datetime.strptime(untagged, "%Y-%m-%d__%H-%M-%S")
    datetime.strptime(
        tagged[: -len("__codex56sol_sklearn")], "%Y-%m-%d__%H-%M-%S"
    )


def test_run_dir_name_keeps_tag_a_single_path_segment() -> None:
    name = run_dir_name("pandas/subtle runs")

    assert "/" not in name
    assert name.endswith("__pandas-subtle-runs")
