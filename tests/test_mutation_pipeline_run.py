from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
_SC_ALIGN = _ROOT / "SC-align"
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mutation_pipeline.models import AtomicSpec, MutationResult, Proof, SpecCategory, SpecSource
import mutation_pipeline.run as run_mod


def _make_result(instance_id: str, spec_id: str = "S1") -> MutationResult:
    return MutationResult(
        schema_version=2,
        instance_id=instance_id,
        spec=AtomicSpec(
            spec_id=spec_id,
            text=f"spec for {instance_id}",
            category=SpecCategory.BEHAVIORAL,
            sources=[SpecSource.DOCSTRING],
            code_mapping=[],
        ),
        mask_patch="mask.patch",
        gold_patch="gold.patch",
        mutant_patch="mutant.patch",
        test_patch="test.patch",
        proof=Proof(),
        metadata={},
    )


def _make_resume_args(run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        llm_a=None,
        llm_b=None,
        output_dir="SC-align/mutation_output/runs",
        instance_ids=None,
        timeout=600,
        hypothesis_timeout=120,
        proxy_port=None,
        gpu_ids=None,
        config_path=None,
        dataset="LiberCoders/FeatureBench",
        split="full",
        resume=str(run_dir),
        _timeout_provided=False,
        _hypothesis_timeout_provided=False,
    )


def _write_resume_metadata(run_dir: Path, instance_ids: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "llm_a_name": "claude-sonnet-4-20250514",
        "llm_b_name": "llm-b",
        "output_dir": str(run_dir),
        "n_concurrent": 1,
        "instance_ids": instance_ids,
        "timeout": 600,
        "hypothesis_timeout": 120,
        "proxy_port": None,
        "gpu_ids": None,
        "dataset": "LiberCoders/FeatureBench",
        "split": "full",
        "preservation_probe_k": 0,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_sync_legacy_output_refreshes_eval_default_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "SC-align" / "mutation_output" / "runs" / "2026-04-12__10-30-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output.jsonl").write_text('{"instance_id":"one"}\n', encoding="utf-8")

    legacy_path = run_mod._sync_legacy_output(run_dir)

    assert legacy_path == tmp_path / "SC-align" / "mutation_output" / "output.jsonl"
    assert legacy_path.read_text(encoding="utf-8") == '{"instance_id":"one"}\n'


def test_load_completed_instance_results_treats_empty_results_as_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-04-12__10-30-00"

    empty_dir = run_dir / "done-empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    (empty_dir / "results.json").write_text("[]", encoding="utf-8")

    full_dir = run_dir / "done-full"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "results.json").write_text(
        json.dumps([_make_result("done-full").to_dict()]),
        encoding="utf-8",
    )

    completed = run_mod._load_completed_instance_results(run_dir)

    assert set(completed) == {"done-empty", "done-full"}
    assert completed["done-empty"] == []
    assert [result.instance_id for result in completed["done-full"]] == ["done-full"]


def test_main_resume_skips_completed_instances_and_merges_existing_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "SC-align" / "mutation_output" / "runs" / "2026-04-12__10-30-00"
    _write_resume_metadata(run_dir, ["done", "todo"])
    done_dir = run_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "results.json").write_text(
        json.dumps([_make_result("done").to_dict()]),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_mod, "parse_args", lambda: _make_resume_args(run_dir))
    monkeypatch.setattr(run_mod, "setup_logging", lambda output_dir: None)
    monkeypatch.setattr(run_mod, "InferConfigLoader", lambda config_path: object())
    monkeypatch.setattr(
        run_mod,
        "load_pipeline_inputs",
        lambda config_loader, config: [
            SimpleNamespace(instance_id="done"),
            SimpleNamespace(instance_id="todo"),
        ],
    )

    class FakePipeline:
        captured_inputs: list[str] = []
        captured_existing_results: list[MutationResult] = []
        captured_completed_instances: int = 0

        def __init__(self, config, config_loader) -> None:
            self.config = config

        def run_all(self, inputs, existing_results=None, completed_instances=0):
            FakePipeline.captured_inputs = [item.instance_id for item in inputs]
            FakePipeline.captured_existing_results = list(existing_results or [])
            FakePipeline.captured_completed_instances = completed_instances
            combined_results = list(existing_results or []) + [_make_result("todo", spec_id="S2")]
            run_mod._write_output_jsonl(self.config.output_dir / "output.jsonl", combined_results)
            return combined_results

    monkeypatch.setattr(run_mod, "MutationPipeline", FakePipeline)

    run_mod.main()

    legacy_output_path = tmp_path / "SC-align" / "mutation_output" / "output.jsonl"
    output_lines = legacy_output_path.read_text(encoding="utf-8").splitlines()

    assert FakePipeline.captured_inputs == ["todo"]
    assert [result.instance_id for result in FakePipeline.captured_existing_results] == ["done"]
    assert FakePipeline.captured_completed_instances == 1
    assert len(output_lines) == 2
    assert [json.loads(line)["instance_id"] for line in output_lines] == ["done", "todo"]


def test_main_resume_rebuilds_outputs_without_rerunning_when_all_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "SC-align" / "mutation_output" / "runs" / "2026-04-12__10-30-00"
    _write_resume_metadata(run_dir, ["done"])
    done_dir = run_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "results.json").write_text(
        json.dumps([_make_result("done").to_dict()]),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_mod, "parse_args", lambda: _make_resume_args(run_dir))
    monkeypatch.setattr(run_mod, "setup_logging", lambda output_dir: None)
    monkeypatch.setattr(run_mod, "InferConfigLoader", lambda config_path: object())
    monkeypatch.setattr(
        run_mod,
        "load_pipeline_inputs",
        lambda config_loader, config: [SimpleNamespace(instance_id="done")],
    )

    class UnexpectedPipeline:
        def __init__(self, config, config_loader) -> None:
            raise AssertionError("resume should not build a pipeline when all instances are complete")

    monkeypatch.setattr(run_mod, "MutationPipeline", UnexpectedPipeline)

    run_mod.main()

    legacy_output_path = tmp_path / "SC-align" / "mutation_output" / "output.jsonl"
    assert legacy_output_path.read_text(encoding="utf-8").splitlines() == [
        _make_result("done").to_jsonl()
    ]


def test_load_resume_config_respects_explicit_default_timeout_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "SC-align" / "mutation_output" / "runs" / "2026-04-12__10-30-00"
    _write_resume_metadata(run_dir, ["done"])

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["timeout"] = 900
    metadata["hypothesis_timeout"] = 240
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    args = _make_resume_args(run_dir)
    args._timeout_provided = True
    args._hypothesis_timeout_provided = True

    config = run_mod._load_resume_config(run_dir, args)

    assert config.timeout == 600
    assert config.hypothesis_timeout == 120
