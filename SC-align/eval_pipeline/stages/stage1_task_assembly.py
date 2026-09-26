"""Stage 1: task assembly."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_pipeline.config import (
    MUTATION_OUTPUT_SCHEMA_VERSION,
    EvalPipelineConfig,
    load_mutation_outputs,
)
from eval_pipeline.coverage_scoping import extract_gold_patch_functions
from eval_pipeline.models import EVAL_SCHEMA_VERSION, EvalTask, SurvivingSpec

logger = logging.getLogger(__name__)


def _parse_test_list(raw: Any) -> List[str]:
    """Parse a test list field from HF data."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return []
    return []


def _parse_repo_settings(raw: Any) -> Dict[str, Any]:
    """Parse repo_settings field from HF data."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}
    return {}


def _to_text(raw: Any) -> str:
    """Normalise optional patch-like fields to plain strings."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        if raw != raw:
            return ""
    except Exception:
        pass
    return str(raw)


def save_eval_tasks(tasks: List[EvalTask], output_dir: Path) -> Path:
    """Write eval tasks to ``eval_tasks.jsonl`` in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eval_tasks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.to_jsonl() + "\n")
    logger.info(f"Saved {len(tasks)} eval tasks to {out_path}")
    return out_path


def load_eval_tasks(output_dir: Path) -> List[EvalTask]:
    """Read eval tasks from ``eval_tasks.jsonl`` in ``output_dir``."""
    in_path = output_dir / "eval_tasks.jsonl"
    if not in_path.exists():
        return []
    tasks: List[EvalTask] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append(EvalTask.from_dict(data))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse eval task line: {e}")
    logger.info(f"Loaded {len(tasks)} eval tasks from {in_path}")
    return tasks


class TaskAssembler:
    """Assembles ``EvalTask`` instances from mutation output and HF metadata.
    """

    def __init__(
        self,
        config: EvalPipelineConfig,
        hf_data: Dict[str, Dict[str, Any]],
    ):
        self.config = config
        self.hf_data = hf_data

    def assemble(self) -> List[EvalTask]:
        """Load mutation output and produce one ``EvalTask`` per instance."""
        mutation_outputs = load_mutation_outputs(self.config.mutation_output)
        if not mutation_outputs:
            logger.warning(f"No mutation outputs found at {self.config.mutation_output}")
            return []

        all_tasks: List[EvalTask] = []
        for iid, mutations in mutation_outputs.items():
            if not mutations:
                continue
            self._validate_mutation_output_rows(iid, mutations)
            hf_item = self.hf_data.get(iid)
            if hf_item is None:
                logger.warning(f"[{iid}] No HF metadata found, skipping")
                continue
            task = self._assemble_instance(iid, mutations, hf_item)
            all_tasks.append(task)
            logger.info(
                f"[{iid}] Assembled 1 task with {len(task.surviving_specs)} surviving specs"
            )

        logger.info(f"Total eval tasks assembled: {len(all_tasks)}")
        return all_tasks

    def _validate_mutation_output_rows(
        self,
        iid: str,
        mutations: List[Dict[str, Any]],
    ) -> None:
        """Validate the schema version and patch fields of mutation output rows."""
        first = mutations[0]
        schema_version = int(first.get("schema_version", 0))
        if schema_version != MUTATION_OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                f"[{iid}] unsupported mutation output schema_version={schema_version}"
            )

        required_patch_keys = ("mask_patch", "gold_patch", "mutant_patch", "test_patch")
        missing_keys = [key for key in required_patch_keys if key not in first]
        if missing_keys:
            raise ValueError(
                f"[{iid}] missing schema v2 patch fields in mutation output: {', '.join(missing_keys)}"
            )

        mask_patch = _to_text(first.get("mask_patch"))
        gold_patch = _to_text(first.get("gold_patch"))
        test_patch = _to_text(first.get("test_patch"))
        if not gold_patch.strip():
            raise ValueError(f"[{iid}] missing gold_patch in mutation output")

        for idx, mut in enumerate(mutations, 1):
            mut_schema_version = int(mut.get("schema_version", 0))
            if mut_schema_version != MUTATION_OUTPUT_SCHEMA_VERSION:
                raise ValueError(
                    f"[{iid}] mutation row {idx} has unsupported schema_version={mut_schema_version}"
                )
            missing_keys = [key for key in required_patch_keys if key not in mut]
            if missing_keys:
                raise ValueError(
                    f"[{iid}] mutation row {idx} is missing schema v2 patch fields: {', '.join(missing_keys)}"
                )
            if _to_text(mut.get("mask_patch")) != mask_patch:
                raise ValueError(f"[{iid}] inconsistent mask_patch across mutation rows")
            if _to_text(mut.get("gold_patch")) != gold_patch:
                raise ValueError(f"[{iid}] inconsistent gold_patch across mutation rows")
            if _to_text(mut.get("test_patch")) != test_patch:
                raise ValueError(f"[{iid}] inconsistent test_patch across mutation rows")
            if not _to_text(mut.get("mutant_patch")).strip():
                raise ValueError(f"[{iid}] mutation row {idx} is missing mutant_patch")

    def _assemble_instance(
        self,
        iid: str,
        mutations: List[Dict[str, Any]],
        hf_item: Dict[str, Any],
    ) -> EvalTask:
        """Assemble one ``EvalTask`` for an instance."""
        first = mutations[0]
        mask_patch = _to_text(first.get("mask_patch"))
        gold_patch = _to_text(first.get("gold_patch"))
        test_patch = _to_text(first.get("test_patch"))
        if not gold_patch.strip():
            raise ValueError(f"[{iid}] missing gold_patch in mutation output")

        seen_spec_ids: set = set()
        surviving_specs: List[SurvivingSpec] = []
        for mut in mutations:
            spec = mut.get("spec", {})
            spec_id = spec.get("spec_id", "")
            if not spec_id:
                continue
            if spec_id in seen_spec_ids:
                continue
            mutant_patch = _to_text(mut.get("mutant_patch"))
            if not mutant_patch.strip():
                raise ValueError(f"[{iid}][{spec_id}] missing mutant_patch in mutation output")
            seen_spec_ids.add(spec_id)
            proof = mut.get("proof", {})
            metadata = mut.get("metadata") or {}
            surviving_specs.append(SurvivingSpec(
                spec=spec,
                mutant_patch=mutant_patch,
                case_contract=_to_text(proof.get("case_contract")),
                witness_factory=_to_text(proof.get("witness_factory")),
                witness_observer=_to_text(proof.get("witness_observer")),
                preservation_probes=list(metadata.get("preservation_probes") or []),
            ))

        f2p_tests = _parse_test_list(hf_item.get("FAIL_TO_PASS"))
        p2p_tests = _parse_test_list(hf_item.get("PASS_TO_PASS"))
        repo_settings = _parse_repo_settings(hf_item.get("repo_settings"))
        image_name = hf_item.get("image_name", "")

        gold_funcs = [
            f.as_dict() for f in extract_gold_patch_functions(gold_patch)
        ]
        if not gold_funcs:
            raise ValueError(
                f"[{iid}] gold_patch contains no newly-added function "
                "definitions; cannot compute metrics C/D. Check that the "
                "gold_patch is non-empty and adds functions (vs. only "
                "modifying existing ones)."
            )

        return EvalTask(
            instance_id=iid,
            image_name=image_name,
            surviving_specs=surviving_specs,
            f2p_tests=f2p_tests,
            p2p_tests=p2p_tests,
            repo_settings=repo_settings,
            mask_patch=mask_patch,
            gold_patch=gold_patch,
            test_patch=test_patch,
            gold_patch_functions=gold_funcs,
            schema_version=EVAL_SCHEMA_VERSION,
        )
