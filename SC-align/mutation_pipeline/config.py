"""Pipeline configuration and instance data loading."""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from featurebench.infer.config import InferConfigLoader, DatasetLoader
from featurebench.llm.llm_caller import LLMCaller

from .models import PipelineInput

logger = logging.getLogger(__name__)


@dataclass
class MutationPipelineConfig:
    """Configuration for a mutation pipeline run."""
    llm_a_name: str
    llm_b_name: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("SC-align/mutation_output/runs"))
    n_concurrent: int = 1
    container_pool_size: int = 8
    instance_ids: Optional[List[str]] = None
    timeout: int = 600
    hypothesis_timeout: int = 120
    proxy_port: Optional[int] = None
    gpu_ids: Optional[str] = None
    dataset: str = "LiberCoders/FeatureBench"
    split: str = "full"
    step2_max_attempts: int = 3
    preservation_probe_k: int = 3
    step6_votes: int = 3
    step6b_concurrency: int = 12


def _reverse_unified_patch(patch_text: str) -> str:
    """Reverse a unified diff so it applies in the opposite direction."""
    if not patch_text:
        return patch_text

    hunk_header_pattern = re.compile(
        r"^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)$"
    )
    reversed_lines: list[str] = []

    for line in patch_text.splitlines():
        if line.startswith("--- "):
            reversed_lines.append("+++ " + line[4:])
            continue
        if line.startswith("+++ "):
            reversed_lines.append("--- " + line[4:])
            continue
        match = hunk_header_pattern.match(line)
        if match:
            old_range, new_range, tail = match.groups()
            reversed_lines.append(f"@@ -{new_range} +{old_range} @@{tail}")
            continue
        if line.startswith("new file mode "):
            reversed_lines.append(line.replace("new file mode ", "deleted file mode ", 1))
            continue
        if line.startswith("deleted file mode "):
            reversed_lines.append(line.replace("deleted file mode ", "new file mode ", 1))
            continue
        if line.startswith("old mode "):
            reversed_lines.append(line.replace("old mode ", "new mode ", 1))
            continue
        if line.startswith("new mode "):
            reversed_lines.append(line.replace("new mode ", "old mode ", 1))
            continue
        if line.startswith("rename from "):
            reversed_lines.append(line.replace("rename from ", "rename to ", 1))
            continue
        if line.startswith("rename to "):
            reversed_lines.append(line.replace("rename to ", "rename from ", 1))
            continue
        if line.startswith("copy from "):
            reversed_lines.append(line.replace("copy from ", "copy to ", 1))
            continue
        if line.startswith("copy to "):
            reversed_lines.append(line.replace("copy to ", "copy from ", 1))
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            reversed_lines.append("-" + line[1:])
            continue
        if line.startswith("-") and not line.startswith("--- "):
            reversed_lines.append("+" + line[1:])
            continue
        reversed_lines.append(line)

    return "\n".join(reversed_lines) + ("\n" if patch_text.endswith("\n") else "")


def _normalize_patch_path(path: str) -> str:
    path = str(path).strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if path.startswith("./"):
        path = path[2:]
    return path


def _parse_fail_to_pass_paths(raw_fail_to_pass: Any) -> set:
    if raw_fail_to_pass is None:
        return set()
    values = raw_fail_to_pass
    if isinstance(raw_fail_to_pass, str):
        try:
            values = json.loads(raw_fail_to_pass)
        except json.JSONDecodeError:
            return set()
    if not isinstance(values, list):
        return set()
    return {_normalize_patch_path(item) for item in values if isinstance(item, str) and item.strip()}


def _remove_patch_operations_for_paths(patch_text: str, excluded_paths: set) -> str:
    if not patch_text or not excluded_paths:
        return patch_text
    normalized_excluded = {_normalize_patch_path(p) for p in excluded_paths if p}
    lines = patch_text.splitlines(keepends=True)
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("diff --git "):
            output.append(line)
            i += 1
            continue
        block_start = i
        i += 1
        while i < len(lines) and not lines[i].startswith("diff --git "):
            i += 1
        block = lines[block_start:i]
        header = block[0].strip()
        parts = header.split()
        if len(parts) >= 4:
            path_a = _normalize_patch_path(parts[2])
            path_b = _normalize_patch_path(parts[3])
            should_drop = path_a in normalized_excluded or path_b in normalized_excluded
        else:
            should_drop = False
        if not should_drop:
            output.extend(block)
    return "".join(output)


def preprocess_hf_patch(patch_text: str, raw_fail_to_pass: Any) -> str:
    """Convert HF dataset patch (corruption direction) to ground truth implementation
    patch.
    """
    reversed_patch = _reverse_unified_patch(patch_text)
    fail_to_pass_paths = _parse_fail_to_pass_paths(raw_fail_to_pass)
    return _remove_patch_operations_for_paths(reversed_patch, fail_to_pass_paths)


class _LLMConfigAdapter:
    """Adapter to make InferConfigLoader's LLM config work with LLMCaller's expected
    interface.
    """
    def __init__(self, llm_config: Dict[str, Any], global_config: Dict[str, Any]):
        self.llm_config = {**global_config, **llm_config}


def create_llm_caller(
    config_loader: InferConfigLoader,
    llm_name: str,
) -> LLMCaller:
    """Create an LLMCaller from a named LLM section in config_mutation.toml."""
    llm_config = config_loader.get_llm_by_name(llm_name)
    if llm_config is None:
        raise ValueError(f"LLM config '{llm_name}' not found in config_mutation.toml [llm.*] sections")

    global_llm_config = config_loader.llm_config
    adapter = _LLMConfigAdapter(llm_config, global_llm_config)
    return LLMCaller(adapter)


def load_pipeline_inputs(
    config_loader: InferConfigLoader,
    config: MutationPipelineConfig,
) -> List[PipelineInput]:
    """Load all instance data needed by the mutation pipeline."""
    sc_align_dir = _PROJECT_ROOT / "SC-align"
    load_start = time.time()

    t0 = time.time()
    extracted_dir = sc_align_dir / "extracted_v2"
    extracted_data: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(extracted_dir.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        iid = data.get("instance_id", "")
        if iid:
            extracted_data[iid] = data

    logger.info(f"Loading extracted specs took {time.time() - t0:.1f}s ({len(extracted_data)} entries)")

    t0 = time.time()
    gt_dir = sc_align_dir / "ground_truth"
    manifest_path = gt_dir / "_manifest.json"
    gt_patches: Dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            iid = entry.get("instance_id", "")
            diff_file = entry.get("file", "")
            if iid and diff_file:
                diff_path = gt_dir / diff_file
                if diff_path.exists():
                    gt_patches[iid] = diff_path.read_text(encoding="utf-8")

    logger.info(f"Loading ground truth patches took {time.time() - t0:.1f}s ({len(gt_patches)} patches)")

    t0 = time.time()
    dataset_loader = DatasetLoader(config_loader)
    hf_instances = dataset_loader.load_dataset(
        dataset=config.dataset,
        split=config.split,
        levels=[1],
        task_ids=config.instance_ids,
    )
    hf_data: Dict[str, Dict[str, Any]] = {}
    for item in hf_instances:
        iid = item.get("instance_id", "")
        if iid:
            hf_data[iid] = item
    logger.info(f"Loading HF dataset took {time.time() - t0:.1f}s ({len(hf_data)} instances)")

    t0 = time.time()
    inputs: List[PipelineInput] = []
    for iid, hf_item in hf_data.items():
        if iid not in extracted_data:
            logger.warning(f"Skipping {iid}: no extracted spec data in extracted_v2/")
            continue

        if iid in gt_patches:
            gold_patch = gt_patches[iid]
        else:
            raw_patch = hf_item.get("patch", "") or ""
            raw_f2p = hf_item.get("FAIL_TO_PASS")
            gold_patch = preprocess_hf_patch(raw_patch, raw_f2p)
            if not gold_patch.strip():
                logger.warning(f"Skipping {iid}: empty gold patch")
                continue

        extracted = extracted_data[iid]

        raw_f2p = hf_item.get("FAIL_TO_PASS", [])
        if isinstance(raw_f2p, str):
            try:
                raw_f2p = json.loads(raw_f2p)
            except json.JSONDecodeError:
                raw_f2p = []
        fail_to_pass = raw_f2p if isinstance(raw_f2p, list) else []

        raw_p2p = hf_item.get("PASS_TO_PASS", [])
        if isinstance(raw_p2p, str):
            try:
                raw_p2p = json.loads(raw_p2p)
            except json.JSONDecodeError:
                raw_p2p = []
        pass_to_pass = raw_p2p if isinstance(raw_p2p, list) else []

        raw_rs = hf_item.get("repo_settings", "{}")
        if isinstance(raw_rs, str):
            try:
                repo_settings = json.loads(raw_rs)
            except json.JSONDecodeError:
                repo_settings = {}
        elif isinstance(raw_rs, dict):
            repo_settings = raw_rs
        else:
            repo_settings = {}

        if config.instance_ids and iid not in config.instance_ids:
            continue

        inputs.append(PipelineInput(
            instance_id=iid,
            level=extracted.get("level", 1),
            task_statement=extracted.get("task_statement", ""),
            interface_descriptions=extracted.get("interface_descriptions", []),
            gold_patch=gold_patch,
            image_name=hf_item.get("image_name", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            repo_settings=repo_settings,
            test_patch=hf_item.get("test_patch"),
            repo=hf_item.get("repo", ""),
            mask_patch=hf_item.get("patch", ""),
        ))

    logger.info(
        f"Loaded {len(inputs)} pipeline inputs (from {len(hf_data)} HF instances, "
        f"{len(extracted_data)} extracted, {len(gt_patches)} ground truth patches) "
        f"— join took {time.time() - t0:.1f}s, total load {time.time() - load_start:.1f}s"
    )
    return inputs
