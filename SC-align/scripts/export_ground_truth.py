#!/usr/bin/env python3
"""Export ground truth patches from the FeatureBench HuggingFace dataset."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def reverse_unified_patch(patch_text: str) -> str:
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
            reversed_lines.append(
                line.replace("new file mode ", "deleted file mode ", 1)
            )
            continue
        if line.startswith("deleted file mode "):
            reversed_lines.append(
                line.replace("deleted file mode ", "new file mode ", 1)
            )
            continue
        if line.startswith("old mode "):
            reversed_lines.append(line.replace("old mode ", "new mode ", 1))
            continue
        if line.startswith("new mode "):
            reversed_lines.append(line.replace("new mode ", "old mode ", 1))
            continue
        if line.startswith("rename from "):
            reversed_lines.append(
                line.replace("rename from ", "rename to ", 1)
            )
            continue
        if line.startswith("rename to "):
            reversed_lines.append(
                line.replace("rename to ", "rename from ", 1)
            )
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


def parse_fail_to_pass_paths(raw_fail_to_pass: Any) -> set[str]:
    """Parse FAIL_TO_PASS field into normalised file-path set."""
    if raw_fail_to_pass is None:
        return set()

    values: Any = raw_fail_to_pass
    if isinstance(raw_fail_to_pass, str):
        try:
            values = json.loads(raw_fail_to_pass)
        except json.JSONDecodeError:
            return set()

    if not isinstance(values, list):
        return set()

    result: set[str] = set()
    for item in values:
        if isinstance(item, str) and item.strip():
            result.add(_normalize_patch_path(item))
    return result


def remove_patch_operations_for_paths(
    patch_text: str, excluded_paths: set[str]
) -> str:
    """Drop diff blocks whose target file matches an excluded path."""
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
            should_drop = (
                path_a in normalized_excluded or path_b in normalized_excluded
            )
        else:
            should_drop = False

        if not should_drop:
            output.extend(block)

    return "".join(output)


def preprocess_hf_patch(patch_text: str, raw_fail_to_pass: Any) -> str:
    """Convert a HF dataset patch into the ground-truth implementation patch."""
    reversed_patch = reverse_unified_patch(patch_text)
    fail_to_pass_paths = parse_fail_to_pass_paths(raw_fail_to_pass)
    return remove_patch_operations_for_paths(reversed_patch, fail_to_pass_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Export FeatureBench ground truth patches to individual files."
    )
    parser.add_argument(
        "--split",
        default="full",
        help="HuggingFace dataset split (default: full)",
    )
    parser.add_argument(
        "--dataset",
        default="LiberCoders/FeatureBench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output-dir",
        default="SC-align/ground_truth",
        help="Output directory, relative to the repo root (default: SC-align/ground_truth)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    for config_name in ("config_eval.toml", "config_mutation.toml", "config.toml"):
        config_path = project_root / config_name
        if not config_path.exists():
            continue
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        env_vars = config.get("env_vars", {})
        if env_vars.get("HF_ENDPOINT"):
            os.environ.setdefault("HF_ENDPOINT", env_vars["HF_ENDPOINT"])
        if env_vars.get("HF_TOKEN"):
            os.environ.setdefault("HF_TOKEN", env_vars["HF_TOKEN"])

    from datasets import load_dataset

    print(f"Loading dataset '{args.dataset}' split='{args.split}' ...")
    ds = load_dataset(
        args.dataset, split=args.split, token=os.environ.get("HF_TOKEN")
    )
    print(f"Loaded {len(ds)} instances.")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped_lv2 = 0
    manifest: list[dict[str, Any]] = []

    for item in ds:
        instance_id = item.get("instance_id", "")
        if not instance_id:
            continue

        if instance_id.endswith(".lv2"):
            skipped_lv2 += 1
            continue

        raw_patch = item.get("patch", "") or ""
        raw_f2p = item.get("FAIL_TO_PASS")
        ground_truth = preprocess_hf_patch(raw_patch, raw_f2p)

        safe_name = instance_id.replace("/", "__")
        filepath = output_dir / f"{safe_name}.diff"
        filepath.write_text(ground_truth, encoding="utf-8")

        manifest.append(
            {
                "instance_id": instance_id,
                "file": f"{safe_name}.diff",
            }
        )
        count += 1

    manifest_path = output_dir / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Exported {count} LV1 ground truth patches to {output_dir}/")
    print(f"Skipped {skipped_lv2} LV2 instances (no ground truth available).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
