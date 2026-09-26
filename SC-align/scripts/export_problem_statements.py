#!/usr/bin/env python3
"""Export problem statements from the FeatureBench HuggingFace dataset."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export FeatureBench problem statements to individual files.")
    parser.add_argument("--split", default="full", help="HuggingFace dataset split (default: full)")
    parser.add_argument("--dataset", default="LiberCoders/FeatureBench", help="HuggingFace dataset name")
    parser.add_argument("--output-dir", default="SC-align/problem_statements", help="Output directory, relative to the repo root (default: SC-align/problem_statements)")
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
    ds = load_dataset(args.dataset, split=args.split, token=os.environ.get("HF_TOKEN"))
    print(f"Loaded {len(ds)} instances.")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in ds:
        instance_id = item.get("instance_id", "")
        problem_statement = item.get("problem_statement", "")
        if not instance_id:
            continue

        safe_name = instance_id.replace("/", "__")
        filepath = output_dir / f"{safe_name}.md"
        filepath.write_text(problem_statement, encoding="utf-8")
        count += 1

    print(f"Exported {count} problem statements to {output_dir}/")


if __name__ == "__main__":
    main()
