#!/usr/bin/env python3
"""Drop specs with no negative control from an existing ``output.jsonl``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mutation_pipeline.control_filter import filter_uncontrolled_specs
from mutation_pipeline.models import MutationResult


def load_results(path: Path) -> List[MutationResult]:
    results: List[MutationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(MutationResult.from_dict(json.loads(line)))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: malformed JSON: {e}")
    return results


def write_results(path: Path, results: List[MutationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.to_jsonl() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drop surviving specs that no other surviving mutant probes, so "
            "eval never scores a certificate with an empty control arm."
        )
    )
    parser.add_argument("path", type=Path, help="output.jsonl to filter")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the filtered rows here instead of reporting only",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite PATH with the filtered rows",
    )
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--in-place and --output are mutually exclusive")
    if not args.path.exists():
        parser.error(f"no such file: {args.path}")

    results = load_results(args.path)
    kept, dropped = filter_uncontrolled_specs(results)

    by_instance: Dict[str, List[str]] = defaultdict(list)
    for result in dropped:
        by_instance[result.instance_id].append(result.spec.spec_id)

    for instance_id in sorted(by_instance):
        total = sum(1 for r in results if r.instance_id == instance_id)
        spec_ids = by_instance[instance_id]
        print(
            f"{instance_id}: dropping {len(spec_ids)}/{total} "
            f"({', '.join(sorted(spec_ids))})"
        )
    print(f"{args.path}: {len(results)} rows -> {len(kept)} eval-ready, {len(dropped)} dropped")

    if results and not kept:
        print(
            "Refusing to write: every row would be dropped, which means this "
            "run carries no preservation-probe records at all. Re-run the "
            "mutation pipeline with Step 5 enabled instead.",
            file=sys.stderr,
        )
        return 1

    destination = args.path if args.in_place else args.output
    if destination is None:
        print("(reported only; pass --in-place or --output to write)")
        return 0

    write_results(destination, kept)
    print(f"Wrote {len(kept)} rows to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
