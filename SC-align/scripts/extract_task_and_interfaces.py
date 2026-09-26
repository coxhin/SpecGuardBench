#!/usr/bin/env python3
"""Extract task_statement and interface descriptions from rendered problem statements.
"""

import argparse
import json
import re
from pathlib import Path


def extract_task_statement(text: str) -> str:
    """Extract the task statement: content between '## Task' and '**NOTE**:'."""
    m = re.search(r"^## Task\n(.*?)(?=\n\*\*NOTE\*\*:)", text, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def _find_outer_code_block(section: str):
    """Find the outermost ```python ..."""
    open_match = re.search(r"```python\n", section)
    if not open_match:
        return None

    code_start = open_match.end()
    block_start = open_match.start()

    depth = 1
    pos = code_start
    fence_re = re.compile(r"^(`{3,})(.*)", re.MULTILINE)

    for m in fence_re.finditer(section[code_start:]):
        backticks = m.group(1)
        after = m.group(2).strip()
        abs_pos = code_start + m.start()

        if after:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                code_end = abs_pos
                block_end = abs_pos + len(backticks)
                code_content = section[code_start:code_end].rstrip("\n")
                return code_content, block_start, block_end

    code_content = section[code_start:].rstrip("\n")
    return code_content, block_start, len(section)


def extract_interface_descriptions(text: str) -> list:
    """Extract all ### Interface Description N sections."""
    results = []

    pattern = r"^### Interface Description (\S+)\s*\n"
    headers = list(re.finditer(pattern, text, re.MULTILINE))

    if not headers:
        return results

    sentinel_pattern = r"\nRemember, \*\*the interface template above is extremely important\*\*"
    sentinel_match = re.search(sentinel_pattern, text)
    sentinel_pos = sentinel_match.start() if sentinel_match else len(text)

    for i, header in enumerate(headers):
        number = header.group(1)
        start = header.end()
        if i + 1 < len(headers):
            end = headers[i + 1].start()
        else:
            end = sentinel_pos

        section = text[start:end]

        entry = {"number": number, "path": None, "description": "", "code": "", "additional_info": ""}

        path_match = re.search(r"^Path:\s*`([^`]+)`", section, re.MULTILINE)
        if path_match:
            entry["path"] = path_match.group(1)

        block = _find_outer_code_block(section)
        if block:
            code_content, block_start, block_end = block
            entry["code"] = code_content

            pre_code = section[:block_start].strip()
            pre_code = re.sub(r"^Below is \*\*Interface Description \S+\*\*\s*", "", pre_code).strip()
            if path_match:
                pre_code = pre_code.replace(f"Path: `{entry['path']}`", "").strip()
            entry["description"] = pre_code

            post_code = section[block_end:].strip()
            if post_code:
                entry["additional_info"] = post_code
        else:
            entry["description"] = section.strip()

        results.append(entry)

    return results


def process_file(filepath: Path) -> dict:
    """Process a single problem statement file."""
    text = filepath.read_text(encoding="utf-8")
    instance_id = filepath.stem

    if instance_id.endswith(".lv1"):
        level = 1
    elif instance_id.endswith(".lv2"):
        level = 2
    else:
        level = None

    task_statement = extract_task_statement(text)
    interfaces = extract_interface_descriptions(text)

    return {
        "instance_id": instance_id,
        "level": level,
        "task_statement": task_statement,
        "interface_descriptions": interfaces,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract task_statement and interface descriptions from problem statement Markdown files."
    )
    parser.add_argument(
        "--input-dir",
        default="SC-align/problem_statements",
        help="Directory of .md problem statements, relative to the repo root (default: SC-align/problem_statements)",
    )
    parser.add_argument(
        "--output-dir",
        default="SC-align/extracted_v2",
        help="Output directory for per-instance JSON files (default: SC-align/extracted_v2)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return

    all_results = []
    for fp in md_files:
        result = process_file(fp)
        all_results.append(result)

        out_path = output_dir / f"{result['instance_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    combined_path = output_dir / "_all.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(all_results)} instances to {output_dir}/")
    print(f"Combined JSON: {combined_path}")

    with_task = sum(1 for r in all_results if r["task_statement"])
    with_iface = sum(1 for r in all_results if r["interface_descriptions"])
    total_ifaces = sum(len(r["interface_descriptions"]) for r in all_results)
    with_ai = sum(1 for r in all_results if any(i["additional_info"].strip() for i in r["interface_descriptions"]))
    total_ai = sum(1 for r in all_results for i in r["interface_descriptions"] if i["additional_info"].strip())
    print(f"  - {with_task}/{len(all_results)} have task_statement")
    print(f"  - {with_iface}/{len(all_results)} have interface_descriptions ({total_ifaces} total)")
    print(f"  - {with_ai}/{len(all_results)} have additional_info ({total_ai} interface sections total)")


if __name__ == "__main__":
    main()
