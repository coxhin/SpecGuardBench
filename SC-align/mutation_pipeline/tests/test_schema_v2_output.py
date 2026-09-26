from __future__ import annotations

import json
import sys
from pathlib import Path

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

stubbed_models = sys.modules.get("mutation_pipeline.models")
if stubbed_models is not None and not hasattr(stubbed_models, "MutationResult"):
    sys.modules.pop("mutation_pipeline.models", None)
    sys.modules.pop("mutation_pipeline", None)

stubbed_runner = sys.modules.get("mutation_pipeline.docker_runner")
if stubbed_runner is not None and not hasattr(stubbed_runner, "_fix_hunk_headers"):
    sys.modules.pop("mutation_pipeline.docker_runner", None)

stubbed_constants = sys.modules.get("featurebench.harness.constants")
if stubbed_constants is not None and not hasattr(stubbed_constants, "DOCKER_USER"):
    for name in [
        "featurebench",
        "featurebench.harness",
        "featurebench.harness.constants",
        "featurebench.harness.container",
        "featurebench.harness.utils",
    ]:
        sys.modules.pop(name, None)

from mutation_pipeline.models import AtomicSpec, MutationResult, Proof, SpecCategory, SpecSource
from mutation_pipeline.docker_runner import _fix_hunk_headers


def _make_spec() -> AtomicSpec:
    return AtomicSpec(
        spec_id="S1",
        text="example spec",
        category=SpecCategory.BEHAVIORAL,
        sources=[SpecSource.DOCSTRING],
        code_mapping=[],
    )


def test_mutation_result_serializes_schema_v2_patch_fields() -> None:
    result = MutationResult(
        schema_version=2,
        instance_id="example.instance.lv1",
        spec=_make_spec(),
        mask_patch="mask.patch",
        gold_patch="gold.patch",
        mutant_patch="mutant.patch",
        test_patch="test.patch",
        proof=Proof(),
        metadata={"mutations": [{"file": "x.py"}]},
    )

    payload = result.to_dict()
    assert payload["schema_version"] == 2
    assert payload["mask_patch"] == "mask.patch"
    assert payload["gold_patch"] == "gold.patch"
    assert payload["mutant_patch"] == "mutant.patch"
    assert payload["test_patch"] == "test.patch"
    assert "aligned_code" not in payload
    assert "misaligned_code" not in payload
    assert "mutation_diff" not in payload

    round_tripped = MutationResult.from_dict(json.loads(result.to_jsonl()))
    assert round_tripped.schema_version == 2
    assert round_tripped.mask_patch == "mask.patch"
    assert round_tripped.gold_patch == "gold.patch"
    assert round_tripped.mutant_patch == "mutant.patch"
    assert round_tripped.test_patch == "test.patch"


def test_fix_hunk_headers_does_not_count_no_newline_marker() -> None:
    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -359,4 +359,4 @@ def _is_model_class(value: Any) -> bool:\n"
        "         from . import v2\n"
        " \n"
        "         return lenient_issubclass(value, v2.BaseModel)  # type: ignore[attr-defined]\n"
        "-    return False\n"
        "+    return False\n"
        "\\ No newline at end of file\n"
    )

    fixed = _fix_hunk_headers(patch)

    assert "@@ -359,4 +359,4 @@ def _is_model_class(value: Any) -> bool:" in fixed
    assert "@@ -359,5 +359,5 @@ def _is_model_class(value: Any) -> bool:" not in fixed
    assert "\\ No newline at end of file\n" in fixed
