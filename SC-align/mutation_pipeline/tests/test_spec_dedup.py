from __future__ import annotations

import sys
from pathlib import Path

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mutation_pipeline.models import (
    AtomicSpec,
    CodeMapping,
    SpecCategory,
    SpecSource,
    StepResult,
)
from mutation_pipeline.pipeline import (
    _dedupe_redundant_specs,
    _minimal_edit,
    _mutation_signature,
)

_FILE = "fastapi/_compat/main.py"

_SHORT_FIND = (
    "    if isinstance(value, may_v1.ModelField):\n"
    "        return True\n"
    "    elif PYDANTIC_V2:"
)
_SHORT_REPLACE = _SHORT_FIND.replace("return True", "return False")

_LONG_FIND = "def _is_model_field(value: Any) -> bool:\n" + _SHORT_FIND
_LONG_REPLACE = "def _is_model_field(value: Any) -> bool:\n" + _SHORT_REPLACE


def _make_spec(
    sid: str,
    function: str = "_is_model_field",
    category: SpecCategory = SpecCategory.BEHAVIORAL,
    text: str | None = None,
) -> AtomicSpec:
    return AtomicSpec(
        spec_id=sid,
        text=text or f"spec text for {sid}",
        category=category,
        sources=[SpecSource.DOCSTRING],
        code_mapping=[CodeMapping(file=_FILE, lines="345-352", function=function)],
    )


def _step2(find: str, replace: str) -> tuple[StepResult, float]:
    return (
        StepResult(
            success=True,
            data={"mutations": [{"file": _FILE, "find": find, "replace": replace}]},
            step_name="mutation_generation",
        ),
        1.0,
    )


def test_minimal_edit_strips_anchor_of_any_length() -> None:
    assert _minimal_edit(_SHORT_FIND, _SHORT_REPLACE) == _minimal_edit(
        _LONG_FIND, _LONG_REPLACE
    )


def test_mutation_signature_ignores_anchor_length() -> None:
    short = _mutation_signature(
        [{"file": _FILE, "find": _SHORT_FIND, "replace": _SHORT_REPLACE}]
    )
    long = _mutation_signature(
        [{"file": _FILE, "find": _LONG_FIND, "replace": _LONG_REPLACE}]
    )
    assert short == long


def test_identical_edit_on_same_function_merges() -> None:
    specs = [_make_spec("S10"), _make_spec("S11"), _make_spec("S12")]
    step2 = {
        "S10": _step2(_LONG_FIND, _LONG_REPLACE),
        "S11": _step2(_SHORT_FIND, _SHORT_REPLACE),
        "S12": _step2(_SHORT_FIND, _SHORT_REPLACE),
    }

    rewritten, groups = _dedupe_redundant_specs(specs, step2)

    assert groups == [["S10", "S11", "S12"]]
    assert [s.spec_id for s in rewritten] == ["S10"]

    merged = rewritten[0]
    assert merged.merged_from == ["S10", "S11", "S12"]
    for sid in ("S10", "S11", "S12"):
        assert f"spec text for {sid}" in merged.text


def test_same_edit_in_different_functions_is_not_merged() -> None:
    specs = [
        _make_spec("S1", function="_is_model_field"),
        _make_spec("S2", function="_is_model_instance"),
    ]
    step2 = {
        "S1": _step2(_SHORT_FIND, _SHORT_REPLACE),
        "S2": _step2(_SHORT_FIND, _SHORT_REPLACE),
    }

    rewritten, groups = _dedupe_redundant_specs(specs, step2)

    assert groups == []
    assert [s.spec_id for s in rewritten] == ["S1", "S2"]


def test_distinct_edits_are_not_merged() -> None:
    specs = [_make_spec("S7", function="_get_model_config"),
             _make_spec("S8", function="_get_model_config")]
    step2 = {
        "S7": _step2("return v1._get_model_config(model)", "return None"),
        "S8": _step2("return v1._get_model_config(model)",
                     "return v2._get_model_config(model)"),
    }

    rewritten, groups = _dedupe_redundant_specs(specs, step2)

    assert groups == []
    assert [s.spec_id for s in rewritten] == ["S7", "S8"]


def test_failed_step2_specs_are_never_merged() -> None:
    specs = [_make_spec("S3"), _make_spec("S4")]
    failed = (
        StepResult(success=False, error="empty mutations",
                   step_name="mutation_generation"),
        0.0,
    )
    step2 = {"S3": failed, "S4": failed}

    rewritten, groups = _dedupe_redundant_specs(specs, step2)

    assert groups == []
    assert [s.spec_id for s in rewritten] == ["S3", "S4"]


def test_merge_keeps_own_id_alongside_model_supplied_labels() -> None:
    first = _make_spec("S1")
    first.merged_from = ["S1a-signature", "S1b-returns-list"]
    specs = [first, _make_spec("S2")]
    step2 = {
        "S1": _step2(_LONG_FIND, _LONG_REPLACE),
        "S2": _step2(_SHORT_FIND, _SHORT_REPLACE),
    }

    rewritten, _ = _dedupe_redundant_specs(specs, step2)

    merged = rewritten[0]
    assert merged.spec_id == "S1"
    assert merged.merged_from == ["S1", "S1a-signature", "S1b-returns-list", "S2"]
    assert merged.spec_id in merged.merged_from


def test_merge_drops_independence_claims_the_merge_refuted() -> None:
    first = _make_spec("S10")
    first.independence_check = {
        "S11": ["change A violates S10 only", "change B violates S11 only"],
        "S1": ["change C violates S10 only", "change D violates S1 only"],
    }
    second = _make_spec("S11")
    second.independence_check = {
        "S10": ["change A violates S10 only", "change B violates S11 only"],
    }
    step2 = {
        "S10": _step2(_LONG_FIND, _LONG_REPLACE),
        "S11": _step2(_SHORT_FIND, _SHORT_REPLACE),
    }

    rewritten, _ = _dedupe_redundant_specs([first, second], step2)

    merged = rewritten[0]
    assert "S11" not in merged.independence_check
    assert "S10" not in merged.independence_check
    assert merged.independence_check == {
        "S1": ["change C violates S10 only", "change D violates S1 only"]
    }


def test_merged_category_takes_the_strongest_constituent() -> None:
    specs = [
        _make_spec("S9", category=SpecCategory.STRUCTURAL),
        _make_spec("S10", category=SpecCategory.BEHAVIORAL),
    ]
    step2 = {
        "S9": _step2(_LONG_FIND, _LONG_REPLACE),
        "S10": _step2(_SHORT_FIND, _SHORT_REPLACE),
    }

    rewritten, _ = _dedupe_redundant_specs(specs, step2)

    assert rewritten[0].category == SpecCategory.BEHAVIORAL
