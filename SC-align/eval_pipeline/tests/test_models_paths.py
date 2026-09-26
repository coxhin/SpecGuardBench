import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_pipeline.models import SurvivingSpec, canonical_testbed_path


def test_canonical_testbed_path():
    assert canonical_testbed_path("/testbed/pkg/mod.py") == "pkg/mod.py"
    assert canonical_testbed_path("pkg/mod.py") == "pkg/mod.py"
    assert canonical_testbed_path("/pkg/mod.py") == "pkg/mod.py"
    assert canonical_testbed_path("/other/root/pkg/mod.py") == "other/root/pkg/mod.py"
    assert canonical_testbed_path("") == ""


def test_surviving_spec_normalises_code_mapping_file_without_mutating_input():
    raw = {
        "spec_id": "S1",
        "code_mapping": [
            {"file": "/testbed/pkg/mod.py", "function": "f", "lines": [1, 2]},
            {"file": "pkg/other.py", "function": "g"},
        ],
    }
    spec = SurvivingSpec(spec=raw, mutant_patch="diff --git a/x b/x")
    assert [cm["file"] for cm in spec.spec["code_mapping"]] == ["pkg/mod.py", "pkg/other.py"]
    assert spec.spec["code_mapping"][0]["lines"] == [1, 2]
    assert raw["code_mapping"][0]["file"] == "/testbed/pkg/mod.py"


def test_surviving_spec_round_trip_is_idempotent():
    raw = {"spec_id": "S1", "code_mapping": [{"file": "/testbed/pkg/mod.py", "function": "f"}]}
    once = SurvivingSpec(spec=raw, mutant_patch="x")
    twice = SurvivingSpec.from_dict(once.to_dict())
    assert twice.spec["code_mapping"] == once.spec["code_mapping"]


def test_surviving_spec_tolerates_missing_or_odd_code_mapping():
    assert SurvivingSpec(spec={"spec_id": "S1"}, mutant_patch="x").spec == {"spec_id": "S1"}
    s = SurvivingSpec(spec={"spec_id": "S1", "code_mapping": ["not-a-dict", {"function": "f"}]}, mutant_patch="x")
    assert s.spec["code_mapping"] == ["not-a-dict", {"function": "f"}]
