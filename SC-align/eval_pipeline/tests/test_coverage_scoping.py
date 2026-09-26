from typing import Any, Dict

import pytest

from eval_pipeline.coverage_scoping import (
    FunctionInfo,
    aggregate_gold_patch_coverage,
    extend_scope_via_call_chain,
    extract_gold_patch_functions,
    scope_to_function,
    spec_breadth,
)


def _make_coverage(files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "branch_coverage_pct": 0.0,
        "combined_coverage_pct": 0.0,
        "num_files": len(files),
        "files": files,
    }


def test_scope_to_function_uses_per_function_block():
    cov = _make_coverage({
        "/testbed/fastapi/_compat/v1.py": {
            "executed_lines": [310, 312],
            "missing_lines": [311],
            "executed_branches": [],
            "missing_branches": [],
            "functions": {
                "get_model_fields": {
                    "executed_lines": [312],
                    "missing_lines": [],
                    "executed_branches": [],
                    "missing_branches": [],
                    "summary": {
                        "num_statements": 1,
                        "covered_lines": 1,
                        "num_branches": 0,
                        "covered_branches": 0,
                    },
                },
                "is_pv1_scalar_field": {
                    "executed_lines": [],
                    "missing_lines": [190, 192, 193, 201, 202],
                    "executed_branches": [],
                    "missing_branches": [[193, 201], [193, 194], [202, 203]],
                    "summary": {
                        "num_statements": 8,
                        "covered_lines": 0,
                        "num_branches": 6,
                        "covered_branches": 0,
                    },
                },
            },
        }
    })
    r = scope_to_function(cov, "fastapi/_compat/v1.py", "get_model_fields")
    assert r is not None
    assert r["source"] == "coverage_functions"
    assert r["covered_lines"] == 1
    assert r["num_statements"] == 1
    assert r["line_coverage_pct"] == 100.0
    assert r["branch_coverage_pct"] is None
    assert r["edge_coverage_pct"] == 100.0
    assert r["covered_edges"] == 1
    assert r["num_edges"] == 1

    r2 = scope_to_function(cov, "fastapi/_compat/v1.py", "is_pv1_scalar_field")
    assert r2["covered_branches"] == 0
    assert r2["num_branches"] == 6
    assert r2["branch_coverage_pct"] == 0.0
    assert r2["covered_edges"] == 0
    assert r2["num_edges"] == 3
    assert r2["edge_coverage_pct"] == 0.0
    assert r2["missing_branches"] == [[193, 201], [193, 194], [202, 203]]


def test_scope_to_function_suffix_match():
    cov = _make_coverage({
        "/testbed/fastapi/_compat/v1.py": {
            "executed_lines": [],
            "missing_lines": [],
            "executed_branches": [],
            "missing_branches": [],
            "functions": {
                "get_model_fields": {
                    "summary": {"num_statements": 1, "covered_lines": 1,
                                "num_branches": 0, "covered_branches": 0},
                    "executed_lines": [312], "missing_lines": [],
                    "executed_branches": [], "missing_branches": [],
                },
            },
        }
    })
    r = scope_to_function(cov, "fastapi/_compat/v1.py", "get_model_fields")
    assert r is not None and r["covered_lines"] == 1


def _one_function_file(name: str = "get_model_fields") -> Dict[str, Any]:
    return {
        "executed_lines": [], "missing_lines": [],
        "executed_branches": [], "missing_branches": [],
        "functions": {
            name: {
                "summary": {"num_statements": 1, "covered_lines": 1,
                            "num_branches": 0, "covered_branches": 0},
                "executed_lines": [312], "missing_lines": [],
                "executed_branches": [], "missing_branches": [],
            },
        },
    }


def test_scope_to_function_absolute_target_relative_key():
    cov = _make_coverage({"fastapi/_compat/v1.py": _one_function_file()})
    r = scope_to_function(cov, "/testbed/fastapi/_compat/v1.py", "get_model_fields")
    assert r is not None and r["source"] == "coverage_functions"
    assert r["covered_lines"] == 1


def test_scope_to_function_both_sides_absolute():
    cov = _make_coverage({"/testbed/fastapi/_compat/v1.py": _one_function_file()})
    r = scope_to_function(cov, "/testbed/fastapi/_compat/v1.py", "get_model_fields")
    assert r is not None and r["source"] == "coverage_functions"


def test_scope_to_function_never_falls_through_to_root_level_file():
    cov = _make_coverage({"conftest.py": _one_function_file("helper")})
    r = scope_to_function(cov, "/testbed/pkg/conftest.py", "helper")
    assert r is not None and r["source"] == "not_found"


def test_scope_to_function_reports_resolved_qualified_name():
    block = {
        "summary": {
            "num_statements": 1,
            "covered_lines": 1,
            "num_branches": 0,
            "covered_branches": 0,
        },
        "executed_lines": [3],
        "missing_lines": [],
        "executed_branches": [],
        "missing_branches": [],
    }
    cov = _make_coverage({
        "/testbed/pkg/m.py": {
            "functions": {"Model.validate": block},
            "executed_lines": [3],
            "missing_lines": [],
            "executed_branches": [],
            "missing_branches": [],
        }
    })

    result = scope_to_function(cov, "pkg/m.py", "validate")

    assert result is not None
    assert result["function"] == "Model.validate"


def test_scope_to_function_file_not_found():
    cov = _make_coverage({})
    r = scope_to_function(cov, "fastapi/missing.py", "foo")
    assert r is not None
    assert r["source"] == "not_found"
    assert r["num_statements"] == 0
    assert r["branch_coverage_pct"] is None
    assert r["edge_coverage_pct"] is None


def test_scope_to_function_returns_none_for_none_coverage():
    assert scope_to_function(None, "x.py", "f") is None


def test_scope_to_function_raises_without_functions_block():
    cov = _make_coverage({
        "/testbed/pkg/m.py": {
            "executed_lines": [1, 2, 3],
            "missing_lines": [4],
            "executed_branches": [[2, 3]],
            "missing_branches": [[2, 4]],
        }
    })
    with pytest.raises(RuntimeError, match="no per-file 'functions' block"):
        scope_to_function(cov, "pkg/m.py", "alpha")


_SYNTHETIC_GOLD = """\
diff --git a/pkg/m.py b/pkg/m.py
index abc..def 100644
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -10,6 +10,12 @@ existing code
 untouched line
+def newly_added(x):
+    return x + 1
+
+
+def another_new():
+    return 42
 def existing_function(y):
     # modified body (body changed but def line not added)
     return y
"""


def test_extract_gold_patch_functions_synthetic():
    funcs = extract_gold_patch_functions(_SYNTHETIC_GOLD)
    assert len(funcs) == 2
    names = sorted(f.name for f in funcs)
    assert names == ["another_new", "newly_added"]
    for f in funcs:
        assert f.file == "pkg/m.py"


def test_extract_gold_patch_functions_with_pre_image(tmp_path):
    pre = tmp_path / "pkg"
    pre.mkdir()
    (pre / "m.py").write_text("def existing_function(y):\n    return y\n")
    patch = """\
diff --git a/pkg/m.py b/pkg/m.py
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -1,2 +1,5 @@
+def existing_function(y):
+    return y * 2
+def truly_new(z):
+    return z
"""
    funcs = extract_gold_patch_functions(patch, pre_image_root=str(tmp_path))
    names = sorted(f.name for f in funcs)
    assert names == ["truly_new"], (
        "existing_function should be filtered as it pre-existed"
    )


def test_extract_gold_patch_functions_empty():
    assert extract_gold_patch_functions("") == []


def test_extract_gold_patch_functions_preserves_method_qualnames():
    patch = """\
diff --git a/pkg/m.py b/pkg/m.py
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -1,2 +1,8 @@
+class A:
+    def helper(self):
+        return 1
+
+class B:
+    def helper(self):
+        return 2
"""

    funcs = extract_gold_patch_functions(patch)

    assert [item.key() for item in funcs] == [
        ("pkg/m.py", "A.helper"),
        ("pkg/m.py", "B.helper"),
    ]


def test_extract_gold_patch_functions_uses_hunk_class_context():
    patch = """\
diff --git a/pkg/m.py b/pkg/m.py
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -10,2 +10,5 @@ class Existing:
+    def added(self):
+        return 1
"""

    funcs = extract_gold_patch_functions(patch)

    assert [item.key() for item in funcs] == [("pkg/m.py", "Existing.added")]


def test_extend_scope_uses_file_and_qualified_name(tmp_path):
    source = """\
class A:
    def entry(self):
        return self.helper()

    def helper(self):
        return 1

class B:
    def helper(self):
        return 2
"""
    (tmp_path / "a.py").write_text(source)

    scope = extend_scope_via_call_chain(
        declared=[FunctionInfo("a.py", "A.entry")],
        gold_patch_funcs=[
            FunctionInfo("a.py", "A.helper"),
            FunctionInfo("a.py", "B.helper"),
        ],
        testbed_root=str(tmp_path),
    )

    assert [item.key() for item in scope] == [
        ("a.py", "A.entry"),
        ("a.py", "A.helper"),
    ]


def test_extend_scope_canonicalizes_unique_bare_method_name(tmp_path):
    (tmp_path / "a.py").write_text(
        "class A:\n"
        "    def entry(self):\n"
        "        return 1\n"
    )

    scope = extend_scope_via_call_chain(
        declared=[FunctionInfo("a.py", "entry")],
        gold_patch_funcs=[],
        testbed_root=str(tmp_path),
    )

    assert [item.key() for item in scope] == [("a.py", "A.entry")]


def test_extend_scope_does_not_conflate_same_name_across_files(tmp_path):
    (tmp_path / "a.py").write_text(
        "def entry():\n"
        "    return helper()\n"
        "def helper():\n"
        "    return 1\n"
    )
    (tmp_path / "b.py").write_text("def helper():\n    return 2\n")

    scope = extend_scope_via_call_chain(
        declared=[FunctionInfo("a.py", "entry")],
        gold_patch_funcs=[
            FunctionInfo("a.py", "helper"),
            FunctionInfo("b.py", "helper"),
        ],
        testbed_root=str(tmp_path),
    )

    assert [item.key() for item in scope] == [
        ("a.py", "entry"),
        ("a.py", "helper"),
    ]


def test_spec_breadth_basic():
    gold = [
        FunctionInfo("a.py", "f1"),
        FunctionInfo("a.py", "f2"),
        FunctionInfo("b.py", "g1"),
    ]
    targeted = [("a.py", "f1"), ("b.py", "g1")]
    r = spec_breadth(gold, targeted)
    assert r["gold_patch_functions_total"] == 3
    assert r["gold_patch_functions_targeted"] == 2
    assert abs(r["ratio"] - 2 / 3) < 1e-9
    assert r["missing"] == [{"file": "a.py", "name": "f2"}]


def test_spec_breadth_invariant_monotone():
    gold = [FunctionInfo("a.py", "f1"), FunctionInfo("a.py", "f2")]
    r1 = spec_breadth(gold, [("a.py", "f1")])
    r2 = spec_breadth(gold, [("a.py", "f1"), ("a.py", "f1")])
    assert r1["ratio"] == r2["ratio"]
    r3 = spec_breadth(gold, [("a.py", "f1"), ("a.py", "f2")])
    assert r3["ratio"] > r1["ratio"]


def test_spec_breadth_compares_paths_relative_to_testbed():
    gold = [FunctionInfo("pkg/a.py", "f1"), FunctionInfo("pkg/a.py", "f2")]
    r = spec_breadth(gold, [("/testbed/pkg/a.py", "f1")])
    assert r["gold_patch_functions_targeted"] == 1
    assert r["ratio"] == 0.5
    assert r["missing"] == [{"file": "pkg/a.py", "name": "f2"}]
    assert r["targeted_but_not_gold"] == []


def test_spec_breadth_targeted_not_in_gold():
    gold = [FunctionInfo("a.py", "f1")]
    targeted = [("a.py", "f1"), ("a.py", "outside")]
    r = spec_breadth(gold, targeted)
    assert r["ratio"] == 1.0
    assert {"file": "a.py", "name": "outside"} in r["targeted_but_not_gold"]


def _trial_with_function(file: str, name: str, fblock: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "coverage": _make_coverage({
            f"/testbed/{file}": {
                "executed_lines": fblock.get("executed_lines", []),
                "missing_lines": fblock.get("missing_lines", []),
                "executed_branches": fblock.get("executed_branches", []),
                "missing_branches": fblock.get("missing_branches", []),
                "functions": {name: fblock},
            }
        })
    }


def test_aggregate_single_cert_matches_scope():
    gold = [FunctionInfo("a.py", "f1")]
    fblock = {
        "executed_lines": [10, 11],
        "missing_lines": [12],
        "executed_branches": [[10, 11]],
        "missing_branches": [[10, 12]],
        "summary": {
            "num_statements": 3, "covered_lines": 2,
            "num_branches": 2, "covered_branches": 1,
        },
    }
    trials = [_trial_with_function("a.py", "f1", fblock)]
    r = aggregate_gold_patch_coverage(trials, gold)
    assert r["covered_edges"] == 1
    assert r["total_edges"] == 2
    assert r["edge_coverage_pct"] == 50.0
    assert r["covered_branches"] == 1
    assert r["total_branches"] == 2
    assert r["branch_coverage_pct"] == 50.0


def test_aggregate_union_across_trials():
    gold = [FunctionInfo("a.py", "f1")]
    t1_block = {
        "executed_lines": [10, 11],
        "missing_lines": [12],
        "executed_branches": [[10, 11]],
        "missing_branches": [[10, 12]],
        "summary": {"num_statements": 3, "covered_lines": 2,
                    "num_branches": 2, "covered_branches": 1},
    }
    t2_block = {
        "executed_lines": [10, 12],
        "missing_lines": [11],
        "executed_branches": [[10, 12]],
        "missing_branches": [[10, 11]],
        "summary": {"num_statements": 3, "covered_lines": 2,
                    "num_branches": 2, "covered_branches": 1},
    }
    trials = [
        _trial_with_function("a.py", "f1", t1_block),
        _trial_with_function("a.py", "f1", t2_block),
    ]
    r = aggregate_gold_patch_coverage(trials, gold)
    assert r["covered_edges"] == 2
    assert r["total_edges"] == 2
    assert r["edge_coverage_pct"] == 100.0
    assert r["covered_branches"] == 2
    assert r["total_branches"] == 2
    assert r["branch_coverage_pct"] == 100.0


def test_aggregate_untargeted_function_penalizes_denominator():
    gold = [
        FunctionInfo("a.py", "covered"),
        FunctionInfo("a.py", "uncovered"),
    ]
    covered_block = {
        "executed_branches": [[1, 2]],
        "missing_branches": [],
        "summary": {"num_statements": 1, "covered_lines": 1,
                    "num_branches": 1, "covered_branches": 1},
        "executed_lines": [1], "missing_lines": [],
    }
    uncovered_block = {
        "executed_branches": [],
        "missing_branches": [[10, 11], [10, 12]],
        "summary": {"num_statements": 3, "covered_lines": 0,
                    "num_branches": 2, "covered_branches": 0},
        "executed_lines": [], "missing_lines": [10, 11, 12],
    }
    coverage = _make_coverage({
        "/testbed/a.py": {
            "executed_lines": [1],
            "missing_lines": [10, 11, 12],
            "executed_branches": [[1, 2]],
            "missing_branches": [[10, 11], [10, 12]],
            "functions": {
                "covered": covered_block,
                "uncovered": uncovered_block,
            },
        }
    })
    trial = {"coverage": coverage}
    r = aggregate_gold_patch_coverage([trial], gold)
    assert r["covered_edges"] == 1
    assert r["total_edges"] == 3
    assert abs(r["edge_coverage_pct"] - 100 / 3) < 1e-9
    assert r["covered_branches"] == 1
    assert r["total_branches"] == 3
    assert abs(r["branch_coverage_pct"] - 100 / 3) < 1e-9


def test_aggregate_empty_gold_funcs():
    r = aggregate_gold_patch_coverage([], [])
    assert r["edge_coverage_pct"] is None
    assert r["branch_coverage_pct"] is None
    assert r["line_coverage_pct"] is None
    assert r["covered_edges"] == 0
    assert r["total_edges"] == 0
    assert r["covered_branches"] == 0
    assert r["total_branches"] == 0


def test_aggregate_branchless_function_has_edge_coverage():
    gold = [FunctionInfo("a.py", "f")]
    fblock = {
        "executed_lines": [10, 11],
        "missing_lines": [],
        "executed_branches": [],
        "missing_branches": [],
        "summary": {
            "num_statements": 2,
            "covered_lines": 2,
            "num_branches": 0,
            "covered_branches": 0,
        },
    }
    r = aggregate_gold_patch_coverage(
        [_trial_with_function("a.py", "f", fblock)],
        gold,
    )
    assert r["branch_coverage_pct"] is None
    assert r["covered_edges"] == 2
    assert r["total_edges"] == 2
    assert r["edge_coverage_pct"] == 100.0


def test_scope_to_function_idempotent():
    cov = _make_coverage({
        "/testbed/a.py": {
            "executed_lines": [1], "missing_lines": [],
            "executed_branches": [], "missing_branches": [],
            "functions": {
                "f": {
                    "summary": {"num_statements": 1, "covered_lines": 1,
                                "num_branches": 0, "covered_branches": 0},
                    "executed_lines": [1], "missing_lines": [],
                    "executed_branches": [], "missing_branches": [],
                }
            },
        }
    })
    r1 = scope_to_function(cov, "a.py", "f")
    r2 = scope_to_function(cov, "a.py", "f")
    assert r1 == r2
