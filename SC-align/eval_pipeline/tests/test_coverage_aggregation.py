from eval_pipeline.models import ProofExecResult
from eval_pipeline.patch_utils import fold_lines_to_statements, mutant_line_hit_rate, parse_mutant_lines
from eval_pipeline.stages.stage5_metrics import _aggregate_mutant_lines


def _make_pr(spec_id, target_state, mutant_hit_rate=None, files=None):
    cov = None
    if mutant_hit_rate is not None or files is not None:
        cov = {"files": files or {}}
        if mutant_hit_rate is not None:
            cov["mutant_lines"] = {
                "hit_rate": mutant_hit_rate,
                "total": 10,
                "hit": int(mutant_hit_rate * 10),
                "detail": {},
            }
    return ProofExecResult(
        instance_id="iid",
        baseline="claude_code",
        model="opus",
        spec_id=spec_id,
        target_state=target_state,
        triggered=False,
        exit_code=0,
        kind="passed",
        proof_output="",
        execution_time_seconds=1.0,
        coverage=cov,
    )


def test_coverage_roundtrip_present():
    pr = _make_pr("S1", "gold", mutant_hit_rate=0.75)
    d = pr.to_dict()
    assert "coverage" in d
    assert d["coverage"]["mutant_lines"]["hit_rate"] == 0.75
    pr2 = ProofExecResult.from_dict(d)
    assert pr2.coverage is not None
    assert pr2.coverage["mutant_lines"]["hit_rate"] == 0.75


def test_coverage_roundtrip_absent():
    pr = _make_pr("S1", "gold")
    d = pr.to_dict()
    assert "coverage" not in d
    pr2 = ProofExecResult.from_dict(d)
    assert pr2.coverage is None


def test_aggregate_mean_per_state():
    prs = [
        _make_pr("S1", "gold", 1.0),
        _make_pr("S1", "mutant", 0.5),
        _make_pr("S2", "gold", 0.6),
        _make_pr("S2", "mutant", 0.0),
    ]
    agg = _aggregate_mutant_lines(prs)
    assert agg["hit_rate_mean"] == (1.0 + 0.5 + 0.6 + 0.0) / 4
    assert agg["hit_rate_mean_gold"] == (1.0 + 0.6) / 2
    assert agg["hit_rate_mean_mutant"] == (0.5 + 0.0) / 2
    assert agg["per_spec"]["iid::claude_code_opus::S1"]["gold"] == 1.0
    assert agg["per_spec"]["iid::claude_code_opus::S1"]["mutant"] == 0.5
    assert agg["counts"]["trials_with_mutant_lines_block"] == 4


def test_aggregate_excludes_none_coverage():
    prs = [
        _make_pr("S1", "gold", 0.8),
        _make_pr("S1", "mutant"),
        _make_pr("S2", "gold", 0.4),
    ]
    agg = _aggregate_mutant_lines(prs)
    assert agg["hit_rate_mean"] == (0.8 + 0.4) / 2
    assert agg["counts"]["trials_with_mutant_lines_block"] == 2
    assert agg["counts"]["trials_total"] == 3


def test_aggregate_empty():
    agg = _aggregate_mutant_lines([])
    assert agg["hit_rate_mean"] is None
    assert agg["hit_rate_mean_gold"] is None
    assert agg["per_spec"] == {}
    assert agg["counts"] == {
        "trials_total": 0, "trials_with_mutant_lines_block": 0,
    }


def _file_block(executed_lines):
    return {"executed_lines": executed_lines}


def _cov(files):
    return {"files": files}


def test_mutant_line_hit_rate_basic():
    cov = _cov({
        "/testbed/fastapi/_compat/shared.py": _file_block([161, 163]),
    })
    r = mutant_line_hit_rate(
        cov,
        mutant_lines={"fastapi/_compat/shared.py": [161, 162, 163]},
    )
    assert r["hit_rate"] == 2 / 3
    assert r["hit"] == 2
    assert r["total"] == 3
    detail = r["detail"]["fastapi/_compat/shared.py"]
    assert detail["hit"] == [161, 163]
    assert detail["missed"] == [162]
    assert detail["lines"] == [161, 162, 163]


def test_mutant_line_hit_rate_file_not_in_coverage():
    cov = _cov({"/testbed/other.py": _file_block([1, 2])})
    r = mutant_line_hit_rate(
        cov,
        mutant_lines={"fastapi/missing.py": [10, 11]},
    )
    assert r["hit_rate"] == 0.0
    assert r["hit"] == 0
    assert r["total"] == 2
    assert r["detail"]["fastapi/missing.py"]["missed"] == [10, 11]


def test_mutant_line_hit_rate_empty_mutant_lines():
    r = mutant_line_hit_rate(_cov({}), mutant_lines={})
    assert r["hit_rate"] is None
    assert r["total"] == 0
    assert r["hit"] == 0
    assert r["detail"] == {}


def test_mutant_line_hit_rate_suffix_match():
    cov = _cov({"/testbed/fastapi/_compat/shared.py": _file_block([161])})
    r = mutant_line_hit_rate(
        cov, mutant_lines={"fastapi/_compat/shared.py": [161, 162]},
    )
    assert r["hit"] == 1
    assert r["total"] == 2


def test_parse_mutant_lines_basic():
    patch = """\
diff --git a/fastapi/_compat/shared.py b/fastapi/_compat/shared.py
index cabf4822..22efd52a 100644
--- a/fastapi/_compat/shared.py
+++ b/fastapi/_compat/shared.py
@@ -160,9 +160,9 @@ def is_uploadfile_or_nonable_uploadfile_annotation(annotation: Any) -> bool:
         return True
     origin = get_origin(annotation)
     if origin is Union or origin is UnionType:
-        for arg in get_args(annotation):
-            if lenient_issubclass(arg, UploadFile):
-                return True
+        args = get_args(annotation)
+        if lenient_issubclass(args[0], bytes):
+            return True
     return False"""
    result = parse_mutant_lines(patch)
    assert "fastapi/_compat/shared.py" in result
    assert result["fastapi/_compat/shared.py"] == [163, 164, 165]


def test_parse_mutant_lines_multi_file():
    patch = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
 line1
-old
+new
 line3
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,3 +5,4 @@
 line5
 line6
+added
 line7"""
    result = parse_mutant_lines(patch)
    assert result["a.py"] == [2]
    assert result["b.py"] == [7]


def test_parse_mutant_lines_empty_patch():
    assert parse_mutant_lines("") == {}


_MULTILINE_SRC = """def build(flag):
    # comment
    parameters_list = [1, 2]
    return dict(
        parameters=parameters_list,
        validate_by_name=flag,
    )
"""


def test_fold_continuation_line_to_its_statement():
    folded = fold_lines_to_statements(_MULTILINE_SRC, [6, 4, 3])
    assert folded == {6: 4, 4: 4, 3: 3}


def test_fold_non_code_lines_to_none():
    folded = fold_lines_to_statements(_MULTILINE_SRC, [2, 7])
    assert folded == {2: None, 7: 4}


def test_fold_ast_fallback_agrees(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_coverage(name, *a, **k):
        if name.startswith("coverage"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_coverage)
    folded = fold_lines_to_statements(_MULTILINE_SRC, [6, 4, 3, 2, 7])
    assert folded == {6: 4, 4: 4, 3: 3, 2: None, 7: 4}


def test_hit_rate_uses_anchors_and_excludes_non_code():
    cov = _cov({"/testbed/m.py": {"executed_lines": [1, 3, 4]}})
    r = mutant_line_hit_rate(
        cov, {"m.py": [6, 2]}, anchors={"m.py": {6: 4, 2: None}}
    )
    assert r["hit"] == 1 and r["total"] == 1 and r["hit_rate"] == 1.0
    assert r["detail"]["m.py"]["not_code"] == [2]
    assert r["detail"]["m.py"]["anchors"] == {6: 4}


def test_parse_mutant_lines_old_side():
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -10,4 +20,5 @@\n"
        " keep\n-old_a\n-old_b\n+new_a\n+new_b\n+new_c\n keep\n"
    )
    assert parse_mutant_lines(patch) == {"m.py": [21, 22, 23]}
    assert parse_mutant_lines(patch, side="old") == {"m.py": [11, 12]}
