from eval_pipeline.fuzz_compliance import validate_fuzz_compliance


def _rule_ids(violations):
    return sorted({v.rule_id for v in violations})


COMPLIANT_SHAPE_A = """
from hypothesis import given, example, settings, HealthCheck, strategies as st


@given(x=st.integers())
@example(x=0)
@example(x=-1)
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property(x):
    result = abs(x)
    assert result >= 0
"""


COMPLIANT_SHAPE_B = """
from hypothesis import given, settings, HealthCheck, strategies as st


@given(x=st.integers())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property(x):
    assert abs(x) >= 0
"""


def test_compliant_shape_a():
    assert validate_fuzz_compliance(COMPLIANT_SHAPE_A) == []


def test_compliant_shape_b():
    assert validate_fuzz_compliance(COMPLIANT_SHAPE_B) == []


COSMETIC_GIVEN = """
from hypothesis import given, settings, strategies as st


@given(x=st.integers())
@settings(deadline=None)
def test_property(x):
    # x is never referenced — cosmetic
    assert 1 + 1 == 2
"""


def test_cosmetic_given_fails_r5():
    v = validate_fuzz_compliance(COSMETIC_GIVEN)
    assert "R5_no_unused_given_arg" in _rule_ids(v)


JUST_ONLY = """
from hypothesis import given, settings, strategies as st


@given(x=st.just(1))
@settings(deadline=None)
def test_property(x):
    assert x == 1
"""


def test_just_only_fails_r3():
    v = validate_fuzz_compliance(JUST_ONLY)
    assert "R3_no_just_only" in _rule_ids(v)


TINY_SAMPLED_FROM = """
from hypothesis import given, settings, strategies as st


@given(x=st.sampled_from([1, 2, 3]))
@settings(deadline=None)
def test_property(x):
    assert x in (1, 2, 3)
"""


def test_tiny_sampled_from_fails_r4():
    v = validate_fuzz_compliance(TINY_SAMPLED_FROM)
    assert "R4_no_tiny_sampled_from" in _rule_ids(v)


PHASE_EXPLICIT = """
from hypothesis import given, settings, Phase, strategies as st


@given(x=st.integers())
@settings(deadline=None, phases=[Phase.explicit])
def test_property(x):
    assert isinstance(x, int)
"""


def test_phase_explicit_fails_r2():
    v = validate_fuzz_compliance(PHASE_EXPLICIT)
    assert "R2_no_explicit_phase" in _rule_ids(v)


MISSING_GIVEN = """
def test_property():
    assert 1 + 1 == 2
"""


def test_missing_given_fails_r1():
    v = validate_fuzz_compliance(MISSING_GIVEN)
    assert "R1_has_given" in _rule_ids(v)


MISSING_SETTINGS = """
from hypothesis import given, strategies as st


@given(x=st.integers())
def test_property(x):
    assert isinstance(x, int)
"""


def test_missing_settings_fails_r6():
    v = validate_fuzz_compliance(MISSING_SETTINGS)
    assert "R6_required_settings" in _rule_ids(v)


DEADLINE_NOT_NONE = """
from hypothesis import given, settings, strategies as st


@given(x=st.integers())
@settings(deadline=1000)
def test_property(x):
    assert isinstance(x, int)
"""


def test_deadline_not_none_fails_r6():
    v = validate_fuzz_compliance(DEADLINE_NOT_NONE)
    assert "R6_required_settings" in _rule_ids(v)


MIXED = """
from hypothesis import given, settings, strategies as st


@given(x=st.just(1), y=st.text())
@settings(deadline=None)
def test_property(x, y):
    assert x == 1 and isinstance(y, str)
"""


def test_mixed_passes_r3():
    v = validate_fuzz_compliance(MIXED)
    assert "R3_no_just_only" not in _rule_ids(v)


MULTI_MIXED = """
from hypothesis import given, settings, strategies as st


@given(x=st.integers())
@settings(deadline=None)
def test_ok(x):
    assert isinstance(x, int)


@given(x=st.just(7))
@settings(deadline=None)
def test_bad(x):
    assert x == 7
"""


def test_multi_funcs_one_bad_fails_overall():
    v = validate_fuzz_compliance(MULTI_MIXED)
    assert any(viol.rule_id == "R3_no_just_only" and viol.test_func == "test_bad"
               for viol in v)
    assert all(viol.test_func != "test_ok" for viol in v)


def test_empty_fails():
    v = validate_fuzz_compliance("")
    assert v
    assert v[0].rule_id == "R1_has_given"


def test_syntax_error_fails():
    v = validate_fuzz_compliance("def test_bad(:\n    pass\n")
    assert v
    assert v[0].rule_id == "R1_has_given"


def test_no_test_functions_fails():
    v = validate_fuzz_compliance("x = 1\ny = 2\n")
    assert v
    assert v[0].rule_id == "R1_has_given"


def test_positional_just_only_fails_r3():
    code = """
from hypothesis import given, settings, strategies as st
@given(st.just(1))
@settings(deadline=None)
def test_x(x):
    assert x == 1
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R3_no_just_only" in rule_ids


def test_positional_tiny_sampled_from_only_fails_r4():
    code = """
from hypothesis import given, settings, strategies as st
@given(st.sampled_from([1, 2, 3]))
@settings(deadline=None)
def test_x(x):
    assert x in [1, 2, 3]
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R4_no_tiny_sampled_from" in rule_ids


def test_mixed_positional_just_and_kwarg_real_passes():
    code = """
from hypothesis import given, settings, strategies as st
@given(st.just(1), x=st.integers())
@settings(deadline=None)
def test_x(_y, x):
    assert x == x
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R3_no_just_only" not in rule_ids


def test_one_of_with_only_just_inside_is_known_gap():
    code = """
from hypothesis import given, settings, strategies as st
@given(x=st.one_of(st.just(1), st.just(2)))
@settings(deadline=None)
def test_x(x):
    assert x in (1, 2)
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R3_no_just_only" not in rule_ids


def test_stacked_given_decorators_each_checked():
    code = """
from hypothesis import given, settings, strategies as st
@given(x=st.integers())
@given(y=st.just(1))
@settings(deadline=None)
def test_x(x, y):
    assert x == x
    assert y == 1
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R3_no_just_only" in rule_ids


def test_arg_used_only_in_decorator_subtree_fails_r5():
    code = """
from hypothesis import given, settings, strategies as st
@given(x=st.integers().filter(lambda x: x > 0))
@settings(deadline=None)
def test_x(x):
    # x is referenced in the @given decorator's filter lambda above,
    # but NOT in the body. R5 should still fire.
    assert True
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R5_no_unused_given_arg" in rule_ids


def test_arg_used_in_comprehension_passes_r5():
    code = """
from hypothesis import given, settings, strategies as st
@given(x=st.integers())
@settings(deadline=None)
def test_x(x):
    out = [x + i for i in range(3)]
    assert len(out) == 3
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R5_no_unused_given_arg" not in rule_ids


def test_phase_explicit_with_given_still_fails_r2():
    code = """
from hypothesis import given, settings, Phase, strategies as st
@given(x=st.integers())
@settings(deadline=None, phases=[Phase.explicit])
def test_x(x):
    assert x == x
"""
    v = validate_fuzz_compliance(code)
    rule_ids = {x.rule_id for x in v}
    assert "R2_no_explicit_phase" in rule_ids
