import json

from eval_pipeline.models import SpecCertificate, repair_double_escaped_code


HEALTHY = (
    '"""Cert."""\n'
    "import re\n"
    "\n"
    "def build_cases():\n"
    "    yield {'args': ('a\\\\nb',), 'kwargs': {}}\n"
    "\n"
    "def check_case(case):\n"
    "    assert re.match(r'a\\\\nb', case['args'][0])\n"
)


def test_healthy_code_is_untouched():
    code, repaired = repair_double_escaped_code(HEALTHY)
    assert code == HEALTHY
    assert repaired is False


def test_json_escaped_once_more_is_unescaped():
    escaped = json.dumps(HEALTHY)[1:-1]
    assert "\n" not in escaped
    code, repaired = repair_double_escaped_code(escaped)
    assert repaired is True
    assert code == HEALTHY


def test_partially_escaped_pandas_shape_is_unescaped():
    escaped = HEALTHY.replace("\\", "\\\\").replace("\n", "\\n")
    code, repaired = repair_double_escaped_code(escaped)
    assert repaired is True
    assert code == HEALTHY


def test_non_ascii_survives_repair():
    src = 'x = "naïve"\ny = "café"\nz = 1\nw = 2\nv = 3\n'
    escaped = src.replace("\n", "\\n")
    code, repaired = repair_double_escaped_code(escaped)
    assert repaired is True
    assert code == src


def test_single_line_code_with_literal_backslash_n_is_untouched():
    src = "PATTERN = 'a\\nb'"
    code, repaired = repair_double_escaped_code(src)
    assert (code, repaired) == (src, False)


def test_unrepairable_garbage_is_left_alone():
    src = "def (:\\n  pass"
    code, repaired = repair_double_escaped_code(src)
    assert (code, repaired) == (src, False)


def test_stray_backslash_typo_in_one_liner_is_not_repaired():
    src = '"""d\\nd"""; x = \\"y\\"'
    assert repair_double_escaped_code(src) == (src, False)


def test_regex_escapes_block_the_repair():
    src = '"""S1\\nspec"""; import re; P = re.compile(r"\\bfoo\\b"); v = \\"bar\\"'
    assert repair_double_escaped_code(src) == (src, False)
    tabbed = HEALTHY.replace("import re", "x = '\t'")
    escaped = json.dumps(tabbed)[1:-1]
    assert "\\t" in escaped
    assert repair_double_escaped_code(escaped) == (escaped, False)


def test_short_decoded_text_is_not_repaired():
    src = '"""a\\nb\\nc"""; x = \\"y\\"'
    assert repair_double_escaped_code(src) == (src, False)


def test_lone_surrogate_and_deep_nesting_do_not_raise():
    surrogate = json.dumps('x = "\ud800"\ny = 1\n' + HEALTHY)[1:-1]
    assert repair_double_escaped_code(surrogate)[1] is False
    deep = "x = " + "(" * 400 + "1" + ")" * 400 + "\\n" + "y = 1"
    code, repaired = repair_double_escaped_code(deep)
    assert code == deep and repaired is False


def test_ensure_ascii_unicode_escapes_are_decoded():
    src = 'x = "naïve"\ny = "日本語 🎉"\nz = 1\nw = 2\nv = 3\n'
    escaped = json.dumps(src)[1:-1]
    assert "\\u" in escaped and "\\ud83c" in escaped
    assert repair_double_escaped_code(escaped) == (src, True)


def test_from_dict_repairs_and_round_trips_flag():
    escaped = json.dumps(HEALTHY)[1:-1]
    cert = SpecCertificate.from_dict({"description": "d", "code": escaped})
    assert cert.code == HEALTHY
    assert cert.code_repaired is True
    d = cert.to_dict()
    assert d["code_repaired"] is True
    again = SpecCertificate.from_dict(d)
    assert again.code == HEALTHY
    assert again.code_repaired is True


def test_from_dict_healthy_has_no_flag():
    cert = SpecCertificate.from_dict({"description": "d", "code": HEALTHY})
    assert cert.code_repaired is False
    assert "code_repaired" not in cert.to_dict()
