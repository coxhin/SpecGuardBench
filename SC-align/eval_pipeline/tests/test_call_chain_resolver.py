import json
import subprocess
import sys
from pathlib import Path

from eval_pipeline.call_chain_resolver import resolve_scope


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _resolve(root: Path, declared, gold):
    return resolve_scope(declared, gold, str(root))


def test_resolves_fastapi_function_local_relative_imports(tmp_path):
    _write(tmp_path, "fastapi/__init__.py", "")
    _write(tmp_path, "fastapi/_compat/__init__.py", "")
    _write(
        tmp_path,
        "fastapi/_compat/main.py",
        "def _get_model_config(model):\n"
        "    if isinstance(model, object):\n"
        "        from fastapi._compat import v1\n"
        "        return v1._get_model_config(model)\n"
        "    else:\n"
        "        from . import v2\n"
        "        return v2._get_model_config(model)\n",
    )
    _write(
        tmp_path,
        "fastapi/_compat/v1.py",
        "def _get_model_config(model):\n    return model.__config__\n",
    )
    _write(
        tmp_path,
        "fastapi/_compat/v2.py",
        "def _get_model_config(model):\n    return model.model_config\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "fastapi/_compat/main.py", "name": "_get_model_config"}],
        [
            {"file": "fastapi/_compat/main.py", "name": "_get_model_config"},
            {"file": "fastapi/_compat/v2.py", "name": "_get_model_config"},
        ],
    )

    assert result["scope"] == [
        {"file": "fastapi/_compat/main.py", "name": "_get_model_config"},
        {"file": "fastapi/_compat/v2.py", "name": "_get_model_config"},
    ]
    assert result["complete"] is True
    assert result["unresolved_calls"] == []


def test_resolves_transitive_module_alias_calls(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/a.py",
        "import pkg.b as backend\n"
        "def entry():\n    return backend.helper()\n",
    )
    _write(
        tmp_path,
        "pkg/b.py",
        "from pkg.c import finish as done\n"
        "def helper():\n    return done()\n",
    )
    _write(tmp_path, "pkg/c.py", "def finish():\n    return 1\n")

    result = _resolve(
        tmp_path,
        [{"file": "pkg/a.py", "name": "entry"}],
        [
            {"file": "pkg/a.py", "name": "entry"},
            {"file": "pkg/b.py", "name": "helper"},
            {"file": "pkg/c.py", "name": "finish"},
        ],
    )

    assert result["scope"] == [
        {"file": "pkg/a.py", "name": "entry"},
        {"file": "pkg/b.py", "name": "helper"},
        {"file": "pkg/c.py", "name": "finish"},
    ]
    assert result["complete"] is True


def test_resolves_same_class_method_without_conflating_classes(tmp_path):
    _write(
        tmp_path,
        "a.py",
        "class A:\n"
        "    def entry(self):\n        return self.helper()\n"
        "    def helper(self):\n        return 1\n"
        "class B:\n"
        "    def helper(self):\n        return 2\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "a.py", "name": "A.entry"}],
        [
            {"file": "a.py", "name": "A.helper"},
            {"file": "a.py", "name": "B.helper"},
        ],
    )

    assert result["scope"] == [
        {"file": "a.py", "name": "A.entry"},
        {"file": "a.py", "name": "A.helper"},
    ]


def test_resolves_package_reexported_class_method(tmp_path):
    _write(
        tmp_path,
        "pkg/__init__.py",
        "from .models import Model\n",
    )
    _write(
        tmp_path,
        "pkg/models.py",
        "class Model:\n"
        "    @staticmethod\n"
        "    def validate():\n        return True\n",
    )
    _write(
        tmp_path,
        "consumer.py",
        "from pkg import Model\n"
        "def entry():\n    return Model.validate()\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "consumer.py", "name": "entry"}],
        [{"file": "pkg/models.py", "name": "Model.validate"}],
    )

    assert result["scope"] == [
        {"file": "consumer.py", "name": "entry"},
        {"file": "pkg/models.py", "name": "Model.validate"},
    ]


def test_resolves_literal_getattr_on_module_alias(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/helpers.py", "def helper():\n    return 1\n")
    _write(
        tmp_path,
        "entry.py",
        "from pkg import helpers as h\n"
        "def entry():\n    return getattr(h, 'helper')()\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "entry.py", "name": "entry"}],
        [{"file": "pkg/helpers.py", "name": "helper"}],
    )

    assert result["scope"] == [
        {"file": "entry.py", "name": "entry"},
        {"file": "pkg/helpers.py", "name": "helper"},
    ]
    assert result["complete"] is True


def test_ambiguous_gold_relevant_call_is_reported_not_guessed(tmp_path):
    _write(tmp_path, "entry.py", "def entry(obj):\n    return obj.helper()\n")
    _write(tmp_path, "a.py", "def helper():\n    return 1\n")
    _write(tmp_path, "b.py", "def helper():\n    return 2\n")

    result = _resolve(
        tmp_path,
        [{"file": "entry.py", "name": "entry"}],
        [
            {"file": "a.py", "name": "helper"},
            {"file": "b.py", "name": "helper"},
        ],
    )

    assert result["scope"] == [{"file": "entry.py", "name": "entry"}]
    assert result["complete"] is False
    assert result["unresolved_calls"][0]["status"] == "ambiguous"
    assert len(result["unresolved_calls"][0]["candidates"]) == 2


def test_container_cli_uses_same_resolver(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/helper.py", "def run():\n    return 1\n")
    _write(
        tmp_path,
        "entry.py",
        "from pkg.helper import run as execute\n"
        "def entry():\n    return execute()\n",
    )
    payload = {
        "declared": [{"file": "entry.py", "name": "entry"}],
        "gold_funcs": [{"file": "pkg/helper.py", "name": "run"}],
        "testbed_root": str(tmp_path),
    }
    script = Path(__file__).resolve().parents[1] / "call_chain_resolver.py"

    completed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result["scope"] == [
        {"file": "entry.py", "name": "entry"},
        {"file": "pkg/helper.py", "name": "run"},
    ]


def test_resolves_modules_under_src_layout(tmp_path):
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/helper.py", "def run():\n    return 1\n")
    _write(
        tmp_path,
        "src/pkg/entry.py",
        "from pkg.helper import run\n"
        "def entry():\n    return run()\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "src/pkg/entry.py", "name": "entry"}],
        [{"file": "src/pkg/helper.py", "name": "run"}],
    )

    assert result["scope"] == [
        {"file": "src/pkg/entry.py", "name": "entry"},
        {"file": "src/pkg/helper.py", "name": "run"},
    ]


def test_conflicting_conditional_alias_is_incomplete(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/a.py", "def helper():\n    return 1\n")
    _write(tmp_path, "pkg/b.py", "def helper():\n    return 2\n")
    _write(
        tmp_path,
        "entry.py",
        "def entry(flag):\n"
        "    if flag:\n"
        "        import pkg.a as backend\n"
        "    else:\n"
        "        import pkg.b as backend\n"
        "    return backend.helper()\n",
    )

    result = _resolve(
        tmp_path,
        [{"file": "entry.py", "name": "entry"}],
        [
            {"file": "pkg/a.py", "name": "helper"},
            {"file": "pkg/b.py", "name": "helper"},
        ],
    )

    assert result["scope"] == [{"file": "entry.py", "name": "entry"}]
    assert result["complete"] is False
    assert result["unresolved_calls"][0]["status"] == "ambiguous"
