"""Step 2: Mutation Generation."""

import ast
import json
import logging
import re
import time
from typing import Any, Dict, Optional

import jinja2

from featurebench.llm.llm_caller import LLMCaller

from ..models import (
    CALL_CASE_CONTRACT,
    OBSERVATION_LINE_PREFIX,
    AtomicSpec,
    PipelineInput,
    StepResult,
)

logger = logging.getLogger(__name__)


class WitnessUnobservable(ValueError):
    pass


_FORBIDDEN_ATTRS = {
    "getsource",
    "getsourcefile",
    "getsourcelines",
    "getfile",
    "findsource",
}
_FORBIDDEN_MODULES = {"linecache"}


class MutationGenerator:
    """Generates spec-violating mutations via LLM."""

    def __init__(self, llm_caller: LLMCaller, prompt_env: jinja2.Environment):
        self.llm_caller = llm_caller
        self.prompt_env = prompt_env

    def generate(
        self,
        spec: AtomicSpec,
        pipeline_input: PipelineInput,
        other_specs: Optional[list] = None,
    ) -> StepResult:
        """Generate mutation artifacts for a single spec."""
        prompt = None
        response_text = None
        try:
            template = self.prompt_env.get_template("mutation_generation.j2")
            prompt = template.render(
                spec=spec,
                gold_patch=pipeline_input.gold_patch,
                fail_to_pass=pipeline_input.fail_to_pass,
                pass_to_pass=pipeline_input.pass_to_pass,
                test_patch=pipeline_input.test_patch,
                other_specs=other_specs or [],
            )
            prompt_len = len(prompt)

            messages = [{"role": "user", "content": prompt}]
            t0 = time.time()
            completion = self.llm_caller.call_llm(messages)
            llm_elapsed = time.time() - t0
            response_text = self.llm_caller.get_llm_response_text(completion)
            logger.info(
                f"[{pipeline_input.instance_id}][{spec.spec_id}] Step 2: LLM call took {llm_elapsed:.1f}s "
                f"(prompt {prompt_len} chars, response {len(response_text)} chars)"
            )

            parsed = self._parse_mutation_response(response_text)
            parsed["prompt"] = prompt
            parsed["response_text"] = response_text

            logger.info(
                f"[{pipeline_input.instance_id}][{spec.spec_id}] Step 2: "
                f"Generated mutation ({len(parsed['mutations'])} edits) — "
                f"strategy: {parsed.get('mutation_strategy', 'N/A')[:80]}"
            )
            return StepResult(
                success=True,
                data=parsed,
                step_name="mutation_generation",
            )

        except Exception as e:
            logger.error(
                f"[{pipeline_input.instance_id}][{spec.spec_id}] Step 2 failed: {e}",
                exc_info=True,
            )
            fail_data = {}
            if prompt:
                fail_data["prompt"] = prompt
            if response_text:
                fail_data["response_text"] = response_text
            return StepResult(
                success=False,
                error=str(e),
                data=fail_data if fail_data else None,
                step_name="mutation_generation",
            )

    def _parse_mutation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured mutation data."""
        json_text = self._extract_json(response_text)
        raw = json.loads(json_text)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object, got {type(raw).__name__}")

        mutations = raw.get("mutations", [])
        if not mutations:
            raise ValueError("Missing or empty 'mutations' list in LLM response")

        for m in mutations:
            if not m.get("file"):
                raise ValueError("Mutation entry missing 'file'")
            if not m.get("find"):
                raise ValueError("Mutation entry missing 'find'")
            if "replace" not in m:
                raise ValueError("Mutation entry missing 'replace'")

        witness_factory = raw.get("witness_factory", "")
        witness_observer = raw.get("witness_observer", "")

        if not (str(witness_factory).strip() or str(witness_observer).strip()):
            raise WitnessUnobservable(
                "LLM declared spec has no observable witness"
            )
        if not all(
            isinstance(value, str) and value.strip()
            for value in (witness_factory, witness_observer)
        ):
            raise ValueError(
                "witness_factory and witness_observer must both be "
                "non-empty strings"
            )

        case_contract = CALL_CASE_CONTRACT
        _validate_factory_module(witness_factory, "build_cases")
        _validate_factory_module(witness_observer, "observe_case")
        _validate_observer_contract(witness_observer)
        witness_script = _build_factory_witness_script(
            witness_factory, witness_observer
        )
        ast.parse(witness_script)

        divergence_type = raw.get("divergence_type", "")
        if not isinstance(divergence_type, str):
            divergence_type = ""

        trigger_condition = raw.get("trigger_condition", "")
        if not isinstance(trigger_condition, str):
            trigger_condition = ""
        if not trigger_condition.strip():
            raise ValueError(
                "trigger_condition is required: state in one sentence when the "
                "mutant diverges, using only concepts reachable from the spec "
                "or the correct implementation"
            )

        return {
            "mutations": mutations,
            "mutation_strategy": raw.get("mutation_strategy", ""),
            "witness_script": witness_script,
            "divergence_type": divergence_type,
            "trigger_condition": trigger_condition,
            "case_contract": case_contract,
            "witness_factory": witness_factory,
            "witness_observer": witness_observer,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may be wrapped in markdown code fences."""
        last_valid: Optional[str] = None
        for m in re.finditer(r"```(\w*)\s*\n(.*?)```", text, re.DOTALL):
            lang = (m.group(1) or "").lower()
            body = m.group(2).strip()
            if lang and lang != "json":
                continue
            try:
                if isinstance(json.loads(body), dict):
                    last_valid = body
            except json.JSONDecodeError:
                continue
        if last_valid is not None:
            return last_valid

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()


_FORBIDDEN_CASE_KEYS = {
    "expected",
    "expected_result",
    "expected_calls",
    "should_raise",
}


_STRAY_WRITE_HINT = (
    "an observation must be RETURNED from observe_case, not written to a "
    "stream. Only the harness writes to stdout, and only lines it wrote are "
    "compared between the gold and mutant runs — anything else on that "
    "stream is stray output the testbed also produces, and it differs "
    "between the two runs for reasons that have nothing to do with the "
    "mutation. To observe a side effect, capture it and put it in the "
    "returned dict: warnings via warnings.catch_warnings(record=True), log "
    "records via a temporary logging handler, the target's own printing via "
    "contextlib.redirect_stdout, in-place argument mutation by reading the "
    "argument back after the call."
)


def _check_no_stray_output(tree: ast.AST, required_function: str) -> None:
    """Reject writes to stdout/stderr from the factory or observer."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            raise ValueError(
                f"{required_function} protocol module must not call print(): "
                + _STRAY_WRITE_HINT
            )
        if not (isinstance(func, ast.Attribute) and func.attr in {"write", "writelines"}):
            continue
        stream = func.value
        name = (
            stream.attr if isinstance(stream, ast.Attribute)
            else stream.id if isinstance(stream, ast.Name)
            else None
        )
        if name in {"stdout", "stderr"}:
            raise ValueError(
                f"{required_function} protocol module must not write to "
                f"{name}: " + _STRAY_WRITE_HINT
            )


def _validate_factory_module(source: str, required_function: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"{required_function} module is not valid Python: {e}")
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if required_function not in functions:
        raise ValueError(
            f"factory protocol module must define {required_function}()"
        )
    _check_blacklist(tree)
    _check_no_stray_output(tree, required_function)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            raise ValueError(
                f"{required_function} protocol module must not contain assertions"
            )
        if required_function == "build_cases":
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and key.value in _FORBIDDEN_CASE_KEYS
                    ):
                        raise ValueError(
                            f"witness_factory case contains forbidden oracle "
                            f"key {key.value!r}"
                        )


def _validate_observer_contract(source: str) -> None:
    """Require an auditable, non-wrapper target binding in the observer."""
    tree = ast.parse(source)
    local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    target_binding: Optional[ast.expr] = None
    imported_target = False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_target = imported_target or any(
                alias.asname == "target" or (
                    alias.asname is None and alias.name == "target"
                )
                for alias in node.names
            )
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(name, ast.Name) and name.id == "target"
                for name in node.targets
            ):
                target_binding = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "target"
        ):
            target_binding = node.value

    if "target" in local_functions:
        raise ValueError(
            "witness_observer must bind global 'target' to the actual callable; "
            "a locally defined wrapper named target is not allowed"
        )
    if target_binding is None and not imported_target:
        raise ValueError(
            "witness_observer must expose the actual callable as module-level "
            "global 'target'"
        )
    if target_binding is not None:
        if not isinstance(target_binding, (ast.Name, ast.Attribute)):
            raise ValueError(
                "witness_observer global 'target' must be a direct callable "
                "reference, not a call, lambda, or wrapper"
            )
        if isinstance(target_binding, ast.Name) and target_binding.id in local_functions:
            raise ValueError(
                "witness_observer global 'target' must reference the actual "
                "implementation callable, not a local wrapper"
            )

    observer = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "observe_case"
    )
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "target"
        for node in ast.walk(observer)
    ):
        raise ValueError(
            "observe_case(case) must call the module-level global 'target'; "
            "setup and monkeypatching may surround that call"
        )


def _build_factory_witness_script(factory: str, observer: str) -> str:
    """Build the harness that execs the LLM's factory/observer and prints one
    observation per case.
    """
    return (
        "import inspect\n"
        "import json\n"
        "import os\n"
        f"_FACTORY_SOURCE = {factory!r}\n"
        f"_OBSERVER_SOURCE = {observer!r}\n"
        "def _validate_call_case(case):\n"
        "    if not isinstance(case, dict):\n"
        "        raise TypeError('case must be a dict')\n"
        "    if set(case) != {'args', 'kwargs'}:\n"
        "        raise ValueError(\"case must contain exactly 'args' and 'kwargs'\")\n"
        "    if not isinstance(case['args'], (tuple, list)):\n"
        "        raise TypeError(\"case['args'] must be a tuple or list\")\n"
        "    if not isinstance(case['kwargs'], dict):\n"
        "        raise TypeError(\"case['kwargs'] must be a dict\")\n"
        "    if not all(isinstance(key, str) for key in case['kwargs']):\n"
        "        raise TypeError(\"case['kwargs'] keys must be strings\")\n"
        "def _validate_receiver(target, args):\n"
        "    if inspect.ismethod(target) and target.__self__ is not None:\n"
        "        return\n"
        "    if not inspect.isfunction(target):\n"
        "        return\n"
        "    qualname = getattr(target, '__qualname__', '')\n"
        "    if '<locals>' in qualname or '.' not in qualname:\n"
        "        return\n"
        "    try:\n"
        "        parameters = list(inspect.signature(target).parameters.values())\n"
        "    except (TypeError, ValueError):\n"
        "        return\n"
        "    if not parameters or parameters[0].name != 'self':\n"
        "        return\n"
        "    owner = inspect.getmodule(target)\n"
        "    for part in qualname.rsplit('.', 1)[0].split('.'):\n"
        "        owner = getattr(owner, part, None)\n"
        "    if isinstance(owner, type) and (not args or not isinstance(args[0], owner)):\n"
        "        raise TypeError(f'unbound method {qualname} requires an instance of {owner.__name__} as the first positional argument')\n"
        "def _bind_target_call(target, case):\n"
        "    args = tuple(case['args'])\n"
        "    kwargs = dict(case['kwargs'])\n"
        "    try:\n"
        "        inspect.signature(target).bind(*args, **kwargs)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    except TypeError as exc:\n"
        "        if os.environ.get('WITNESS_ALLOW_BIND_MISMATCH') != '1':\n"
        "            raise TypeError(f'case does not match target signature: {exc}') from exc\n"
        "    _validate_receiver(target, args)\n"
        "def test_factory_witness():\n"
        "    factory_ns = {}\n"
        "    observer_ns = {}\n"
        "    exec(_FACTORY_SOURCE, factory_ns)\n"
        "    exec(_OBSERVER_SOURCE, observer_ns)\n"
        "    target = observer_ns.get('target')\n"
        "    if not callable(target):\n"
        "        raise TypeError(\"witness_observer must expose callable global 'target'\")\n"
        "    for case in factory_ns['build_cases']():\n"
        "        _validate_call_case(case)\n"
        "        _bind_target_call(target, case)\n"
        "        observation = observer_ns['observe_case'](case)\n"
        f"        print({OBSERVATION_LINE_PREFIX!r} + json.dumps("
        "observation, sort_keys=True, ensure_ascii=False), flush=True)\n"
    )


def _check_blacklist(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise ValueError(
                f"witness_script uses forbidden source-text introspection: '.{node.attr}'"
            )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ATTRS:
            raise ValueError(
                f"witness_script uses forbidden source-text introspection: '{node.id}'"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_MODULES:
                    raise ValueError(
                        f"witness_script imports forbidden module: '{alias.name}'"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module in _FORBIDDEN_MODULES:
                raise ValueError(
                    f"witness_script imports forbidden module: '{node.module}'"
                )
        if isinstance(node, ast.Call):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.endswith(".py")
                ):
                    raise ValueError(
                        f"witness_script appears to read a .py source file: {arg.value!r}"
                    )
