"""Data models for the evaluation pipeline."""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_SC_ALIGN = Path(__file__).resolve().parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))

EVAL_SCHEMA_VERSION = 3

TESTBED_ROOT = "/testbed"


def canonical_testbed_path(path: str) -> str:
    """``path`` as coverage.py reports it inside the eval container."""
    p = path or ""
    prefix = TESTBED_ROOT + "/"
    if p.startswith(prefix):
        p = p[len(prefix):]
    return p.lstrip("/")


@dataclass
class SurvivingSpec:
    """One surviving spec for an instance: spec definition + the mutant that targets it.
    """

    spec: Dict[str, Any]
    mutant_patch: str
    case_contract: str = ""
    witness_factory: str = ""
    witness_observer: str = ""
    preservation_probes: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cm = self.spec.get("code_mapping")
        if isinstance(cm, list):
            fixed = []
            for entry in cm:
                if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                    entry = {**entry, "file": canonical_testbed_path(entry["file"])}
                fixed.append(entry)
            self.spec = {**self.spec, "code_mapping": fixed}

    def to_dict(self) -> Dict[str, Any]:
        out = {"spec": self.spec, "mutant_patch": self.mutant_patch}
        if self.case_contract:
            out["case_contract"] = self.case_contract
        if self.witness_factory:
            out["witness_factory"] = self.witness_factory
        if self.witness_observer:
            out["witness_observer"] = self.witness_observer
        if self.preservation_probes:
            out["preservation_probes"] = self.preservation_probes
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurvivingSpec":
        return cls(
            spec=data.get("spec", {}),
            mutant_patch=data.get("mutant_patch", ""),
            case_contract=data.get("case_contract", ""),
            witness_factory=data.get("witness_factory", ""),
            witness_observer=data.get("witness_observer", ""),
            preservation_probes=list(data.get("preservation_probes") or []),
        )

    @property
    def spec_id(self) -> str:
        return self.spec.get("spec_id", "")

    @property
    def control_spec_ids(self) -> List[str]:
        """Specs whose certificates may use this mutant as a negative control."""
        return [
            str(probe.get("spec_id"))
            for probe in self.preservation_probes
            if probe.get("spec_id")
        ]


@dataclass
class EvalTask:
    """The evaluation input for one instance: its patches and its surviving specs."""

    instance_id: str
    image_name: str
    surviving_specs: List[SurvivingSpec]
    f2p_tests: List[str]
    p2p_tests: List[str]
    repo_settings: Dict[str, Any]
    mask_patch: str = ""
    gold_patch: str = ""
    test_patch: str = ""
    gold_patch_functions: List[Dict[str, str]] = field(default_factory=list)
    schema_version: int = EVAL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "image_name": self.image_name,
            "surviving_specs": [s.to_dict() for s in self.surviving_specs],
            "f2p_tests": self.f2p_tests,
            "p2p_tests": self.p2p_tests,
            "repo_settings": self.repo_settings,
            "mask_patch": self.mask_patch,
            "gold_patch": self.gold_patch,
            "test_patch": self.test_patch,
            "gold_patch_functions": list(self.gold_patch_functions),
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalTask":
        raw_schema_version = data.get("schema_version", EVAL_SCHEMA_VERSION)
        try:
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError):
            schema_version = EVAL_SCHEMA_VERSION
        return cls(
            schema_version=schema_version,
            instance_id=data.get("instance_id", ""),
            image_name=data.get("image_name", ""),
            surviving_specs=[
                SurvivingSpec.from_dict(s) for s in data.get("surviving_specs", [])
            ],
            f2p_tests=data.get("f2p_tests", []),
            p2p_tests=data.get("p2p_tests", []),
            repo_settings=data.get("repo_settings", {}),
            mask_patch=data.get("mask_patch", ""),
            gold_patch=data.get("gold_patch", ""),
            test_patch=data.get("test_patch", ""),
            gold_patch_functions=list(data.get("gold_patch_functions", [])),
        )

    @property
    def all_specs(self) -> List[Dict[str, Any]]:
        """Convenience: just the spec dicts (without mutant_patch)."""
        return [s.spec for s in self.surviving_specs]


@dataclass
class SpecCertificate:
    """An executable certificate for one spec."""

    description: str
    code: str
    saved_inputs: Any = field(default_factory=list)
    inputs_file: str = ""
    code_repaired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = {"description": self.description, "code": self.code}
        if self.saved_inputs:
            out["saved_inputs"] = self.saved_inputs
        if self.inputs_file:
            out["inputs_file"] = self.inputs_file
        if self.code_repaired:
            out["code_repaired"] = True
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecCertificate":
        code, repaired = repair_double_escaped_code(data.get("code", ""))
        return cls(
            description=data.get("description", ""),
            code=code,
            saved_inputs=data.get("saved_inputs", []),
            inputs_file=data.get("inputs_file", ""),
            code_repaired=repaired or bool(data.get("code_repaired", False)),
        )


_JSON_LIKE_ESCAPE = re.compile(r'\\(?:n|"|\'|\\|u[0-9a-fA-F]{4})')
_SIMPLE_UNESCAPES = {'\\n': '\n', '\\"': '"', "\\'": "'", '\\\\': '\\'}
_MIN_DECODED_LINES = 5


def _escapes_are_json_like(code: str) -> bool:
    i = 0
    while True:
        i = code.find("\\", i)
        if i == -1:
            return True
        m = _JSON_LIKE_ESCAPE.match(code, i)
        if m is None:
            return False
        i = m.end()


def _decode_json_like(code: str) -> str:
    def one(m: "re.Match") -> str:
        s = m.group(0)
        return _SIMPLE_UNESCAPES.get(s) or chr(int(s[2:], 16))

    text = _JSON_LIKE_ESCAPE.sub(one, code)
    return text.encode("utf-16", "surrogatepass").decode("utf-16")


def repair_double_escaped_code(code: str) -> tuple:
    """Undo one level of string escaping on a certificate that was written escaped
    twice.
    """
    if not code or "\n" in code or "\\n" not in code:
        return code, False
    if not _escapes_are_json_like(code):
        return code, False
    try:
        compile(code, "<certificate>", "exec")
        return code, False
    except SyntaxError:
        pass
    except Exception:
        return code, False
    try:
        fixed = _decode_json_like(code)
        if fixed.count("\n") < _MIN_DECODED_LINES:
            return code, False
        compile(fixed, "<certificate>", "exec")
    except Exception:
        return code, False
    return fixed, True


@dataclass
class LLMCallError:
    """A detected LLM call error in the baseline run output."""

    error_type: str
    count: int
    sample_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "count": self.count,
            "sample_message": self.sample_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMCallError":
        return cls(
            error_type=data.get("error_type", ""),
            count=data.get("count", 0),
            sample_message=data.get("sample_message", ""),
        )


@dataclass
class BaselineResult:
    """Output of one baseline run on one instance: a cert per surviving spec."""

    instance_id: str
    baseline: str
    model: str
    tool_version: str
    raw_output: str
    certificates: Dict[str, SpecCertificate]
    cost: Dict[str, float]
    wall_time_seconds: float
    llm_errors: List[LLMCallError] = field(default_factory=list)
    completed_normally: bool = True
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    fuzz_compliance: Optional[Dict[str, Any]] = None
    provider_pending: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "baseline": self.baseline,
            "model": self.model,
            "tool_version": self.tool_version,
            "raw_output": self.raw_output,
            "certificates": {
                spec_id: c.to_dict() for spec_id, c in self.certificates.items()
            },
            "cost": self.cost,
            "wall_time_seconds": self.wall_time_seconds,
            "llm_errors": [e.to_dict() for e in self.llm_errors],
            "completed_normally": self.completed_normally,
            "attempts": self.attempts,
        }
        if self.fuzz_compliance is not None:
            d["fuzz_compliance"] = self.fuzz_compliance
        if self.provider_pending is not None:
            d["provider_pending"] = self.provider_pending
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineResult":
        certificates = {
            spec_id: SpecCertificate.from_dict(c)
            for spec_id, c in data.get("certificates", {}).items()
        }
        llm_errors = [
            LLMCallError.from_dict(e) for e in data.get("llm_errors", [])
        ]
        return cls(
            instance_id=data.get("instance_id", ""),
            baseline=data.get("baseline", ""),
            model=data.get("model", ""),
            tool_version=data.get("tool_version", ""),
            raw_output=data.get("raw_output", ""),
            certificates=certificates,
            cost=data.get("cost", {}),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            llm_errors=llm_errors,
            completed_normally=data.get("completed_normally", True),
            attempts=data.get("attempts") or [],
            fuzz_compliance=data.get("fuzz_compliance"),
            provider_pending=data.get("provider_pending"),
        )


TARGET_STATE_GOLD = "gold"
TARGET_STATE_MUTANT = "mutant"
TARGET_STATE_CONTROL = "control"

KIND_PASSED = "passed"
KIND_ASSERTION = "assertion"
KIND_TEST_EXCEPTION = "test_exception"
KIND_COLLECTION_ERROR = "collection_error"
KIND_NO_TESTS = "no_tests"
KIND_INVALID_CASE = "invalid_case"
KIND_INVALID_CERTIFICATE = "invalid_certificate"
KIND_ERROR = "error"
KIND_TIMEOUT = "timeout"
KIND_INFRA = "infra_error"

SEMANTIC_FAILURE_KINDS = frozenset({KIND_ASSERTION, KIND_TEST_EXCEPTION})

NON_EXECUTION_KINDS = frozenset({
    KIND_COLLECTION_ERROR,
    KIND_NO_TESTS,
    KIND_INVALID_CASE,
    KIND_INVALID_CERTIFICATE,
    KIND_INFRA,
    KIND_TIMEOUT,
})


def classify_pytest_output(exit_code: int, output: str) -> str:
    """Classify a pytest run from its exit code + textual output."""
    if exit_code == 0:
        return KIND_PASSED
    if exit_code == 5:
        return KIND_NO_TESTS

    text = output or ""

    collection_markers = (
        "errors during collection",
        "ERROR collecting",
        "ImportError while importing test module",
        "ModuleNotFoundError while importing test module",
    )
    if any(m in text for m in collection_markers):
        return KIND_COLLECTION_ERROR

    if "SyntaxError" in text and ("collecting" in text or "ERROR" in text):
        return KIND_COLLECTION_ERROR

    if exit_code == 1:
        return KIND_ASSERTION

    return KIND_ERROR


def classify_hypofuzz_output(exit_code: int, output_text: str) -> str:
    """Classify a hypofuzz two-stage run."""
    return classify_pytest_output(exit_code, output_text)


@dataclass
class ProofExecResult:
    """Result of executing one certificate against one code state."""

    instance_id: str
    baseline: str
    model: str
    spec_id: str
    target_state: str
    triggered: bool
    exit_code: int
    proof_output: str
    execution_time_seconds: float
    kind: str = KIND_PASSED
    error: Optional[str] = None
    coverage: Optional[Dict[str, Any]] = None
    control_spec_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "baseline": self.baseline,
            "model": self.model,
            "spec_id": self.spec_id,
            "target_state": self.target_state,
            "triggered": self.triggered,
            "exit_code": self.exit_code,
            "kind": self.kind,
            "proof_output": self.proof_output,
            "execution_time_seconds": self.execution_time_seconds,
            "error": self.error,
        }
        if self.coverage is not None:
            out["coverage"] = self.coverage
        if self.control_spec_id is not None:
            out["control_spec_id"] = self.control_spec_id
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofExecResult":
        return cls(
            instance_id=data.get("instance_id", ""),
            baseline=data.get("baseline", ""),
            model=data.get("model", ""),
            spec_id=data.get("spec_id", ""),
            target_state=data.get("target_state", ""),
            triggered=data.get("triggered", False),
            exit_code=data.get("exit_code", 0),
            kind=data.get("kind", KIND_PASSED),
            proof_output=data.get("proof_output", ""),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            error=data.get("error"),
            coverage=data.get("coverage"),
            control_spec_id=data.get("control_spec_id"),
        )


@dataclass
class SpecCertOutcome:
    """Per-spec quality of one baseline's certificate."""

    spec_id: str
    has_certificate: bool
    gold_triggered: Optional[bool]
    mutant_triggered: Optional[bool]
    gold_kind: Optional[str] = None
    mutant_kind: Optional[str] = None
    gold_error: Optional[str] = None
    mutant_error: Optional[str] = None

    @property
    def cert_broken(self) -> Optional[bool]:
        """True when the cert never judged a case on gold."""
        if self.gold_kind is None:
            return None
        return self.gold_kind in NON_EXECUTION_KINDS

    @property
    def valid_cert(self) -> Optional[bool]:
        """True when the cert passes gold and has a test-level mutant failure."""
        if self.gold_triggered is None or self.mutant_triggered is None:
            return None
        if self.gold_error or self.mutant_error:
            return None
        if self.cert_broken:
            return None
        passed_on_gold = (self.gold_kind == KIND_PASSED)
        fired_on_mutant = self.mutant_kind in SEMANTIC_FAILURE_KINDS
        return passed_on_gold and fired_on_mutant

    @property
    def gold_firing(self) -> Optional[bool]:
        """True when a test-level certificate failure occurs on gold."""
        if self.gold_kind is None or self.gold_error:
            return None
        if self.cert_broken:
            return None
        return self.gold_kind in SEMANTIC_FAILURE_KINDS

    @property
    def missed(self) -> Optional[bool]:
        """True when the mutant produced no test-level certificate failure."""
        if self.mutant_triggered is None or self.mutant_error:
            return None
        if self.cert_broken:
            return None
        return self.mutant_kind not in SEMANTIC_FAILURE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "has_certificate": self.has_certificate,
            "gold_triggered": self.gold_triggered,
            "mutant_triggered": self.mutant_triggered,
            "gold_kind": self.gold_kind,
            "mutant_kind": self.mutant_kind,
            "gold_error": self.gold_error,
            "mutant_error": self.mutant_error,
            "cert_broken": self.cert_broken,
            "valid_cert": self.valid_cert,
            "gold_firing": self.gold_firing,
            "missed": self.missed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecCertOutcome":
        return cls(
            spec_id=data.get("spec_id", ""),
            has_certificate=data.get("has_certificate", False),
            gold_triggered=data.get("gold_triggered"),
            mutant_triggered=data.get("mutant_triggered"),
            gold_kind=data.get("gold_kind"),
            mutant_kind=data.get("mutant_kind"),
            gold_error=data.get("gold_error"),
            mutant_error=data.get("mutant_error"),
        )


@dataclass
class DifferentialResult:
    """Per-spec cert quality for one (instance, baseline, model)."""

    instance_id: str
    baseline: str
    model: str
    per_spec: Dict[str, SpecCertOutcome]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "baseline": self.baseline,
            "model": self.model,
            "per_spec": {sid: o.to_dict() for sid, o in self.per_spec.items()},
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DifferentialResult":
        return cls(
            instance_id=data.get("instance_id", ""),
            baseline=data.get("baseline", ""),
            model=data.get("model", ""),
            per_spec={
                sid: SpecCertOutcome.from_dict(o)
                for sid, o in data.get("per_spec", {}).items()
            },
        )
