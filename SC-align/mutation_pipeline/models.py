"""Data models for the mutation pipeline."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


OBSERVATION_LINE_PREFIX = "__WITNESS_OBS__ "

CALL_CASE_CONTRACT = """Every case is a dict with exactly two keys:

- `args`: a tuple or list of positional arguments for the target callable.
- `kwargs`: a dict mapping string keys to keyword argument values for the target
  callable.

`check_case(case)` must invoke the target using
`target(*case["args"], **case["kwargs"])`. The evaluator may provide unseen
cases using the same representation. Argument values are constrained only by
the natural-language specification and the target's public calling interface.
The target must be the actual implementation callable, not a wrapper invented
by the witness or certificate. If an instance method is taken from its class,
`args` includes the receiver instance first; a method already taken from an
instance is bound and does not need that extra argument.
Cases may exercise normal, boundary, or error-handling behavior when that
behavior is covered by the specification or public interface. Whether an
exception or any other observed behavior is a violation depends on what the
specification requires for that case.
Cases contain stimulus only and must not contain expected outcomes, expected
exceptions, failure flags, or mutant-specific information."""


class SpecCategory(str, Enum):
    """Spec category per Section 2.2."""
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    IMPLEMENTATION_CONSTRAINT = "implementation_constraint"


class SpecSource(str, Enum):
    """Where the spec text was extracted from."""
    TASK_STATEMENT = "task_statement"
    DOCSTRING = "docstring"
    INTERFACE_DESCRIPTION = "interface_description"
    ADDITIONAL_INFO = "additional_info"


@dataclass
class CodeMapping:
    """Maps a spec to its implementation location in the gold patch."""
    file: str
    lines: str
    function: str

    def to_dict(self) -> Dict[str, str]:
        return {"file": self.file, "lines": self.lines, "function": self.function}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeMapping":
        return cls(
            file=data.get("file", ""),
            lines=data.get("lines", ""),
            function=data.get("function", ""),
        )


@dataclass
class AtomicSpec:
    """A single decomposed spec with category, source, and code mapping."""
    spec_id: str
    text: str
    category: SpecCategory
    sources: List[SpecSource]
    code_mapping: List[CodeMapping]
    merged_from: Optional[List[str]] = None
    independence_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "spec_id": self.spec_id,
            "text": self.text,
            "category": self.category.value,
            "sources": [s.value for s in self.sources],
            "code_mapping": [cm.to_dict() for cm in self.code_mapping],
        }
        if self.merged_from:
            d["merged_from"] = self.merged_from
        if self.independence_check:
            d["independence_check"] = self.independence_check
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicSpec":
        return cls(
            spec_id=data.get("spec_id", ""),
            text=data.get("text", ""),
            category=SpecCategory(data.get("category", "behavioral")),
            sources=[SpecSource(s) for s in data.get("sources", [])],
            code_mapping=[CodeMapping.from_dict(cm) for cm in data.get("code_mapping", [])],
            merged_from=data.get("merged_from"),
            independence_check=data.get("independence_check"),
        )


@dataclass
class Proof:
    """Witness-based proof of spec violation."""
    witness_script: str = ""
    actual_gold_output: str = ""
    actual_mutant_output: str = ""
    divergence_type: str = ""
    trigger_condition: str = ""
    case_contract: str = ""
    witness_factory: str = ""
    witness_observer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "witness_script": self.witness_script,
            "actual_gold_output": self.actual_gold_output,
            "actual_mutant_output": self.actual_mutant_output,
            "divergence_type": self.divergence_type,
        }
        if self.trigger_condition:
            out["trigger_condition"] = self.trigger_condition
        if self.case_contract:
            out["case_contract"] = self.case_contract
        if self.witness_factory:
            out["witness_factory"] = self.witness_factory
        if self.witness_observer:
            out["witness_observer"] = self.witness_observer
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proof":
        return cls(
            witness_script=data.get("witness_script", ""),
            actual_gold_output=data.get("actual_gold_output", ""),
            actual_mutant_output=data.get("actual_mutant_output", ""),
            divergence_type=data.get("divergence_type", ""),
            trigger_condition=data.get("trigger_condition", ""),
            case_contract=data.get("case_contract", ""),
            witness_factory=data.get("witness_factory", ""),
            witness_observer=data.get("witness_observer", ""),
        )


@dataclass
class PreservationProbe:
    """One reading of another spec's witness against this spec's mutant."""
    spec_id: str
    verdict: str
    gold_observation: str = ""
    mutant_observation: str = ""
    error: str = ""
    shared_functions: int = 0
    shared_lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"spec_id": self.spec_id, "verdict": self.verdict}
        if self.gold_observation:
            out["gold_observation"] = self.gold_observation
        if self.mutant_observation:
            out["mutant_observation"] = self.mutant_observation
        if self.error:
            out["error"] = self.error
        out["shared_functions"] = self.shared_functions
        out["shared_lines"] = self.shared_lines
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreservationProbe":
        return cls(
            spec_id=data.get("spec_id", ""),
            verdict=data.get("verdict", ""),
            gold_observation=data.get("gold_observation", ""),
            mutant_observation=data.get("mutant_observation", ""),
            error=data.get("error", ""),
            shared_functions=int(data.get("shared_functions") or 0),
            shared_lines=int(data.get("shared_lines") or 0),
        )


@dataclass
class FilterSurvival:
    """Tracks which pipeline filters a mutation survived."""
    tests_passed: Optional[bool] = None
    witness_diverged: Optional[bool] = None
    verifier_approved: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tests_passed": self.tests_passed,
            "witness_diverged": self.witness_diverged,
            "verifier_approved": self.verifier_approved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterSurvival":
        return cls(**{k: data.get(k) for k in ["tests_passed", "witness_diverged", "verifier_approved"]})


@dataclass
class MutationResult:
    """Full pipeline output per (instance, spec) pair."""
    schema_version: int
    instance_id: str
    spec: AtomicSpec
    mask_patch: str
    gold_patch: str
    mutant_patch: str
    test_patch: str
    proof: Proof
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "spec": self.spec.to_dict(),
            "mask_patch": self.mask_patch,
            "gold_patch": self.gold_patch,
            "mutant_patch": self.mutant_patch,
            "test_patch": self.test_patch,
            "proof": self.proof.to_dict(),
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutationResult":
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            instance_id=data.get("instance_id", ""),
            spec=AtomicSpec.from_dict(data.get("spec", {})),
            mask_patch=data.get("mask_patch", ""),
            gold_patch=data.get("gold_patch", ""),
            mutant_patch=data.get("mutant_patch", ""),
            test_patch=data.get("test_patch", ""),
            proof=Proof.from_dict(data.get("proof", {})),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PipelineInput:
    """All data needed to process one FeatureBench instance through the mutation
    pipeline.
    """
    instance_id: str
    level: int
    task_statement: str
    interface_descriptions: List[Dict[str, Any]]
    gold_patch: str
    image_name: str
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    repo_settings: Dict[str, Any] = field(default_factory=dict)
    test_patch: Optional[str] = None
    repo: str = ""
    mask_patch: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "level": self.level,
            "task_statement": self.task_statement,
            "interface_descriptions": self.interface_descriptions,
            "gold_patch": self.gold_patch,
            "image_name": self.image_name,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "repo_settings": self.repo_settings,
            "test_patch": self.test_patch,
            "repo": self.repo,
            "mask_patch": self.mask_patch,
        }


@dataclass
class StepResult:
    """Generic envelope for a pipeline step's output."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    step_name: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "step_name": self.step_name,
        }
        if self.raw_response:
            d["raw_response"] = self.raw_response
        return d
