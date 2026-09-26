from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import pytest

_SC_ALIGN = Path(__file__).resolve().parent.parent.parent
if str(_SC_ALIGN) not in sys.path:
    sys.path.insert(0, str(_SC_ALIGN))
_PROJECT_ROOT = _SC_ALIGN.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mutation_pipeline.models import (
    OBSERVATION_LINE_PREFIX,
    AtomicSpec,
    CodeMapping,
    PipelineInput,
    PreservationProbe,
    FilterSurvival,
    Proof,
    SpecCategory,
    SpecSource,
    StepResult,
)
from mutation_pipeline.pipeline import MutationPipeline, symmetrise_probe_selection
from mutation_pipeline.steps.step5_preservation_probe import (
    PreservationProber,
    probes_from_result,
    rank_probe_candidates,
    select_probe_specs,
)
from mutation_pipeline.steps.step6_llm_verification import LLMVerifier, _render_probes


def _witness(tag: str) -> str:
    return f"def test_w():\n    print({tag!r})\n"


def _make_pipeline_input() -> PipelineInput:
    return PipelineInput(
        instance_id="example.instance.lv1",
        level=1,
        task_statement="t",
        interface_descriptions=[],
        gold_patch="gold",
        image_name="img",
    )


class _FakeRunner:
    def __init__(self, observations: Dict[Tuple[str, str], Optional[str]]):
        self.observations = observations
        self.state = "gold"
        self.resets = 0
        self.setups = 0
        self.cleanups = 0
        self.applied: List[List[dict]] = []
        self.runs: List[Tuple[str, str, Dict[str, str]]] = []
        self.apply_succeeds = True

    def setup_instance(self, pipeline_input: PipelineInput) -> Any:
        self.setups += 1
        self.state = "gold"
        return object()

    def cleanup(self, container: Any) -> None:
        self.cleanups += 1

    def reset_testbed(self, container: Any, pipeline_input: PipelineInput) -> None:
        self.resets += 1
        self.state = "gold"

    def apply_mutation(self, container: Any, mutations: List[dict]) -> bool:
        self.applied.append(mutations)
        if not self.apply_succeeds:
            return False
        self.state = "mutant"
        return True

    def run_witness_script(
        self,
        container: Any,
        code: str,
        timeout: int,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        tag = next(t for (_, t) in self.observations if f"'{t}'" in code)
        self.runs.append((self.state, tag, dict(env or {})))
        observation = self.observations.get((self.state, tag))
        if observation is None:
            return True, "E   RuntimeError: boom"
        return False, (
            "=== __WITNESS_OBS_BEGIN__ ===\n"
            "[0/1] Regenerating build files.\n"
            f"{OBSERVATION_LINE_PREFIX}{observation}\n"
            "=== __WITNESS_OBS_END__ ===\n"
        )


def _run(
    observations: Dict[Tuple[str, str], Optional[str]],
    tags: List[str],
    prober: Optional[PreservationProber] = None,
) -> Tuple[List[PreservationProbe], _FakeRunner, PreservationProber]:
    runner = _FakeRunner(observations)
    prober = prober or PreservationProber(runner)
    prober.docker_runner = runner
    for tag in tags:
        gold = observations.get(("gold", tag))
        if gold is not None:
            prober.record_gold_observation(
                "example.instance.lv1", _witness(tag), gold
            )
    probes = [
        prober.probe_one(
            _make_pipeline_input(),
            "S1",
            [{"file": "a.py", "find": "x", "replace": "y"}],
            f"S{i + 2}",
            _witness(tag),
            timeout=30,
        )
        for i, tag in enumerate(tags)
    ]
    return probes, runner, prober


def _mapped(spec_id: str, *mappings: Tuple[str, str, str]):
    return AtomicSpec(
        spec_id=spec_id,
        text=f"spec {spec_id}",
        category=SpecCategory.BEHAVIORAL,
        sources=[SpecSource.DOCSTRING],
        code_mapping=[
            CodeMapping(file=f, lines=lines, function=fn) for f, lines, fn in mappings
        ],
    )


def test_select_prefers_the_most_entangled_sibling() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"), ("a.py", "30-40", "g"))
    specs = [
        target,
        _mapped("S2", ("a.py", "10-20", "f")),
        _mapped("S3", ("a.py", "10-20", "f"), ("a.py", "30-40", "g")),
        _mapped("S4", ("a.py", "12-14", "")),
        _mapped("S5", ("a.py", "900-950", "z")),
    ]
    assert select_probe_specs("S1", specs, 4) == ["S3", "S2", "S4", "S5"]


def test_the_pair_score_is_the_same_from_either_side() -> None:
    a = _mapped("S1", ("a.py", "10-20", "f"), ("a.py", "30-40", "g"))
    b = _mapped("S2", ("a.py", "15-35", "f"))
    specs = [a, b]

    forward = {cid: (sf, sl) for cid, sf, sl in rank_probe_candidates("S1", specs)}
    backward = {cid: (sf, sl) for cid, sf, sl in rank_probe_candidates("S2", specs)}
    assert forward["S2"] == backward["S1"]


def test_ranking_exposes_the_scores_selection_orders_by() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"), ("a.py", "30-40", "g"))
    specs = [
        target,
        _mapped("S2", ("a.py", "10-20", "f")),
        _mapped("S3", ("a.py", "10-20", "f"), ("a.py", "30-40", "g")),
        _mapped("S5", ("a.py", "900-950", "z")),
    ]
    ranked = rank_probe_candidates("S1", specs)

    assert [cid for cid, _, _ in ranked] == select_probe_specs("S1", specs, 3)
    assert ranked[0][:2] == ("S3", 2)
    assert ranked[-1] == ("S5", 0, 0)


def test_select_falls_back_from_function_to_lines_to_file() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"))
    specs = [target, _mapped("S2", ("a.py", "800-900", "z")), _mapped("S3", ("a.py", "15-25", "F"))]
    assert select_probe_specs("S1", specs, 2) == ["S3", "S2"]


def test_closure_gives_the_widely_picked_spec_more_probes_than_k() -> None:
    selection = {
        "S1": ["S2"],
        "S2": ["S1"],
        "S3": ["S1"],
        "S4": ["S1"],
    }
    witnesses = {sid: "w" for sid in selection}

    added = symmetrise_probe_selection(selection, witnesses)

    assert added == 2
    assert sorted(selection["S1"]) == ["S2", "S3", "S4"]
    assert selection["S2"] == ["S1"]


def test_closure_gives_an_unpicked_spec_the_in_degree_it_needs() -> None:
    selection = {"S1": ["S3"], "S2": ["S1"], "S3": ["S1"]}
    witnesses = {sid: "w" for sid in selection}

    def probed_by(sid):
        return {other for other, ids in selection.items() if sid in ids}

    assert probed_by("S2") == set()
    symmetrise_probe_selection(selection, witnesses)
    assert probed_by("S2") == {"S1"}


def test_closure_skips_a_spec_with_no_witness() -> None:
    selection = {"S1": ["S2"], "S2": []}

    assert symmetrise_probe_selection(selection, {"S2": "w"}) == 0
    assert selection["S2"] == []


def test_select_refuses_unrelated_specs() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"))
    specs = [target, _mapped("S2", ("b.py", "10-20", "f")), _mapped("S3", ("c.py", "1-5", "h"))]
    assert select_probe_specs("S1", specs, 2) == []


def test_select_is_deterministic_and_clamped() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"))
    specs = [
        target,
        _mapped("S4", ("a.py", "10-20", "f")),
        _mapped("S2", ("a.py", "10-20", "f")),
        _mapped("S3", ("a.py", "10-20", "f")),
    ]
    assert select_probe_specs("S1", specs, 2) == ["S2", "S3"]
    assert select_probe_specs("S1", list(reversed(specs)), 2) == ["S2", "S3"]
    assert select_probe_specs("S1", specs, 0) == []
    assert select_probe_specs("S1", [target], 2) == []
    assert select_probe_specs("S9", specs, 2) == []


def test_select_tolerates_freeform_line_text() -> None:
    target = _mapped("S1", ("a.py", "L10-L20", "f"))
    specs = [
        target,
        _mapped("S2", ("a.py", "lines 18 through 24", "")),
        _mapped("S3", ("a.py", "see above", "")),
    ]
    assert select_probe_specs("S1", specs, 2) == ["S2", "S3"]


def test_select_ignores_specs_without_a_code_mapping() -> None:
    target = _mapped("S1", ("a.py", "10-20", "f"))
    assert select_probe_specs("S1", [target, _mapped("S2")], 2) == []
    assert select_probe_specs("S1", [_mapped("S1"), _mapped("S2", ("a.py", "10-20", "f"))], 2) == []


def test_preserved_when_observation_is_unchanged() -> None:
    probes, _, _ = _run({("gold", "a"): "same", ("mutant", "a"): "same"}, ["a"])
    assert [p.verdict for p in probes] == ["preserved"]


def test_diverged_when_observation_moves() -> None:
    probes, _, _ = _run({("gold", "a"): "one", ("mutant", "a"): "two"}, ["a"])
    assert probes[0].verdict == "diverged"
    assert probes[0].gold_observation == "one"
    assert probes[0].mutant_observation == "two"


def test_stray_output_does_not_make_a_probe_diverge() -> None:
    probes, _, _ = _run({("gold", "a"): "same", ("mutant", "a"): "same"}, ["a"])
    assert probes[0].verdict == "preserved"
    assert probes[0].gold_observation == "same"


def test_mutant_crash_is_its_own_verdict_not_a_violation() -> None:
    probes, _, _ = _run({("gold", "a"): "one", ("mutant", "a"): None}, ["a"])
    assert probes[0].verdict == "mutant_error"
    assert "boom" in probes[0].error


def test_a_witness_with_no_gold_reading_is_inconclusive() -> None:
    prober = PreservationProber(_FakeRunner({}))
    runner = _FakeRunner({("mutant", "a"): "two"})
    prober.docker_runner = runner
    probe = prober.probe_one(
        _make_pipeline_input(), "S1",
        [{"file": "a.py", "find": "x", "replace": "y"}],
        "S2", _witness("a"), timeout=30,
    )
    assert probe.verdict == "inconclusive"
    assert "never passed Step 4" in probe.error
    assert runner.setups == 0
    assert runner.runs == []


def test_bind_mismatch_is_muted_on_the_mutant_run() -> None:
    _, runner, _ = _run({("gold", "a"): "one", ("mutant", "a"): "two"}, ["a"])
    assert [state for state, _, _ in runner.runs] == ["mutant"]
    assert runner.runs[0][2] == {"WITNESS_ALLOW_BIND_MISMATCH": "1"}


def test_a_diverged_probe_is_still_a_successful_probe() -> None:
    probes, _, _ = _run({("gold", "a"): "one", ("mutant", "a"): "two"}, ["a"])
    assert probes[0].verdict == "diverged"
    assert probes[0].error == ""


def test_failed_mutation_apply_is_recorded_not_raised() -> None:
    prober = PreservationProber(_FakeRunner({}))
    prober.record_gold_observation("example.instance.lv1", _witness("a"), "one")
    runner = _FakeRunner({("gold", "a"): "one"})
    runner.apply_succeeds = False
    prober.docker_runner = runner
    probe = prober.probe_one(
        _make_pipeline_input(), "S1",
        [{"file": "a.py", "find": "x", "replace": "y"}],
        "S2", _witness("a"), timeout=30,
    )
    assert probe.verdict == "inconclusive"
    assert "Could not apply the mutation" in probe.error
    assert runner.setups == 1 and runner.cleanups == 1


def test_each_probe_gets_its_own_container() -> None:
    _, runner, _ = _run(
        {("gold", "a"): "one", ("mutant", "a"): "two",
         ("gold", "b"): "x", ("mutant", "b"): "x"},
        ["a", "b"],
    )
    assert runner.setups == 2
    assert runner.cleanups == 2


def test_the_probe_phase_has_no_gold_arm() -> None:
    _, runner, _ = _run({("gold", "a"): "one", ("mutant", "a"): "two"}, ["a"])
    assert [state for state, _, _ in runner.runs] == ["mutant"]
    assert runner.resets == 0


def test_step4_gold_observation_seeds_the_cache() -> None:
    prober = PreservationProber(_FakeRunner({}))
    prober.record_gold_observation("example.instance.lv1", _witness("a"), "one")
    runner = _FakeRunner({("mutant", "a"): "one"})
    prober.docker_runner = runner
    probe = prober.probe_one(
        _make_pipeline_input(), "S1",
        [{"file": "a.py", "find": "x", "replace": "y"}],
        "S2", _witness("a"), timeout=30,
    )
    assert probe.verdict == "preserved"
    assert [state for state, _, _ in runner.runs] == ["mutant"]


def test_cache_does_not_cross_witness_sources() -> None:
    prober = PreservationProber(_FakeRunner({}))
    prober.record_gold_observation("example.instance.lv1", _witness("a"), "one")
    runner = _FakeRunner({("mutant", "b"): "x"})
    prober.docker_runner = runner
    probe = prober.probe_one(
        _make_pipeline_input(), "S1",
        [{"file": "a.py", "find": "x", "replace": "y"}],
        "S2", _witness("b"), timeout=30,
    )
    assert probe.verdict == "inconclusive"


def test_empty_gold_observation_is_not_cached() -> None:
    prober = PreservationProber(_FakeRunner({}))
    prober.record_gold_observation("example.instance.lv1", _witness("a"), "")
    runner = _FakeRunner({("mutant", "a"): "one"})
    prober.docker_runner = runner
    probe = prober.probe_one(
        _make_pipeline_input(), "S1",
        [{"file": "a.py", "find": "x", "replace": "y"}],
        "S2", _witness("a"), timeout=30,
    )
    assert probe.verdict == "inconclusive"


def _spec(sid: str, text: str) -> AtomicSpec:
    return AtomicSpec(
        spec_id=sid, text=text, category=SpecCategory.BEHAVIORAL,
        sources=[SpecSource.DOCSTRING], code_mapping=[],
    )


def test_render_probes_attaches_spec_text() -> None:
    rows = _render_probes(
        [PreservationProbe("S2", "diverged", gold_observation="a", mutant_observation="b")],
        [_spec("S2", "must stay sorted")],
    )
    assert rows == [{
        "spec_id": "S2",
        "spec_text": "must stay sorted",
        "verdict": "diverged",
        "gold_observation": "a",
        "mutant_observation": "b",
        "error": "",
    }]


def test_render_probes_never_trims_an_observation() -> None:
    observation = "HEAD" + ("x" * 50_000) + "TAIL"
    rows = _render_probes(
        [PreservationProbe("S2", "diverged", gold_observation=observation)],
        [_spec("S2", "must stay sorted")],
    )
    assert rows[0]["gold_observation"] == observation


def test_render_probes_keeps_both_ends_of_oversized_stray_output() -> None:
    from mutation_pipeline.steps.step6_llm_verification import (
        _STRAY_PROMPT_CHARS,
        _window,
    )

    shown = _window("HEAD" + ("x" * _STRAY_PROMPT_CHARS) + "TAIL")
    assert shown.startswith("HEAD")
    assert shown.endswith("TAIL")
    assert "elided" in shown


def test_render_probes_drops_specs_missing_from_the_preserve_list() -> None:
    assert _render_probes([PreservationProbe("S9", "diverged")], [_spec("S2", "t")]) == []


class _MockLLMCaller:
    def __init__(self, response: str):
        self.response = response
        self.prompts: List[str] = []

    def call_llm(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        return {"_response": self.response}

    def get_llm_response_text(self, completion):
        return completion["_response"]


def _prompt_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            str(_SC_ALIGN / "mutation_pipeline" / "prompts")
        ),
        undefined=jinja2.StrictUndefined,
    )


def test_the_probe_reaches_the_preservation_prompt() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "reject", "category": "wrong_spec", '
        '"reasoning": "the probed spec constrains the part that moved"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = verifier.verify_preservation(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
        other_specs=[_spec("S2", "must stay sorted")],
        probe=PreservationProbe("S2", "diverged", gold_observation="[1, 2]",
                                mutant_observation="[2, 1]"),
    )
    assert result.success is False
    assert result.data["category"] == "wrong_spec"
    prompt = llm.prompts[0]
    assert "Preservation probe: S2" in prompt
    assert "must stay sorted" in prompt
    assert "[2, 1]" in prompt


def test_one_call_carries_one_probe() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", "reasoning": "ok"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    verifier.verify_preservation(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
        other_specs=[_spec("S2", "sorted"), _spec("S3", "non-empty")],
        probe=PreservationProbe("S2", "diverged", gold_observation="[1, 2]",
                                mutant_observation="[2, 1]"),
    )
    prompt = llm.prompts[0]
    assert "Preservation probe: S2" in prompt
    assert "Preservation probe: S3" not in prompt
    assert "Judge only the probed spec named below." in prompt


def test_preservation_prompt_renders_without_a_probe() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", "reasoning": "ok"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = verifier.verify_preservation(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
        other_specs=[_spec("S2", "must stay sorted")],
        probe=None,
    )
    assert result.success is True
    assert "No probe for this audit" in llm.prompts[0]
    assert "must stay sorted" in llm.prompts[0]


def test_target_prompt_carries_no_probes_and_no_other_specs() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", "reasoning": "ok"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = verifier.verify_target(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.success is True
    prompt = llm.prompts[0]
    assert "Preservation probe" not in prompt
    assert "Other specs (must remain intact)" not in prompt
    assert "wrong_spec" not in prompt


def test_each_half_refuses_the_other_half_s_category() -> None:
    target = LLMVerifier(
        llm_caller=_MockLLMCaller(
            '```json\n{"verdict": "reject", "category": "wrong_spec", '
            '"reasoning": "not this call\'s question"}\n```'
        ),
        prompt_env=_prompt_env(),
    )
    result = target.verify_target(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.data["category"] == "infra_error"

    preservation = LLMVerifier(
        llm_caller=_MockLLMCaller(
            '```json\n{"verdict": "reject", "category": "noise", '
            '"reasoning": "not this call\'s question"}\n```'
        ),
        prompt_env=_prompt_env(),
    )
    result = preservation.verify_preservation(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
        other_specs=[_spec("S2", "sorted")],
        probe=None,
    )
    assert result.data["category"] == "infra_error"


def test_stray_output_reaches_the_prompt_labelled_as_not_evidence() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", "reasoning": "ok"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    verifier.verify_target(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
        gold_stray_output="[0/1] Regenerating build files.",
        mutant_stray_output="[1/1] Generating write_version_file",
    )
    prompt = llm.prompts[0]
    assert "NOT part of the observation" in prompt
    assert "[0/1] Regenerating build files." in prompt
    assert "[1/1] Generating write_version_file" in prompt


def test_prompt_omits_the_stray_section_when_both_runs_were_quiet() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "approve", "category": "valid", "reasoning": "ok"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    verifier.verify_target(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
    )
    assert "NOT part of the observation" not in llm.prompts[0]


def test_unreachable_trigger_is_an_accepted_category() -> None:
    llm = _MockLLMCaller(
        '```json\n{"verdict": "reject", "category": "unreachable_trigger", '
        '"reasoning": "trigger exists only inside the mutation"}\n```'
    )
    verifier = LLMVerifier(llm_caller=llm, prompt_env=_prompt_env())
    result = verifier.verify_target(
        spec=_spec("S1", "targeted"),
        mutant_patch="diff",
        proof=Proof(witness_script="def test_x(): pass"),
        pipeline_input=_make_pipeline_input(),
    )
    assert result.success is False
    assert result.data["category"] == "unreachable_trigger"


class _RecordingVerifier:
    def __init__(self, verdicts: Dict[str, Tuple[bool, str]]):
        self.verdicts = verdicts
        self.asked: List[Optional[str]] = []

    def verify_preservation(self, spec, mutant_patch, proof, pipeline_input,
                            other_specs, probe=None, **kwargs):
        key = probe.spec_id if probe is not None else "patch-audit"
        self.asked.append(key)
        success, category = self.verdicts.get(key, (True, "valid"))
        return StepResult(
            success=success,
            error=None if success else f"rejected ({category})",
            data={"category": category, "verdict": "approve" if success else "reject"},
            step_name="llm_verification",
        )


def _audit_pipeline(verifier: _RecordingVerifier, tmp_path: Path):
    pipeline = MutationPipeline.__new__(MutationPipeline)
    pipeline.llm_verifier = verifier
    pipeline.output_dir = tmp_path
    pipeline._stats = {}
    pipeline.config = SimpleNamespace(step6b_concurrency=1)
    return pipeline


def _audit(pipeline, probes: List[PreservationProbe]) -> Optional[str]:
    return pipeline._run_preservation_audit(
        _spec("S1", "targeted"),
        "diff",
        Proof(witness_script="def test_x(): pass"),
        _make_pipeline_input(),
        [_spec("S2", "sorted"), _spec("S3", "non-empty")],
        probes,
        "strategy",
        "",
        "",
        1,
    )


def test_every_probe_gets_a_call_whatever_its_verdict(tmp_path: Path) -> None:
    verifier = _RecordingVerifier({})
    pipeline = _audit_pipeline(verifier, tmp_path)
    broken = _audit(pipeline, [
        PreservationProbe("S2", "diverged", gold_observation="[1]",
                          mutant_observation="[2]"),
        PreservationProbe("S3", "preserved", gold_observation="x",
                          mutant_observation="x"),
    ])
    assert broken is None
    assert verifier.asked == ["S2", "S3", "patch-audit"]


def test_a_preserved_probe_can_still_be_judged_wrong_spec(
    tmp_path: Path,
) -> None:
    verifier = _RecordingVerifier({"S3": (False, "wrong_spec")})
    pipeline = _audit_pipeline(verifier, tmp_path)
    broken = _audit(pipeline, [
        PreservationProbe("S3", "preserved", gold_observation="x",
                          mutant_observation="x"),
    ])
    assert broken == "S3"


def test_the_patch_audit_runs_even_with_no_probes(tmp_path: Path) -> None:
    verifier = _RecordingVerifier({})
    pipeline = _audit_pipeline(verifier, tmp_path)
    assert _audit(pipeline, []) is None
    assert verifier.asked == ["patch-audit"]


def test_a_wrong_spec_verdict_names_the_spec_it_broke(tmp_path: Path) -> None:
    verifier = _RecordingVerifier({"S2": (False, "wrong_spec")})
    pipeline = _audit_pipeline(verifier, tmp_path)
    broken = _audit(pipeline, [
        PreservationProbe("S2", "diverged", gold_observation="[1]",
                          mutant_observation="[2]"),
    ])
    assert broken == "S2"


def test_an_infra_failure_is_not_a_preservation_verdict(tmp_path: Path) -> None:
    verifier = _RecordingVerifier({"patch-audit": (False, "infra_error")})
    pipeline = _audit_pipeline(verifier, tmp_path)
    assert _audit(pipeline, []) is None


def test_each_call_lands_in_its_own_artifact(tmp_path: Path) -> None:
    verifier = _RecordingVerifier({})
    pipeline = _audit_pipeline(verifier, tmp_path)
    _audit(pipeline, [
        PreservationProbe("S2", "diverged", gold_observation="[1]",
                          mutant_observation="[2]"),
        PreservationProbe("S3", "diverged", gold_observation="a",
                          mutant_observation="b"),
    ])
    written = {
        p.name
        for p in (tmp_path / "example.instance.lv1" / "artifacts").glob("*.json")
    }
    assert written == {
        "S1_step6b_preservation_S2.json",
        "S1_step6b_preservation_S3.json",
        "S1_step6b_preservation_patch-audit.json",
    }


def _verified_record(spec_id: str, witness_tag: str):
    from mutation_pipeline.pipeline import _VerifiedSpec

    spec = _spec(spec_id, f"spec {spec_id}")
    return _VerifiedSpec(
        spec=spec,
        attempt=1,
        step2_result=StepResult(
            success=True, error=None,
            data={"mutations": [{"file": "a.py", "find": "x", "replace": "y"}]},
            step_name="mutation_generation",
        ),
        step2_time=0.0,
        mutations=[{"file": "a.py", "find": "x", "replace": "y"}],
        mutant_patch="diff",
        proof=Proof(witness_script=_witness(witness_tag)),
        mutation_strategy="s",
        gold_stray_output="",
        mutant_stray_output="",
        filter_survival=FilterSurvival(),
        step_timings={},
    )


def test_only_verified_witnesses_are_frozen_for_probing(tmp_path: Path) -> None:
    from mutation_pipeline.pipeline import MutationPipeline

    pipeline = MutationPipeline.__new__(MutationPipeline)
    pipeline.output_dir = tmp_path
    pipeline._stats = {"specs_merged": 0}
    pipeline.config = SimpleNamespace(
        container_pool_size=4, preservation_probe_k=2, step2_max_attempts=3,
    )
    specs = [_spec("S1", "one"), _spec("S2", "two"), _spec("S3", "three")]
    verified = {
        "S1": _verified_record("S1", "a"),
        "S3": _verified_record("S3", "c"),
    }
    pipeline._run_verification_phase = lambda *a, **k: verified

    seen: Dict[str, Any] = {}

    def fake_probe_phase(kept, pipeline_input, records, witness_by_spec):
        seen["kept"] = [s.spec_id for s in kept]
        seen["witnesses"] = sorted(witness_by_spec)
        return []

    pipeline._run_probe_phase = fake_probe_phase
    pipeline._run_pipelined(specs, _make_pipeline_input())

    assert seen["kept"] == ["S1", "S3"]
    assert seen["witnesses"] == ["S1", "S3"]


def test_worker_counts_are_folded_without_loss(tmp_path: Path) -> None:
    from mutation_pipeline.pipeline import MutationPipeline

    pipeline = MutationPipeline.__new__(MutationPipeline)
    pipeline.output_dir = tmp_path
    pipeline._stats = {"step3_pass": 0}
    pipeline.config = SimpleNamespace(container_pool_size=8, step2_max_attempts=3)
    specs = [_spec(f"S{i}", "t") for i in range(1, 41)]

    def fake_verify_spec(spec, pipeline_input, all_specs):
        return None, {"step3_pass": 1, "step4_diverged": 2}

    pipeline._verify_spec = fake_verify_spec
    pipeline._run_verification_phase(specs, _make_pipeline_input())

    assert pipeline._stats["step3_pass"] == 40
    assert pipeline._stats["step4_diverged"] == 80
