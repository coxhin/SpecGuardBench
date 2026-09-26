"""Step 6: LLM-B Verification."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import jinja2

from featurebench.llm.llm_caller import LLMCaller

from ..models import AtomicSpec, PipelineInput, PreservationProbe, Proof, StepResult

logger = logging.getLogger(__name__)


_VALID_VERDICTS = {"approve", "reject"}

_INFRA = "infra_error"


def _tally(
    ballots: List[StepResult],
    instance_id: str,
    spec_id: str,
    label: str,
    votes: int,
) -> StepResult:
    """Reduce independent ballots to one verdict, keeping all of them."""
    scored = [b for b in ballots if (b.data or {}).get("category") != _INFRA]
    approve = [b for b in scored if b.success]
    reject = [b for b in scored if not b.success]

    ballot_records = [
        {
            "verdict": (b.data or {}).get("verdict"),
            "category": (b.data or {}).get("category"),
            "reasoning": (b.data or {}).get("reasoning", ""),
            "response_text": (b.data or {}).get("response_text", ""),
        }
        for b in ballots
    ]
    tally = {
        "votes_requested": votes,
        "approve": len(approve),
        "reject": len(reject),
        "infra_error": len(ballots) - len(scored),
        "unanimous": bool(scored) and (not approve or not reject),
    }

    if not scored:
        winner = ballots[0]
    elif len(approve) > len(reject):
        winner = approve[0]
    else:
        winner = reject[0] if reject else ballots[0]

    if not tally["unanimous"] and scored:
        logger.info(
            f"[{instance_id}][{spec_id}] {label}: split panel "
            f"{tally['approve']}-{tally['reject']}"
            f"{' (+%d infra)' % tally['infra_error'] if tally['infra_error'] else ''}"
            f" -> {'approve' if winner.success else 'reject'}"
        )

    data = dict(winner.data or {})
    data["ballots"] = ballot_records
    data["vote_tally"] = tally
    return StepResult(
        success=winner.success,
        error=winner.error,
        data=data,
        step_name=winner.step_name,
    )
_TARGET_CATEGORIES = {
    "valid", "overreach", "unreachable_trigger", "noise", "non_behavioral",
}
_PRESERVATION_CATEGORIES = {"valid", "wrong_spec"}
_VALID_CATEGORIES = _TARGET_CATEGORIES | _PRESERVATION_CATEGORIES

_STRAY_PROMPT_CHARS = 4000


def _window(text: str, limit: int = _STRAY_PROMPT_CHARS) -> str:
    """Trim ``text`` to ``limit`` characters, keeping both ends."""
    if not text or len(text) <= limit:
        return text or ""
    head = limit // 2
    tail = limit - head
    dropped = len(text) - limit
    return f"{text[:head]}\n... [{dropped} characters elided] ...\n{text[-tail:]}"


class LLMVerifier:
    """Verifies mutation/proof validity using a different LLM (LLM-B)."""

    def __init__(
        self,
        llm_caller: LLMCaller,
        prompt_env: jinja2.Environment,
        votes: int = 3,
    ):
        self.llm_caller = llm_caller
        self.prompt_env = prompt_env
        self.votes = max(1, votes)

    def verify_target(
        self,
        spec: AtomicSpec,
        mutant_patch: str,
        proof: Proof,
        pipeline_input: PipelineInput,
        mutation_strategy: str = "",
        gold_stray_output: str = "",
        mutant_stray_output: str = "",
    ) -> StepResult:
        """Step 6a: does this mutation violate the spec it targeted?"""
        return self._verify(
            template_name="llm_verification_target.j2",
            label="Step 6a",
            allowed_categories=_TARGET_CATEGORIES,
            pipeline_input=pipeline_input,
            spec_id=spec.spec_id,
            context={
                "spec": spec,
                "other_specs": [],
                "mutation_strategy": mutation_strategy,
                "trigger_condition": proof.trigger_condition,
                "mutant_patch": mutant_patch,
                "witness_script": proof.witness_script,
                "case_contract": proof.case_contract,
                "witness_factory": proof.witness_factory,
                "witness_observer": proof.witness_observer,
                "actual_gold_output": proof.actual_gold_output,
                "actual_mutant_output": proof.actual_mutant_output,
                "gold_stray_output": _window(gold_stray_output),
                "mutant_stray_output": _window(mutant_stray_output),
            },
        )

    def verify_preservation(
        self,
        spec: AtomicSpec,
        mutant_patch: str,
        proof: Proof,
        pipeline_input: PipelineInput,
        other_specs: List[AtomicSpec],
        probe: Optional[PreservationProbe] = None,
        mutation_strategy: str = "",
        gold_stray_output: str = "",
        mutant_stray_output: str = "",
    ) -> StepResult:
        """Step 6b: did this mutation break a spec it was supposed to preserve?"""
        return self._verify(
            template_name="llm_verification_preservation.j2",
            label=(
                f"Step 6b[{probe.spec_id}]" if probe is not None
                else "Step 6b[patch-audit]"
            ),
            allowed_categories=_PRESERVATION_CATEGORIES,
            pipeline_input=pipeline_input,
            spec_id=spec.spec_id,
            context={
                "spec": spec,
                "other_specs": other_specs,
                "probe": (
                    _render_probe(probe, other_specs) if probe is not None
                    else None
                ),
                "mutation_strategy": mutation_strategy,
                "trigger_condition": proof.trigger_condition,
                "mutant_patch": mutant_patch,
                "witness_script": proof.witness_script,
                "case_contract": proof.case_contract,
                "witness_factory": proof.witness_factory,
                "witness_observer": proof.witness_observer,
                "actual_gold_output": proof.actual_gold_output,
                "actual_mutant_output": proof.actual_mutant_output,
                "gold_stray_output": _window(gold_stray_output),
                "mutant_stray_output": _window(mutant_stray_output),
            },
        )

    def _verify(
        self,
        template_name: str,
        label: str,
        allowed_categories: set,
        pipeline_input: PipelineInput,
        spec_id: str,
        context: Dict[str, Any],
    ) -> StepResult:
        """Poll the judge ``self.votes`` times and return the majority verdict."""
        ballots = [
            self._verify_once(
                template_name, label, allowed_categories,
                pipeline_input, spec_id, context, ballot=i + 1,
            )
            for i in range(self.votes)
        ]
        return _tally(ballots, pipeline_input.instance_id, spec_id, label, self.votes)

    def _verify_once(
        self,
        template_name: str,
        label: str,
        allowed_categories: set,
        pipeline_input: PipelineInput,
        spec_id: str,
        context: Dict[str, Any],
        ballot: int = 1,
    ) -> StepResult:
        """Render one LLM-B prompt, call, and parse the verdict."""
        iid = pipeline_input.instance_id
        sid = spec_id
        label = f"{label}[vote {ballot}]"

        try:
            template = self.prompt_env.get_template(template_name)
            prompt = template.render(**context)
            messages = [{"role": "user", "content": prompt}]
            t0 = time.time()
            completion = self.llm_caller.call_llm(messages)
            llm_elapsed = time.time() - t0
            response_text = self.llm_caller.get_llm_response_text(completion)
            logger.info(
                f"[{iid}][{sid}] {label}: LLM-B call took {llm_elapsed:.1f}s "
                f"(prompt {len(prompt)} chars, response {len(response_text)} chars)"
            )

            parsed = _parse_verdict(response_text, allowed_categories)
            verdict = parsed["verdict"]
            category = parsed["category"]
            approved = verdict == "approve" and category == "valid"

            logger.info(
                f"[{iid}][{sid}] {label}: LLM-B {verdict} ({category}) — "
                f"{parsed['reasoning'][:160]}"
            )

            return StepResult(
                success=approved,
                error=None if approved else (
                    f"LLM-B rejected ({category}): {parsed['reasoning'][:200]}"
                ),
                data={
                    "verdict": verdict,
                    "category": category,
                    "reasoning": parsed["reasoning"],
                    "prompt": prompt,
                    "response_text": response_text,
                },
                step_name="llm_verification",
            )

        except Exception as e:
            logger.error(
                f"[{iid}][{sid}] {label} LLM-B failed: {e}", exc_info=True
            )
            return StepResult(
                success=False,
                error=f"LLM-B verifier crashed: {e}",
                data={"verdict": "reject", "category": "infra_error",
                      "reasoning": str(e)},
                step_name="llm_verification",
            )


def _render_probes(
    probes: List[PreservationProbe],
    other_specs: List[AtomicSpec],
) -> List[Dict[str, str]]:
    """Flatten Step 5 probes into template rows, each carrying its spec text."""
    text_by_id = {s.spec_id: s.text for s in other_specs}
    rows: List[Dict[str, str]] = []
    for probe in probes:
        if probe.spec_id not in text_by_id:
            continue
        rows.append({
            "spec_id": probe.spec_id,
            "spec_text": text_by_id[probe.spec_id],
            "verdict": probe.verdict,
            "gold_observation": probe.gold_observation,
            "mutant_observation": probe.mutant_observation,
            "error": probe.error,
        })
    return rows


def _render_probe(
    probe: PreservationProbe,
    other_specs: List[AtomicSpec],
) -> Optional[Dict[str, str]]:
    """One probe's template row, or None when its spec text is unavailable."""
    rows = _render_probes([probe], other_specs)
    return rows[0] if rows else None


def _parse_verdict(
    response_text: str,
    allowed_categories: set = _VALID_CATEGORIES,
) -> Dict[str, Any]:
    """Extract the LLM-B verdict JSON from the response."""
    raw = _extract_json(response_text)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM-B response is not a JSON object: {type(parsed).__name__}")

    verdict = (parsed.get("verdict") or "").strip().lower()
    category = (parsed.get("category") or "").strip().lower()
    reasoning = parsed.get("reasoning") or ""

    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"LLM-B verdict must be one of {sorted(_VALID_VERDICTS)}, got: {verdict!r}"
        )
    if category not in allowed_categories:
        raise ValueError(
            f"LLM-B category must be one of {sorted(allowed_categories)}, "
            f"got: {category!r}"
        )
    if verdict == "approve" and category != "valid":
        raise ValueError(
            f"LLM-B verdict 'approve' requires category 'valid', got: {category!r}"
        )

    return {
        "verdict": verdict,
        "category": category,
        "reasoning": str(reasoning),
    }


def _extract_json(text: str) -> str:
    """Return the last fenced JSON block parseable as a dict."""
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
