"""Step 1: Spec Decomposition + Mapping."""

import json
import logging
import re
import time
from typing import List

import jinja2

from featurebench.llm.llm_caller import LLMCaller

from ..models import (
    AtomicSpec,
    CodeMapping,
    PipelineInput,
    SpecCategory,
    SpecSource,
    StepResult,
)

logger = logging.getLogger(__name__)


class SpecDecomposer:
    """Decomposes an instance's specs into atomic specs via LLM."""

    def __init__(self, llm_caller: LLMCaller, prompt_env: jinja2.Environment):
        self.llm_caller = llm_caller
        self.prompt_env = prompt_env

    def decompose(self, pipeline_input: PipelineInput) -> StepResult:
        """Run spec decomposition on a single instance."""
        try:
            template = self.prompt_env.get_template("spec_decomposition.j2")
            prompt = template.render(
                task_statement=pipeline_input.task_statement,
                interface_descriptions=pipeline_input.interface_descriptions,
                gold_patch=pipeline_input.gold_patch,
            )
            prompt_len = len(prompt)

            messages = [{"role": "user", "content": prompt}]
            t0 = time.time()
            completion = self.llm_caller.call_llm(messages)
            llm_elapsed = time.time() - t0
            response_text = self.llm_caller.get_llm_response_text(completion)
            logger.info(
                f"[{pipeline_input.instance_id}] Step 1: LLM call took {llm_elapsed:.1f}s "
                f"(prompt {prompt_len} chars, response {len(response_text)} chars)"
            )

            specs = self._parse_specs(response_text)

            logger.info(
                f"[{pipeline_input.instance_id}] Step 1: Decomposed into {len(specs)} specs"
            )
            return StepResult(
                success=True,
                data=specs,
                step_name="spec_decomposition",
                raw_response=response_text,
            )

        except Exception as e:
            logger.error(
                f"[{pipeline_input.instance_id}] Step 1 failed: {e}", exc_info=True
            )
            return StepResult(
                success=False,
                error=str(e),
                step_name="spec_decomposition",
            )

    def _parse_specs(self, response_text: str) -> List[AtomicSpec]:
        """Extract JSON array of specs from LLM response."""
        json_text = self._extract_json(response_text)
        raw_specs = json.loads(json_text)

        if not isinstance(raw_specs, list):
            raise ValueError(f"Expected JSON array, got {type(raw_specs).__name__}")

        specs = []
        for raw in raw_specs:
            spec = self._parse_single_spec(raw)
            specs.append(spec)
        return specs

    def _parse_single_spec(self, raw: dict) -> AtomicSpec:
        """Parse a single spec dict into an AtomicSpec."""
        cat_str = raw.get("category", "behavioral")
        try:
            category = SpecCategory(cat_str)
        except ValueError:
            category = SpecCategory.BEHAVIORAL

        sources = []
        for s in raw.get("sources", []):
            try:
                sources.append(SpecSource(s))
            except ValueError:
                pass
        if not sources:
            sources = [SpecSource.TASK_STATEMENT]

        code_mapping = []
        for cm in raw.get("code_mapping", []):
            code_mapping.append(CodeMapping.from_dict(cm))

        independence_check = raw.get("independence_check")
        if not isinstance(independence_check, dict) or not independence_check:
            independence_check = None

        return AtomicSpec(
            spec_id=raw.get("spec_id", ""),
            text=raw.get("text", ""),
            category=category,
            sources=sources,
            code_mapping=code_mapping,
            merged_from=raw.get("merged_from"),
            independence_check=independence_check,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may be wrapped in markdown code fences."""
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()
