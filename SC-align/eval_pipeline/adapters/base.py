"""Base adapter for evaluation pipeline baselines."""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docker.models.containers import Container

from featurebench.harness.container import (
    copy_to_container,
    exec_run_with_timeout,
)
from featurebench.harness.constants import UTF8

import re

from ..models import BaselineResult, EvalTask, LLMCallError, SpecCertificate
from ..trajectory import session_chains

logger = logging.getLogger(__name__)

STEP_GATE_SRC = (
    Path(__file__).resolve().parents[3]
    / "featurebench"
    / "resources"
    / "step_gate.py"
)
STEP_GATE_PATH = "/opt/eval/step_gate.py"
STEP_CAP_EVENT_TYPE = "step_cap_reached"
STEP_CAP_REFUSAL_MARKER = "step cap reached"
STEP_GATE_STDERR_PREFIX = "step_gate:"
AGENT_PID_PATH = "/tmp/eval_agent.pid"

SUCCESSFUL_LLM_CALLS = "successful_llm_calls"


class EvalBaselineAdapter(ABC):
    """Abstract base for evaluation baseline adapters."""

    def __init__(
        self,
        name: str,
        model: str,
        tool_version: str,
        agent_config: Dict[str, Any],
        env_vars: Dict[str, str],
        max_steps: Optional[int] = None,
    ):
        self.name = name
        self.model = model
        self.tool_version = tool_version
        self.agent_config = agent_config
        self.env_vars = env_vars
        self.max_steps = max_steps
        self.egress: Optional[Any] = None

    @abstractmethod
    def get_install_script(self) -> str:
        """Return shell script to install the tool at self.tool_version."""
        ...

    @abstractmethod
    def get_run_command(self, prompt_path: str) -> str:
        """Return shell command to invoke the tool with the prompt file."""
        ...

    def setup_container(
        self,
        container: Container,
        task: EvalTask,
        prompt_text: str,
    ) -> bool:
        """Set up the container for baseline execution."""
        self._write_file_to_container(
            container, prompt_text, "/tmp/eval_prompt.txt"
        )

        if self.egress is not None and self.egress.is_restricted(container):
            logger.info(
                f"[{self.name}/{self.model}] tool already installed before "
                f"egress was restricted; not reinstalling"
            )
        elif not self._install_tool(container):
            return False

        return self._install_step_gate(container)

    def _with_egress(self, cmd: str) -> str:
        """Prefix a CLI command with the proxy variables when egress is restricted."""
        if self.egress is None:
            return cmd
        return self.egress.shell_prelude() + cmd

    def enforced_max_steps(self) -> Optional[int]:
        """The cap this baseline will actually be held to, or None."""
        if not self.max_steps or self.step_gate_unit() is None:
            return None
        return self.max_steps

    def _remaining_steps(self, spent: int) -> Optional[int]:
        """What is left of the cap after ``spent`` model calls, or None."""
        cap = self.enforced_max_steps()
        return None if cap is None else cap - spent

    def _install_step_gate(self, container: Container) -> bool:
        """Put the step gate in the container, when this run is capped."""
        if not self.max_steps:
            return True

        unit = self.step_gate_unit()
        if unit is None:
            logger.warning(
                f"[{self.name}/{self.model}] max_steps={self.max_steps} is set "
                f"but this adapter has no step-counting rule; running uncapped"
            )
            return True

        exit_code, output = exec_run_with_timeout(
            container, f"mkdir -p {Path(STEP_GATE_PATH).parent}", timeout=30
        )
        if exit_code != 0:
            logger.error(
                f"[{self.name}/{self.model}] could not create the step gate's "
                f"directory: {output.decode(UTF8, errors='replace')[:300]}"
            )
            return False

        self._write_file_to_container(
            container,
            STEP_GATE_SRC.read_text(encoding=UTF8),
            STEP_GATE_PATH,
        )
        exit_code, output = exec_run_with_timeout(
            container,
            f"python3 {STEP_GATE_PATH} 2>&1 | grep -q usage:",
            timeout=30,
        )
        if exit_code != 0:
            logger.error(
                f"[{self.name}/{self.model}] step gate is not runnable in this "
                f"container (max_steps={self.max_steps}): "
                f"{output.decode(UTF8, errors='replace')[:300]}"
            )
            return False

        logger.info(
            f"[{self.name}/{self.model}] step gate installed "
            f"(unit={unit}, max_steps={self.max_steps})"
        )
        return True

    def step_gate_unit(self) -> Optional[str]:
        """Which counting rule in ``step_gate.py`` holds this CLI to the cap."""
        return None

    def step_gate_source(self) -> Optional[str]:
        """Container path of the count a wire unit polls, or None."""
        return None

    def _piped_cli(
        self,
        cli_cmd: str,
        transcript_path: str,
        max_steps: Optional[int] = None,
    ) -> str:
        """The tail of a run command: merge stderr, tee, and cap."""
        if max_steps is None:
            max_steps = self.enforced_max_steps()
        if max_steps is None:
            return f"{cli_cmd} 2>&1 | tee {transcript_path}"

        source = self.step_gate_source()
        return (
            f"( {cli_cmd} 2>&1 & echo $! > {AGENT_PID_PATH}; wait $! ) "
            f"| tee {transcript_path} "
            f"| python3 {STEP_GATE_PATH} {max_steps} "
            f"{self.step_gate_unit()} {AGENT_PID_PATH}"
            + (f" {source}" if source else "")
        )

    MAX_RUN_RETRIES = 3
    RETRY_WAIT_SECONDS = 15

    def run(
        self,
        container: Container,
        task: EvalTask,
        prompt_text: str,
        timeout: Optional[int],
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> BaselineResult:
        """Execute the baseline and collect results."""
        t0 = time.time()

        try:
            exec_run_with_timeout(
                container, "rm -f /tmp/eval_output.json", timeout=10
            )
        except Exception as e:
            logger.debug(
                f"[{self.name}/{self.model}] pre-run cleanup of "
                f"/tmp/eval_output.json failed (continuing): {e}"
            )

        logger.info(
            f"[{self.name}/{self.model}] Setting up container for "
            f"{task.instance_id} "
            f"(container={container.short_id})"
        )
        if not self.setup_container(container, task, prompt_text):
            return self._error_result(task, "Installation failed", time.time() - t0)
        logger.info(
            f"[{self.name}/{self.model}] Container setup completed in "
            f"{time.time() - t0:.1f}s"
        )

        if self.egress is not None and not self.egress.is_restricted(container):
            try:
                self.egress.restrict(container)
            except Exception as e:
                logger.error(
                    f"[{self.name}/{self.model}] could not restrict egress: {e}"
                )
                return self._error_result(
                    task, f"Egress restriction failed: {e}", time.time() - t0
                )

        run_cmd = self._with_egress(self.get_run_command("/tmp/eval_prompt.txt"))
        logger.debug(f"[{self.name}/{self.model}] Run command: {run_cmd}")

        all_llm_errors: List[LLMCallError] = []
        prior = resume_from or {}
        attempts_log: List[Dict[str, Any]] = list(prior.get("attempts") or [])
        attempt_offset = len(attempts_log)
        last_session_id: Optional[str] = prior.get("session_id") or None
        chain_steps = int(prior.get("chain_steps") or 0)
        budget_spent = False
        attempt_no = attempt_offset
        raw_output = ""
        exit_code = -1
        run_duration = 0.0
        certificates: Dict[str, Any] = {}
        cost: Dict[str, float] = {}
        step_cap: Optional[Dict[str, Any]] = None
        llm_errors: List[LLMCallError] = []

        for attempt in range(1, self.MAX_RUN_RETRIES + 1):
            attempt_no = attempt + attempt_offset
            cmd = run_cmd
            resumed = False
            resume_of: Optional[str] = None
            if last_session_id:
                remaining = self._remaining_steps(chain_steps)
                if remaining is not None and remaining <= 0:
                    logger.info(
                        f"[{self.name}/{self.model}] Budget spent "
                        f"({chain_steps}/{self.enforced_max_steps()} model "
                        f"calls); not resuming"
                    )
                    budget_spent = True
                    break
                resume_cmd = self.get_resume_command(
                    "/tmp/eval_resume_prompt.txt",
                    last_session_id,
                    max_steps=remaining,
                )
                if resume_cmd:
                    self._write_file_to_container(
                        container,
                        self._build_resume_prompt(task),
                        "/tmp/eval_resume_prompt.txt",
                    )
                    cmd = self._with_egress(resume_cmd)
                    resumed = True
                    resume_of = last_session_id
                    logger.info(
                        f"[{self.name}/{self.model}] Resuming session "
                        f"{last_session_id} for attempt {attempt}"
                        + (
                            f" with {remaining} of "
                            f"{self.enforced_max_steps()} model calls left"
                            if remaining is not None else ""
                        )
                    )

            run_t0 = time.time()
            exit_code, output = exec_run_with_timeout(
                container, cmd, timeout=timeout
            )
            raw_output = output.decode(UTF8, errors="replace")
            run_duration = time.time() - run_t0

            if resumed and exit_code != 0 and not self._has_stream_events(raw_output):
                logger.warning(
                    f"[{self.name}/{self.model}] Resume rejected "
                    f"(exit_code={exit_code}, no events after "
                    f"{run_duration:.1f}s); falling back to a fresh session. "
                    f"Output (first 300 chars): {raw_output[:300]}"
                )
                attempts_log.append(self._attempt_record(
                    attempt_no, resumed=True, resume_of_session=resume_of,
                    resume_rejected=True, exit_code=exit_code,
                    duration=run_duration, raw_output=raw_output,
                    llm_errors=[],
                ))
                run_t0 = time.time()
                exit_code, output = exec_run_with_timeout(
                    container, run_cmd, timeout=timeout
                )
                raw_output = output.decode(UTF8, errors="replace")
                run_duration = time.time() - run_t0
                resumed = False
                resume_of = None

            logger.info(
                f"[{self.name}/{self.model}] Run finished "
                f"(attempt {attempt}/{self.MAX_RUN_RETRIES}"
                f"{', resumed' if resumed else ''}): "
                f"exit_code={exit_code}, output_length={len(raw_output)}, "
                f"run_duration={run_duration:.1f}s"
            )
            if exit_code != 0:
                logger.warning(
                    f"[{self.name}/{self.model}] Non-zero exit code. "
                    f"Output (first 1000 chars): {raw_output[:1000]}"
                )

            session_id = self._extract_session_id(raw_output)
            if session_id:
                last_session_id = session_id

            step_cap = self._step_cap_event(raw_output)
            if step_cap:
                logger.warning(
                    f"[{self.name}/{self.model}] Step cap reached: "
                    f"{step_cap.get('count')}/{step_cap.get('limit')} "
                    f"{step_cap.get('unit')} model calls — agent stopped"
                )

            llm_errors = self._detect_llm_errors(raw_output)
            all_llm_errors.extend(llm_errors)

            record = self._attempt_record(
                attempt_no, resumed=resumed, resume_of_session=resume_of,
                resume_rejected=False, exit_code=exit_code,
                duration=run_duration, raw_output=raw_output,
                llm_errors=llm_errors, step_cap=step_cap,
            )
            attempts_log.append(record)
            spent = record.get("steps", {}).get(SUCCESSFUL_LLM_CALLS) or 0
            chain_steps = chain_steps + spent if resumed else spent

            if llm_errors:
                for err in llm_errors:
                    logger.warning(
                        f"[{self.name}/{self.model}] LLM error detected: "
                        f"{err.error_type} x{err.count} — "
                        f"{err.sample_message[:200]}"
                    )

            certificates, cost = self._parse_output(container, raw_output)

            if certificates:
                break

            if step_cap:
                logger.info(
                    f"[{self.name}/{self.model}] Step cap reached with no "
                    f"certificate; not retrying"
                )
                break

            if llm_errors and attempt < self.MAX_RUN_RETRIES:
                wait = self.RETRY_WAIT_SECONDS * attempt
                logger.info(
                    f"[{self.name}/{self.model}] No certificates produced "
                    f"and LLM errors detected. Retrying in {wait}s "
                    f"(attempt {attempt + 1}/{self.MAX_RUN_RETRIES})..."
                )
                time.sleep(wait)
                continue

            break

        wall_time = time.time() - t0 + float(prior.get("wall_time_seconds") or 0)

        logger.info(
            f"[{self.name}/{self.model}] Parsed {len(certificates)} certificates"
        )
        repaired = sorted(s for s, c in certificates.items() if c.code_repaired)
        if repaired:
            logger.warning(
                f"[{self.name}/{self.model}] Unescaped double-escaped code for "
                f"{len(repaired)} certificate(s): {', '.join(repaired)}"
            )

        completed_normally = exit_code == 0 and not step_cap
        if timeout is not None and run_duration >= timeout - 5:
            completed_normally = False
            logger.warning(
                f"[{self.name}/{self.model}] Run likely killed by timeout "
                f"(run_duration={run_duration:.1f}s, timeout={timeout}s)"
            )

        provider_pending: Optional[Dict[str, Any]] = None
        remaining_after = self._remaining_steps(chain_steps)
        lost_to_provider = (
            not certificates
            and bool(llm_errors)
            and not step_cap
            and not budget_spent
            and (remaining_after is None or remaining_after > 0)
        )
        if lost_to_provider:
            completed_normally = False
            provider_pending = {
                "session_id": last_session_id,
                "chain_steps": chain_steps,
                "error_types": sorted({e.error_type for e in llm_errors}),
                "wall_time_seconds": wall_time,
                "container_id": None,
                "container_name": None,
                "spec_id": None,
            }
            logger.warning(
                f"[{self.name}/{self.model}] Lost to provider after "
                f"{attempt_no} attempt(s) ({', '.join(provider_pending['error_types'])}); "
                f"session {last_session_id or '<none>'} left to resume"
            )

        attempts_log = self._retain_last_chain(attempts_log)

        chain_cost = self._chain_cost(attempts_log)
        extracted_cost = self._extract_cost(container)
        if chain_cost:
            cost.update(chain_cost)
            for key in ("api_requests", SUCCESSFUL_LLM_CALLS, "refused_llm_calls"):
                if extracted_cost and key in extracted_cost:
                    cost[key] = extracted_cost[key]
        elif extracted_cost:
            cost.update(extracted_cost)

        if SUCCESSFUL_LLM_CALLS not in cost:
            per_attempt = [
                record.get("steps", {}).get(SUCCESSFUL_LLM_CALLS)
                for record in attempts_log
            ]
            if any(value is not None for value in per_attempt):
                cost[SUCCESSFUL_LLM_CALLS] = sum(v for v in per_attempt if v)
        logger.info(
            f"[{self.name}/{self.model}] Cost: {cost}"
            f"{' (summed over resumed chain)' if chain_cost and len(attempts_log) > 1 else ''}"
        )

        return BaselineResult(
            instance_id=task.instance_id,
            baseline=self.name,
            model=self.model,
            tool_version=self.tool_version,
            raw_output=raw_output,
            certificates=certificates,
            cost=cost,
            wall_time_seconds=wall_time,
            llm_errors=all_llm_errors,
            completed_normally=completed_normally,
            attempts=attempts_log,
            provider_pending=provider_pending,
        )

    @staticmethod
    def _retain_last_chain(
        attempts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop abandoned attempts, keep the conversation that produced the result."""
        chains = session_chains(attempts)
        return chains[-1] if chains else []

    @staticmethod
    def _chain_cost(
        attempts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, float]]:
        """Sum per-attempt cost over the retained chain."""
        per_attempt = [
            record["cost"] for record in attempts if record.get("cost")
        ]
        if not per_attempt:
            return None
        totals: Dict[str, float] = {}
        for entry in per_attempt:
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        return totals

    def _attempt_record(
        self,
        attempt: int,
        *,
        resumed: bool,
        resume_of_session: Optional[str],
        resume_rejected: bool,
        exit_code: int,
        duration: float,
        raw_output: str,
        llm_errors: List[LLMCallError],
        step_cap: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """One CLI invocation, kept whole so retried attempts stay auditable."""
        return {
            "attempt": attempt,
            "resumed": resumed,
            "resume_of_session": resume_of_session,
            "resume_rejected": resume_rejected,
            "session_id": self._extract_session_id(raw_output),
            "exit_code": exit_code,
            "duration_seconds": round(duration, 1),
            "raw_output": raw_output,
            "llm_errors": [e.to_dict() for e in llm_errors],
            "steps": self._extract_steps(raw_output),
            "cost": self._cost_from_stream(raw_output),
            "step_cap": step_cap,
        }

    @staticmethod
    def _step_cap_event(raw_output: str) -> Optional[Dict[str, Any]]:
        """The step gate's marker, if it stopped this invocation."""
        for line in reversed(raw_output.splitlines()):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                event = json.loads(stripped)
            except (ValueError, TypeError):
                continue
            if (
                isinstance(event, dict)
                and event.get("type") == STEP_CAP_EVENT_TYPE
            ):
                return event
        return None

    def _cost_from_stream(self, raw_output: str) -> Optional[Dict[str, float]]:
        """Token cost of one invocation, from its own event stream."""
        return None

    def _extract_steps(self, raw_output: str) -> Dict[str, int]:
        """Count the work one invocation did, from its event stream."""
        return {}

    def _parse_output(
        self,
        container: Container,
        raw_output: str,
    ) -> Tuple[Dict[str, SpecCertificate], Dict[str, float]]:
        """Parse the structured JSON output from the baseline."""
        certificates: Dict[str, SpecCertificate] = {}
        cost: Dict[str, float] = {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}

        try:
            exit_code, content = exec_run_with_timeout(
                container, "cat /tmp/eval_output.json", timeout=10
            )
            if exit_code == 0:
                data = json.loads(content.decode(UTF8, errors="replace"))
                raw_certs = data.get("certificates") or {}
                for spec_id, cdata in raw_certs.items():
                    certificates[spec_id] = SpecCertificate.from_dict(cdata)
                cost = data.get("cost") or cost
                return certificates, cost
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse /tmp/eval_output.json: {e}")

        search_texts = [raw_output]

        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "message" and event.get("role") in ("model", "assistant"):
                    content = event.get("content", "")
                    if isinstance(content, str) and "certificate" in content.lower():
                        search_texts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                text = block.get("text", "")
                                if text and "certificate" in text.lower():
                                    search_texts.append(text)
                if event.get("type") == "result":
                    for field_name in ("content", "text", "result"):
                        val = event.get(field_name, "")
                        if isinstance(val, str) and "certificate" in val.lower():
                            search_texts.append(val)
            except (json.JSONDecodeError, TypeError):
                continue

        for text in search_texts:
            try:
                result = self._extract_certificates_from_text(text)
                if result:
                    certificates, cost = result
                    return certificates, cost
            except Exception:
                continue

        logger.warning("Failed to extract certificates from raw output")

        return certificates, cost

    def _extract_certificates_from_text(
        self, text: str
    ) -> Optional[Tuple[Dict[str, SpecCertificate], Dict[str, float]]]:
        """Try to find and parse a certificates JSON block in text."""
        cost_default = {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}

        start = text.rfind('{"certificates"')
        if start != -1:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                data = json.loads(text[start:end])
                raw_certs = data.get("certificates") or {}
                certificates = {
                    spec_id: SpecCertificate.from_dict(cdata)
                    for spec_id, cdata in raw_certs.items()
                }
                if certificates:
                    return certificates, data.get("cost") or cost_default
            except json.JSONDecodeError:
                pass

        for i in range(len(text) - 1, -1, -1):
            if text[i] == '{':
                try:
                    data = json.loads(text[i:])
                    raw_certs = data.get("certificates") or {}
                    if raw_certs:
                        certificates = {
                            spec_id: SpecCertificate.from_dict(cdata)
                            for spec_id, cdata in raw_certs.items()
                        }
                        return certificates, data.get("cost") or cost_default
                except json.JSONDecodeError:
                    continue

        return None

    INSTALL_ATTEMPTS = 3
    INSTALL_RETRY_PAUSE_SECONDS = (10.0, 30.0)

    def _install_tool(self, container: Container) -> bool:
        """Install the baseline tool inside the container."""
        import tempfile, os

        script = self.get_install_script()
        logger.info(f"[{self.name}/{self.model}] Installing tool (version={self.tool_version})...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            temp_path = f.name

        try:
            copy_to_container(container, temp_path, "/tmp/install_baseline.sh")
        finally:
            os.unlink(temp_path)

        cmd = "chmod +x /tmp/install_baseline.sh && /tmp/install_baseline.sh"
        for attempt in range(1, self.INSTALL_ATTEMPTS + 1):
            t0 = time.time()
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=1800)
            install_output = output.decode(UTF8, errors="replace")
            elapsed = time.time() - t0
            if exit_code == 0:
                logger.info(
                    f"[{self.name}/{self.model}] Tool installed successfully in "
                    f"{elapsed:.1f}s (attempt {attempt}/{self.INSTALL_ATTEMPTS})"
                )
                logger.debug(
                    f"[{self.name}/{self.model}] Install output (last 300 chars): "
                    f"{install_output[-300:]}"
                )
                return True
            last = attempt == self.INSTALL_ATTEMPTS
            logger.log(
                logging.ERROR if last else logging.WARNING,
                f"[{self.name}/{self.model}] Tool install failed "
                f"(attempt {attempt}/{self.INSTALL_ATTEMPTS}, exit_code={exit_code}, "
                f"{elapsed:.1f}s): {install_output[-500:]}"
            )
            if not last:
                pauses = self.INSTALL_RETRY_PAUSE_SECONDS
                time.sleep(pauses[min(attempt - 1, len(pauses) - 1)])
        return False

    def _write_file_to_container(
        self, container: Container, content: str, container_path: str
    ) -> None:
        """Write a string to a file inside the container."""
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name
        try:
            copy_to_container(container, temp_path, container_path)
        finally:
            os.unlink(temp_path)

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """Extract cost/token usage from tool-specific output."""
        return None

    _LLM_ERROR_PATTERNS = [
        ("rate_limit", re.compile(
            r"(?:status[:\s]*429|\"code\":\s*429|rate.?limit|负载已饱和|too many requests)",
            re.IGNORECASE,
        )),
        ("server_error", re.compile(
            r"(?:status[:\s]*5\d{2}|\"code\":\s*5\d{2}|internal.?server.?error|"
            r"bad.?gateway|service.?unavailable|upstream.?error)",
            re.IGNORECASE,
        )),
        ("capacity", re.compile(
            r"(?:at capacity|try a different model)",
            re.IGNORECASE,
        )),
        ("model_not_found", re.compile(
            r"(?:model.?not.?found|model_not_found|does not exist)",
            re.IGNORECASE,
        )),
        ("auth_error", re.compile(
            r"(?:status[:\s]*401|status[:\s]*403|unauthorized|forbidden|"
            r"invalid.?api.?key|authentication)",
            re.IGNORECASE,
        )),
        ("context_overflow", re.compile(
            r"(?:context.?(?:length|window|limit).{0,20}(?:exceeded|error|reached|too\s)|"
            r"too.?many.?tokens|maximum.?context.{0,10}exceeded|"
            r"token.?limit.{0,10}(?:exceeded|reached))",
            re.IGNORECASE,
        )),
        ("connection_error", re.compile(
            r"(?:ECONNREFUSED|ECONNRESET|ETIMEDOUT|connection.?refused|"
            r"network.?error|socket.?hang.?up)",
            re.IGNORECASE,
        )),
        ("stream_disconnect", re.compile(
            r"(?:stream disconnected|stream closed before|"
            r"disconnected before completion|Reconnecting\.\.\.)",
            re.IGNORECASE,
        )),
    ]

    _ERROR_BEARING_EVENT_TYPES = frozenset({
        "error",
        "turn.failed",
        "stream_error",
    })

    @staticmethod
    def _error_bearing_segments(raw_output: str) -> List[str]:
        """Yield only the parts of a transcript that can hold provider errors."""
        segments: List[str] = []
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if (
                STEP_CAP_REFUSAL_MARKER in stripped
                or stripped.startswith(STEP_GATE_STDERR_PREFIX)
            ):
                continue
            if not stripped.startswith("{"):
                segments.append(stripped)
                continue
            try:
                event = json.loads(stripped)
            except (ValueError, TypeError):
                segments.append(stripped)
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type in EvalBaselineAdapter._ERROR_BEARING_EVENT_TYPES:
                segments.append(json.dumps(event, ensure_ascii=False))
            elif event_type == "system" and event.get("subtype") == "error":
                segments.append(json.dumps(event, ensure_ascii=False))
            elif event_type == "result" and event.get("is_error"):
                segments.append(json.dumps(event, ensure_ascii=False))
        return segments

    def _detect_llm_errors(self, raw_output: str) -> List[LLMCallError]:
        """Scan the error-bearing parts of raw output for LLM call errors."""
        haystack = "\n".join(self._error_bearing_segments(raw_output))
        errors: List[LLMCallError] = []
        if haystack:
            for error_type, pattern in self._LLM_ERROR_PATTERNS:
                matches = pattern.findall(haystack)
                if matches:
                    first_match = pattern.search(haystack)
                    start = max(0, first_match.start() - 80)
                    end = min(len(haystack), first_match.end() + 120)
                    sample = haystack[start:end].replace("\n", " ").strip()
                    errors.append(LLMCallError(
                        error_type=error_type,
                        count=len(matches),
                        sample_message=sample,
                    ))
        abort = self._detect_stream_abort(raw_output)
        if abort is not None:
            errors.append(abort)
        return errors

    def _detect_stream_abort(self, raw_output: str) -> Optional[LLMCallError]:
        """Detect a silently truncated response stream."""
        return None

    def get_resume_command(
        self,
        prompt_path: str,
        session_id: str,
        max_steps: Optional[int] = None,
    ) -> Optional[str]:
        """Command that continues ``session_id`` instead of restarting."""
        return None

    def _extract_session_id(self, raw_output: str) -> Optional[str]:
        """Pull the session identifier out of a finished run's output."""
        return None

    @staticmethod
    def _has_stream_events(raw_output: str) -> bool:
        """Whether the output contains at least one structured event."""
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                event = json.loads(stripped)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict) and event.get("type"):
                return True
        return False

    def _build_resume_prompt(self, task: EvalTask) -> str:
        """The message handed to a resumed session."""
        return (
            "Your previous turn was cut off by an infrastructure failure, not "
            "by anything you did wrong. The container is unchanged, so any "
            "files you already wrote are still on disk.\n\n"
            "Do not start over. Pick up from where you stopped."
        )

    def _error_result(
        self, task: EvalTask, error_msg: str, wall_time: float
    ) -> BaselineResult:
        return BaselineResult(
            instance_id=task.instance_id,
            baseline=self.name,
            model=self.model,
            tool_version=self.tool_version,
            raw_output=f"ERROR: {error_msg}",
            certificates={},
            cost={"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0},
            wall_time_seconds=wall_time,
            llm_errors=[],
            completed_normally=False,
        )
