"""OpenCode CLI evaluation adapter."""

import json
import logging
import shlex
from typing import Any, Dict, List, Optional, Tuple

from docker.models.containers import Container

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

from . import tool_filter
from .base import SUCCESSFUL_LLM_CALLS, EvalBaselineAdapter
from .tool_filter import TOOL_FILTER_AUDIT_PATH
from ..models import LLMCallError

logger = logging.getLogger(__name__)

PROVIDER_ID = "proxy"

WIRE_API_NPM = {
    "anthropic": "@ai-sdk/anthropic",
    "openai-compatible": "@ai-sdk/openai-compatible",
    "openai": "@ai-sdk/openai",
}
DEFAULT_WIRE_API = "anthropic"

DEFAULT_CONTEXT_LIMIT = 200_000
DEFAULT_OUTPUT_LIMIT = 64_000

KEY_ENV_VAR = "OPENCODE_PROXY_API_KEY"

SESSION_TITLE = "sc-align eval"

CONFIG_DIR = "/root/.config/opencode"
CONFIG_PATH = f"{CONFIG_DIR}/opencode.json"


class OpenCodeAdapter(EvalBaselineAdapter):
    def _wire_api(self) -> str:
        wire = (self.env_vars.get("OPENCODE_WIRE_API") or "").strip()
        if not wire:
            return DEFAULT_WIRE_API
        if wire not in WIRE_API_NPM:
            raise ValueError(
                f"OPENCODE_WIRE_API={wire!r} is not one of {sorted(WIRE_API_NPM)}"
            )
        return wire

    def _model_id(self) -> str:
        """The bare model id, such as ``claude-opus-5``."""
        return self.env_vars.get("OPENCODE_MODEL", "") or self.model

    def _qualified_model(self) -> str:
        return f"{PROVIDER_ID}/{self._model_id()}"

    def _upstream_base_url(self) -> str:
        return self.env_vars.get("OPENCODE_BASE_URL", "")

    def _loopback_base_url(self) -> str:
        """Where opencode.json points the provider: the shim, when there is an upstream
        for it to forward to.
        """
        base_url = self._upstream_base_url()
        return tool_filter.loopback_base_url(base_url) if base_url else base_url

    def _config_json(self) -> str:
        """The opencode.json this baseline runs under."""
        model_id = self._model_id()

        def _limit(key: str, fallback: int) -> int:
            raw = str(self.env_vars.get(key, "")).strip()
            return int(raw) if raw else fallback

        config: Dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "snapshot": False,
            "model": f"{PROVIDER_ID}/{model_id}",
            "small_model": f"{PROVIDER_ID}/{model_id}",
            "provider": {
                PROVIDER_ID: {
                    "npm": WIRE_API_NPM[self._wire_api()],
                    "name": "SC-align eval proxy",
                    "options": {
                        "baseURL": self._loopback_base_url(),
                        "apiKey": f"{{env:{KEY_ENV_VAR}}}",
                    },
                    "models": {
                        model_id: {
                            "name": model_id,
                            "limit": {
                                "context": _limit(
                                    "OPENCODE_CONTEXT_LIMIT",
                                    DEFAULT_CONTEXT_LIMIT,
                                ),
                                "output": _limit(
                                    "OPENCODE_OUTPUT_LIMIT",
                                    DEFAULT_OUTPUT_LIMIT,
                                ),
                            },
                        }
                    },
                }
            },
            "tools": {"webfetch": False},
        }
        return json.dumps(config, indent=2)

    def get_install_script(self) -> str:
        version = self.tool_version or "latest"
        api_key = shlex.quote(self.env_vars.get("OPENCODE_API_KEY", ""))
        model = self._qualified_model()
        base_url = self._upstream_base_url()
        shim_install = tool_filter.install_snippet(base_url) if base_url else ""

        return f"""#!/bin/bash
set -e
apt-get update && apt-get install -y curl ca-certificates tar xz-utils

NVM_DIR="/opt/eval/nvm"
mkdir -p "$NVM_DIR"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash
export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22 && nvm use 22

npm install -g opencode-ai@{version}
opencode --version

mkdir -p {CONFIG_DIR}
cat > {CONFIG_PATH} << 'OPENCODE_CFG'
{self._config_json()}
OPENCODE_CFG

{shim_install}
export {KEY_ENV_VAR}={api_key}
if ! opencode models | grep -qx {shlex.quote(model)}; then
    echo "FATAL: opencode cannot resolve {model}; it offers:" >&2
    opencode models >&2 || true
    exit 1
fi

echo "OpenCode {version} installed"
echo "=== opencode.json ==="
cat {CONFIG_PATH}
echo "====================="
"""

    def get_run_command(self, prompt_path: str) -> str:
        api_key = shlex.quote(self.env_vars.get("OPENCODE_API_KEY", ""))
        model = shlex.quote(self._qualified_model())

        variant = (self.env_vars.get("OPENCODE_VARIANT") or "").strip()
        variant_flag = f" --variant {shlex.quote(variant)}" if variant else ""

        prelude = (
            tool_filter.prelude(self.enforced_max_steps())
            if self._upstream_base_url() else ""
        )
        return (
            f'NVM_DIR="/opt/eval/nvm"; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'{prelude}'
            f'export {KEY_ENV_VAR}={api_key} && '
            f'cd /testbed && '
            + self._piped_cli(
                f'opencode run --auto --format json '
                f'--print-logs --log-level ERROR '
                f'--title {shlex.quote(SESSION_TITLE)} '
                f'--model {model}{variant_flag} '
                f'"$(cat {prompt_path})"',
                "/tmp/opencode_output.txt",
            )
        )

    def step_gate_unit(self) -> Optional[str]:
        """Counted on the wire by the loopback shim; see ``step_gate.py``."""
        return "opencode_wire" if self._upstream_base_url() else "opencode"

    def step_gate_source(self) -> Optional[str]:
        return TOOL_FILTER_AUDIT_PATH if self._upstream_base_url() else None

    @staticmethod
    def _parse_events(raw_output: str) -> List[dict]:
        """The NDJSON event stream, with the log lines filtered out."""
        events: List[dict] = []
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                event = json.loads(stripped)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _extract_session_id(self, raw_output: str) -> Optional[str]:
        """The session id, which every event carries."""
        for event in self._parse_events(raw_output):
            session_id = event.get("sessionID")
            if session_id:
                return str(session_id)
        return None

    def _extract_steps(self, raw_output: str) -> Dict[str, int]:
        """Count this invocation's work from the ``--format json`` events."""
        model_responses = 0
        tool_calls = 0
        text_parts = 0
        error_events = 0
        for event in self._parse_events(raw_output):
            event_type = event.get("type")
            part = event.get("part") or {}
            if event_type == "step_finish":
                model_responses += 1
            elif event_type == "tool_use":
                state = part.get("state") or {}
                if state.get("status") == "completed":
                    tool_calls += 1
            elif event_type == "text":
                text_parts += 1
            elif event_type == "error":
                error_events += 1
        return {
            "model_responses": model_responses,
            SUCCESSFUL_LLM_CALLS: model_responses,
            "tool_calls": tool_calls,
            "text_parts": text_parts,
            "error_events": error_events,
        }

    def _detect_stream_abort(self, raw_output: str) -> Optional[LLMCallError]:
        """Detect a run that stopped without the model finishing a turn."""
        for event in reversed(self._parse_events(raw_output)):
            if event.get("type") != "step_finish":
                continue
            reason = (event.get("part") or {}).get("reason")
            if reason == "stop":
                return None
            return LLMCallError(
                error_type="stream_abort",
                count=1,
                sample_message=(
                    f"run ended on step_finish reason={reason!r} rather than "
                    f"'stop'; the model's turn did not close"
                ),
            )

        tail = raw_output.strip()[-300:].replace("\n", " ")
        return LLMCallError(
            error_type="stream_abort",
            count=1,
            sample_message=(
                "no step_finish event in the transcript -- the CLI exited "
                f"without the model completing a turn. Output tail: {tail}"
            ),
        )

    def _cost_from_stream(self, text: str) -> Optional[Dict[str, float]]:
        """Cost for one invocation, summed over its own ``step_finish`` events."""
        price_input = self.agent_config.get("price_input_per_mtok", 0.0)
        price_output = self.agent_config.get("price_output_per_mtok", 0.0)
        price_cache_read = self.agent_config.get("price_cache_read_per_mtok", 0.0)
        price_cache_write = self.agent_config.get("price_cache_write_per_mtok", 0.0)

        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        seen = False

        for event in self._parse_events(text):
            if event.get("type") != "step_finish":
                continue
            tokens = (event.get("part") or {}).get("tokens") or {}
            if not tokens:
                continue
            seen = True
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input", 0) or 0)
            output_tokens += int(tokens.get("output", 0) or 0)
            reasoning_tokens += int(tokens.get("reasoning", 0) or 0)
            cache_read_tokens += int(cache.get("read", 0) or 0)
            cache_write_tokens += int(cache.get("write", 0) or 0)

        if not seen:
            return None

        M = 1_000_000
        billed_output = output_tokens + reasoning_tokens
        cost_with_cache = (
            input_tokens * price_input / M
            + billed_output * price_output / M
            + cache_read_tokens * price_cache_read / M
            + cache_write_tokens * price_cache_write / M
        )
        cost_no_cache = (
            (input_tokens + cache_read_tokens + cache_write_tokens)
            * price_input / M
            + billed_output * price_output / M
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_write_tokens,
            "total_cost_usd": round(cost_with_cache, 7),
            "total_cost_usd_no_cache": round(cost_no_cache, 7),
        }

    MAX_SUBAGENT_SESSIONS = 32

    @staticmethod
    def _task_session_ids(node: Any) -> List[str]:
        """Child session ids spawned by ``task``, anywhere in a document."""
        found: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "tool" and value.get("tool") == "task":
                    metadata = (value.get("state") or {}).get("metadata") or {}
                    session_id = metadata.get("sessionId")
                    if session_id:
                        found.append(str(session_id))
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(node)
        return found

    @staticmethod
    def _step_finish_lines(node: Any) -> List[str]:
        """``step-finish`` parts anywhere in a document, in stream event shape."""
        lines: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "step-finish":
                    lines.append(json.dumps(
                        {"type": "step_finish", "part": value},
                        ensure_ascii=False,
                    ))
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(node)
        return lines

    def _export_session(self, container: Container, session_id: str) -> Optional[Any]:
        """One session as JSON, read out of the CLI's own store."""
        cmd = (
            'NVM_DIR="/opt/eval/nvm"; '
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'opencode export {shlex.quote(session_id)} '
            f'> /tmp/opencode_export.json 2>/dev/null '
            f'&& cat /tmp/opencode_export.json'
        )
        try:
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)
            if exit_code != 0:
                logger.warning(
                    f"[{self.name}/{self.model}] opencode export {session_id} "
                    f"exited {exit_code}"
                )
                return None
            return json.loads(output.decode(UTF8, errors="replace"))
        except Exception as e:
            logger.warning(
                f"[{self.name}/{self.model}] Failed to export session "
                f"{session_id}: {e}"
            )
            return None

    def _subagent_steps(
        self, container: Container, root_text: str
    ) -> Tuple[List[str], int]:
        """Every step taken below the run's own session."""
        pending = list(self._task_session_ids(self._parse_events(root_text)))
        seen: set = set()
        lines: List[str] = []

        while pending:
            session_id = pending.pop(0)
            if session_id in seen:
                continue
            if len(seen) >= self.MAX_SUBAGENT_SESSIONS:
                logger.warning(
                    f"[{self.name}/{self.model}] subagent walk stopped at "
                    f"{self.MAX_SUBAGENT_SESSIONS} sessions; {len(pending) + 1} "
                    f"left unread, so calls and tokens are undercounted"
                )
                break
            seen.add(session_id)
            document = self._export_session(container, session_id)
            if document is None:
                continue
            lines.extend(self._step_finish_lines(document))
            pending.extend(self._task_session_ids(document))

        return lines, len(seen)

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """The run's whole spend: its own session plus every subagent it opened."""
        try:
            exit_code, output = exec_run_with_timeout(
                container, "cat /tmp/opencode_output.txt", timeout=10
            )
        except Exception as e:
            logger.warning(f"Failed to read OpenCode transcript: {e}")
            return None
        if exit_code != 0:
            return None

        text = output.decode(UTF8, errors="replace")
        parent_lines = self._step_finish_lines(
            [event.get("part") for event in self._parse_events(text)
             if event.get("type") == "step_finish"]
        )
        if not parent_lines:
            session_id = self._extract_session_id(text)
            document = (
                self._export_session(container, session_id) if session_id else None
            )
            if document is not None:
                parent_lines = self._step_finish_lines(document)
                if parent_lines:
                    logger.info(
                        f"[{self.name}/{self.model}] Recovered "
                        f"{len(parent_lines)} steps from the session store "
                        f"(stream carried none)"
                    )

        child_lines, sessions = self._subagent_steps(container, text)
        if sessions:
            logger.info(
                f"[{self.name}/{self.model}] {len(child_lines)} calls from "
                f"{sessions} subagent session(s) folded into this run"
            )

        wire = (
            tool_filter.read_audit_counters(self, container)
            if self._upstream_base_url() else {}
        )
        cost = self._cost_from_stream("\n".join(parent_lines + child_lines))
        if cost is None:
            return dict(wire) or None
        cost[SUCCESSFUL_LLM_CALLS] = len(parent_lines) + len(child_lines)
        cost["subagent_llm_calls"] = len(child_lines)
        cost["subagent_sessions"] = sessions
        if SUCCESSFUL_LLM_CALLS in wire:
            cost["session_llm_calls"] = cost[SUCCESSFUL_LLM_CALLS]
            cost.update(wire)
        return cost

    @staticmethod
    def _chain_cost(attempts: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Not used by this adapter."""
        return None
