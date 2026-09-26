"""Claude Code evaluation adapter."""

import json
import logging
import shlex
from typing import Dict, List, Optional

from docker.models.containers import Container

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

from .base import SUCCESSFUL_LLM_CALLS, EvalBaselineAdapter
from ..models import LLMCallError

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter(EvalBaselineAdapter):
    _SETTINGS = json.dumps({
        "alwaysThinkingEnabled": True,
    }, indent=2)

    def get_install_script(self) -> str:
        version = self.tool_version or "latest"
        api_key = shlex.quote(self.env_vars.get("ANTHROPIC_API_KEY", ""))
        base_url = shlex.quote(self.env_vars.get("ANTHROPIC_BASE_URL", ""))
        base_url_raw = self.env_vars.get("ANTHROPIC_BASE_URL", "")

        settings_block = f"""
mkdir -p ~/.claude ~/.config/claude-code
cat > ~/.claude/settings.json <<'JSONEOF'
{self._SETTINGS}
JSONEOF
cat > ~/.config/claude-code/settings.json <<'JSONEOF'
{self._SETTINGS}
JSONEOF
echo "Wrote Claude Code settings.json to ~/.claude and ~/.config/claude-code"
"""

        return f"""#!/bin/bash
set -e
apt-get update && apt-get install -y curl ca-certificates tar xz-utils

NVM_DIR="/opt/eval/nvm"
mkdir -p "$NVM_DIR"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash
export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22 && nvm use 22

npm install -g @anthropic-ai/claude-code@{version}
claude --version

export ANTHROPIC_API_KEY={api_key}
{f"export ANTHROPIC_BASE_URL={base_url}" if base_url_raw else ""}
{settings_block}
echo "Claude Code {version} installed"
echo "=== Claude Code env ==="
echo "ANTHROPIC_BASE_URL=${{ANTHROPIC_BASE_URL:-<not set>}}"
echo "ANTHROPIC_MODEL=${{ANTHROPIC_MODEL:-<not set>}}"
echo "========================"
"""

    def _command_prelude(self) -> tuple:
        """Shared shell prelude and model flag for run and resume commands."""
        api_key = shlex.quote(self.env_vars.get("ANTHROPIC_API_KEY", ""))
        base_url = shlex.quote(self.env_vars.get("ANTHROPIC_BASE_URL", ""))
        model = shlex.quote(self.env_vars.get("ANTHROPIC_MODEL", ""))

        env_prefix = f"export ANTHROPIC_API_KEY={api_key}"
        if self.env_vars.get("ANTHROPIC_BASE_URL", ""):
            env_prefix += f" && export ANTHROPIC_BASE_URL={base_url}"
        if self.env_vars.get("ANTHROPIC_MODEL", ""):
            env_prefix += f" && export ANTHROPIC_MODEL={model}"

        model_raw = self.env_vars.get("ANTHROPIC_MODEL", "")
        if model_raw and "[1m]" not in model_raw and "opus" in model_raw.lower():
            model = shlex.quote(model_raw + "[1m]")
        model_flag = ""
        if model_raw:
            model_flag = f" --model {model}"

        return env_prefix, model_flag

    def step_gate_unit(self) -> Optional[str]:
        """Counted in distinct assistant message ids; see ``step_gate.py``."""
        return "claude_code"

    def get_run_command(self, prompt_path: str) -> str:
        env_prefix, model_flag = self._command_prelude()

        return (
            f'NVM_DIR="/opt/eval/nvm"; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'{env_prefix} && '
            f'export DISABLE_TELEMETRY=1 && '
            f'cd /testbed && '
            + self._piped_cli(
                f'claude -p "$(cat {prompt_path})" '
                f'--allowedTools Bash Edit Write Read Glob Grep'
                f'{model_flag} '
                f'--verbose --output-format stream-json',
                "/tmp/claude_output.txt",
            )
        )

    def get_resume_command(
        self,
        prompt_path: str,
        session_id: str,
        max_steps: Optional[int] = None,
    ) -> Optional[str]:
        """Continue a previous session rather than restarting the task."""
        env_prefix, model_flag = self._command_prelude()

        return (
            f'NVM_DIR="/opt/eval/nvm"; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'{env_prefix} && '
            f'export DISABLE_TELEMETRY=1 && '
            f'cd /testbed && '
            + self._piped_cli(
                f'claude -p "$(cat {prompt_path})" '
                f'--resume {shlex.quote(session_id)} '
                f'--allowedTools Bash Edit Write Read Glob Grep'
                f'{model_flag} '
                f'--verbose --output-format stream-json',
                "/tmp/claude_output_resume.txt",
                max_steps=max_steps,
            )
        )

    def _extract_session_id(self, raw_output: str) -> Optional[str]:
        """Read the session id the CLI reports for this run."""
        for event in self._parse_stream_events(raw_output):
            if event.get("type") == "system" and event.get("session_id"):
                return str(event["session_id"])
        for event in reversed(self._parse_stream_events(raw_output)):
            if event.get("session_id"):
                return str(event["session_id"])
        return None

    def _detect_stream_abort(self, raw_output: str) -> Optional[LLMCallError]:
        """Detect a response stream that was cut mid-turn."""
        events = self._parse_stream_events(raw_output)
        if not events:
            return None

        result_events = [e for e in events if e.get("type") == "result"]
        if not result_events or (result_events[-1].get("result") or "") != "":
            return None

        last_block = None
        for event in events:
            if event.get("type") != "assistant":
                continue
            content = (event.get("message") or {}).get("content") or []
            if content:
                last_block = content[-1]
        if not isinstance(last_block, dict):
            return None
        if last_block.get("type") != "thinking" or last_block.get("signature"):
            return None

        tail = (last_block.get("thinking") or "")[-200:].replace("\n", " ").strip()
        return LLMCallError(
            error_type="stream_abort",
            count=1,
            sample_message=(
                "response stream cut mid-thinking; CLI reported end_turn with "
                f"an empty result. Truncated at: ...{tail}"
            ),
        )

    def _extract_steps(self, raw_output: str) -> Dict[str, int]:
        """Count this invocation's work from the stream-json events."""
        events = self._parse_stream_events(raw_output)
        message_ids = set()
        truncated_ids = set()
        assistant_events = 0
        tool_calls = 0
        num_turns = None
        for event in events:
            if event.get("type") == "result":
                turns = event.get("num_turns")
                if isinstance(turns, int):
                    num_turns = turns
            if event.get("type") != "assistant":
                continue
            assistant_events += 1
            message = event.get("message") or {}
            message_id = message.get("id")
            if message_id:
                message_ids.add(message_id)
            blocks = message.get("content") or []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls += 1
            if message_id and blocks and isinstance(blocks[-1], dict):
                last = blocks[-1]
                if last.get("type") == "thinking" and not last.get("signature"):
                    truncated_ids.add(message_id)

        responses = len(message_ids) or assistant_events
        steps = {
            "api_responses": responses,
            SUCCESSFUL_LLM_CALLS: responses - len(truncated_ids),
            "assistant_events": assistant_events,
            "tool_calls": tool_calls,
        }
        if num_turns is not None:
            steps["num_turns"] = num_turns
        return steps

    @staticmethod
    def _parse_stream_events(raw_output: str) -> List[dict]:
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

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """Parse cost from Claude Code stream-json output."""
        try:
            exit_code, output = exec_run_with_timeout(
                container, "cat /tmp/claude_output.txt", timeout=10
            )
            if exit_code != 0:
                return None
            return self._cost_from_stream(output.decode(UTF8, errors="replace"))
        except Exception as e:
            logger.warning(f"Failed to extract cost from Claude Code output: {e}")
        return None

    def _cost_from_stream(self, text: str) -> Optional[Dict[str, float]]:
        """Cost for one invocation, computed from its own event stream."""
        price_input = self.agent_config.get("price_input_per_mtok", 0.0)
        price_output = self.agent_config.get("price_output_per_mtok", 0.0)
        price_cache_read = self.agent_config.get("price_cache_read_per_mtok", 0.0)
        price_cache_write = self.agent_config.get("price_cache_write_per_mtok", 0.0)

        try:
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "result":
                    continue

                model_usage = event.get("modelUsage", {})
                input_tokens = 0
                output_tokens = 0
                cache_read_tokens = 0
                cache_write_tokens = 0
                for usage in model_usage.values():
                    input_tokens += usage.get("inputTokens", 0)
                    output_tokens += usage.get("outputTokens", 0)
                    cache_read_tokens += usage.get("cacheReadInputTokens", 0)
                    cache_write_tokens += usage.get("cacheCreationInputTokens", 0)

                if not (input_tokens or output_tokens or cache_read_tokens or cache_write_tokens):
                    return None

                M = 1_000_000
                cost_with_cache = (
                    input_tokens * price_input / M
                    + output_tokens * price_output / M
                    + cache_read_tokens * price_cache_read / M
                    + cache_write_tokens * price_cache_write / M
                )
                cost_no_cache = (
                    (input_tokens + cache_read_tokens + cache_write_tokens)
                    * price_input / M
                    + output_tokens * price_output / M
                )
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                    "cache_creation_input_tokens": cache_write_tokens,
                    "total_cost_usd": round(cost_with_cache, 7),
                    "total_cost_usd_no_cache": round(cost_no_cache, 7),
                }
        except Exception as e:
            logger.warning(f"Failed to extract cost from Claude Code output: {e}")
        return None
