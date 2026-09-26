"""Codex CLI evaluation adapter."""

import json
import logging
import shlex
from typing import Dict, Optional

from docker.models.containers import Container

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

from . import tool_filter
from .base import EvalBaselineAdapter
from .tool_filter import (
    TOOL_FILTER_AUDIT_PATH,
    TOOL_FILTER_PORT,
    TOOL_FILTER_SEGMENT_PREFIX,
    TOOL_FILTER_SRC,
)

logger = logging.getLogger(__name__)


class CodexAdapter(EvalBaselineAdapter):
    def get_install_script(self) -> str:
        version = self.tool_version or "latest"
        model = self.env_vars.get("OPENAI_MODEL", "")
        base_url = self.env_vars.get("OPENAI_BASE_URL", "")
        api_key = self.env_vars.get("OPENAI_API_KEY", "")

        reasoning_effort = str(
            self.env_vars.get("CODEX_REASONING_EFFORT")
            or self.env_vars.get("MODEL_REASONING_EFFORT")
            or ""
        ).strip()
        effort_line = (
            f'model_reasoning_effort = "{reasoning_effort}"\n' if reasoning_effort else ""
        )

        auth_json = json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False)

        effective_base_url = (
            tool_filter.loopback_base_url(base_url) if base_url else base_url
        )
        shim_install = tool_filter.install_snippet(base_url) if base_url else ""

        return f"""#!/bin/bash
set -e
apt-get update && apt-get install -y curl ca-certificates tar xz-utils bubblewrap

NVM_DIR="/opt/eval/nvm"
mkdir -p "$NVM_DIR"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash
export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22 && nvm use 22

npm install -g @openai/codex@{version}
codex --version || true

mkdir -p /root/.codex
cat > /root/.codex/auth.json << 'CODEX_AUTH'
{auth_json}
CODEX_AUTH

cat > /root/.codex/config.toml << 'CODEX_CFG'
model = "{model}"
model_provider = "litellm"
{effort_line}disable_response_storage = true

[model_providers.litellm]
name = "litellm"
base_url = "{effective_base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
CODEX_CFG

{shim_install}
echo "Codex CLI {version} installed"
echo "=== Codex config.toml ==="
cat /root/.codex/config.toml
echo "========================="
"""

    def _uses_tool_filter(self) -> bool:
        """Whether Codex talks through the loopback shim."""
        return bool(self.env_vars.get("OPENAI_BASE_URL", ""))

    def _tool_filter_prelude(self, max_steps: Optional[int] = None) -> str:
        """Start a fresh shim for this invocation; see ``tool_filter.prelude``."""
        if not self._uses_tool_filter():
            return ""
        return tool_filter.prelude(max_steps)

    def step_gate_unit(self) -> Optional[str]:
        """Counted on the wire by the loopback shim; see ``step_gate.py``."""
        return "codex_wire" if self._uses_tool_filter() else "codex"

    def step_gate_source(self) -> Optional[str]:
        return TOOL_FILTER_AUDIT_PATH if self._uses_tool_filter() else None

    def get_run_command(self, prompt_path: str) -> str:
        api_key = shlex.quote(self.env_vars.get("OPENAI_API_KEY", ""))

        return (
            f'NVM_DIR="/opt/eval/nvm"; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'{self._tool_filter_prelude(self.enforced_max_steps())}'
            f'export OPENAI_API_KEY={api_key} && '
            f'cd /testbed && '
            + self._piped_cli(
                f'codex exec --dangerously-bypass-approvals-and-sandbox '
                f'--skip-git-repo-check --json "$(cat {prompt_path})"',
                "/tmp/codex_output.txt",
            )
        )

    def _extract_steps(self, raw_output: str) -> Dict[str, int]:
        """Count this invocation's work from the ``--json`` event stream."""
        counts = {"agent_messages": 0, "reasoning_items": 0, "tool_calls": 0}
        by_item_type = {
            "agent_message": "agent_messages",
            "reasoning": "reasoning_items",
            "command_execution": "tool_calls",
        }
        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict) or event.get("type") != "item.completed":
                continue
            key = by_item_type.get((event.get("item") or {}).get("type"))
            if key:
                counts[key] += 1
        return counts

    _AUDIT_REQUESTS_RE = tool_filter.AUDIT_REQUESTS_RE
    _AUDIT_OK_RE = tool_filter.AUDIT_OK_RE
    _AUDIT_REFUSED_RE = tool_filter.AUDIT_REFUSED_RE

    @staticmethod
    def _last_counted_audit_line(text: str) -> Optional[str]:
        return tool_filter.last_counted_line(text)

    def _log_tool_filter_audit(
        self, container: Container
    ) -> Dict[str, int]:
        """The run's call counts off the shim's audit log."""
        return tool_filter.read_audit_counters(self, container)

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """Parse cost from Codex CLI --json JSONL events."""
        counters = self._log_tool_filter_audit(container)
        try:
            exit_code, output = exec_run_with_timeout(
                container, "cat /tmp/codex_output.txt", timeout=10
            )
            if exit_code != 0:
                return None
            cost = self._cost_from_stream(output.decode(UTF8, errors="replace"))
            if cost is not None:
                cost.update(counters)
            return cost
        except Exception as e:
            logger.warning(f"Failed to extract cost from Codex output: {e}")
        return None

    def _cost_from_stream(self, text: str) -> Optional[Dict[str, float]]:
        """Cost for one invocation, computed from its own event stream."""
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage", {})
            if not usage:
                continue
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
            cached_input_tokens += usage.get("cached_input_tokens", 0)
        if input_tokens or output_tokens:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "total_cost_usd": 0.0,
            }
        return None
