"""Gemini CLI evaluation adapter."""

import json
import logging
import shlex
from typing import Dict, Optional

from docker.models.containers import Container

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

from .base import EvalBaselineAdapter

logger = logging.getLogger(__name__)


class GeminiCLIAdapter(EvalBaselineAdapter):
    def get_install_script(self) -> str:
        version = self.tool_version or "latest"
        model = self.env_vars.get("GEMINI_MODEL", "")
        return f"""#!/bin/bash
set -e
apt-get update && apt-get install -y curl ca-certificates tar xz-utils

NVM_DIR="/opt/eval/nvm"
mkdir -p "$NVM_DIR"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash
export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22 && nvm use 22

npm install -g @google/gemini-cli@{version}
gemini --version || true

mkdir -p /root/.gemini
cat > /root/.gemini/settings.json << 'JSONEOF'
{{
  "maxAttempts": 10,
  "retryFetchErrors": true,
  "modelConfigs": {{
    "overrides": [
      {{ "match": {{ "model": "loop-detection" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "loop-detection-double-check" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "classifier" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "prompt-completion" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "summarizer-default" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "summarizer-shell" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "next-speaker-checker" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "llm-edit-fixer" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "edit-corrector" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "web-search" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "web-fetch" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "web-fetch-fallback" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-default" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-2.5-pro" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-2.5-flash" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-2.5-flash-lite" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-3-pro" }}, "modelConfig": {{ "model": "{model}" }} }},
      {{ "match": {{ "model": "chat-compression-3-flash" }}, "modelConfig": {{ "model": "{model}" }} }}
    ]
  }}
}}
JSONEOF

echo "Gemini CLI {version} installed with model overrides for {model}"
"""

    def get_run_command(self, prompt_path: str) -> str:
        api_key = shlex.quote(self.env_vars.get("GEMINI_API_KEY", ""))
        base_url = shlex.quote(self.env_vars.get("GOOGLE_GEMINI_BASE_URL", ""))
        model = shlex.quote(self.env_vars.get("GEMINI_MODEL", ""))

        env_prefix = f"export GEMINI_API_KEY={api_key}"
        if self.env_vars.get("GOOGLE_GEMINI_BASE_URL", ""):
            env_prefix += f" && export GOOGLE_GEMINI_BASE_URL={base_url}"

        model_flag = ""
        if self.env_vars.get("GEMINI_MODEL", ""):
            model_flag = f" -m {model}"

        return (
            f'NVM_DIR="/opt/eval/nvm"; '
            f'[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true; '
            f'{env_prefix} && '
            f'export GEMINI_SANDBOX=false && '
            f'cd /testbed && '
            f'gemini -p "$(cat {prompt_path})"{model_flag} -y '
            f'--output-format stream-json '
            f'2>&1 | tee /tmp/gemini_output.txt'
        )

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """Parse cost from Gemini CLI stream-json output."""
        try:
            exit_code, output = exec_run_with_timeout(
                container, "cat /tmp/gemini_output.txt", timeout=10
            )
            if exit_code != 0:
                return None
            text = output.decode(UTF8, errors="replace")
            input_tokens = 0
            output_tokens = 0
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    tokens_used = event.get("totalTokensUsed", {})
                    input_tokens = tokens_used.get("inputTokens", 0)
                    output_tokens = tokens_used.get("outputTokens", 0)
                usage_meta = event.get("usageMetadata", {})
                if usage_meta:
                    input_tokens += usage_meta.get("promptTokenCount", 0)
                    output_tokens += usage_meta.get("candidatesTokenCount", 0)
            if input_tokens or output_tokens:
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_cost_usd": 0.0,
                }
        except Exception as e:
            logger.warning(f"Failed to extract cost from Gemini CLI output: {e}")
        return None
