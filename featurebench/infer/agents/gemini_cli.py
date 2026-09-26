"""Gemini CLI agent implementation."""

import json
import re
import shlex
from pathlib import Path
from typing import Dict

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY


class GeminiCliAgent(BaseAgent):
    """Gemini CLI agent for FeatureBench inference."""

    NODE_VERSION = "22"

    @property
    def name(self) -> str:
        return "gemini_cli"

    @property
    def install_script(self) -> str:
        """Installation script for Gemini CLI."""
        version = self._kwargs.get("version")
        if not version:
            version = self.env_vars.get("GEMINI_CLI_VERSION")
        if not version or not str(version).strip():
            version = "latest"

        npm_pkg = f"@google/gemini-cli@{version}"

        return f"""#!/bin/bash
set -e

echo "Installing Gemini CLI agent..."

apt-get update
apt-get install -y curl ca-certificates tar xz-utils

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/npm"

export npm_config_cache="$CACHE_ROOT/npm"
export NPM_CONFIG_CACHE="$CACHE_ROOT/npm"

NVM_DIR="/opt/featurebench/nvm"
mkdir -p "$NVM_DIR"

mkdir -p "$NVM_DIR/.cache"

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash

export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if [ -z "$(command -v nvm)" ]; then
    echo "nvm not available after install" >&2
    exit 1
fi

nvm install "{self.NODE_VERSION}"
nvm use "{self.NODE_VERSION}"

npm -v

npm install -g {npm_pkg}

command -v gemini >/dev/null 2>&1 || (echo "gemini not found after install" >&2 && exit 1)
gemini --version || true

echo "Gemini CLI installation complete"
"""

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Gemini CLI."""
        escaped_instruction = shlex.quote(instruction)

        model = self._kwargs.get("model")
        if model and "/" in model:
            model = model.split("/")[-1]
        model_arg = f"-m {shlex.quote(model)}" if model else ""

        return (
            "NVM_DIR=${NVM_DIR:-/opt/featurebench/nvm}; "
            "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            f"gemini -p {escaped_instruction} --record-responses /agent-logs/gemini_cli_responses.jsonl --output-format stream-json -y {model_arg} | tee /agent-logs/gemini_cli_stream_output.jsonl".rstrip()
        )

    def pre_run_hook(self, container, log_file) -> bool:
        """Create agent logs directory before running."""
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """Save gemini_cli_stream_output.jsonl Check if the last line of
        gemini_cli_stream_output.jsonl is a success If not, return False If success,
        return True
        """
        log_dir = Path(log_file).parent

        gemini_cli_responses_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/gemini_cli_responses.jsonl",
            log_dir / "gemini_cli_responses.jsonl"
        )

        if not gemini_cli_responses_copied:
            self.logger.error("Failed to copy gemini_cli_responses.jsonl from container")
            return False

        gemini_cli_stream_output_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/gemini_cli_stream_output.jsonl",
            log_dir / "gemini_cli_stream_output.jsonl"
        )

        if not gemini_cli_stream_output_copied:
            self.logger.error("Failed to copy gemini_cli_stream_output.jsonl from container")
            return False

        with open(log_dir / "gemini_cli_stream_output.jsonl", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
            last_line_json = json.loads(last_line)
            if "type" in last_line_json and last_line_json["type"] == "result" and "status" in last_line_json and last_line_json["status"] == "success":
                return True
            else:
                self.logger.error("Last line of gemini_cli_stream_output.jsonl is not a success, agent may not have finished properly")
                return False

    def get_env_setup_script(self) -> str:
        """Get environment setup script for Gemini CLI."""
        lines = ["#!/bin/bash", ""]

        env_settings: Dict[str, str] = {}

        if self.env_vars.get("GEMINI_API_KEY"):
            env_settings["GEMINI_API_KEY"] = self.env_vars["GEMINI_API_KEY"]

        for key in [
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
        ]:
            if self.env_vars.get(key):
                env_settings[key] = self.env_vars[key]

        model = self._kwargs.get("model")
        if model and "/" in model:
            model = model.split("/")[-1]
        if model:
            env_settings["GEMINI_MODEL"] = str(model)

        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                if "localhost" in value_str or "127.0.0.1" in value_str:
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())

        lines.extend(
            [
                "",
                "# Load NVM",
                'export NVM_DIR="/opt/featurebench/nvm"',
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true',
            ]
        )

        lines.extend(
            [
                "",
                "# Configure workspace model overrides (avoid internal gemini-2.5-flash calls)",
                'if [ -n "${GEMINI_MODEL:-}" ]; then',
                '  mkdir -p /root/.gemini',
                "  cat > /root/.gemini/settings.json << 'JSONEOF'",
                "{",
                '  "modelConfigs": {',
                '    "overrides": [',
                "      {",
                '        "match": { "model": "loop-detection" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "loop-detection-double-check" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "classifier" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "prompt-completion" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "summarizer-default" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "summarizer-shell" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "next-speaker-checker" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "llm-edit-fixer" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "edit-corrector" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-search" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-fetch" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-fetch-fallback" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-default" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-pro" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-flash" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-flash-lite" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-3-pro" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-3-flash" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      }",
                "    ]",
                "  }",
                "}",
                "JSONEOF",
                "fi",
            ]
        )

        return "\n".join(lines)

    def failure_hook(self, container, log_file: Path) -> None:
        """Persist gemini-cli detailed error reports on failures."""

        candidates: list[str] = []
        try:
            with open(log_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 64 * 1024), 0)
                tail = f.read().decode("utf-8", errors="ignore")

            candidates = re.findall(r"(/tmp/gemini-client-error-[^\s]+?\.json)", tail)
            seen: set[str] = set()
            candidates = [p for p in candidates if not (p in seen or seen.add(p))]
        except Exception:
            candidates = []

        if not candidates:
            exit_code, output = self.cm.exec_command(
                container,
                "ls -1t /tmp/gemini-client-error-*.json 2>/dev/null | head -n 5 || true",
            )
            if exit_code != 0:
                return

            candidates = [line.strip() for line in output.splitlines() if line.strip()]
            if not candidates:
                return

        output_dir = log_file.parent
