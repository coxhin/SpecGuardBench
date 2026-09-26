"""Codex agent implementation."""

import shlex
from pathlib import Path
from typing import Dict, List
import json
from urllib.parse import urlsplit

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY

TOOL_FILTER_PORT = 8123

TOOL_FILTER_SRC = (
    Path(__file__).resolve().parents[2] / "resources" / "hosted_tool_filter.py"
)


class CodexAgent(BaseAgent):
    """Codex agent for FeatureBench inference."""

    NODE_VERSION = "22"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def install_script(self) -> str:
        """Installation script for Codex CLI."""
        version = self._kwargs.get("version")
        if not version:
            version = self.env_vars.get("CODEX_VERSION")
        if not version or not str(version).strip():
            version = "latest"

        npm_pkg = f"@openai/codex@{version}"

        return f"""#!/bin/bash
set -e

echo "Installing Codex agent..."

if command -v codex >/dev/null 2>&1; then
    echo "codex already available; skipping installation"
    codex --version || true
    exit 0
fi

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

command -v codex >/dev/null 2>&1 || (echo "codex not found after install" >&2 && exit 1)
codex --version || true

codex --version || true

echo "Codex installation complete"
"""

    def pre_run_hook(self, container, log_file) -> bool:
        """Create log dir."""

        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """Collect outputs and validate completion."""

        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(container, "/agent-logs/codex_events.jsonl", log_dir / "codex_events.jsonl")
        except Exception:
            pass
        try:
            events_path = log_dir / "codex_events.jsonl"
            if not events_path.is_file():
                self.logger.error("codex_events.jsonl not found; treating run as failed")
                return False

            with events_path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            last_line = ""
            for raw in reversed(lines):
                s = raw.strip()
                if s:
                    last_line = s
                    break
            if not last_line:
                self.logger.error("codex_events.jsonl is empty; treating run as failed")
                return False

            last_obj = json.loads(last_line)
            if not isinstance(last_obj, dict):
                self.logger.error("codex_events.jsonl last line is not a JSON object; treating run as failed")
                return False

            if last_obj.get("type") != "turn.completed":
                self.logger.error(
                    f"Codex did not finish with turn.completed (last type={last_obj.get('type')}); run failed"
                )
                return False

            if not isinstance(last_obj.get("usage"), dict):
                self.logger.error("Codex turn.completed missing usage; run failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate codex_events.jsonl completion: {e}")
            return False

        return True

    def failure_hook(self, container, log_file: Path) -> None:
        """Collect outputs on failures (best-effort)."""

        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(container, "/agent-logs/codex_events.jsonl", log_dir / "codex_events.jsonl")
        except Exception:
            pass

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Codex."""
        escaped_instruction = shlex.quote(instruction)

        model = self._kwargs.get("model")

        model_arg = f"--model {shlex.quote(str(model))} " if model else ""

        return (
            "NVM_DIR=${NVM_DIR:-/opt/featurebench/nvm}; "
            "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            f"{self._tool_filter_prelude()}"
            "set -o pipefail; "
            f"codex exec "
            f"--sandbox danger-full-access "
            f"--skip-git-repo-check "
            f"--json "
            f"{model_arg}"
            f"-- {escaped_instruction}"
            f" | tee /agent-logs/codex_events.jsonl"
        ).rstrip()

    @staticmethod
    def _tool_filter_base_url(base_url: str) -> str:
        """Loopback URL that fronts ``base_url``, preserving its path prefix."""
        if not base_url:
            return base_url
        path = urlsplit(base_url).path.rstrip("/")
        return f"http://127.0.0.1:{TOOL_FILTER_PORT}{path}"

    @staticmethod
    def _tool_filter_setup_lines(base_url: str) -> List[str]:
        """Shell lines installing the hosted-tool filter into the container."""
        if not base_url:
            return []
        return [
            "",
            "# Hosted-tool filter (strips provider-executed web tools)",
            'mkdir -p /opt/featurebench',
            "cat > /opt/featurebench/hosted_tool_filter.py <<'TOOL_FILTER_PY'",
            TOOL_FILTER_SRC.read_text(encoding="utf-8"),
            "TOOL_FILTER_PY",
            "cat > /opt/featurebench/start_tool_filter.sh <<'TOOL_FILTER_SH'",
            "#!/bin/bash",
            f"export UPSTREAM_BASE_URL={shlex.quote(base_url)}",
            f"export LISTEN_PORT={TOOL_FILTER_PORT}",
            "export TOOL_FILTER_AUDIT=/agent-logs/hosted_tool_filter.log",
            "exec python3 /opt/featurebench/hosted_tool_filter.py",
            "TOOL_FILTER_SH",
            "chmod +x /opt/featurebench/start_tool_filter.sh",
        ]

    def _tool_filter_prelude(self) -> str:
        """Start the filter and abort the run if it will not serve."""
        if not self.env_vars.get("OPENAI_BASE_URL", ""):
            return ""
        return (
            "mkdir -p /agent-logs; "
            'if ! pgrep -f "[h]osted_tool_filter.py" > /dev/null 2>&1; then '
            "(setsid /opt/featurebench/start_tool_filter.sh "
            "> /tmp/hosted_tool_filter.out 2>&1 &); "
            "fi; "
            "for _ in $(seq 1 100); do "
            "grep -q READY /tmp/hosted_tool_filter.out 2>/dev/null && break; "
            "sleep 0.1; "
            "done; "
            "grep -q READY /tmp/hosted_tool_filter.out 2>/dev/null || "
            '{ echo "FATAL: hosted_tool_filter did not start" >&2; '
            "cat /tmp/hosted_tool_filter.out >&2 2>/dev/null; exit 1; }; "
            'export NO_PROXY="127.0.0.1,localhost,${NO_PROXY}"; '
            'export no_proxy="127.0.0.1,localhost,${no_proxy}"; '
        )

    def get_env_setup_script(self) -> str:
        """Get environment setup script for Codex."""
        lines = ["#!/bin/bash", ""]

        api_key = self.env_vars.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for codex agent")

        raw_model = self._kwargs.get("model") or ""
        base_url = self.env_vars.get("OPENAI_BASE_URL") or ""
        is_azure = raw_model.startswith("azure/") or (".openai.azure.com" in base_url)

        env_settings: Dict[str, str] = {
            "OPENAI_API_KEY": api_key,
        }

        if is_azure:
            env_settings["AZURE_OPENAI_API_KEY"] = self.env_vars.get("AZURE_OPENAI_API_KEY") or api_key

        for key in [
            "OPENAI_BASE_URL",
            "OPENAI_ORG",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT",
            "OPENAI_API_VERSION",
            "CODEX_REASONING_EFFORT",
        ]:
            if self.env_vars.get(key):
                env_settings[key] = self.env_vars[key]

        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                _no_rewrite_keys = {
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "ALL_PROXY",
                    "all_proxy",
                    "NO_PROXY",
                    "no_proxy",
                }
                if key not in _no_rewrite_keys and ("localhost" in value_str or "127.0.0.1" in value_str):
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())

        reasoning_effort = (
            self.env_vars.get("CODEX_REASONING_EFFORT")
            or self.env_vars.get("MODEL_REASONING_EFFORT")
            or ""
        )
        reasoning_effort = str(reasoning_effort).strip()
        if reasoning_effort:
            reasoning_effort = reasoning_effort.replace("\\", "\\\\").replace('"', '\\"')

        if is_azure:
            deployment = raw_model.split("/", 1)[1] if raw_model.startswith("azure/") else raw_model

            base = (base_url or "").rstrip("/")
            if base and not base.endswith("/openai/v1"):
                if base.endswith("/openai"):
                    base = f"{base}/v1"
                else:
                    base = f"{base}/openai/v1"

            lines.extend(
                [
                    "",
                    "# Codex CLI config (auto-generated by FeatureBench)",
                    'mkdir -p "$HOME/.codex"',
                    "cat > \"$HOME/.codex/auth.json\" <<'CODEX_AUTH'",
                    json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False),
                    "CODEX_AUTH",
                    "cat > \"$HOME/.codex/config.toml\" <<'CODEX_TOML'",
                    f'model = "{deployment}"',
                    'model_provider = "azure"',
                    *(
                        [f'model_reasoning_effort = "{reasoning_effort}"']
                        if reasoning_effort
                        else []
                    ),
                    "",
                    "[model_providers.azure]",
                    'name = "Azure OpenAI"',
                    f'base_url = "{self._tool_filter_base_url(base)}"',
                    'env_key = "AZURE_OPENAI_API_KEY"',
                    'wire_api = "responses"',
                    "CODEX_TOML",
                    *self._tool_filter_setup_lines(base),
                ]
            )

        else:
            model_name = raw_model
            if model_name.startswith("openai/"):
                model_name = model_name.split("/", 1)[1]

            provider_id = "featurebench"
            base = (base_url or "").rstrip("/")

            lines.extend(
                [
                    "",
                    "# Codex CLI config (auto-generated by FeatureBench)",
                    'mkdir -p "$HOME/.codex"',
                    "cat > \"$HOME/.codex/auth.json\" <<'CODEX_AUTH'",
                    json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False),
                    "CODEX_AUTH",
                    "cat > \"$HOME/.codex/config.toml\" <<'CODEX_TOML'",
                    f'model_provider = "{provider_id}"',
                    *([f'model = "{model_name}"'] if model_name else []),
                    *([f'model_reasoning_effort = "{reasoning_effort}"'] if reasoning_effort else []),
                    "disable_response_storage = true",
                    "",
                    f"[model_providers.{provider_id}]",
                    f'name = "{provider_id}"',
                    *(
                        [f'base_url = "{self._tool_filter_base_url(base)}"']
                        if base
                        else []
                    ),
                    'wire_api = "responses"',
                    "CODEX_TOML",
                    *self._tool_filter_setup_lines(base),
                ]
            )

        lines.extend(
            [
                "",
                "# Load NVM",
                'export NVM_DIR="/opt/featurebench/nvm"',
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true',
            ]
        )

        return "\n".join(lines)
