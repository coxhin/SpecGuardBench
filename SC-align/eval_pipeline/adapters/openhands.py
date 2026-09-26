"""OpenHands evaluation adapter."""

import json
import logging
import shlex
from typing import Dict, Optional

from docker.models.containers import Container

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

from .base import EvalBaselineAdapter

logger = logging.getLogger(__name__)


class OpenHandsAdapter(EvalBaselineAdapter):
    def get_install_script(self) -> str:
        version = self.tool_version
        if version and version != "latest":
            pkg = f"openhands-ai=={version}"
        else:
            pkg = "openhands-ai"
        return f"""#!/bin/bash
set -e

pip install {pkg}

echo "OpenHands installed"
"""

    def get_run_command(self, prompt_path: str) -> str:
        api_key = shlex.quote(self.env_vars.get("LLM_API_KEY", ""))
        base_url = shlex.quote(self.env_vars.get("LLM_BASE_URL", ""))
        model = shlex.quote(self.env_vars.get("LLM_MODEL", ""))

        env_prefix = f"export LLM_API_KEY={api_key}"
        if self.env_vars.get("LLM_BASE_URL", ""):
            env_prefix += f" && export LLM_BASE_URL={base_url}"
        if self.env_vars.get("LLM_MODEL", ""):
            env_prefix += f" && export LLM_MODEL={model}"

        return (
            f'{env_prefix} && '
            f'cd /testbed && '
            f'python -m openhands.core.main '
            f'-t "$(cat {prompt_path})" '
            f'2>&1 | tee /tmp/openhands_output.txt'
        )

    def _extract_cost(self, container: Container) -> Optional[Dict[str, float]]:
        """Parse cost from OpenHands trajectory.json."""
        try:
            exit_code, output = exec_run_with_timeout(
                container, "cat /tmp/trajectory.json", timeout=10
            )
            if exit_code != 0:
                return None
            text = output.decode(UTF8, errors="replace")
            trajectory = json.loads(text)
            if not isinstance(trajectory, list) or not trajectory:
                return None
            last_event = trajectory[-1]
            metrics = last_event.get("metrics", {})
            accumulated_cost = metrics.get("accumulated_cost", 0.0)
            if accumulated_cost:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": accumulated_cost,
                }
        except Exception as e:
            logger.warning(f"Failed to extract cost from OpenHands trajectory: {e}")
        return None
