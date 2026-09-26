"""The loopback shim a CLI talks to instead of its provider."""

import logging
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from featurebench.harness.container import exec_run_with_timeout
from featurebench.harness.constants import UTF8

logger = logging.getLogger(__name__)

TOOL_FILTER_PORT = 8123

TOOL_FILTER_SRC = (
    Path(__file__).resolve().parents[3]
    / "featurebench"
    / "resources"
    / "hosted_tool_filter.py"
)
TOOL_FILTER_PATH = "/opt/eval/hosted_tool_filter.py"
TOOL_FILTER_LAUNCHER = "/opt/eval/start_tool_filter.sh"
TOOL_FILTER_AUDIT_PATH = "/tmp/hosted_tool_filter.log"
TOOL_FILTER_STDOUT_PATH = "/tmp/hosted_tool_filter.out"
TOOL_FILTER_SEGMENT_PREFIX = "hosted_tool_filter listening"

AUDIT_REQUESTS_RE = re.compile(r"(\d+)/(\d+) requests")
AUDIT_OK_RE = re.compile(r"\bok=(\d+)")
AUDIT_REFUSED_RE = re.compile(r"\brefused=(\d+)")


def loopback_base_url(base_url: str) -> str:
    """Where the CLI is pointed: the shim, with the upstream's path kept."""
    upstream_path = urlsplit(base_url).path.rstrip("/")
    return f"http://127.0.0.1:{TOOL_FILTER_PORT}{upstream_path}"


def install_snippet(base_url: str) -> str:
    """Shell for an install script: write the shim and its launcher."""
    filter_src = TOOL_FILTER_SRC.read_text(encoding="utf-8")
    return f"""mkdir -p /opt/eval
cat > {TOOL_FILTER_PATH} << 'TOOL_FILTER_PY'
{filter_src}
TOOL_FILTER_PY

cat > {TOOL_FILTER_LAUNCHER} << 'TOOL_FILTER_SH'
#!/bin/bash
export UPSTREAM_BASE_URL={shlex.quote(base_url)}
export LISTEN_PORT={TOOL_FILTER_PORT}
export TOOL_FILTER_AUDIT={TOOL_FILTER_AUDIT_PATH}
exec python3 {TOOL_FILTER_PATH}
TOOL_FILTER_SH
chmod +x {TOOL_FILTER_LAUNCHER}

python3 -c 'import json, urllib.request, http.server' || {{
    echo "python3 with stdlib http/json is required for hosted_tool_filter" >&2
    exit 1
}}
"""


def prelude(max_steps: Optional[int]) -> str:
    """Shell that starts a fresh shim for one invocation and waits for it."""
    budget = (
        f"export MAX_MODEL_CALLS={int(max_steps)}; "
        if max_steps
        else "unset MAX_MODEL_CALLS; "
    )
    return (
        'pkill -f "[h]osted_tool_filter.py" > /dev/null 2>&1; '
        'for _ in $(seq 1 50); do '
        'pgrep -f "[h]osted_tool_filter.py" > /dev/null 2>&1 || break; '
        'sleep 0.1; '
        'done; '
        f'rm -f {TOOL_FILTER_STDOUT_PATH}; '
        + budget +
        f'(setsid {TOOL_FILTER_LAUNCHER} '
        f'> {TOOL_FILTER_STDOUT_PATH} 2>&1 &); '
        'for _ in $(seq 1 100); do '
        f'grep -q READY {TOOL_FILTER_STDOUT_PATH} 2>/dev/null && break; '
        'sleep 0.1; '
        'done; '
        f'grep -q READY {TOOL_FILTER_STDOUT_PATH} 2>/dev/null || '
        '{ echo "FATAL: hosted_tool_filter did not start" >&2; '
        f'cat {TOOL_FILTER_STDOUT_PATH} >&2 2>/dev/null; exit 1; ' '}; '
        'export NO_PROXY="127.0.0.1,localhost,${NO_PROXY}" && '
        'export no_proxy="127.0.0.1,localhost,${no_proxy}" && '
    )


def last_counted_line(text: str) -> Optional[str]:
    """The audit line carrying this invocation's running totals."""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.startswith(TOOL_FILTER_SEGMENT_PREFIX):
            start = index + 1
    for line in reversed(lines[start:]):
        if AUDIT_REQUESTS_RE.search(line):
            return line
    return None


def audit_counters(text: str) -> Dict[str, int]:
    """Call counts off the audit log's last counted line."""
    counters: Dict[str, int] = {}
    last = last_counted_line(text)
    if last is None:
        return counters
    match = AUDIT_REQUESTS_RE.search(last)
    if match:
        counters["api_requests"] = int(match.group(2))
    match = AUDIT_OK_RE.search(last)
    if match:
        counters["successful_llm_calls"] = int(match.group(1))
    match = AUDIT_REFUSED_RE.search(last)
    if match:
        counters["refused_llm_calls"] = int(match.group(1))
    return counters


def read_audit_counters(adapter: Any, container: Any) -> Dict[str, int]:
    """Read the shim's audit log out of the container and log its last line."""
    label = f"[{adapter.name}/{adapter.model}]"
    try:
        exit_code, output = exec_run_with_timeout(
            container, f"cat {TOOL_FILTER_AUDIT_PATH}", timeout=10
        )
        if exit_code != 0:
            logger.warning(
                f"{label} hosted-tool filter audit log missing — no wire "
                f"count for this run"
            )
            return {}
        text = output.decode(UTF8, errors="replace").strip()
        last = last_counted_line(text) or (
            text.splitlines()[-1] if text else "(empty)"
        )
        logger.info(f"{label} hosted-tool filter: {last}")
        return audit_counters(text)
    except Exception as e:
        logger.warning(
            f"{label} Failed to read hosted-tool filter audit log: {e}"
        )
        return {}
