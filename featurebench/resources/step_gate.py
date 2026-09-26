#!/usr/bin/env python3
"""Stop an agent CLI once it has spent a fixed number of model calls."""

import json
import os
import re
import signal
import sys
import threading
import time

STEP_CAP_EXIT_CODE = 111

STEP_CAP_EVENT_TYPE = "step_cap_reached"

TERM_GRACE_SECONDS = 10.0

DRAIN_TIMEOUT_SECONDS = 60

AUDIT_POLL_SECONDS = 0.2

AUDIT_MISSING_WARN_SECONDS = 10.0

AUDIT_SEGMENT_PREFIX = "hosted_tool_filter listening"

_AUDIT_OK_RE = re.compile(r"\bok=(\d+)")


def _count_claude_code(event, state):
    """Distinct assistant message ids -- one per model response."""
    if event.get("type") != "assistant":
        return 0
    message_id = (event.get("message") or {}).get("id")
    if not message_id or message_id in state:
        return 0
    state.add(message_id)
    return 1


def _count_codex(event, _state):
    """Completed items that end a response: a tool call or a spoken message."""
    if event.get("type") != "item.completed":
        return 0
    item_type = (event.get("item") or {}).get("type")
    return 1 if item_type in ("command_execution", "agent_message") else 0


def _count_opencode(event, _state):
    """``step_finish`` -- one per response that closed."""
    return 1 if event.get("type") == "step_finish" else 0


COUNTERS = {
    "claude_code": _count_claude_code,
    "codex": _count_codex,
    "opencode": _count_opencode,
}


def _wire_count_shim(text):
    """Whole responses the current shim has relayed, from its audit log."""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.startswith(AUDIT_SEGMENT_PREFIX):
            start = index + 1
    for line in reversed(lines[start:]):
        match = _AUDIT_OK_RE.search(line)
        if match:
            return int(match.group(1))
    return None


WIRE_UNITS = {
    "codex_wire": _wire_count_shim,
    "opencode_wire": _wire_count_shim,
}

KNOWN_UNITS = frozenset(COUNTERS) | frozenset(WIRE_UNITS)


def _read_pid(pid_file):
    try:
        with open(pid_file, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill(pid):
    """SIGTERM the CLI, then SIGKILL it if it is still there."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + TERM_GRACE_SECONDS
    while time.time() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _arm_drain_timeout(seconds):
    """Bound the post-kill drain."""
    def _stop():
        sys.stderr.write(
            f"step_gate: transcript still open {seconds}s after the kill; "
            f"closing the stream\n"
        )
        sys.stdout.flush()
        os._exit(STEP_CAP_EXIT_CODE)

    timer = threading.Timer(seconds, _stop)
    timer.daemon = True
    timer.start()


UNREADABLE = object()


def _read_wire_count(parse, audit_path):
    try:
        with open(audit_path, "r", encoding="utf-8", errors="replace") as handle:
            return parse(handle.read())
    except OSError:
        return UNREADABLE


class Gate:
    """The cap itself: one marker, one kill, whichever thread gets there."""

    def __init__(self, max_steps, unit, pid_file):
        self.max_steps = max_steps
        self.unit = unit
        self.pid_file = pid_file
        self.out_lock = threading.Lock()
        self.capped = False

    def relay(self, line):
        with self.out_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def fire(self, count, kill=True):
        """Write the marker and stop the CLI."""
        with self.out_lock:
            if self.capped:
                return True
            self.capped = True
            sys.stdout.write(json.dumps({
                "type": STEP_CAP_EVENT_TYPE,
                "unit": self.unit,
                "limit": self.max_steps,
                "count": count,
            }) + "\n")
            sys.stdout.flush()
        sys.stderr.write(
            f"step_gate: {self.unit} reached {count}/{self.max_steps} model "
            f"calls; stopping the agent\n"
        )
        if not kill:
            return True

        pid = _read_pid(self.pid_file)
        if pid is None:
            sys.stderr.write(
                f"step_gate: no pid in {self.pid_file}; closing the stream "
                f"instead\n"
            )
            return False
        _arm_drain_timeout(DRAIN_TIMEOUT_SECONDS)
        _kill(pid)
        return True


def _poll_audit(gate, parse, audit_path):
    """Fire the gate once the shim's count reaches the cap."""
    started = time.time()
    warned = False
    while not gate.capped:
        count = _read_wire_count(parse, audit_path)
        if count is UNREADABLE:
            if not warned and time.time() - started > AUDIT_MISSING_WARN_SECONDS:
                warned = True
                sys.stderr.write(
                    f"step_gate: cannot read {audit_path} after "
                    f"{AUDIT_MISSING_WARN_SECONDS:.0f}s; the cap cannot fire "
                    f"until it can\n"
                )
        elif count is not None and count >= gate.max_steps:
            if not gate.fire(count):
                os._exit(STEP_CAP_EXIT_CODE)
            return
        time.sleep(AUDIT_POLL_SECONDS)


def _usage():
    sys.stderr.write(
        "usage: step_gate.py <max_steps> <unit> <pid_file> [<audit_log>]\n"
        f"       stream units {sorted(COUNTERS)} count stdin;\n"
        f"       wire units {sorted(WIRE_UNITS)} poll <audit_log>\n"
    )
    return 2


def main(argv):
    if len(argv) not in (3, 4):
        return _usage()

    max_steps = int(argv[0])
    unit = argv[1]
    pid_file = argv[2]
    audit_path = argv[3] if len(argv) == 4 else None

    count_one = COUNTERS.get(unit)
    parse_wire = WIRE_UNITS.get(unit)
    if count_one is None and parse_wire is None:
        sys.stderr.write(f"step_gate: unknown unit {unit!r}\n")
        return 2
    if parse_wire is not None and audit_path is None:
        sys.stderr.write(f"step_gate: unit {unit!r} needs the audit log path\n")
        return _usage()

    gate = Gate(max_steps, unit, pid_file)
    if parse_wire is not None:
        poller = threading.Thread(
            target=_poll_audit, args=(gate, parse_wire, audit_path), daemon=True,
        )
        poller.start()

    state = set()
    steps = 0

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        gate.relay(line)

        if gate.capped or count_one is None:
            continue

        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue

        steps += count_one(event, state)
        if steps < max_steps:
            continue
        if not gate.fire(steps):
            return STEP_CAP_EXIT_CODE

    if parse_wire is not None and not gate.capped:
        count = _read_wire_count(parse_wire, audit_path)
        if count is not UNREADABLE and count is not None and count >= max_steps:
            gate.fire(count, kill=False)

    return STEP_CAP_EXIT_CODE if gate.capped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
