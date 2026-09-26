"""Reassembling a baseline's per-attempt transcripts."""

import json
from typing import Any, Dict, Iterator, List, Optional

from .models import BaselineResult


def session_chains(attempts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group attempt records into continuous-conversation chains."""
    chains: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for record in attempts:
        previous_session = current[-1].get("session_id") if current else None
        resumes_previous = (
            bool(current)
            and record.get("resume_of_session") is not None
            and record.get("resume_of_session") == previous_session
        )
        if resumes_previous:
            current.append(record)
            continue
        if current:
            chains.append(current)
        current = [record]

    if current:
        chains.append(current)
    return chains


def chain_events(chain: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield every stream event in a chain, in order, tagged with its attempt."""
    for record in chain:
        attempt = record.get("attempt")
        for line in (record.get("raw_output") or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict):
                event["_attempt"] = attempt
                yield event


def joined_trajectory(result: BaselineResult) -> List[Dict[str, Any]]:
    """Events of the chain that produced the result, as one timeline."""
    chains = session_chains(result.attempts)
    if not chains:
        return []
    return list(chain_events(chains[-1]))


def total_steps(attempts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the per-attempt step counts, keeping each adapter's own keys."""
    totals: Dict[str, int] = {}
    for record in attempts:
        for key, value in (record.get("steps") or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def describe_attempts(result: BaselineResult) -> str:
    """One line per attempt, for eyeballing what a retried task actually did."""
    lines: List[str] = []
    for record in result.attempts:
        session: Optional[str] = record.get("session_id")
        tag = "fresh"
        if record.get("resume_rejected"):
            tag = f"resume-REFUSED of {record.get('resume_of_session')}"
        elif record.get("resumed"):
            tag = f"resume of {record.get('resume_of_session')}"
        errors = ",".join(
            e.get("error_type", "?") for e in record.get("llm_errors") or []
        )
        lines.append(
            f"attempt {record.get('attempt')} [{tag}] "
            f"session={session} exit={record.get('exit_code')} "
            f"{record.get('duration_seconds')}s "
            f"steps={record.get('steps')}"
            + (f" errors={errors}" if errors else "")
        )
    return "\n".join(lines)
