#!/usr/bin/env python3
"""Loopback shim that strips provider-hosted tools from Responses API requests and, when
asked, holds the CLI to a budget of model calls.
"""

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

DEFAULT_STRIP_TYPES = "web_search,web_search_preview,web_search_2025_08_26"

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "0"))
STRIP_TYPES = {
    t.strip()
    for t in os.environ.get("STRIP_TOOL_TYPES", DEFAULT_STRIP_TYPES).split(",")
    if t.strip()
}
AUDIT_PATH = os.environ.get("TOOL_FILTER_AUDIT", "/tmp/hosted_tool_filter.log")
MAX_MODEL_CALLS = int(os.environ.get("MAX_MODEL_CALLS", "0") or 0)

AUDIT_SEGMENT_PREFIX = "hosted_tool_filter listening"

MODEL_ENDPOINT_RE = re.compile(
    r"/(responses|chat/completions|messages)/?(?:\?|$)"
)

_RESPONSE_END_RE = re.compile(
    rb'"type"\s*:\s*"(response\.completed|response\.failed|'
    rb'response\.incomplete|message_stop)"'
)
_CHAT_DONE_RE = re.compile(rb"data:\s*\[DONE\]")


def is_model_invocation(method, path):
    """Whether this request is a model call rather than housekeeping."""
    return method == "POST" and bool(MODEL_ENDPOINT_RE.search(path or ""))


class StreamWatch:
    """Whether a relayed SSE body reached its terminal event."""

    TAIL_BYTES = 64

    def __init__(self):
        self.outcome = None
        self._tail = b""

    def feed(self, chunk):
        window = self._tail + chunk
        match = _RESPONSE_END_RE.search(window)
        if match:
            self.outcome = match.group(1).decode("ascii").replace("response.", "")
        elif _CHAT_DONE_RE.search(window):
            self.outcome = "done"
        self._tail = window[-self.TAIL_BYTES:]

    @property
    def whole(self):
        return self.outcome in ("completed", "done", "message_stop")


def _stats_phrase():
    """The running totals, in the form every counted audit line carries."""
    return (
        f"total {_stats['stripped_tools']} in "
        f"{_stats['stripped_requests']}/{_stats['requests']} requests; "
        f"ok={_stats['ok']} failed={_stats['failed']}; "
        f"refused={_stats['refused']}"
    )


def _record_outcome(method, path, stripped, ok, tools=None):
    """Tally one finished model call and write its audit line."""
    with _lock:
        _stats["ok" if ok else "failed"] += 1
        reached = (
            ok
            and MAX_MODEL_CALLS > 0
            and _stats["ok"] >= MAX_MODEL_CALLS
            and not _stats["capped"]
        )
        if reached:
            _stats["capped"] = True
        phrase = _stats_phrase()
    offered = "" if tools is None else f"tools={tools}; "
    audit(
        f"stripped {stripped} hosted tool(s) from {method} {path} "
        f"({offered}{phrase})"
    )
    if reached:
        audit(f"step_cap_reached limit={MAX_MODEL_CALLS} ({phrase})")


def _budget_spent():
    with _lock:
        return MAX_MODEL_CALLS > 0 and _stats["ok"] >= MAX_MODEL_CALLS


if not UPSTREAM:
    sys.exit("UPSTREAM_BASE_URL is required")

_split = urlsplit(UPSTREAM)
UPSTREAM_ORIGIN = f"{_split.scheme}://{_split.netloc}"

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

_lock = threading.Lock()
_audit_lock = threading.Lock()
_stats = {
    "requests": 0,
    "stripped_tools": 0,
    "stripped_requests": 0,
    "ok": 0,
    "failed": 0,
    "refused": 0,
    "capped": False,
}


def audit(message):
    """Append one line to the audit log; never let logging break a request."""
    with _audit_lock:
        try:
            with open(AUDIT_PATH, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            pass


def filter_body(raw):
    """Return (body, n_stripped, n_tools)."""
    if not raw:
        return raw, 0, 0
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw, 0, 0
    if not isinstance(payload, dict):
        return raw, 0, 0

    tools = payload.get("tools")
    if not isinstance(tools, list):
        return raw, 0, 0

    kept = [
        t
        for t in tools
        if not (isinstance(t, dict) and t.get("type") in STRIP_TYPES)
    ]
    n = len(tools) - len(kept)
    if n == 0:
        return raw, 0, len(kept)

    payload["tools"] = kept
    return json.dumps(payload).encode("utf-8"), n, len(kept)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refuse(self, method):
        """Turn a model call away because the budget is spent."""
        with _lock:
            _stats["refused"] += 1
            phrase = _stats_phrase()
        audit(
            f"refused {method} {self.path}: step cap limit={MAX_MODEL_CALLS} "
            f"reached ({phrase})"
        )
        self._send_json(403, {
            "error": {
                "message": (
                    f"step cap reached: this run's budget of "
                    f"{MAX_MODEL_CALLS} model calls is spent"
                ),
                "type": "step_cap_reached",
                "code": "step_cap_reached",
            }
        })

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        counted = is_model_invocation(method, self.path)
        if counted and _budget_spent():
            self._refuse(method)
            return

        body, stripped, tools = filter_body(raw)
        if counted:
            with _lock:
                _stats["requests"] += 1
                if stripped:
                    _stats["stripped_tools"] += stripped
                    _stats["stripped_requests"] += 1

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "content-length")
        }
        headers["Accept-Encoding"] = "identity"

        req = urllib.request.Request(
            UPSTREAM_ORIGIN + self.path,
            data=body if body else None,
            headers=headers,
            method=method,
        )

        try:
            upstream = urllib.request.urlopen(req, timeout=None)
        except urllib.error.HTTPError as e:
            upstream = e
        except Exception as e:
            audit(f"upstream error for {method} {self.path}: {e!r}")
            if counted:
                _record_outcome(
                    method, self.path, stripped, ok=False, tools=tools
                )
            self._send_json(
                502,
                {"error": {"message": f"hosted_tool_filter upstream error: {e}"}},
            )
            return

        self.send_response(upstream.status)
        for k, v in upstream.headers.items():
            if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        status_ok = 200 <= getattr(upstream, "status", 0) < 300
        content_type = (upstream.headers.get("Content-Type") or "").lower()
        watch = StreamWatch() if "text/event-stream" in content_type else None
        drained = False
        client_gone = False
        try:
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                if watch is not None:
                    watch.feed(chunk)
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            drained = True
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            client_gone = True
        except Exception as e:
            audit(f"upstream stream failed for {method} {self.path}: {e!r}")
        finally:
            upstream.close()
            if counted and not client_gone:
                whole = drained and (watch is None or watch.whole)
                if status_ok and drained and not whole:
                    audit(
                        f"upstream stream for {method} {self.path} ended "
                        f"without a terminal event "
                        f"(last seen: {watch.outcome or 'none'})"
                    )
                _record_outcome(
                    method, self.path, stripped, ok=status_ok and whole,
                    tools=tools,
                )

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def do_DELETE(self):
        self._proxy("DELETE")

    def log_message(self, *args):
        pass


def open_segment(port):
    """Write the line that opens this shim's segment of the audit log."""
    budget = (
        f"; budget: {MAX_MODEL_CALLS} model calls" if MAX_MODEL_CALLS else ""
    )
    audit(
        f"{AUDIT_SEGMENT_PREFIX} on 127.0.0.1:{port} -> {UPSTREAM_ORIGIN} "
        f"(stripping: {sorted(STRIP_TYPES)}{budget})"
    )


def main():
    server = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    server.daemon_threads = True
    open_segment(server.server_address[1])
    print(f"READY {server.server_address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
