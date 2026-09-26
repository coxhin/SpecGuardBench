"""Allow-list forward proxy that runs *inside* the egress sidecar container."""

import asyncio
import os
import sys
from urllib.parse import urlsplit

ALLOW_HOSTS = {
    h.strip().lower()
    for h in os.environ.get("EGRESS_ALLOW_HOSTS", "").split(",")
    if h.strip()
}
PORT = int(os.environ.get("EGRESS_PORT", "3128"))
CONNECT_TIMEOUT = 30.0


def _log(decision: str, method: str, host: str, port: int) -> None:
    sys.stdout.write(f"{decision} {method} {host}:{port}\n")
    sys.stdout.flush()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _reply(writer: asyncio.StreamWriter, status: str) -> None:
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
    )
    try:
        await writer.drain()
    finally:
        writer.close()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=CONNECT_TIMEOUT)
    except Exception:
        writer.close()
        return

    request_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
    parts = request_line.split()
    if len(parts) != 3:
        await _reply(writer, "400 Bad Request")
        return
    method, target, version = parts

    if method == "CONNECT":
        host, _, port_text = target.rpartition(":")
        port = int(port_text) if port_text.isdigit() else 443
        upstream_head = b""
    else:
        url = urlsplit(target)
        host = url.hostname or ""
        port = url.port or (443 if url.scheme == "https" else 80)
        path = url.path or "/"
        if url.query:
            path += "?" + url.query
        upstream_head = head.replace(
            request_line.encode("latin-1"),
            f"{method} {path} {version}".encode("latin-1"),
            1,
        )

    host = host.lower()
    if not host or host not in ALLOW_HOSTS:
        _log("DENY", method, host or "?", port)
        await _reply(writer, "403 Forbidden")
        return
    _log("ALLOW", method, host, port)

    try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=CONNECT_TIMEOUT
        )
    except Exception:
        await _reply(writer, "502 Bad Gateway")
        return

    if method == "CONNECT":
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
    else:
        up_writer.write(upstream_head)
        await up_writer.drain()

    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def _main() -> None:
    server = await asyncio.start_server(_handle, host="0.0.0.0", port=PORT)
    sys.stdout.write(
        f"listening on 0.0.0.0:{PORT}; allowed hosts: {', '.join(sorted(ALLOW_HOSTS)) or '<none>'}\n"
    )
    sys.stdout.flush()
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
