"""Restrict a baseline container's network to the LLM API, after tool install."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Set
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PROXY_PORT = 3128
PROXY_SCRIPT = Path(__file__).parent / "egress_proxy.py"
PROXY_SCRIPT_IN_CONTAINER = "/egress_proxy.py"
DEFAULT_PROXY_IMAGE = "python:3.12-alpine"
_LISTEN_TIMEOUT_SECONDS = 30.0


def hosts_from_env(env: Mapping[str, str]) -> Set[str]:
    """The host names of every URL-valued entry in an agent environment."""
    hosts: Set[str] = set()
    for value in env.values():
        text = str(value or "").strip()
        if not text.lower().startswith(("http://", "https://")):
            continue
        host = urlsplit(text).hostname
        if host:
            hosts.add(host.lower())
    return hosts


class ApiOnlyEgress:
    """One run's egress sidecar, internal network, and the containers on it."""

    def __init__(
        self,
        allow_hosts: Iterable[str],
        run_id: str,
        proxy_image: str = DEFAULT_PROXY_IMAGE,
        client: Any = None,
    ):
        self.allow_hosts = sorted({h.strip().lower() for h in allow_hosts if h and h.strip()})
        if not self.allow_hosts:
            raise ValueError(
                "egress=api_only with an empty allow-list would cut the agent "
                "off from its model; no *_BASE_URL in the agent environment and "
                "no egress_allow_hosts in config"
            )
        self.run_id = run_id
        self.proxy_image = proxy_image
        self.network_name = f"sc-egress-{run_id}"
        self.proxy_name = f"sc-egress-proxy-{run_id}"
        self._client = client
        self._network: Any = None
        self._proxy: Any = None
        self._restricted: Set[str] = set()
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    def start(self) -> None:
        """Create the internal network and the proxy sidecar on it."""
        client = self.client
        try:
            client.images.get(self.proxy_image)
        except Exception:
            logger.info("Egress: pulling proxy image %s", self.proxy_image)
            client.images.pull(self.proxy_image)

        self._network = client.networks.create(
            self.network_name, driver="bridge", internal=True
        )
        self._proxy = client.containers.create(
            self.proxy_image,
            command=["python3", "-u", PROXY_SCRIPT_IN_CONTAINER],
            name=self.proxy_name,
            environment={
                "EGRESS_ALLOW_HOSTS": ",".join(self.allow_hosts),
                "EGRESS_PORT": str(PROXY_PORT),
            },
            detach=True,
        )
        self._ship_proxy_script()
        self._network.connect(self._proxy)
        self._proxy.start()
        self._wait_until_listening()
        logger.info(
            "Egress: api_only — proxy %s on network %s, allowed hosts: %s",
            self.proxy_name, self.network_name, ", ".join(self.allow_hosts),
        )

    def _ship_proxy_script(self) -> None:
        from featurebench.harness.container import copy_to_container

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(PROXY_SCRIPT.read_text(encoding="utf-8"))
            temp_path = f.name
        try:
            copy_to_container(self._proxy, temp_path, PROXY_SCRIPT_IN_CONTAINER)
        finally:
            os.unlink(temp_path)

    def _wait_until_listening(self) -> None:
        deadline = time.time() + _LISTEN_TIMEOUT_SECONDS
        while time.time() < deadline:
            logs = self._proxy.logs().decode("utf-8", errors="replace")
            if "listening on" in logs:
                return
            self._proxy.reload()
            if self._proxy.status not in ("created", "running"):
                raise RuntimeError(
                    f"egress proxy {self.proxy_name} exited before listening: "
                    f"{logs[-500:]}"
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"egress proxy {self.proxy_name} did not start listening within "
            f"{_LISTEN_TIMEOUT_SECONDS:.0f}s"
        )

    def stop(self, log_path: Optional[Path] = None) -> None:
        """Remove the sidecar and the network; save the proxy's decision log."""
        if self._proxy is not None:
            try:
                if log_path is not None:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_bytes(self._proxy.logs())
            except Exception as e:
                logger.warning("Egress: could not save proxy log: %s", e)
            try:
                self._proxy.remove(force=True)
            except Exception as e:
                logger.warning("Egress: could not remove proxy %s: %s", self.proxy_name, e)
            self._proxy = None
        if self._network is not None:
            try:
                self._network.remove()
            except Exception as e:
                logger.warning("Egress: could not remove network %s: %s", self.network_name, e)
            self._network = None

    def restrict(self, container: Any) -> None:
        """Move ``container`` from the bridge onto the internal network."""
        if self._network is None:
            raise RuntimeError("ApiOnlyEgress.restrict() before start()")
        bridge = self.client.networks.get("bridge")
        bridge.disconnect(container, force=True)
        self._network.connect(container)
        with self._lock:
            self._restricted.add(container.id)
        logger.info("Egress: %s restricted to %s", container.short_id, self.proxy_name)

    def is_restricted(self, container: Any) -> bool:
        with self._lock:
            return container.id in self._restricted

    def release(self, container: Any) -> None:
        """Put a container this run is keeping back on the bridge."""
        if self._network is not None:
            try:
                self._network.disconnect(container, force=True)
            except Exception as e:
                logger.warning(
                    "Egress: could not detach %s from %s: %s",
                    container.short_id, self.network_name, e,
                )
        try:
            self.client.networks.get("bridge").connect(container)
        except Exception as e:
            logger.warning(
                "Egress: could not return %s to the bridge: %s", container.short_id, e
            )
        with self._lock:
            self._restricted.discard(container.id)
        logger.info("Egress: %s released to the bridge (kept for resume)", container.short_id)

    @property
    def proxy_url(self) -> str:
        return f"http://{self.proxy_name}:{PROXY_PORT}"

    def shell_prelude(self) -> str:
        """``export ...`` for the proxy variables, to prefix a run command with."""
        no_proxy = "localhost,127.0.0.1"
        return (
            f"export HTTP_PROXY={self.proxy_url} HTTPS_PROXY={self.proxy_url} "
            f"http_proxy={self.proxy_url} https_proxy={self.proxy_url} "
            f"NO_PROXY={no_proxy} no_proxy={no_proxy} && "
        )
