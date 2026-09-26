"""Docker container management for FeatureBench evaluation."""

import io
import logging
import os
import tarfile
import time
import threading
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

import docker
from docker.models.containers import Container

from featurebench.harness.constants import (
    DOCKER_USER,
    DOCKER_WORKDIR,
    UTF8,
)

DOCKER_HOST_GATEWAY = "172.17.0.1"


def exec_run_with_timeout(
    container: Container,
    cmd: str,
    timeout: int | None = None,
    **kwargs
) -> tuple[int, bytes]:
    """Execute command in container with timeout."""
    user = kwargs.get("user", DOCKER_USER)
    workdir = kwargs.get("workdir", DOCKER_WORKDIR)
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in ["user", "workdir"]}

    inner_cmd = f"source ~/.bashrc && {cmd}"
    if timeout is not None:
        inner_cmd = f"timeout -k 10 {int(timeout)}s bash -lc {shlex.quote(inner_cmd)}"

    full_cmd: list[str] = ["/bin/bash", "-lc", inner_cmd]

    result: Any | None = None
    error: Exception | None = None

    def _run_exec() -> None:
        nonlocal result, error
        try:
            result = container.exec_run(
                full_cmd,
                user=user,
                workdir=workdir,
                stream=False,
                demux=False,
                **extra_kwargs,
            )
        except Exception as e:
            error = e

    thread = threading.Thread(target=_run_exec, daemon=True)
    thread.start()

    join_timeout = None if timeout is None else max(1, int(timeout) + 30)
    thread.join(join_timeout)

    if thread.is_alive():
        try:
            container.kill()
        except Exception:
            pass
        return -1, f"Timeout after {timeout}s (container killed)".encode(UTF8)

    if error is not None:
        return -1, str(error).encode(UTF8)
    if result is None:
        return -1, b"Unknown exec error"

    try:
        out = result.output or b""
    except Exception:
        out = b""
    if timeout is not None and getattr(result, "exit_code", None) == 127 and b"timeout" in out.lower():
        full_cmd = ["/bin/bash", "-lc", f"source ~/.bashrc && {cmd}"]
        result = None
        error = None
        thread = threading.Thread(target=_run_exec, daemon=True)
        thread.start()
        thread.join(max(1, int(timeout) + 30))
        if thread.is_alive():
            try:
                container.kill()
            except Exception:
                pass
            return -1, f"Timeout after {timeout}s (container killed)".encode(UTF8)
        if error is not None:
            return -1, str(error).encode(UTF8)
        if result is None:
            return -1, b"Unknown exec error"
        return result.exit_code, result.output

    return result.exit_code, result.output


def copy_to_container(
    container: Container,
    src_path: str | Path,
    dst_path: str
) -> None:
    """Copy file to container."""
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(str(src_path), arcname=os.path.basename(dst_path))

    tar_stream.seek(0)

    dst_dir = os.path.dirname(dst_path)
    container.put_archive(dst_dir, tar_stream)


class EvalContainerManager:
    """Manages Docker containers for evaluation."""

    def __init__(self, logger: logging.Logger):
        """Initialize container manager."""
        self.logger = logger
        self.client = docker.from_env()

    def pull_image_if_needed(self, image_name: str) -> None:
        """Pull Docker image if not exists locally."""
        try:
            self.client.images.get(image_name)
            self.logger.info(f"Image {image_name} found locally")
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling image {image_name}")
            self.client.images.pull(image_name)

    def create_container(
        self,
        image_name: str,
        instance_id: str,
        n_attempt: int = 1,
        gpu_ids: str | None = None,
        proxy_port: int | None = None,
        docker_runtime_config: dict | None = None,
    ) -> Container:
        """Create and start a Docker container for evaluation."""
        docker_runtime_config = docker_runtime_config or {}

        need_gpu = bool(docker_runtime_config.get("need_gpu"))
        number_once = docker_runtime_config.get("number_once", 1)
        if not isinstance(number_once, int) or number_once <= 0:
            number_once = 1
        shm_size = docker_runtime_config.get("shm_size")

        sanitized_id = instance_id.replace("/", "-").replace(".", "-").replace("--", "-")
        container_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        container_name = f"fb-eval-{sanitized_id}-attempt-{n_attempt}-{container_timestamp}"
        self.logger.info(f"Creating container: {container_name}")

        try:
            existing = self.client.containers.get(container_name)
            self.logger.info(f"Container {container_name} already exists, stopping and removing it first.")
            try:
                existing.stop(timeout=20)
            except Exception as stop_exc:
                self.logger.warning(f"Failed to stop existing container {container_name}: {stop_exc}")
            try:
                existing.remove(force=True)
                self.logger.info(f"Removed existing container: {container_name}")
            except Exception as rm_exc:
                self.logger.warning(f"Failed to remove existing container {container_name}: {rm_exc}")
        except docker.errors.NotFound:
            pass

        device_requests = None
        environment: dict[str, Any] = {}
        if need_gpu:
            if gpu_ids is not None:
                device_requests = [
                    docker.types.DeviceRequest(device_ids=gpu_ids.split(','), capabilities=[["gpu"]])
                ]
                self.logger.info(f"GPU access requested for specific GPUs: {gpu_ids}")
                nvidia_visible_devices = gpu_ids
            else:
                device_requests = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
                self.logger.info("GPU access requested for all available GPUs")
                nvidia_visible_devices = "all"

            environment.update({
                "NVIDIA_VISIBLE_DEVICES": nvidia_visible_devices,
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            })
        else:
            self.logger.info("This task does not require GPU; starting container without GPU access")

        env_vars_from_config = docker_runtime_config.get("env_vars", {})
        if env_vars_from_config:
            environment.update(env_vars_from_config)
            self.logger.info(f"Added environment variables from config: {list(env_vars_from_config.keys())}")

        network_mode = "bridge"
        if proxy_port is not None:
            proxy_url = f"http://{DOCKER_HOST_GATEWAY}:{proxy_port}"
            environment.update({
                'HTTP_PROXY': proxy_url,
                'HTTPS_PROXY': proxy_url,
                'http_proxy': proxy_url,
                'https_proxy': proxy_url,
                'NO_PROXY': f'localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}',
                'no_proxy': f'localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}',
            })
            self.logger.info(f"Using bridge network with proxy via Docker host gateway: {proxy_url}")

        if shm_size:
            self.logger.info(f"Using shared memory size: {shm_size}")

        container = self.client.containers.run(
            image_name,
            command="/bin/bash -c 'sleep infinity'",
            name=container_name,
            detach=True,
            remove=False,
            user=DOCKER_USER,
            working_dir=DOCKER_WORKDIR,
            device_requests=device_requests,
            environment=environment,
            network_mode=network_mode,
            shm_size=shm_size,
        )

        if need_gpu:
            exit_code, output = container.exec_run("nvidia-smi --list-gpus")
            if exit_code == 0:
                gpu_list = output.decode().strip().split("\n")
                gpu_count = len(gpu_list)
                if gpu_count < number_once:
                    raise RuntimeError(f"Container can only access {gpu_count} GPU(s), but {number_once} are required")
                else:
                    self.logger.info(f"Container can access {gpu_count} GPU(s)")
            else:
                raise RuntimeError(f"This task needs GPU, but failed to query GPUs: {output.decode()}")

        self.logger.info(f"Container {container_name} created successfully")

        env_exports = docker_runtime_config.get("env_exports", [])
        if env_exports:
            self._apply_env_exports(container, env_exports)

        return container

    def _apply_env_exports(self, container: Container, env_exports: list[str]) -> None:
        if not env_exports:
            return

        try:
            heredoc_lines = [
                "",
                "# Custom environment variables from repo_settings",
            ]
            heredoc_lines.extend(env_exports)

            heredoc_content = "\n".join(heredoc_lines)

            full_cmd = f"cat >> ~/.bashrc << 'EOF'\n{heredoc_content}\nEOF\n"

            exit_code, output = exec_run_with_timeout(
                container,
                full_cmd,
                timeout=60
            )

            if exit_code == 0:
                self.logger.info(f"Applied {len(env_exports)} environment exports to .bashrc")
            else:
                self.logger.warning(f"Failed to apply env exports: {output.decode('utf-8', errors='replace')}")
        except Exception as e:
            self.logger.warning(f"Error applying env exports: {e}")

    def cleanup_container(self, container: Container) -> None:
        """Kill and remove a container."""
        try:
            self.logger.info("Killing and removing container")
            container.remove(force=True)
            self.logger.info("Container removed successfully")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup container: {e}")
