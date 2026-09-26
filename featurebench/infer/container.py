"""Container management for inference."""

import io
import logging
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
from docker.models.containers import Container

import selectors
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')

DOCKER_HOST_GATEWAY = "172.17.0.1"


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ANSI_ESCAPE_PATTERN.sub('', text)


class ContainerManager:
    """Manages Docker containers for inference."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        env_vars: Optional[Dict[str, str]] = None
    ):
        """Initialize the container manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.env_vars = env_vars or {}

        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError(
                f"Failed to connect to Docker: {e}. "
                "Please ensure Docker is installed and running."
            )

    def pull_image(self, image_name: str) -> bool:
        """Pull a Docker image if it doesn't exist locally."""
        try:
            try:
                self.client.images.get(image_name)
                self.logger.info(f"Image {image_name} already exists locally")
                return True
            except docker.errors.ImageNotFound:
                pass

            short_name = image_name.split("/")[-1]
            try:
                short_img = self.client.images.get(short_name)
                self.logger.info(
                    f"Image {image_name} not found, but found local image {short_name}. "
                    f"Tagging it as {image_name}"
                )

                short_img.tag(image_name)
                return True

            except docker.errors.ImageNotFound:
                pass

            self.logger.info(f"Pulling image {image_name}...")
            self.client.images.pull(image_name)
            self.logger.info(f"Successfully pulled image {image_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to pull image {image_name}: {e}")
            raise

    def create_container(
        self,
        image_name: str,
        container_name: Optional[str] = None,
        working_dir: str = "/testbed",
        extra_env: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict]] = None,
        use_host_network: bool = False,
        proxy_port: Optional[int] = None,
        gpu_ids: Optional[str] = None,
        docker_runtime_config: Optional[Dict[str, Any]] = None
    ) -> Container:
        """Create and start a Docker container."""
        docker_runtime_config = docker_runtime_config or {}

        env = dict(self.env_vars)
        if extra_env:
            env.update(extra_env)

        env_vars_from_config = docker_runtime_config.get("env_vars", {})
        if env_vars_from_config:
            env.update(env_vars_from_config)
            self.logger.info(f"Added environment variables from config: {list(env_vars_from_config.keys())}")

        processed_env = {}
        for k, v in env.items():
            if v:
                v_str = str(v)
                if 'localhost' in v_str or '127.0.0.1' in v_str:
                    original = v_str
                    v_str = v_str.replace('localhost', DOCKER_HOST_GATEWAY)
                    v_str = v_str.replace('127.0.0.1', DOCKER_HOST_GATEWAY)
                    if original != v_str:
                        self.logger.info(f"Replaced host reference in {k}: {original} -> {v_str}")
                processed_env[k] = v_str

        env_list = [f"{k}={v}" for k, v in processed_env.items()]

        if proxy_port is not None:
            self.logger.info(f"Using proxy port {proxy_port} via Docker host gateway ({DOCKER_HOST_GATEWAY})")
            use_host_network = False
            env_list.append(f"http_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"https_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"HTTP_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"HTTPS_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"no_proxy=localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}")
            env_list.append(f"NO_PROXY=localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}")
            env_list.append(f"FB_UPSTREAM_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"fb_upstream_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")

        need_gpu = bool(docker_runtime_config.get("need_gpu"))
        number_once = docker_runtime_config.get("number_once", 1)
        if not isinstance(number_once, int) or number_once <= 0:
            number_once = 1
        shm_size = docker_runtime_config.get("shm_size")

        device_requests = None
        if need_gpu:
            if gpu_ids is not None:
                device_requests = [
                    docker.types.DeviceRequest(device_ids=gpu_ids.split(','), capabilities=[["gpu"]])
                ]
                self.logger.info(f"GPU access requested for specific GPUs: {gpu_ids}")
                env_list.append(f"NVIDIA_VISIBLE_DEVICES={gpu_ids}")
            else:
                device_requests = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
                self.logger.info("GPU access requested for all available GPUs")
                env_list.append("NVIDIA_VISIBLE_DEVICES=all")

            env_list.append("NVIDIA_DRIVER_CAPABILITIES=compute,utility")
        else:
            self.logger.info("This task does not require GPU; starting container without GPU access")

        if shm_size:
            self.logger.info(f"Using shared memory size: {shm_size}")

        if container_name:
            try:
                existing_container = self.client.containers.get(container_name)
                self.logger.info(f"Found existing container with name '{container_name}', removing it...")
                existing_container.stop(timeout=20)
                existing_container.remove(force=True)
                self.logger.info(f"Removed existing container '{container_name}'")
            except docker.errors.NotFound:
                pass
            except Exception as e:
                self.logger.warning(f"Error removing existing container '{container_name}': {e}")
                try:
                    existing_container.remove(force=True)
                except:
                    pass

        try:
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir=working_dir,
                environment=env_list,
                volumes=volumes,
                command="tail -f /dev/null",
                platform="linux/amd64",
                network_mode="host" if use_host_network else "bridge",
                device_requests=device_requests,
                shm_size=shm_size
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

            self.logger.info(f"Created container {container.short_id} (network: {'host' if use_host_network else 'bridge'})")

            env_exports = docker_runtime_config.get("env_exports", [])
            if env_exports:
                self._apply_env_exports(container, env_exports)

            return container

        except Exception as e:
            self.logger.error(f"Failed to create container: {e}")
            raise

    def _apply_env_exports(self, container: Container, env_exports: List[str]) -> None:
        """Apply environment exports to container's .bashrc."""
        if not env_exports:
            return

        try:
            export_cmds = [
                'echo "" >> ~/.bashrc',
                'echo "# Custom environment variables from repo_settings" >> ~/.bashrc'
            ]
            for export_stmt in env_exports:
                escaped_stmt = export_stmt.replace("'", "'\\''")
                export_cmds.append(f"echo '{escaped_stmt}' >> ~/.bashrc")
            export_cmds.append('source ~/.bashrc')

            full_cmd = " && ".join(export_cmds)
            exit_code, output = self.exec_command(container, full_cmd)

            if exit_code == 0:
                self.logger.info(f"Applied {len(env_exports)} environment exports to .bashrc")
            else:
                self.logger.warning(f"Failed to apply env exports: {output}")
        except Exception as e:
            self.logger.warning(f"Error applying env exports: {e}")

    def exec_command(
        self,
        container: Container,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[int, str]:
        """Execute a command inside a container."""
        full_command = f"source ~/.bashrc && conda activate testbed 2>/dev/null || true && {command}"

        if workdir:
            full_command = f"cd {workdir} && {full_command}"

        try:
            exec_result = container.exec_run(
                cmd=["bash", "-c", full_command],
                workdir=workdir,
                demux=True
            )

            exit_code = exec_result.exit_code
            stdout = exec_result.output[0] if exec_result.output[0] else b""
            stderr = exec_result.output[1] if exec_result.output[1] else b""

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace")

            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(output)
                    f.write(f"\n[Exit code: {exit_code}]\n\n")

            return exit_code, output

        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            self.logger.error(error_msg)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(f"ERROR: {error_msg}\n\n")
            return -1, error_msg

    def exec_command_stream(
        self,
        container: Container,
        command: str,
        log_file: Path,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        skip_bashrc: bool = False
    ) -> int:
        """Execute a command with streaming output to log file (real-time)."""
        if skip_bashrc:
            full_command = command
        else:
            full_command = f"source ~/.bashrc && conda activate testbed 2>/dev/null || true && {command}"

        if workdir:
            full_command = f"cd {workdir} && {full_command}"

        docker_cmd = [
            "docker", "exec",
            container.id,
            "bash", "-c", full_command
        ]

        process = None
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"$ {command}\n")
                f.flush()

                process = subprocess.Popen(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                start_time = time.time()

                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)

                def _timed_out() -> bool:
                    return bool(timeout) and (time.time() - start_time) > timeout

                try:
                    while True:
                        if _timed_out():
                            process.kill()
                            try:
                                process.wait(timeout=5)
                            except Exception:
                                pass
                            f.write(f"\n[TIMEOUT after {timeout} seconds]\n\n")
                            f.flush()
                            return -1

                        if process.poll() is not None:
                            while True:
                                events = selector.select(timeout=0)
                                if not events:
                                    break
                                chunk = process.stdout.read()
                                if not chunk:
                                    break
                                f.write(strip_ansi_codes(chunk))
                                f.flush()
                            break

                        events = selector.select(timeout=0.1)
                        if not events:
                            continue

                        line = process.stdout.readline()
                        if line:
                            f.write(strip_ansi_codes(line))
                            f.flush()

                finally:
                    try:
                        selector.unregister(process.stdout)
                    except Exception:
                        pass
                    try:
                        selector.close()
                    except Exception:
                        pass

                f.write(f"\n[Exit code: {process.returncode}]\n\n")
                f.flush()
                return process.returncode

        except Exception as e:
            self.logger.error(f"Stream execution failed: {e}")
            import traceback
            if process and process.poll() is None:
                process.kill()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            return -1

    def copy_to_container(
        self,
        container: Container,
        src_path: Path,
        dest_path: str
    ) -> None:
        """Copy a file or directory to a container."""
        dest_dir = str(Path(dest_path).parent)
        container.exec_run(["mkdir", "-p", dest_dir])

        tar_stream = io.BytesIO()
        dest_name = Path(dest_path).name
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if src_path.is_file():
                tar.add(src_path, arcname=dest_name)
            else:
                for item in src_path.rglob("*"):
                    tar.add(item, arcname=str(Path(dest_name) / item.relative_to(src_path)))

        tar_stream.seek(0)
        container.put_archive(dest_dir, tar_stream.read())

    def copy_from_container(
        self,
        container: Container,
        src_path: str,
        dest_path: Path
    ) -> bool:
        """Copy a file or directory from a container."""
        try:
            bits, stat = container.get_archive(src_path)

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)

            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                members = tar.getmembers()

                if not members:
                    self.logger.warning(f"No files found in archive from {src_path}")
                    return False

                root_member = members[0]
                is_directory = root_member.isdir() or len(members) > 1

                if is_directory:
                    extracted_count = 0
                    for member in members:
                        if member == root_member and member.isdir():
                            continue

                        if member.name.startswith(root_member.name + '/'):
                            relative_path = member.name[len(root_member.name) + 1:]
                        elif member.name == root_member.name:
                            continue
                        else:
                            relative_path = member.name

                        if not relative_path:
                            continue

                        target_path = dest_path / relative_path

                        if member.isdir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                with open(target_path, "wb") as f:
                                    f.write(file_obj.read())
                                extracted_count += 1

                    if extracted_count > 0:
                        self.logger.info(f"Extracted {extracted_count} files from {src_path} to {dest_path}")
                        return True
                    else:
                        self.logger.warning(f"No files extracted from {src_path}")
                        return False
                else:
                    if root_member.isfile():
                        file_obj = tar.extractfile(root_member)
                        if file_obj:
                            with open(dest_path, "wb") as f:
                                f.write(file_obj.read())
                            self.logger.info(f"Copied file {src_path} to {dest_path}")
                            return True

                    return False

        except Exception as e:
            self.logger.warning(f"Failed to copy {src_path} from container: {e}")
            return False

    def stop_container(self, container: Container, force: bool = False) -> None:
        """Stop and remove a container."""
        try:
            if force:
                container.kill()
            else:
                container.stop(timeout=10)
            container.remove(force=True)
            self.logger.info(f"Stopped and removed container {container.short_id}")
        except Exception as e:
            self.logger.warning(f"Error stopping container: {e}")
            try:
                container.remove(force=True)
            except:
                pass

    def get_container_logs(self, container: Container) -> str:
        """Get container logs."""
        try:
            return container.logs().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Failed to get logs: {e}"
