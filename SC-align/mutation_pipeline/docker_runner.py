"""Docker execution helpers for the mutation pipeline."""

import json
import logging
import os
import shlex
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docker.models.containers import Container

from featurebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR, UTF8
from featurebench.harness.container import (
    EvalContainerManager,
    copy_to_container,
    exec_run_with_timeout,
)
from featurebench.harness.utils import (
    build_test_command,
    get_test_config_from_repo_settings,
    parse_repo_settings,
)

from .models import PipelineInput

logger = logging.getLogger(__name__)


def _normalize_patch_content(patch_content: str) -> str:
    """Ensure patch ends with newline for git apply."""
    if patch_content and not patch_content.endswith("\n"):
        return patch_content + "\n"
    return patch_content


def _filter_diff_to_files(patch_text: str, allowed_files: set) -> str:
    """Keep only ``diff --git`` blocks whose target file is in the whitelist."""
    import re
    if not patch_text.strip():
        return patch_text
    parts = re.split(r"(?m)^(?=diff --git )", patch_text)
    kept = []
    for block in parts:
        if not block.strip():
            continue
        m = re.match(r"diff --git a/\S+ b/(\S+)", block)
        target = m.group(1) if m else None
        if target and target in allowed_files:
            kept.append(block)
    return "".join(kept)


def _fix_hunk_headers(patch_text: str) -> str:
    """Recompute @@ hunk header line counts to match actual content."""
    import re

    lines = patch_text.splitlines(keepends=True)
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        hunk_match = re.match(
            r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)\n?$", line
        )
        if not hunk_match:
            result.append(line)
            i += 1
            continue

        old_start = hunk_match.group(1)
        new_start = hunk_match.group(2)
        tail = hunk_match.group(3)

        i += 1
        hunk_lines: list[str] = []
        while i < len(lines):
            l = lines[i]
            if l.startswith("@@ ") or l.startswith("diff --git "):
                break
            hunk_lines.append(l)
            i += 1

        while hunk_lines and hunk_lines[-1].strip() == "":
            hunk_lines.pop()

        old_count = 0
        new_count = 0
        for hl in hunk_lines:
            if hl.startswith("-"):
                old_count += 1
            elif hl.startswith("+"):
                new_count += 1
            elif hl.startswith("\\ "):
                continue
            else:
                old_count += 1
                new_count += 1

        result.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{tail}\n")
        result.extend(hunk_lines)

    return "".join(result)


class MutationDockerRunner:
    """High-level Docker operations for the mutation pipeline."""

    def __init__(
        self,
        proxy_port: Optional[int] = None,
        gpu_ids: Optional[str] = None,
    ):
        self.proxy_port = proxy_port
        self.gpu_ids = gpu_ids
        self._container_manager = EvalContainerManager(logger)

    def setup_instance(self, pipeline_input: PipelineInput) -> Container:
        """Create and set up a Docker container for an instance."""
        image_name = pipeline_input.image_name
        instance_id = pipeline_input.instance_id

        t0 = time.time()
        self._container_manager.pull_image_if_needed(image_name)
        t_pull = time.time()
        logger.info(
            f"[{instance_id}] Docker: image pull check took {t_pull - t0:.1f}s"
        )

        container = self._container_manager.create_container(
            image_name,
            instance_id,
            1,
            self.gpu_ids,
            self.proxy_port,
            {},
        )
        t_create = time.time()
        logger.info(
            f"[{instance_id}] Docker: container create took {t_create - t_pull:.1f}s"
        )

        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(container, cmd, timeout=600)
        t_restore = time.time()
        logger.info(
            f"[{instance_id}] Docker: project restore (cp -r) took {t_restore - t_create:.1f}s"
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to restore project for {instance_id}: "
                f"{output.decode(UTF8, errors='replace')}"
            )

        logger.info(
            f"[{instance_id}] Docker: setup_instance total {t_restore - t0:.1f}s"
        )
        return container

    def apply_patch(self, container: Container, patch_content: str) -> bool:
        """Apply a unified diff patch inside the container."""
        if not patch_content or not patch_content.strip():
            return True

        patch_content = _fix_hunk_headers(_normalize_patch_content(patch_content))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", delete=False, encoding=UTF8
        ) as f:
            f.write(patch_content)
            temp_path = f.name

        try:
            container_path = "/tmp/mutation_patch.diff"
            copy_to_container(container, temp_path, container_path)

            cmd = f"cd /testbed && git apply --whitespace=fix {container_path}"
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=120)

            if exit_code == 0:
                return True

            logger.info("git apply failed, falling back to patch -p1")
            cmd = f"cd /testbed && patch -p1 --no-backup-if-mismatch < {container_path}"
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=120)

            if exit_code != 0:
                logger.warning(
                    f"Patch apply failed (both git apply and patch -p1): "
                    f"{output.decode(UTF8, errors='replace')}"
                )
                return False
            return True
        finally:
            os.unlink(temp_path)
            rm_code, rm_out = exec_run_with_timeout(
                container, f"rm -f {container_path}", timeout=30
            )
            if rm_code != 0:
                logger.warning(
                    f"Could not remove {container_path} after apply: "
                    f"{rm_out.decode(UTF8, errors='replace')}"
                )

    def apply_mutation(
        self,
        container: Container,
        mutations: List[dict],
    ) -> bool:
        """Apply mutation via find/replace pairs."""
        return self.apply_text_mutations(container, mutations)

    def apply_text_mutations(
        self,
        container: Container,
        mutations: List[dict],
    ) -> bool:
        """Apply find/replace mutations to source files in the container."""
        if not mutations:
            return True

        t0 = time.time()
        script = textwrap.dedent("""\
            import json, sys

            mutations = json.loads(sys.argv[1])
            for m in mutations:
                filepath = m["file"]
                if not filepath.startswith("/testbed/"):
                    filepath = "/testbed/" + filepath.lstrip("/")

                find_text = m["find"]
                replace_text = m["replace"]

                with open(filepath, "r") as f:
                    content = f.read()

                if find_text not in content:
                    print(f"ERROR: find text not found in {filepath}")
                    print(f"  find={repr(find_text[:120])}")
                    sys.exit(1)

                count = content.count(find_text)
                if count > 1:
                    print(f"WARN: find text appears {count} times in {filepath}, replacing first")

                content = content.replace(find_text, replace_text, 1)

                with open(filepath, "w") as f:
                    f.write(content)

                print(f"OK: mutated {filepath}")
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding=UTF8
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            container_script = "/tmp/apply_mutations.py"
            copy_to_container(container, script_path, container_script)

            mutations_json = json.dumps(mutations)
            cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"python {container_script} {shlex.quote(mutations_json)}"
            )
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)
            output_text = output.decode(UTF8, errors="replace")

            if exit_code != 0:
                logger.warning(f"Mutation apply failed: {output_text}")
                return False

            elapsed = time.time() - t0
            logger.info(f"Docker: apply_text_mutations took {elapsed:.1f}s ({len(mutations)} mutations)")
            return True
        finally:
            os.unlink(script_path)

    def get_mutation_diff(
        self,
        container: Container,
        allowed_files: Optional[set] = None,
    ) -> str:
        """Get a git diff of the current mutations in the container."""
        diff_cmd = "cd /testbed && git diff --no-color"
        exit_code, output = exec_run_with_timeout(container, diff_cmd, timeout=60)
        if exit_code != 0:
            raise RuntimeError(
                "Failed to capture mutation diff: "
                f"{output.decode(UTF8, errors='replace')}"
            )
        text = output.decode(UTF8, errors="replace")
        if allowed_files:
            text = _filter_diff_to_files(text, allowed_files)
        return text

    def restore_test_files(
        self,
        container: Container,
        pipeline_input: PipelineInput,
    ) -> None:
        """Ensure F2P test files are present in /testbed/."""
        t0 = time.time()
        for f2p_test in pipeline_input.fail_to_pass:
            test_path = f2p_test
            if not test_path.startswith("/testbed/"):
                test_path = f"/testbed/{test_path}"
            backup_path = test_path.replace("/testbed/", "/root/my_repo/", 1)

            cmd = (
                f"if [ ! -f {shlex.quote(test_path)} ]; then "
                f"  cp {shlex.quote(backup_path)} {shlex.quote(test_path)} && "
                f"  echo 'Restored {test_path}'; "
                f"else echo 'Already present: {test_path}'; fi"
            )
            exec_run_with_timeout(container, cmd, timeout=30)
        logger.info(
            f"Docker: restore_test_files took {time.time() - t0:.1f}s "
            f"({len(pipeline_input.fail_to_pass)} f2p tests)"
        )

    def run_existing_tests(
        self,
        container: Container,
        pipeline_input: PipelineInput,
        timeout: int,
    ) -> Tuple[bool, str]:
        """Run f2p and p2p tests against the container state."""
        repo_settings = pipeline_input.repo_settings
        test_config = get_test_config_from_repo_settings(repo_settings)
        base_test_cmd = test_config.get("test_cmd", "pytest -rA --color=no")
        timeout_one = test_config.get("timeout_one")
        effective_timeout = test_config.get("timeout_run", timeout) or timeout

        base_cmd = build_test_command(base_test_cmd, timeout_one)

        outputs = []
        all_passed = True

        all_test_paths = []
        for t in pipeline_input.fail_to_pass:
            path = t if t.startswith("/testbed/") else f"/testbed/{t}"
            all_test_paths.append(("F2P", t, path))
        for t in pipeline_input.pass_to_pass:
            path = t if t.startswith("/testbed/") else f"/testbed/{t}"
            all_test_paths.append(("P2P", t, path))

        tests_start = time.time()
        for label, orig, test_path in all_test_paths:
            full_cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"cd /testbed && {base_cmd} {test_path}"
            )
            t0 = time.time()
            exit_code, output = exec_run_with_timeout(
                container, full_cmd, timeout=effective_timeout
            )
            test_elapsed = time.time() - t0
            output_text = output.decode(UTF8, errors="replace")
            outputs.append(f"[{label}: {orig}] exit={exit_code}\n{output_text}")
            logger.info(
                f"Docker: test [{label}: {orig}] took {test_elapsed:.1f}s, exit={exit_code}"
            )
            if exit_code != 0:
                all_passed = False

        total_tests = time.time() - tests_start
        logger.info(
            f"Docker: run_existing_tests total {total_tests:.1f}s "
            f"({len(all_test_paths)} tests), all_passed={all_passed}"
        )
        return all_passed, "\n".join(outputs)

    def run_hypothesis_test(
        self,
        container: Container,
        hypothesis_code: str,
        timeout: int,
        disable_capture: bool = False,
        env: Optional[Dict[str, str]] = None,
        run_all_tests: bool = False,
    ) -> Tuple[bool, str]:
        """Run a pytest file inside the container."""
        t0 = time.time()
        install_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "pip install hypothesis -q"
        )
        exec_run_with_timeout(container, install_cmd, timeout=120)
        t_install = time.time()
        logger.info(f"Docker: pip install hypothesis took {t_install - t0:.1f}s")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding=UTF8
        ) as f:
            f.write(hypothesis_code)
            temp_path = f.name

        try:
            container_test_path = "/tmp/test_hypothesis_proof.py"
            copy_to_container(container, temp_path, container_test_path)

            capture_flag = "-s " if disable_capture else ""
            env_prefix = "".join(
                f"export {key}={shlex.quote(value)} && "
                for key, value in (env or {}).items()
            )
            cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"{env_prefix}"
                f"cd /testbed && python -m pytest {container_test_path} "
                f"{capture_flag}{'' if run_all_tests else '-x '}-v "
                "--tb=long 2>&1"
            )
            t_run = time.time()
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=timeout)
            t_done = time.time()
            output_text = output.decode(UTF8, errors="replace")
            triggered = exit_code != 0
            logger.info(
                f"Docker: run_hypothesis_test total {t_done - t0:.1f}s "
                f"(install={t_install - t0:.1f}s, run={t_done - t_run:.1f}s, "
                f"triggered={triggered}, disable_capture={disable_capture})"
            )
            return triggered, output_text
        finally:
            os.unlink(temp_path)

    def run_witness_script(
        self,
        container: Container,
        witness_code: str,
        timeout: int,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Run a witness script inside the container as a plain program."""
        t0 = time.time()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding=UTF8
        ) as f:
            f.write(witness_code)
            temp_path = f.name

        try:
            container_script_path = "/tmp/witness_script.py"
            copy_to_container(container, temp_path, container_script_path)

            env_prefix = "".join(
                f"export {key}={shlex.quote(value)} && "
                for key, value in (env or {}).items()
            )
            cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"{env_prefix}"
                f"cd /testbed && python3 -u {container_script_path} 2>&1"
            )
            t_run = time.time()
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=timeout)
            t_done = time.time()
            output_text = output.decode(UTF8, errors="replace")
            failed = exit_code != 0
            logger.info(
                f"Docker: run_witness_script total {t_done - t0:.1f}s "
                f"(run={t_done - t_run:.1f}s, failed={failed})"
            )
            return failed, output_text
        finally:
            os.unlink(temp_path)

    _HYPOFUZZ_DRIVER_LOCAL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eval_pipeline", "hypofuzz_driver.py",
    )
    _HYPOFUZZ_DRIVER_CONTAINER = "/tmp/hypofuzz_driver.py"
    _HYPOFUZZ_RESULT_MARKER = "__HYPOFUZZ_DRIVER_RESULT__"
    _SAVED_INPUTS_DRIVER_LOCAL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eval_pipeline", "saved_inputs_driver.py",
    )
    _SAVED_INPUTS_DRIVER_CONTAINER = "/tmp/saved_inputs_driver.py"
    _SAVED_INPUTS_RESULT_MARKER = "__SAVED_INPUTS_DRIVER_RESULT__"
    _FACTORY_ORACLE_DRIVER_LOCAL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eval_pipeline", "factory_oracle_driver.py",
    )
    _FACTORY_ORACLE_DRIVER_CONTAINER = "/tmp/factory_oracle_driver.py"
    _FACTORY_ORACLE_RESULT_MARKER = "__FACTORY_ORACLE_DRIVER_RESULT__"

    def run_hypofuzz_campaign(
        self,
        container: Container,
        cert_code: str,
        timeout_seconds: int,
        run_all_tests: bool = False,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Run a coverage-guided hypofuzz campaign + pytest replay + coverage."""
        t0 = time.time()
        check_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "pip show hypofuzz >/dev/null 2>&1 || "
            "pip install hypofuzz hypothesis -q; "
            "pip show coverage >/dev/null 2>&1 || "
            "pip install coverage -q"
        )
        exec_run_with_timeout(container, check_cmd, timeout=300)
        t_install = time.time()
        logger.info(
            f"Docker: hypofuzz install/check took {t_install - t0:.1f}s"
        )

        if not os.path.isfile(self._HYPOFUZZ_DRIVER_LOCAL):
            raise FileNotFoundError(
                "hypofuzz_driver.py not found at "
                f"{self._HYPOFUZZ_DRIVER_LOCAL}"
            )
        copy_to_container(
            container, self._HYPOFUZZ_DRIVER_LOCAL,
            self._HYPOFUZZ_DRIVER_CONTAINER,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding=UTF8
        ) as f:
            f.write(cert_code)
            temp_path = f.name

        try:
            container_test_path = "/tmp/cert.py"
            copy_to_container(container, temp_path, container_test_path)

            fuzz_cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"cd /testbed && python3 -u "
                f"{self._HYPOFUZZ_DRIVER_CONTAINER} "
                f"{int(timeout_seconds)} {container_test_path} 2>&1"
            )
            t_fuzz = time.time()
            fuzz_exit, fuzz_output = exec_run_with_timeout(
                container, fuzz_cmd, timeout=timeout_seconds + 60
            )
            t_fuzz_done = time.time()
            fuzz_text = fuzz_output.decode(UTF8, errors="replace")
            driver_summary = self._parse_driver_result(fuzz_text)

            replay_cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"cd /testbed && python -m pytest {container_test_path} "
                f"{'' if run_all_tests else '-x '}-v --tb=long 2>&1"
            )
            t_replay = time.time()
            replay_timeout = max(60, timeout_seconds // 2)
            replay_exit, replay_output = exec_run_with_timeout(
                container, replay_cmd, timeout=replay_timeout
            )
            t_done_replay = time.time()
            replay_text = replay_output.decode(UTF8, errors="replace")

            triggered = replay_exit == 1

            coverage = self._extract_coverage(container)
            t_done = time.time()

            driver_brief = self._format_driver_brief(driver_summary, fuzz_text)
            output_text = (
                f"=== hypofuzz stage 1 (driver, "
                f"{t_fuzz_done - t_fuzz:.1f}s, exit={fuzz_exit}) ===\n"
                f"{driver_brief}\n"
                f"=== hypofuzz stage 2 (pytest replay, "
                f"{t_done_replay - t_replay:.1f}s, exit={replay_exit}) ===\n"
                f"{replay_text}"
            )

            cov_pct = (
                coverage.get("edge_coverage_pct")
                if isinstance(coverage, dict) else None
            )
            n_examples = (
                driver_summary.get("examples_count")
                if driver_summary else None
            )
            logger.info(
                f"Docker: run_hypofuzz_campaign total {t_done - t0:.1f}s "
                f"(install={t_install - t0:.1f}s, "
                f"fuzz={t_fuzz_done - t_fuzz:.1f}s/exit={fuzz_exit}, "
                f"examples={n_examples}, "
                f"replay={t_done_replay - t_replay:.1f}s/exit={replay_exit}, "
                f"cov={cov_pct}, triggered={triggered})"
            )
            return triggered, output_text, replay_exit, coverage
        finally:
            os.unlink(temp_path)

    def _parse_driver_result(self, fuzz_text: str) -> Optional[Dict[str, Any]]:
        """Pull the RESULT_MARKER JSON line out of the driver's stdout."""
        for line in fuzz_text.splitlines():
            if line.startswith(self._HYPOFUZZ_RESULT_MARKER):
                payload = line[len(self._HYPOFUZZ_RESULT_MARKER):]
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Docker: driver result marker present but JSON "
                        "parse failed: %s; payload=%r", e, payload[:200]
                    )
                    return None
        return None

    def run_saved_inputs_replay(
        self,
        container: Container,
        cert_code: str,
        saved_inputs: Any,
        timeout_seconds: int,
        saved_inputs_file: Optional[Path] = None,
        run_all_tests: bool = False,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Replay an agent-provided input corpus under pytest + coverage."""
        t0 = time.time()
        check_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "pip show coverage >/dev/null 2>&1 || pip install coverage -q"
        )
        exec_run_with_timeout(container, check_cmd, timeout=300)
        t_install = time.time()
        logger.info(
            f"Docker: saved-input replay install/check took {t_install - t0:.1f}s"
        )

        if not os.path.isfile(self._SAVED_INPUTS_DRIVER_LOCAL):
            raise FileNotFoundError(
                "saved_inputs_driver.py not found at "
                f"{self._SAVED_INPUTS_DRIVER_LOCAL}"
            )
        copy_to_container(
            container, self._SAVED_INPUTS_DRIVER_LOCAL,
            self._SAVED_INPUTS_DRIVER_CONTAINER,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding=UTF8
        ) as cert_file:
            cert_file.write(cert_code)
            cert_temp_path = cert_file.name
        if saved_inputs_file is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding=UTF8
            ) as inputs_file:
                json.dump(saved_inputs if saved_inputs is not None else [], inputs_file)
                inputs_temp_path = inputs_file.name
        else:
            inputs_temp_path = str(saved_inputs_file)

        try:
            container_cert_path = "/tmp/saved_inputs_cert.py"
            container_inputs_path = (
                "/tmp/agent_saved_inputs.jsonl"
                if saved_inputs_file is not None
                else "/tmp/agent_saved_inputs.json"
            )
            copy_to_container(container, cert_temp_path, container_cert_path)
            copy_to_container(container, inputs_temp_path, container_inputs_path)

            replay_cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                f"cd /testbed && "
                f"SAVED_INPUTS_RUN_ALL_TESTS={'1' if run_all_tests else '0'} "
                f"python3 -u "
                f"{self._SAVED_INPUTS_DRIVER_CONTAINER} "
                f"{container_cert_path} {container_inputs_path} /testbed 2>&1"
            )
            t_replay = time.time()
            replay_exit, replay_output = exec_run_with_timeout(
                container, replay_cmd, timeout=timeout_seconds
            )
            t_done_replay = time.time()
            replay_text = replay_output.decode(UTF8, errors="replace")
            driver_summary = self._parse_saved_inputs_result(replay_text)

            coverage = self._extract_coverage(container)
            t_done = time.time()

            driver_brief = self._format_saved_inputs_brief(
                driver_summary, replay_text
            )
            output_text = (
                f"=== saved-input replay "
                f"({t_done_replay - t_replay:.1f}s, exit={replay_exit}) ===\n"
                f"{driver_brief}\n"
                f"{replay_text}"
            )

            triggered = replay_exit != 0
            cov_pct = (
                coverage.get("edge_coverage_pct")
                if isinstance(coverage, dict) else None
            )
            n_inputs = (
                driver_summary.get("inputs_count")
                if driver_summary else None
            )
            logger.info(
                f"Docker: run_saved_inputs_replay total {t_done - t0:.1f}s "
                f"(install={t_install - t0:.1f}s, "
                f"replay={t_done_replay - t_replay:.1f}s/exit={replay_exit}, "
                f"inputs={n_inputs}, cov={cov_pct}, triggered={triggered})"
            )
            return triggered, output_text, replay_exit, coverage
        finally:
            os.unlink(cert_temp_path)
            if saved_inputs_file is None:
                os.unlink(inputs_temp_path)

    def _parse_saved_inputs_result(
        self, replay_text: str
    ) -> Optional[Dict[str, Any]]:
        """Pull the saved-input driver summary marker out of stdout."""
        for line in replay_text.splitlines():
            if line.startswith(self._SAVED_INPUTS_RESULT_MARKER):
                payload = line[len(self._SAVED_INPUTS_RESULT_MARKER):]
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Docker: saved-input result marker present but JSON "
                        "parse failed: %s; payload=%r", e, payload[:200]
                    )
                    return None
        return None

    def run_factory_oracle(
        self,
        container: Container,
        cert_code: str,
        timeout_seconds: int,
        run_all_cases: bool = False,
    ) -> Tuple[bool, str, int, Optional[Dict[str, Any]]]:
        """Run a frozen certificate oracle over its own generated cases."""
        check_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "pip show coverage >/dev/null 2>&1 || pip install coverage -q"
        )
        exec_run_with_timeout(container, check_cmd, timeout=300)
        if not os.path.isfile(self._FACTORY_ORACLE_DRIVER_LOCAL):
            raise FileNotFoundError(
                f"factory_oracle_driver.py not found at "
                f"{self._FACTORY_ORACLE_DRIVER_LOCAL}"
            )
        copy_to_container(
            container,
            self._FACTORY_ORACLE_DRIVER_LOCAL,
            self._FACTORY_ORACLE_DRIVER_CONTAINER,
        )

        cert_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding=UTF8
            ) as cert_file:
                cert_file.write(cert_code)
                cert_path = cert_file.name
            copy_to_container(container, cert_path, "/tmp/factory_oracle_cert.py")

            case_budget = max(1, int(timeout_seconds * 0.8))
            cmd = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && cd /testbed && "
                f"FACTORY_ORACLE_BUDGET_SECONDS={case_budget} "
                f"FACTORY_ORACLE_RUN_ALL_CASES={'1' if run_all_cases else '0'} "
                f"python3 -u {self._FACTORY_ORACLE_DRIVER_CONTAINER} "
                "/tmp/factory_oracle_cert.py /testbed 2>&1"
            )
            exit_code, raw = exec_run_with_timeout(
                container, cmd, timeout=timeout_seconds
            )
            text = raw.decode(UTF8, errors="replace")
            coverage = self._extract_coverage(container)
            return exit_code != 0, text, exit_code, coverage
        finally:
            if cert_path:
                os.unlink(cert_path)

    def _format_saved_inputs_brief(
        self,
        summary: Optional[Dict[str, Any]],
        replay_text: str,
    ) -> str:
        """Compact one-line summary for saved-input replay output."""
        if summary is None:
            tail = replay_text.strip().splitlines()
            return "(saved-input result marker missing) tail: " + " | ".join(
                tail[-3:] if tail else ["(no output)"]
            )
        parts = [
            f"inputs={summary.get('inputs_count')}",
            f"pytest_exit={summary.get('pytest_exit_code')}",
            f"elapsed={summary.get('elapsed')}s",
        ]
        if summary.get("error"):
            parts.append(f"error={summary['error']}")
        return ", ".join(parts)

    def _format_driver_brief(
        self,
        summary: Optional[Dict[str, Any]],
        fuzz_text: str,
    ) -> str:
        """Compact one-line driver summary for the stitched output_text."""
        if summary is None:
            tail = fuzz_text.strip().splitlines()
            return "(driver result marker missing) tail: " + " | ".join(
                tail[-3:] if tail else ["(no output)"]
            )
        parts = [
            f"n_targets={summary.get('n_targets')}",
            f"examples={summary.get('examples_count')}",
            f"timed_out={summary.get('timed_out')}",
            f"fuzz_elapsed={summary.get('fuzz_elapsed')}s",
        ]
        if summary.get("error"):
            parts.append(f"error={summary['error']}")
        return ", ".join(parts)

    def _extract_coverage(self, container: Container) -> Optional[Dict[str, Any]]:
        """Convert ``/testbed/.coverage`` into a compact JSON summary."""
        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "cd /testbed && "
            "(test -f .coverage || echo '__NO_COVERAGE_DATA__'; "
            "python3 - <<'PY'\n"
            "import json\n"
            "import coverage\n"
            "\n"
            "cov = coverage.Coverage("
            "data_file='/testbed/.coverage', config_file=False)\n"
            "cov.load()\n"
            "cov.json_report(outfile='/tmp/cov_fuzz.json', ignore_errors=True)\n"
            "with open('/tmp/cov_fuzz.json', 'r', encoding='utf-8') as f:\n"
            "    data = json.load(f)\n"
            "files = data.get('files') or {}\n"
            "def _maybe_call(value):\n"
            "    return value() if callable(value) else value\n"
            "for filename in list(cov.get_data().measured_files()):\n"
            "    block = files.get(filename)\n"
            "    if block is None:\n"
            "        continue\n"
            "    analysis = cov._analyze(filename)\n"
            "    block['possible_arcs'] = [list(a) for a in "
            "_maybe_call(analysis.arc_possibilities)]\n"
            "    block['executed_arcs'] = [list(a) for a in "
            "_maybe_call(analysis.arcs_executed)]\n"
            "    block['missing_arcs'] = [list(a) for a in "
            "_maybe_call(analysis.arcs_missing)]\n"
            "with open('/tmp/cov_fuzz.json', 'w', encoding='utf-8') as f:\n"
            "    json.dump(data, f)\n"
            "PY\n"
            ") 2>&1 || true; "
            "cat /tmp/cov_fuzz.json 2>&1 || echo '__NO_COVERAGE_JSON__'"
        )
        exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)
        text = output.decode(UTF8, errors="replace")
        start = text.rfind("\n{")
        if start == -1 and not text.lstrip().startswith("{"):
            logger.warning(
                f"Docker: coverage harvest produced no JSON "
                f"(exit={exit_code}, head={text[:200]!r})"
            )
            return None
        json_blob = text[start + 1:] if start != -1 else text.lstrip()
        try:
            data = json.loads(json_blob)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Docker: coverage JSON parse failed ({e}); "
                f"head={text[:200]!r}"
            )
            return None

        files_summary: Dict[str, Any] = {}
        for path, fdata in (data.get("files") or {}).items():
            files_summary[path] = {
                "executed_lines": fdata.get("executed_lines", []),
                "missing_lines": fdata.get("missing_lines", []),
                "executed_branches": fdata.get("executed_branches", []),
                "missing_branches": fdata.get("missing_branches", []),
                "possible_arcs": fdata.get("possible_arcs", []),
                "executed_arcs": fdata.get("executed_arcs", []),
                "missing_arcs": fdata.get("missing_arcs", []),
                "functions": fdata.get("functions", {}),
                "summary": {
                    "percent_covered": fdata.get("summary", {}).get(
                        "percent_covered"
                    ),
                    "num_statements": fdata.get("summary", {}).get(
                        "num_statements"
                    ),
                    "covered_lines": fdata.get("summary", {}).get(
                        "covered_lines"
                    ),
                    "num_branches": fdata.get("summary", {}).get(
                        "num_branches"
                    ),
                    "covered_branches": fdata.get("summary", {}).get(
                        "covered_branches"
                    ),
                },
            }

        totals = data.get("totals", {})
        num_branches = totals.get("num_branches") or 0
        covered_branches = totals.get("covered_branches") or 0
        total_edges = 0
        covered_edges = 0
        for fdata in files_summary.values():
            possible_edges = {
                tuple(edge) for edge in fdata.get("possible_arcs", [])
                if isinstance(edge, list) and len(edge) >= 2
            }
            executed_edges = {
                tuple(edge) for edge in fdata.get("executed_arcs", [])
                if isinstance(edge, list) and len(edge) >= 2
            }
            total_edges += len(possible_edges)
            covered_edges += len(possible_edges & executed_edges)

        return {
            "combined_coverage_pct": totals.get("percent_covered"),
            "edge_coverage_pct": (
                (covered_edges / total_edges * 100)
                if total_edges else None
            ),
            "branch_coverage_pct": (
                (covered_branches / num_branches * 100)
                if num_branches else None
            ),
            "num_files": len(files_summary),
            "num_edges": total_edges,
            "covered_edges": covered_edges,
            "num_statements": totals.get("num_statements"),
            "covered_lines": totals.get("covered_lines"),
            "num_branches": num_branches,
            "covered_branches": covered_branches,
            "files": files_summary,
        }

    def reset_testbed(self, container: Container, pipeline_input: PipelineInput) -> None:
        """Reset /testbed/ to gold state by re-copying from /root/my_repo/."""
        t0 = time.time()
        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(container, cmd, timeout=600)
        elapsed = time.time() - t0
        logger.info(
            f"[{pipeline_input.instance_id}] Docker: reset_testbed took {elapsed:.1f}s"
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to reset testbed for {pipeline_input.instance_id}: "
                f"{output.decode(UTF8, errors='replace')}"
            )

    def cleanup(self, container: Container) -> None:
        """Remove and clean up a container."""
        t0 = time.time()
        self._container_manager.cleanup_container(container)
        logger.info(f"Docker: container cleanup took {time.time() - t0:.1f}s")

    def get_container(self, container_id: str) -> Optional[Container]:
        """A container by id, started if it had stopped; None if it is gone."""
        import docker

        try:
            container = self._container_manager.client.containers.get(container_id)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logger.warning(f"Docker: could not look up container {container_id}: {e}")
            return None
        container.reload()
        if container.status != "running":
            logger.info(
                f"Docker: kept container {container.short_id} is {container.status}; starting"
            )
            container.start()
            container.reload()
        return container
