"""
Docker-based sandbox executor.

Phase 1 implementation:
- Ephemeral containers per test execution
- Supports installing packages (when combined with richer agent loop in future phases)
- Basic resource limits via Docker
- Automatic cleanup with --rm

This is a drop-in replacement for SandboxedExecutor in terms of interface.
"""

import asyncio
import tempfile
import time
import tarfile
import io
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

try:
    import docker
    from docker.errors import DockerException, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore

from agent.models import CodeArtifact, TestResults
from agent.safety.checker import SafetyChecker
from agent.executor.sandbox import ExecutionConfig, _is_test_path
from agent.pytest_report import build_test_results


class DockerExecutor:
    """
    Docker-based code executor.

    Supports two modes:
    - Ephemeral (Phase 1): Fresh container per execution (original behavior).
    - Persistent (Phase 2): One long-lived container per task. Supports
      installing packages via `install_packages()` and re-using the
      environment for subsequent executions.
    """

    def __init__(
        self,
        config: ExecutionConfig = None,
        safety_checker: SafetyChecker = None,
        docker_image: str = "python:3.12-slim",
        docker_client: Optional["docker.DockerClient"] = None,
        # Phase 2 persistent settings
        persistent: bool = False,
        enable_network: bool = True,
        install_build_tools: bool = True,
    ):
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "Docker executor requires the 'docker' package. "
                "Install it with: pip install docker"
            )

        self.config = config or ExecutionConfig()
        self.safety_checker = safety_checker or SafetyChecker()
        self.docker_image = docker_image
        self.persistent = persistent
        self.enable_network = enable_network
        self.install_build_tools = install_build_tools

        self._persistent_container = None  # type: ignore
        self._workspace_id: Optional[str] = None
        self._workspace_path: Optional[str] = None
        self._logger = logging.getLogger(__name__)

        try:
            self.client = docker_client or docker.from_env()
            self.client.ping()
        except DockerException as e:
            raise RuntimeError(
                f"Failed to connect to Docker daemon. Is Docker running?\n"
                f"Original error: {e}"
            ) from e

    # ==================================================================
    # Phase 2: Persistent Container Support (Restricted Scope)
    # ==================================================================

    def start_persistent(
        self, workspace_id: Optional[str] = None, publish_port: Optional[int] = None
    ) -> None:
        """
        Start a long-lived container for the entire task with a dedicated workspace.

        The workspace will be created at /workspace/<workspace_id>/ inside the container.
        Safe to call multiple times (idempotent / re-entrant).

        If `publish_port` is given, that container port is published to the same
        host port so a server started inside the container is reachable from the
        host. Port mapping must be set at creation time, so this is a no-op on a
        re-entrant call to an already-running container.
        """
        if workspace_id is None:
            workspace_id = "default"

        # Always store the workspace info (even on re-entrant calls)
        self._workspace_id = workspace_id
        self._workspace_path = f"/workspace/{workspace_id}"

        if self._persistent_container is not None:
            # Container already running — just ensure workspace dir exists
            self._persistent_container.exec_run(["mkdir", "-p", self._workspace_path])
            return

        run_kwargs = {
            "image": self.docker_image,
            "command": ["sleep", "3600"],
            "detach": True,
            "mem_limit": f"{self.config.memory_limit_mb}m",
            "nano_cpus": int(1_000_000_000 * 1.0),
            "network_disabled": not self.enable_network,
            "working_dir": self._workspace_path,
        }
        if publish_port is not None:
            run_kwargs["ports"] = {f"{publish_port}/tcp": publish_port}

        container = self.client.containers.run(**run_kwargs)
        self._persistent_container = container

        # Ensure the workspace directory exists
        container.exec_run(["mkdir", "-p", self._workspace_path])

        # Set up a richer environment
        if self.install_build_tools:
            self._run_setup_commands()

        self._log(f"Persistent container started with workspace: {self._workspace_path}")

    def stop_persistent(self) -> None:
        """Stop and remove the persistent container."""
        if self._persistent_container is None:
            return
        try:
            self._persistent_container.stop(timeout=2)
        except Exception:
            pass
        try:
            self._persistent_container.remove(force=True)
        except Exception:
            pass
        self._persistent_container = None

    def launch_app(self, command: str) -> None:
        """
        Start a long-running command (e.g. a web server) detached in the workspace.

        Returns immediately; the process keeps running inside the container until
        the container is stopped. Output is not captured (it's a server, not a
        one-shot command).
        """
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            raise RuntimeError("launch_app requires a running persistent container.")
        self._persistent_container.exec_run(
            command, workdir=self._workspace_path, detach=True
        )

    def get_container_short_id(self) -> Optional[str]:
        """Short Docker id of the persistent container (for stop instructions)."""
        if self._persistent_container is None:
            return None
        return self._persistent_container.short_id

    def install_packages(self, packages: list[str]) -> bool:
        """
        Install Python packages inside the persistent container.
        Restricted interface (no arbitrary shell for now).

        Thin bool wrapper over `install_packages_detailed` — kept for callers
        that only care whether the install succeeded.
        """
        ok, _stdout, stderr = self.install_packages_detailed(packages)
        if not ok:
            print(f"[Docker] pip install failed:\n{stderr}")
        return ok

    def install_packages_detailed(self, packages: list[str]) -> tuple[bool, str, str]:
        """
        Install Python packages and return (success, stdout, stderr).

        The detailed return lets DependencyManager categorize *why* an install
        failed (not_found / build_error / network / permission) instead of
        collapsing everything to a bool.
        """
        if not self._persistent_container:
            raise RuntimeError("Persistent container is not running. Call start_persistent() first.")

        if not packages:
            return True, "", ""

        cmd = ["pip", "install", "--no-cache-dir"] + packages
        result = self._persistent_container.exec_run(cmd, demux=True)

        stdout_b, stderr_b = result.output or (b"", b"")
        stdout = (stdout_b or b"").decode(errors="replace")
        stderr = (stderr_b or b"").decode(errors="replace")
        return result.exit_code == 0, stdout, stderr

    def _run_setup_commands(self) -> None:
        """Run one-time setup inside the persistent container (richer environment).

        If the image is already provisioned (e.g. the prebaked `crucible-runtime`
        image, which bakes the C toolchain + pytest tooling), this detects the
        tooling and skips the expensive apt/pip work entirely — dropping per-task
        container_setup from ~15s to ~1s.
        """
        if not self._persistent_container:
            return

        # Fast path: tooling already present (prebaked image) → skip setup.
        check = self._persistent_container.exec_run(
            ["python", "-c", "import pytest, pytest_jsonreport"]
        )
        if getattr(check, "exit_code", 1) == 0:
            self._log("Test tooling already present in image; skipping container setup.")
            return

        commands = [
            "apt-get update -qq",
            "apt-get install -y --no-install-recommends build-essential gcc python3-dev",
            "pip install --upgrade pip setuptools wheel",
            # Needed for the real-pytest success gate.
            "pip install --no-cache-dir pytest pytest-json-report",
        ]

        for cmd in commands:
            result = self._persistent_container.exec_run(cmd, demux=True)
            if result.exit_code != 0:
                # Non-fatal for now — many packages work without build tools
                pass

    def _log(self, message: str, level: str = "info") -> None:
        """Internal logging helper."""
        if level == "error":
            self._logger.error(f"[DockerExecutor] {message}")
        elif level == "warning":
            self._logger.warning(f"[DockerExecutor] {message}")
        else:
            self._logger.info(f"[DockerExecutor] {message}")

    # ==================================================================
    # Phase 2: Workspace / Filesystem Methods (Persistent Mode Only)
    # ==================================================================

    def get_workspace_path(self) -> Optional[str]:
        """Return the workspace root path inside the container (e.g. /workspace/<task_id>)."""
        if not self.persistent or not self._persistent_container:
            return None
        return self._workspace_path

    def get_workspace_id(self) -> Optional[str]:
        """Return the current workspace ID."""
        return self._workspace_id if self.persistent else None

    def _resolve_workspace_path(self, relative_path: str) -> Optional[str]:
        """Internal helper to safely resolve a path inside the workspace."""
        if not self._workspace_path:
            return None

        if ".." in relative_path or relative_path.startswith("/"):
            return None

        return f"{self._workspace_path}/{relative_path.lstrip('/')}"

    def write_file(self, relative_path: str, content: str) -> bool:
        """Write (or overwrite) a file relative to the workspace root."""
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            self._log("write_file called but no persistent workspace is active", level="warning")
            return False

        # Basic path safety
        safe_path = self._resolve_workspace_path(relative_path)
        if not safe_path:
            self._log(f"write_file rejected unsafe path: {relative_path}", level="warning")
            return False

        try:
            parent_dir = str(Path(safe_path).parent)

            self._persistent_container.exec_run(["mkdir", "-p", parent_dir])

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=Path(safe_path).name)
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content.encode("utf-8")))
            tar_stream.seek(0)

            self._persistent_container.put_archive(parent_dir, tar_stream.getvalue())
            return True

        except Exception as e:
            self._log(f"write_file failed for {relative_path}: {e}", level="error")
            return False

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read a file relative to the workspace root."""
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            return None

        safe_path = self._resolve_workspace_path(relative_path)
        if not safe_path:
            self._log(f"read_file rejected unsafe path: {relative_path}", level="warning")
            return None

        try:
            full_path = safe_path
            result = self._persistent_container.exec_run(["cat", full_path], demux=True)
            if result.exit_code == 0 and result.output:
                stdout, _ = result.output
                return stdout.decode("utf-8") if stdout else ""
            return None
        except Exception as e:
            self._log(f"read_file failed for {relative_path}: {e}", level="error")
            return None

    def list_dir(self, relative_path: str = ".") -> List[str]:
        """List contents of a directory relative to the workspace root."""
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            return []

        safe_path = self._resolve_workspace_path(relative_path)
        if not safe_path:
            self._log(f"list_dir rejected unsafe path: {relative_path}", level="warning")
            return []

        try:
            target = safe_path
            result = self._persistent_container.exec_run(["ls", "-1", target], demux=True)
            if result.exit_code == 0 and result.output:
                stdout, _ = result.output
                if stdout:
                    return [line.strip() for line in stdout.decode().splitlines() if line.strip()]
            return []
        except Exception as e:
            self._log(f"list_dir failed for {relative_path}: {e}", level="error")
            return []

    def create_directory(self, relative_path: str) -> bool:
        """Create a directory (and parents) relative to the workspace root."""
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            return False

        if ".." in relative_path or relative_path.startswith("/"):
            self._log(f"create_directory rejected unsafe path: {relative_path}", level="warning")
            return False

        try:
            target = f"{self._workspace_path}/{relative_path.lstrip('/')}"
            result = self._persistent_container.exec_run(["mkdir", "-p", target])
            return result.exit_code == 0
        except Exception as e:
            self._log(f"create_directory failed for {relative_path}: {e}", level="error")
            return False

    # ------------------------------------------------------------------
    # Convenience Methods (Phase 2)
    # ------------------------------------------------------------------

    def run_command_in_workspace(
        self, cmd: str, timeout: Optional[int] = None
    ) -> tuple[int, str, str]:
        """
        Run an arbitrary shell command inside the workspace directory.
        Returns (exit_code, stdout, stderr).
        """
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            return -1, "", "No persistent container is running."

        try:
            result = self._persistent_container.exec_run(
                cmd,
                workdir=self._workspace_path,
                demux=True,
            )
            exit_code = result.exit_code
            stdout, stderr = result.output or (b"", b"")

            return (
                exit_code,
                stdout.decode(errors="replace") if stdout else "",
                stderr.decode(errors="replace") if stderr else "",
            )
        except Exception as e:
            self._log(f"run_command_in_workspace failed: {e}")
            return -1, "", str(e)

    def run_python_file(self, relative_path: str) -> tuple[int, str, str]:
        """
        Convenience method to run a Python file inside the workspace.
        Returns (exit_code, stdout, stderr).
        """
        return self.run_command_in_workspace(f"python {relative_path}")

    def run_in_workspace(self, cmd: str) -> tuple[int, str, str]:
        """
        Short alias for run_command_in_workspace. Preferred for readability.
        """
        return self.run_command_in_workspace(cmd)

    def execute_python_in_workspace(self, relative_path: str) -> tuple[int, str, str]:
        """
        Run a Python file inside the workspace. Cleaner name than run_python_file.
        """
        return self.run_python_file(relative_path)

    def install_requirements_file(self, relative_path: str = "requirements.txt") -> bool:
        """
        Install Python packages from a requirements.txt file inside the workspace.

        Thin bool wrapper over `install_requirements_file_detailed`.
        """
        ok, _stdout, stderr = self.install_requirements_file_detailed(relative_path)
        if not ok and stderr:
            self._log(f"install_requirements_file failed:\n{stderr}", level="error")
        return ok

    def install_requirements_file_detailed(
        self, relative_path: str = "requirements.txt"
    ) -> tuple[bool, str, str]:
        """
        Install from a requirements.txt and return (success, stdout, stderr) so
        the caller can categorize failures.
        """
        if not self.persistent or not self._persistent_container:
            return False, "", "requires a persistent container"

        full_path = f"{self._workspace_path}/{relative_path.lstrip('/')}"
        cmd = ["pip", "install", "--no-cache-dir", "-r", full_path]
        result = self._persistent_container.exec_run(cmd, workdir=self._workspace_path, demux=True)

        stdout_b, stderr_b = result.output or (b"", b"")
        stdout = (stdout_b or b"").decode(errors="replace")
        stderr = (stderr_b or b"").decode(errors="replace")
        return result.exit_code == 0, stdout, stderr

    def run_tests(self, command: str = "python -m pytest") -> tuple[int, str, str]:
        """
        Run tests inside the workspace using the provided command.
        Defaults to pytest.
        """
        return self.run_command_in_workspace(command)

    async def run_pytest(self, files: dict[str, str]) -> TestResults:
        """
        Write `files` into the persistent workspace and run a real pytest suite.

        Mirrors SandboxedExecutor.run_pytest so the loop can stay
        executor-agnostic. Requires persistent mode (a long-lived workspace);
        ephemeral containers return a clear failure instead of a hollow pass.
        """
        impl_blob = "\n\n".join(
            src for path, src in files.items() if not _is_test_path(path)
        )
        safety_report = self.safety_checker.analyze(
            CodeArtifact(source=impl_blob, file_path="solution.py", language="python")
        )
        if safety_report.level.value == "dangerous":
            return TestResults(
                passed=False,
                stderr=f"Safety violation: {safety_report.warnings}",
                exit_code=-2,
                error_type="SafetyError",
            )

        if not (self.persistent and self._persistent_container):
            return TestResults(
                passed=False,
                stderr=(
                    "Docker pytest gating requires persistent mode. "
                    "Run with --docker-persistent."
                ),
                exit_code=-1,
                error_type="DockerExecutionError",
                from_pytest=True,
            )

        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._run_pytest_sync(files)
        )

    def _run_pytest_sync(self, files: dict[str, str]) -> TestResults:
        start_time = time.time()
        self.write_files(files)
        report_rel = ".report.json"
        cmd = (
            f"python -m pytest . -p no:cacheprovider "
            f"--json-report --json-report-file={report_rel} -q"
        )
        exit_code, stdout, stderr = self.run_command_in_workspace(cmd)
        report_text = self.read_file(report_rel)
        return build_test_results(
            report_text, stdout, stderr, exit_code,
            execution_time=time.time() - start_time,
        )

    def capture_environment(self) -> dict:
        """
        Snapshot the persistent container's environment at this moment.

        Returns a dict with:
          - installed_packages: list[str] (pip names, no versions for stability)
          - workspace_files: list[str] (file paths relative to workspace root,
            shallow listing — top-level entries only)

        Used by LongTermMemory to attach env context to successful patterns.
        Returns an empty payload if no persistent container is active.
        """
        empty = {"installed_packages": [], "workspace_files": []}
        if not self.persistent or not self._persistent_container:
            return empty

        # Installed packages via `pip list --format=freeze` — strip versions
        # so retrieval bonuses are stable across point releases.
        packages: List[str] = []
        try:
            exit_code, stdout, _ = self.run_command_in_workspace("pip list --format=freeze")
            if exit_code == 0 and stdout:
                for line in stdout.splitlines():
                    if "==" in line:
                        name = line.split("==", 1)[0].strip().lower()
                        if name:
                            packages.append(name)
        except Exception as e:
            self._log(f"capture_environment: pip list failed: {e}", level="warning")

        # Workspace files (top level only — keeps the snapshot small)
        files = self.list_dir(".")

        return {
            "installed_packages": packages,
            "workspace_files": files,
        }

    def write_files(self, files: dict[str, str]) -> bool:
        """
        Write multiple files at once (very useful for multi-file generation).
        Returns True only if all files were written successfully.
        """
        if not self.persistent or not self._persistent_container or not self._workspace_path:
            self._log("write_files called but no persistent workspace is active", level="warning")
            return False

        success = True
        for relative_path, content in files.items():
            if not self.write_file(relative_path, content):
                success = False
        return success

    # ==================================================================
    # Existing Ephemeral Execution (kept for compatibility)
    # ==================================================================

    async def execute(
        self,
        code: CodeArtifact,
        test_command: Optional[str] = None,
    ) -> TestResults:
        """
        Execute code.

        - If persistent mode is active and a container is running → reuse it.
        - Otherwise → use the old ephemeral container path (Phase 1 behavior).
        """
        # Always run safety analysis first
        safety_report = self.safety_checker.analyze(code)
        if safety_report.level.value == "dangerous":
            return TestResults(
                passed=False,
                stderr=f"Safety violation: {safety_report.warnings}",
                exit_code=-2,
                error_type="SafetyError",
            )

        start_time = time.time()

        # === Phase 2: Reuse persistent container when available ===
        if self.persistent and self._persistent_container is not None:
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._execute_in_persistent_container(code.source),
                )
                execution_time = time.time() - start_time

                return TestResults(
                    passed=result["exit_code"] == 0,
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    exit_code=result["exit_code"],
                    execution_time=execution_time,
                    error_type=self._classify_error(result["exit_code"], result.get("stderr", "")),
                )
            except Exception as e:
                return TestResults(
                    passed=False,
                    stderr=f"Docker execution error (persistent): {str(e)}",
                    exit_code=-1,
                    execution_time=time.time() - start_time,
                    error_type="DockerExecutionError",
                )

        # === Phase 1 fallback: Ephemeral container ===
        container_dir = "/tmp/code"
        container_code_path = f"{container_dir}/main.py"

        run_kwargs = {
            "image": self.docker_image,
            "command": ["python", "-c", f"exec(open('{container_code_path}').read())"],
            "mem_limit": f"{self.config.memory_limit_mb}m",
            "nano_cpus": int(1_000_000_000 * 1.0),
            "network_disabled": not self.config.network_enabled,
            "working_dir": "/tmp",
        }

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._run_container_with_timeout(
                    run_kwargs,
                    self.config.timeout_seconds,
                    code.source,
                    container_dir,
                ),
            )

            execution_time = time.time() - start_time
            return TestResults(
                passed=result["exit_code"] == 0,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                exit_code=result["exit_code"],
                execution_time=execution_time,
                error_type=self._classify_error(result["exit_code"], result.get("stderr", "")),
            )

        except ImageNotFound:
            return TestResults(
                passed=False,
                stderr=f"Docker image not found: {self.docker_image}. "
                       f"Try: docker pull {self.docker_image}",
                exit_code=-1,
                error_type="DockerImageNotFound",
            )
        except ContainerError as e:
            execution_time = time.time() - start_time
            return TestResults(
                passed=False,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else str(e),
                exit_code=e.exit_status,
                execution_time=execution_time,
                error_type="ContainerError",
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResults(
                passed=False,
                stderr=f"Docker execution error: {str(e)}",
                exit_code=-1,
                execution_time=execution_time,
                error_type="DockerExecutionError",
            )

    def _execute_in_persistent_container(self, source_code: str) -> dict:
        """
        Execute user code inside the already-running persistent container.
        Used by the new persistent execution path.
        """
        container = self._persistent_container
        container_dir = "/tmp/code"

        # Ensure target directory exists
        container.exec_run(["mkdir", "-p", container_dir])

        # Inject code
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name="main.py")
            tarinfo.size = len(source_code)
            tar.addfile(tarinfo, io.BytesIO(source_code.encode("utf-8")))
        tar_stream.seek(0)

        container.put_archive(container_dir, tar_stream.getvalue())

        # Run the code
        exec_result = container.exec_run(
            ["python", "-c", f"exec(open('{container_dir}/main.py').read())"],
            demux=True
        )

        exit_code = exec_result.exit_code
        stdout, stderr = exec_result.output or (b"", b"")

        return {
            "exit_code": exit_code,
            "stdout": stdout.decode(errors="replace") if stdout else "",
            "stderr": stderr.decode(errors="replace") if stderr else "",
        }

    def _run_container_with_timeout(
        self,
        run_kwargs: dict,
        timeout: int,
        source_code: str,
        container_dir: str,
    ) -> dict:
        """
        Start container, inject the source code using put_archive (docker cp equivalent),
        then wait for completion with timeout.

        This approach is much more reliable on macOS (Colima / Docker Desktop)
        than bind-mounting temporary files/directories.
        """
        container = None
        try:
            # Start the container with a long-running dummy command so it stays alive.
            # This lets us reliably inject the file with put_archive before running anything.
            run_kwargs = dict(run_kwargs)
            run_kwargs["detach"] = True
            run_kwargs.pop("stdout", None)
            run_kwargs.pop("stderr", None)
            run_kwargs["command"] = ["sleep", "3600"]   # keep container alive

            container = self.client.containers.run(**run_kwargs)

            # Ensure the target directory exists inside the container before put_archive.
            # This is often required to avoid 404 errors on the archive endpoint,
            # especially on macOS Docker/Colima where filesystem operations can be finicky.
            container.exec_run(["mkdir", "-p", container_dir])

            # Inject the code file using put_archive (equivalent to `docker cp`)
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name="main.py")
                tarinfo.size = len(source_code)
                tar.addfile(tarinfo, io.BytesIO(source_code.encode("utf-8")))
            tar_stream.seek(0)

            container.put_archive(container_dir, tar_stream.getvalue())

            # Now run the actual code via exec_run (so we control when it executes)
            exec_result = container.exec_run(
                ["python", "-c", f"exec(open('{container_dir}/main.py').read())"],
                demux=True
            )

            exit_code = exec_result.exit_code
            stdout, stderr = exec_result.output or (b"", b"")

            # Stop the sleeper container
            try:
                container.stop(timeout=1)
            except Exception:
                pass

            return {
                "exit_code": exit_code,
                "stdout": stdout.decode(errors="replace") if stdout else "",
                "stderr": stderr.decode(errors="replace") if stderr else "",
            }

        except Exception as e:
            if container is not None:
                try:
                    container.stop(timeout=2)
                except Exception:
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    pass

            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise TimeoutError(f"Container execution timed out after {timeout}s") from e

            raise RuntimeError(f"Container run failed: {e}") from e
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _classify_error(self, exit_code: int, stderr: str) -> Optional[str]:
        """Classify common errors from container output."""
        if exit_code == 0:
            return None

        stderr_lower = stderr.lower()

        error_patterns = [
            ("ModuleNotFoundError", "modulenotfounderror"),
            ("ImportError", "importerror"),
            ("SyntaxError", "syntaxerror"),
            ("IndentationError", "indentationerror"),
            ("NameError", "nameerror"),
            ("TypeError", "typeerror"),
            ("ValueError", "valueerror"),
            ("TimeoutError", "timeout"),
            ("MemoryError", "memoryerror"),
        ]

        for error_type, pattern in error_patterns:
            if pattern in stderr_lower:
                return error_type

        return "UnknownError"

    def close(self):
        """Close the Docker client (optional cleanup)."""
        if hasattr(self, "client"):
            self.client.close()
