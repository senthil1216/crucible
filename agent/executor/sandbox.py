"""
Sandboxed execution environment with resource limits.
"""

import asyncio
import subprocess
import tempfile
import sys
import resource
import signal
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
import time

from agent.models import CodeArtifact, TestResults, AgentConfig
from agent.safety.checker import SafetyChecker
from agent.pytest_report import build_test_results


def _apply_rlimits(cpu_seconds: int) -> None:
    """preexec_fn for subprocesses: best-effort CPU/core limits (POSIX).

    Ergonomic isolation, not a security boundary. We deliberately do NOT cap
    RLIMIT_AS here: it counts virtual address space, and a tight cap kills the
    pytest runner itself (especially with native deps on Linux). Runaway memory
    is bounded instead by the wall-clock timeout (and by Docker mem_limit in the
    container path).
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def _is_test_path(rel_path: str) -> bool:
    """True for files that are part of the test suite (not implementation)."""
    name = Path(rel_path).name
    parts = Path(rel_path).parts
    return (
        "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


@dataclass
class ExecutionConfig:
    """Configuration for sandboxed execution."""
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_time_limit_seconds: int = 10
    network_enabled: bool = False
    filesystem_readonly: bool = True


class SandboxedExecutor:
    """
    Executes code in an isolated environment with resource limits.
    Uses subprocess for isolation with ulimit for resource constraints.
    """
    
    def __init__(self, config: ExecutionConfig = None, safety_checker: SafetyChecker = None):
        self.config = config or ExecutionConfig()
        self.safety_checker = safety_checker or SafetyChecker()
    
    async def execute(
        self,
        code: CodeArtifact,
        test_command: Optional[str] = None
    ) -> TestResults:
        """
        Execute code in sandbox and return results.
        """
        # Safety check first
        safety_report = self.safety_checker.analyze(code)
        if safety_report.level.value == "dangerous":
            return TestResults(
                passed=False,
                stderr=f"Safety violation: {safety_report.warnings}",
                exit_code=-2,
                error_type="SafetyError"
            )
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            code_path = code.write_to_disk(workspace)
            
            # Set up execution
            if code.language == "python":
                return await self._execute_python(code_path, workspace)
            elif code.language == "javascript":
                return await self._execute_javascript(code_path, workspace)
            else:
                return TestResults(
                    passed=False,
                    stderr=f"Unsupported language: {code.language}",
                    exit_code=-1,
                    error_type="UnsupportedLanguage"
                )
    
    async def run_pytest(self, files: Dict[str, str]) -> TestResults:
        """
        Write `files` to a fresh temp workspace and run a real pytest suite.

        `files` maps relative paths (e.g. "solution.py",
        "tests/test_solution.py") to their contents. Returns a TestResults
        built from the pytest JSON report — `passed` is true only if pytest
        collected at least one test and none failed.
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

        return await asyncio.to_thread(self._run_pytest_sync, files)

    def _run_pytest_sync(self, files: Dict[str, str]) -> TestResults:
        start_time = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            for rel_path, content in files.items():
                target = workspace / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            report_path = workspace / ".report.json"
            cmd = [
                sys.executable, "-m", "pytest", ".",
                "-p", "no:cacheprovider",
                "--json-report", f"--json-report-file={report_path}",
                "-q",
            ]

            preexec = None
            if os.name == "posix":
                cpu = self.config.cpu_time_limit_seconds
                preexec = lambda: _apply_rlimits(cpu)  # noqa: E731

            try:
                result = subprocess.run(
                    cmd,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    preexec_fn=preexec,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired:
                return TestResults(
                    passed=False,
                    stderr=f"pytest timeout after {self.config.timeout_seconds}s",
                    exit_code=-1,
                    error_type="TimeoutError",
                    execution_time=self.config.timeout_seconds,
                    from_pytest=True,
                )

            report_text = report_path.read_text() if report_path.exists() else None
            return build_test_results(
                report_text,
                result.stdout,
                result.stderr,
                result.returncode,
                execution_time=time.time() - start_time,
            )

    async def _execute_python(
        self,
        code_path: Path,
        workspace: Path
    ) -> TestResults:
        """Execute Python code with resource limits."""
        
        start_time = time.time()
        
        # Build the command with resource limits
        cmd = [
            sys.executable,
            "-c",
            self._build_resource_limited_runner(code_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            return TestResults(
                passed=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                error_type=self._classify_error(result.returncode, result.stderr)
            )
            
        except subprocess.TimeoutExpired:
            return TestResults(
                passed=False,
                stderr=f"Execution timeout after {self.config.timeout_seconds}s",
                exit_code=-1,
                error_type="TimeoutError",
                execution_time=self.config.timeout_seconds
            )
        except Exception as e:
            return TestResults(
                passed=False,
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                error_type="ExecutionError"
            )
    
    def _build_resource_limited_runner(self, code_path: Path) -> str:
        """
        Build a Python script that sets resource limits and runs the code.
        """
        return f'''
import resource
import sys
import traceback

# Set resource limits
try:
    # CPU time limit (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (
        {self.config.cpu_time_limit_seconds},
        {self.config.cpu_time_limit_seconds}
    ))
except (ValueError, OSError):
    pass

# Memory limit (bytes) - RLIMIT_AS not available on macOS
try:
    memory_bytes = {self.config.memory_limit_mb} * 1024 * 1024
    if hasattr(resource, 'RLIMIT_AS'):
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    elif hasattr(resource, 'RLIMIT_VMEM'):
        resource.setrlimit(resource.RLIMIT_VMEM, (memory_bytes, memory_bytes))
    elif hasattr(resource, 'RLIMIT_RSS'):
        resource.setrlimit(resource.RLIMIT_RSS, (memory_bytes, memory_bytes))
except (ValueError, OSError):
    pass

# Disable core dumps
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ValueError, OSError):
    pass

# Limit number of open files
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
except (ValueError, OSError):
    pass

try:
    # Read and execute the code
    with open("{code_path}", "r") as f:
        code = f.read()
    
    # Execute in isolated namespace
    namespace = {{}}
    exec(code, namespace)
    
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
'''
    
    async def _execute_javascript(
        self,
        code_path: Path,
        workspace: Path
    ) -> TestResults:
        """Execute JavaScript code using Node.js."""
        
        start_time = time.time()
        
        # Use timeout command for resource limits
        cmd = [
            "timeout",
            f"{self.config.timeout_seconds}s",
            "node",
            str(code_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds + 2  # Slightly longer for timeout cmd
            )
            
            execution_time = time.time() - start_time
            
            # timeout command returns 124 on timeout
            if result.returncode == 124:
                return TestResults(
                    passed=False,
                    stderr=f"Execution timeout after {self.config.timeout_seconds}s",
                    exit_code=-1,
                    error_type="TimeoutError",
                    execution_time=self.config.timeout_seconds
                )
            
            return TestResults(
                passed=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                error_type=self._classify_js_error(result.returncode, result.stderr)
            )
            
        except FileNotFoundError:
            return TestResults(
                passed=False,
                stderr="Node.js not found. Cannot execute JavaScript.",
                exit_code=-1,
                error_type="RuntimeNotFound"
            )
        except subprocess.TimeoutExpired:
            return TestResults(
                passed=False,
                stderr="Execution timeout",
                exit_code=-1,
                error_type="TimeoutError"
            )
    
    def _classify_error(self, exit_code: int, stderr: str) -> Optional[str]:
        """Classify Python error from exit code and stderr."""
        if exit_code == 0:
            return None
        
        stderr_lower = stderr.lower()
        
        error_patterns = [
            ("SyntaxError", "syntaxerror"),
            ("IndentationError", "indentationerror"),
            ("NameError", "nameerror"),
            ("TypeError", "typeerror"),
            ("ValueError", "valueerror"),
            ("KeyError", "keyerror"),
            ("IndexError", "indexerror"),
            ("AttributeError", "attributeerror"),
            ("ImportError", "importerror"),
            ("ModuleNotFoundError", "modulenotfounderror"),
            ("ZeroDivisionError", "zerodivisionerror"),
            ("RecursionError", "recursionerror"),
            ("MemoryError", "memoryerror"),
            ("TimeoutError", "timeout"),
            ("AssertionError", "assertionerror"),
        ]
        
        for error_type, pattern in error_patterns:
            if pattern in stderr_lower:
                return error_type
        
        return "UnknownError"
    
    def _classify_js_error(self, exit_code: int, stderr: str) -> Optional[str]:
        """Classify JavaScript error."""
        if exit_code == 0:
            return None
        
        stderr_lower = stderr.lower()
        
        if "syntaxerror" in stderr_lower:
            return "SyntaxError"
        elif "referenceerror" in stderr_lower:
            return "ReferenceError"
        elif "typeerror" in stderr_lower:
            return "TypeError"
        elif "rangeerror" in stderr_lower:
            return "RangeError"
        elif "assertionerror" in stderr_lower:
            return "AssertionError"
        
        return "UnknownError"
