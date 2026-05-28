"""
Tester: Runs tests and validates code.
"""

from typing import Dict, List
from agent.models import Plan, CodeArtifact, TestResults
from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig


class Tester:
    """
    Runs tests on generated code using the sandboxed executor.
    """

    __test__ = False  # not a pytest test class

    def __init__(self, executor: SandboxedExecutor = None):
        self.executor = executor or SandboxedExecutor(
            config=ExecutionConfig(
                timeout_seconds=30,
                memory_limit_mb=512
            )
        )

    async def run_pytest(
        self,
        impl_files: Dict[str, str],
        test_files: Dict[str, str],
    ) -> TestResults:
        """
        Run a real pytest suite over the implementation + frozen test files.

        Delegates to the executor's `run_pytest`, which writes everything to an
        isolated workspace and returns a TestResults whose `passed` is true only
        when pytest collected >= 1 test and none failed. The implementation and
        test files are merged before running.
        """
        files = {**impl_files, **test_files}
        if not hasattr(self.executor, "run_pytest"):
            return TestResults(
                passed=False,
                stderr="Executor does not support pytest-based testing.",
                error_type="ExecutorError",
            )
        return await self.executor.run_pytest(files)
    
    async def run_tests(
        self,
        code: CodeArtifact,
        plan: Plan,
        run_from_workspace: bool = False
    ) -> TestResults:
        """
        Run tests on the generated code.

        When run_from_workspace=True (Phase 2 multi-file mode), the executor
        should run commands from within the project workspace.
        """
        if run_from_workspace and hasattr(self.executor, "run_in_workspace"):
            # For multi-file projects, prefer running via workspace command
            # The code's file_path tells us the entry point
            cmd = f"python {code.file_path}"
            exit_code, stdout, stderr = self.executor.run_in_workspace(cmd)

            return TestResults(
                passed=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )

        # Default path (works for both single-file and legacy)
        results = await self.executor.execute(code)
        return results
    
    async def validate_syntax(
        self,
        code: CodeArtifact
    ) -> TestResults:
        """
        Quick syntax validation without full execution.
        
        Args:
            code: Code to validate
        
        Returns:
            TestResults with syntax check results
        """
        if code.language == "python":
            import ast
            try:
                ast.parse(code.source)
                return TestResults(passed=True)
            except SyntaxError as e:
                return TestResults(
                    passed=False,
                    stderr=f"SyntaxError: {e.msg} at line {e.lineno}",
                    error_type="SyntaxError"
                )
        elif code.language == "javascript":
            # Use node --check for syntax validation.
            import asyncio
            try:
                proc = await asyncio.create_subprocess_exec(
                    "node", "--check", "-e", code.source,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(), timeout=5
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return TestResults(
                        passed=False,
                        stderr="node --check timeout",
                        error_type="TimeoutError",
                    )
                stderr = stderr_b.decode("utf-8", errors="replace")
                return TestResults(
                    passed=proc.returncode == 0,
                    stderr=stderr,
                    error_type="SyntaxError" if proc.returncode != 0 else None
                )
            except Exception as e:
                return TestResults(
                    passed=False,
                    stderr=str(e),
                    error_type="ValidationError"
                )
        
        return TestResults(passed=True)  # Unknown language, assume valid
    
    def summarize_results(self, results: TestResults) -> str:
        """Create a human-readable summary of test results."""
        lines = []
        
        if results.passed:
            lines.append("✅ All tests passed")
        else:
            lines.append("❌ Tests failed")
        
        if results.error_type:
            lines.append(f"Error Type: {results.error_type}")
        
        if results.execution_time:
            lines.append(f"Execution Time: {results.execution_time:.2f}s")
        
        if results.failed_tests:
            lines.append(f"Failed Tests: {', '.join(results.failed_tests)}")
        
        if results.warnings:
            lines.append(f"Warnings: {len(results.warnings)}")
        
        return "\n".join(lines)
