"""
Tester: Runs tests and validates code.
"""

from typing import List
from agent.models import Plan, CodeArtifact, TestResults
from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig


class Tester:
    """
    Runs tests on generated code using the sandboxed executor.
    """
    
    def __init__(self, executor: SandboxedExecutor = None):
        self.executor = executor or SandboxedExecutor(
            config=ExecutionConfig(
                timeout_seconds=30,
                memory_limit_mb=512
            )
        )
    
    async def run_tests(
        self,
        code: CodeArtifact,
        plan: Plan
    ) -> TestResults:
        """
        Run tests on the generated code.
        
        Args:
            code: The code artifact to test
            plan: The plan containing test cases
        
        Returns:
            TestResults with pass/fail status and details
        """
        # The sandboxed executor runs the code and returns results
        # The code should include its own tests/assertions
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
            # Use node --check for syntax validation
            import subprocess
            try:
                result = subprocess.run(
                    ["node", "--check", "-e", code.source],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return TestResults(
                    passed=result.returncode == 0,
                    stderr=result.stderr,
                    error_type="SyntaxError" if result.returncode != 0 else None
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
