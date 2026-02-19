"""
Reflector: Analyzes failures and extracts learnings.
"""

import json
import re
from typing import Protocol, List, Optional

from agent.models import (
    TestResults, CodeArtifact, Reflection, ErrorSignature, Plan
)
from agent.memory.failure_memory import FailureMemory


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str: ...


class Reflector:
    """
    Analyzes test failures to extract insights and suggest fixes.
    Uses failure memory to avoid repeating mistakes.
    """
    
    SYSTEM_PROMPT = """You are an expert debugging assistant. Analyze test failures carefully.

Your task is to:
1. Identify the root cause of the failure
2. Formulate a hypothesis about why it failed
3. Suggest a specific fix
4. Determine if continued attempts are worthwhile

Respond in valid JSON format:
{
    "success": false,
    "analysis": "brief description of what went wrong",
    "root_cause": "the underlying reason for the failure",
    "hypothesis": "explanation of why the code failed",
    "suggested_fix": "specific instructions for fixing",
    "should_continue": true,
    "confidence": 0.8
}

Be honest about confidence. If the error seems hopeless or unclear, set should_continue to false."""

    def __init__(
        self,
        llm: LLMClient,
        failure_memory: Optional[FailureMemory] = None
    ):
        self.llm = llm
        self.failure_memory = failure_memory
    
    async def analyze(
        self,
        test_results: TestResults,
        code: CodeArtifact,
        plan: Plan,
        iteration: int,
        previous_reflections: List[Reflection] = None
    ) -> Reflection:
        """
        Analyze test results and generate reflection.
        
        Args:
            test_results: Results from test execution
            code: The code that was tested
            plan: The original plan
            iteration: Current iteration number
            previous_reflections: Previous reflections for context
        
        Returns:
            Reflection with analysis and recommendations
        """
        # Success case - no deep analysis needed
        if test_results.passed:
            return Reflection(
                success=True,
                analysis="All tests passed successfully",
                should_continue=False,
                confidence=1.0
            )
        
        # Extract error signature
        error_sig = self._extract_error_signature(test_results)
        
        # Check failure memory for similar errors
        similar_failures = []
        if self.failure_memory:
            similar_failures = await self.failure_memory.find_similar_failures(error_sig)
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            test_results=test_results,
            code=code,
            plan=plan,
            error_sig=error_sig,
            similar_failures=similar_failures,
            iteration=iteration,
            previous_reflections=previous_reflections
        )
        
        # Get LLM analysis
        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.5
        )
        
        # Parse response
        analysis = self._parse_response(response)
        
        # Build reflection
        reflection = Reflection(
            success=analysis.get("success", False),
            analysis=analysis.get("analysis", "Unknown error"),
            error_signature=error_sig,
            root_cause=analysis.get("root_cause"),
            hypothesis=analysis.get("hypothesis"),
            suggested_fix=analysis.get("suggested_fix"),
            should_continue=analysis.get("should_continue", True),
            confidence=analysis.get("confidence", 0.5)
        )
        
        # Store in failure memory if it's a real failure
        if not test_results.passed and self.failure_memory:
            await self.failure_memory.store_failure(
                error_signature=error_sig,
                attempt=code,
                root_cause=reflection.root_cause or "Unknown",
                fix=reflection.suggested_fix or "No fix suggested",
                goal=plan.goal
            )
        
        return reflection
    
    def _extract_error_signature(self, results: TestResults) -> ErrorSignature:
        """Extract normalized error signature from test results."""
        error_type = results.error_type or "UnknownError"
        error_message = results.stderr[:500] if results.stderr else ""
        
        # Try to extract line number
        line_number = None
        if error_message:
            # Common patterns for line numbers
            patterns = [
                r'line (\d+)',
                r':(\d+):',
                r'Line (\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, error_message)
                if match:
                    line_number = int(match.group(1))
                    break
        
        return ErrorSignature(
            error_type=error_type,
            error_message=error_message,
            line_number=line_number
        )
    
    def _build_analysis_prompt(
        self,
        test_results: TestResults,
        code: CodeArtifact,
        plan: Plan,
        error_sig: ErrorSignature,
        similar_failures: List[dict],
        iteration: int,
        previous_reflections: List[Reflection]
    ) -> str:
        """Build the analysis prompt for the LLM."""
        lines = [
            f"Analysis Request - Iteration {iteration}",
            "",
            "Task:",
            plan.goal,
            "",
            "Generated Code:",
            "```",
            code.source[:2000],  # Limit code length
            "```",
            "",
            "Test Results:",
            f"Passed: {test_results.passed}",
            f"Exit Code: {test_results.exit_code}",
            f"Error Type: {test_results.error_type}",
            "",
            "Standard Output:",
            test_results.stdout[:1000] if test_results.stdout else "(empty)",
            "",
            "Standard Error:",
            test_results.stderr[:1000] if test_results.stderr else "(empty)",
        ]
        
        if similar_failures:
            lines.extend([
                "",
                "Similar Past Failures:",
            ])
            for i, failure in enumerate(similar_failures[:2], 1):
                lines.extend([
                    f"\nPast Failure {i}:",
                    f"Error: {failure['error_type']}",
                    f"Root Cause: {failure['root_cause']}",
                    f"Fix: {failure['fix']}",
                ])
        
        if previous_reflections:
            lines.extend([
                "",
                "Previous Attempts:",
            ])
            for ref in previous_reflections[-2:]:  # Last 2 reflections
                lines.extend([
                    f"- Analysis: {ref.analysis[:100]}",
                    f"  Fix tried: {ref.suggested_fix[:100] if ref.suggested_fix else 'N/A'}",
                ])
        
        lines.append("\nProvide detailed analysis and fix suggestion.")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response into structured analysis."""
        # Try to extract JSON
        try:
            # Look for JSON in code blocks
            pattern = r'```(?:json)?\s*\n(.*?)\n```'
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
            
            # Try to find JSON object directly
            pattern = r'\{[\s\S]*\}'
            match = re.search(pattern, response)
            if match:
                return json.loads(match.group())
            
            # Direct parse
            return json.loads(response)
            
        except json.JSONDecodeError:
            # Fallback: extract key information manually
            return {
                "success": False,
                "analysis": response[:500],
                "should_continue": "hopeless" not in response.lower(),
                "confidence": 0.5
            }
    
    def _is_hopeless_case(
        self,
        test_results: TestResults,
        iteration: int,
        error_sig: ErrorSignature
    ) -> bool:
        """
        Heuristic to detect hopeless cases early.
        """
        # Too many iterations with same error
        if iteration > 5 and error_sig.error_type:
            return True
        
        # Syntax errors after iteration 2 are concerning
        if error_sig.error_type == "SyntaxError" and iteration > 2:
            return True
        
        # Timeout errors might indicate infinite loops
        if error_sig.error_type == "TimeoutError" and iteration > 3:
            return True
        
        return False
