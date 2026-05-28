"""
Reflector: Analyzes failures and extracts learnings.
"""

import json
import re
from typing import Protocol, List, Optional

from agent.models import (
    TestResults, CodeArtifact, Reflection, ErrorSignature, Plan, Learning, Prediction
)
from agent.memory.failure_memory import FailureMemory
from agent.memory.predictions import PredictionMemory


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

Be honest about confidence. If the error seems hopeless or unclear, set should_continue to false.

If a test suite is shown, treat it as a fixed, authoritative specification: never
suggest modifying, relaxing, or rewriting the tests. The fix must always change the
code under test so it satisfies the tests exactly as written."""

    LEARNING_SYSTEM_PROMPT = """You are reviewing a coding task that just succeeded.
Extract 0–3 short, reusable lessons that would help on similar future tasks.

A good lesson is concrete and transferable. Bad: "the code worked." Good: "For
FastAPI projects, expose a /health endpoint for liveness probes." Skip the
boilerplate; do not restate the task description.

Respond in valid JSON:
{
  "learnings": [
    {"lesson": "...", "tags": ["fastapi", "health-check"]},
    ...
  ]
}

If nothing non-obvious was learned, return {"learnings": []}."""

    PREDICTION_SYSTEM_PROMPT = """You are inspecting a coding task that just FAILED.
Your job is to emit 0–3 falsifiable predictions: concrete inputs that would
exercise the same kind of bug, paired with the error type you expect.

A good prediction is testable. Bad: "edge cases will fail." Good:
  trigger_input: "[]"
  predicted_error_type: "IndexError"
  predicted_explanation: "Empty list dereferenced without bounds check."

Requirements:
- trigger_input MUST be a concrete value as a Python repr-style string
  (e.g. "-1", "''", "[]", "{'k': None}"). NEVER vague ("any negative number").
- predicted_error_type MUST be a Python exception class name
  (ValueError, TypeError, IndexError, KeyError, AttributeError, ...).
- Skip predictions you are not confident about; better to emit none.

Respond in valid JSON only:
{
  "predictions": [
    {
      "trigger_input": "...",
      "predicted_error_type": "...",
      "predicted_explanation": "...",
      "confidence": 0.7
    }
  ]
}

If no useful prediction can be made, return {"predictions": []}."""

    def __init__(
        self,
        llm: LLMClient,
        failure_memory: Optional[FailureMemory] = None,
        prediction_memory: Optional[PredictionMemory] = None,
    ):
        self.llm = llm
        self.failure_memory = failure_memory
        self.prediction_memory = prediction_memory
    
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
        
        # Store in failure memory if it's a real failure. The loop reads
        # reflection.failure_id on the next iteration's success to mark this
        # entry was_fixed=True.
        if not test_results.passed and self.failure_memory:
            reflection.failure_id = await self.failure_memory.store_failure(
                error_signature=error_sig,
                attempt=code,
                root_cause=reflection.root_cause or "Unknown",
                fix=reflection.suggested_fix or "No fix suggested",
                goal=plan.goal
            )

        # Track D phase 1: emit falsifiable predictions about *why* this
        # failed. Best-effort and isolated — if the LLM gives us nothing
        # useful, or the call itself errors, the reflection still returns
        # normally. A second LLM call is intentional: keeping prediction
        # extraction out of the main reflection prompt avoids regressing
        # reflection quality with extra format pressure.
        if (
            not test_results.passed
            and self.prediction_memory is not None
            and reflection.failure_id
        ):
            try:
                predictions = await self.extract_predictions(
                    code=code,
                    plan=plan,
                    error_signature=error_sig,
                    source_failure_id=reflection.failure_id,
                    task_id=None,
                )
                for pred in predictions:
                    await self.prediction_memory.store(pred)
            except Exception as e:
                # Predictions are an experimental signal; never let them
                # break the reflection path.
                print(f"⚠️  Prediction extraction failed (non-fatal): {e}")

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

        # When the result came from a real pytest run, the per-test failures are
        # far more actionable than the raw streams — surface them explicitly.
        if getattr(test_results, "from_pytest", False):
            lines.extend([
                "",
                "IMPORTANT: The pytest suite is the FROZEN, authoritative specification "
                "for this task. It is immutable — do NOT suggest changing, relaxing, or "
                "rewriting any test or assertion. Diagnose why the IMPLEMENTATION fails "
                "and propose a fix to the implementation only. If a test expects a "
                "specific output shape (e.g. plain text vs JSON, or an exact string), "
                "change the implementation to produce exactly that.",
                "",
                "Pytest Summary:",
                f"collected={test_results.tests_collected} "
                f"passed={test_results.tests_passed} "
                f"failed={test_results.tests_failed} "
                f"errors={test_results.tests_errors}",
            ])
            if test_results.tests_collected == 0:
                lines.append(
                    "No tests were collected — the suite failed to import or is empty."
                )
            for failure in (test_results.test_failures or [])[:4]:
                lines.extend([
                    "",
                    f"FAILED {failure.get('nodeid', '?')} ({failure.get('outcome', '')}):",
                    (failure.get("message", "") or "")[:600],
                ])

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
    
    async def extract_learnings(
        self,
        plan: Plan,
        code: CodeArtifact,
        task_id: Optional[str] = None,
    ) -> List[Learning]:
        """
        Ask the LLM to extract reusable lessons from a successful task.

        Returns 0-3 Learning entries. Failures during extraction (LLM errors,
        unparseable response) are swallowed and return an empty list — learnings
        are best-effort and must not break the success path.
        """
        prompt_parts = [
            "Successful Task",
            "",
            "Goal:",
            plan.goal,
            "",
            f"Project type: {getattr(plan, 'project_type', 'general')}",
            f"Language: {getattr(plan, 'language', 'python')}",
        ]
        if plan.dependencies:
            prompt_parts.extend(["", "Dependencies: " + ", ".join(plan.dependencies)])
        prompt_parts.extend([
            "",
            "Solution Code:",
            "```",
            code.source[:2000],
            "```",
            "",
            "Extract reusable lessons.",
        ])
        prompt = "\n".join(prompt_parts)

        try:
            response = await self.llm.complete(
                system=self.LEARNING_SYSTEM_PROMPT,
                prompt=prompt,
                temperature=0.3,
            )
        except Exception:
            return []

        parsed = self._parse_response(response) or {}
        raw_items = parsed.get("learnings") or []
        if not isinstance(raw_items, list):
            return []

        learnings: List[Learning] = []
        for item in raw_items[:3]:  # cap at 3
            if not isinstance(item, dict):
                continue
            lesson_text = (item.get("lesson") or "").strip()
            if not lesson_text:
                continue
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            learnings.append(Learning(
                lesson=lesson_text,
                project_type=getattr(plan, "project_type", "general"),
                language=getattr(plan, "language", "python"),
                tags=[str(t) for t in tags],
                source_task_id=task_id,
                source_goal=plan.goal,
            ))
        return learnings

    async def extract_predictions(
        self,
        code: CodeArtifact,
        plan: Plan,
        error_signature: ErrorSignature,
        source_failure_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Prediction]:
        """
        Ask the LLM to emit falsifiable predictions about *why* this failure
        happened — concrete inputs that should trigger the same kind of bug
        if the code is left unfixed.

        Returns 0-3 Predictions, schema-gated: each must have a concrete
        trigger_input and a Python exception type. Predictions are the seed
        of the Track D self-improvement loop — phase 2's replay engine will
        test them and write back times_tested / times_confirmed.

        Best-effort. Failures (LLM error, unparseable response) return [].
        """
        prompt_parts = [
            "Failed Task",
            "",
            "Goal:",
            plan.goal,
            "",
            f"Project type: {getattr(plan, 'project_type', 'general')}",
            f"Language: {getattr(plan, 'language', 'python')}",
            "",
            "Error:",
            f"  type: {error_signature.error_type}",
            f"  message: {error_signature.error_message[:300]}",
            "",
            "Failing Code:",
            "```",
            code.source[:2000],
            "```",
            "",
            (
                "Emit predictions: for each candidate, give a CONCRETE "
                "input (Python repr style) and the exception type you "
                "expect when that input is fed to this code. If you have "
                "no high-confidence prediction, return an empty list."
            ),
        ]
        prompt = "\n".join(prompt_parts)

        try:
            response = await self.llm.complete(
                system=self.PREDICTION_SYSTEM_PROMPT,
                prompt=prompt,
                temperature=0.3,
            )
        except Exception:
            return []

        parsed = self._parse_response(response) or {}
        raw_items = parsed.get("predictions") or []
        if not isinstance(raw_items, list):
            return []

        predictions: List[Prediction] = []
        for item in raw_items[:3]:  # cap at 3
            if not isinstance(item, dict):
                continue
            pred = Prediction(
                trigger_input=str(item.get("trigger_input") or "").strip(),
                predicted_error_type=str(item.get("predicted_error_type") or "").strip(),
                predicted_explanation=str(item.get("predicted_explanation") or "").strip(),
                confidence=float(item.get("confidence", 0.5) or 0.5),
                source_failure_id=source_failure_id,
                source_task_id=task_id,
                source_goal=plan.goal,
                language=getattr(plan, "language", "python"),
            )
            # Strict schema gate — falsifiability requires a concrete input
            # and a predicted error type. Drop anything else.
            if pred.is_well_formed():
                predictions.append(pred)
        return predictions

