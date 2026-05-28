"""
Planner: Breaks down goals into executable plans.
"""

import json
from typing import List, Dict, Any, Protocol, Optional
from dataclasses import dataclass

from agent.models import Plan
from agent.memory.long_term import LongTermMemory


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str: ...


class Planner:
    """
    Creates execution plans from goals.
    Uses long-term memory to retrieve similar past solutions.
    """
    
    SYSTEM_PROMPT = """You are an expert coding agent planner. 
Given a coding goal, break it down into clear, actionable steps.

Your response must be valid JSON in this exact format:
{
    "steps": ["step 1 description", "step 2 description", ...],
    "test_cases": ["test case 1", "test case 2", ...],
    "language": "python",
    "dependencies": ["package1", "package2"],
    "estimated_complexity": "low|medium|high",
    "project_type": "general|fastapi|python_package|cli_tool|web_app",
    "use_multi_file": true|false
}

Guidelines:
- Steps should be specific and actionable
- Test cases should cover main functionality and edge cases
- Language should be one of: python, javascript
- Dependencies should be standard pip/npm packages only
- Set "use_multi_file" to true if the task would benefit from multiple files (e.g. web apps, libraries, CLI tools with multiple modules).
- Choose an appropriate "project_type" when use_multi_file is true."""

    def __init__(
        self,
        llm: LLMClient,
        memory: Optional[LongTermMemory] = None
    ):
        self.llm = llm
        self.memory = memory
    
    async def create_plan(
        self,
        goal: str,
        similar_solutions: List[Dict[str, Any]] = None,
        relevant_learnings: List[Dict[str, Any]] = None,
        relevant_predictions: List[Dict[str, Any]] = None,
    ) -> Plan:
        """
        Create a plan for achieving the goal.

        Args:
            goal: The coding task to accomplish
            similar_solutions: Optional list of similar past solutions (patterns)
            relevant_learnings: Optional list of structured Learnings retrieved
                from long-term memory (Phase B/C). Surfaced verbatim so the
                planner can apply concrete, transferable advice.
            relevant_predictions: Optional list of Predictions (Track D) emitted
                on similar past failures. Each is a concrete adversarial input
                paired with the exception type it should trigger if mishandled —
                actionable signal that tells the planner what corner cases to
                cover defensively.

        Returns:
            Plan object with steps and test cases
        """
        # Build prompt with context from similar solutions
        prompt_parts = [f"Goal: {goal}\n"]

        # Predictions go first — they're the most actionable: concrete corner
        # cases that broke similar past code. The model should see them before
        # the more abstract "lessons" or example-approach sections.
        if relevant_predictions:
            prompt_parts.append(
                "\nKnown failure modes from similar past tasks. Cover each "
                "of these defensively in the plan:"
            )
            for pred in relevant_predictions[:5]:
                trig = pred.get("trigger_input", "?")
                err = pred.get("predicted_error_type", "?")
                why = (pred.get("predicted_explanation") or "").strip()
                tail = f" — {why}" if why else ""
                prompt_parts.append(f"- input {trig} should NOT raise {err}{tail}")

        if relevant_learnings:
            prompt_parts.append("\nRelevant lessons from past successful tasks:")
            for i, learning in enumerate(relevant_learnings[:5], 1):
                prompt_parts.append(f"- {learning.get('lesson', '').strip()}")

        if similar_solutions:
            prompt_parts.append("\nHere are similar problems that were solved successfully:")
            for i, sol in enumerate(similar_solutions[:2], 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Goal: {sol['goal']}")
                prompt_parts.append(f"Approach: {' -> '.join(sol['plan']['steps'][:3])}")

        prompt_parts.append("\nCreate a detailed plan to implement this.")
        prompt = "\n".join(prompt_parts)

        # Get LLM response
        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.7
        )

        # Parse JSON response
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            data = self._extract_json_from_markdown(response)

        return Plan(
            goal=goal,
            steps=data.get("steps", []),
            test_cases=data.get("test_cases", []),
            language=data.get("language", "python"),
            dependencies=data.get("dependencies", []),
            context={
                "estimated_complexity": data.get("estimated_complexity", "medium"),
                "similar_examples_used": len(similar_solutions) if similar_solutions else 0,
                "learnings_surfaced": len(relevant_learnings) if relevant_learnings else 0,
                "predictions_surfaced": len(relevant_predictions) if relevant_predictions else 0,
            },
            project_type=data.get("project_type", "general"),
            use_multi_file=data.get("use_multi_file", False),
        )
    
    def _extract_json_from_markdown(self, text: str) -> Dict[str, Any]:
        """Extract JSON from markdown code blocks."""
        import re
        
        # Look for JSON in code blocks
        pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON object directly
        pattern = r'\{[\s\S]*\}'
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Return default if parsing fails
        return {
            "steps": ["Implement the requested functionality"],
            "test_cases": ["Test the main functionality"],
            "language": "python",
            "dependencies": []
        }
    
    async def refine_plan(
        self,
        plan: Plan,
        reflection: str,
        previous_attempts: int
    ) -> Plan:
        """
        Refine a plan based on reflection from previous attempts.
        
        Args:
            plan: The original plan
            reflection: Reflection text with what went wrong
            previous_attempts: Number of previous attempts
        
        Returns:
            Refined Plan
        """
        prompt = f"""The previous plan didn't work. Please refine it.

Original Goal: {plan.goal}

Previous Plan Steps:
{chr(10).join(f"- {s}" for s in plan.steps)}

Test Cases:
{chr(10).join(f"- {t}" for t in plan.test_cases)}

What went wrong (from reflection):
{reflection}

Number of previous attempts: {previous_attempts}

Please provide an improved plan that addresses the issues.
"""
        
        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.8  # Slightly higher for creativity in fixing
        )
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = self._extract_json_from_markdown(response)
        
        return Plan(
            goal=plan.goal,
            steps=data.get("steps", plan.steps),
            test_cases=data.get("test_cases", plan.test_cases),
            language=data.get("language", plan.language),
            dependencies=data.get("dependencies", plan.dependencies),
            context={
                **plan.context,
                "refined_from": plan.to_dict(),
                "refinement_attempt": previous_attempts + 1
            }
        )
