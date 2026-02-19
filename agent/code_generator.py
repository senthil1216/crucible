"""
Code Generator: Generates code based on plans.
"""

import re
from typing import Protocol

from agent.models import Plan, CodeArtifact


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str: ...


class CodeGenerator:
    """
    Generates code based on execution plans.
    Incorporates feedback from previous attempts.
    """
    
    SYSTEM_PROMPT = """You are an expert programmer. Write clean, correct, well-tested code.

Your code should:
- Be complete and runnable
- Include all necessary imports
- Handle edge cases
- Include the test cases provided in the plan
- Output results to stdout for verification

Respond with ONLY the code, no explanations or markdown formatting outside the code block."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    async def generate(
        self,
        plan: Plan,
        previous_attempt: str = None,
        error_feedback: str = None,
        similar_solutions: list = None
    ) -> CodeArtifact:
        """
        Generate code based on a plan.
        
        Args:
            plan: The execution plan
            previous_attempt: Previous code attempt (if iterating)
            error_feedback: Error message from previous attempt
            similar_solutions: Similar working solutions for reference
        
        Returns:
            CodeArtifact with generated code
        """
        prompt_parts = [
            f"Task: {plan.goal}",
            "",
            "Implementation Steps:",
            *[f"- {step}" for step in plan.steps],
            "",
            "Test Cases to Include:",
            *[f"- {test}" for test in plan.test_cases],
            "",
            f"Language: {plan.language}",
        ]
        
        if plan.dependencies:
            prompt_parts.extend([
                "",
                "Dependencies:",
                *[f"- {dep}" for dep in plan.dependencies]
            ])
        
        if similar_solutions:
            prompt_parts.extend([
                "",
                "Reference Implementation (similar problem):",
                similar_solutions[0]['code']['source'][:1000] if similar_solutions else ""
            ])
        
        if previous_attempt and error_feedback:
            prompt_parts.extend([
                "",
                "Previous Attempt (failed):",
                "```",
                previous_attempt,
                "```",
                "",
                "Error Feedback:",
                error_feedback,
                "",
                "Please fix the issues and generate corrected code."
            ])
        
        prompt_parts.extend([
            "",
            "Generate complete, runnable code:"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.5 if not previous_attempt else 0.7  # Higher temp for fixes
        )
        
        # Extract code from response
        source = self._extract_code(response)
        
        # Determine file extension
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "go": ".go",
            "rust": ".rs"
        }
        ext = extensions.get(plan.language, ".txt")
        
        return CodeArtifact(
            source=source,
            file_path=f"main{ext}",
            language=plan.language,
            metadata={
                "is_fix_attempt": previous_attempt is not None,
                "has_error_feedback": error_feedback is not None
            }
        )
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks or raw text."""
        # Try to extract from markdown code blocks
        pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return the first (and likely only) code block
            return matches[0].strip()
        
        # No code blocks found, return stripped text
        return text.strip()
    
    async def generate_fix(
        self,
        plan: Plan,
        broken_code: str,
        error_type: str,
        error_message: str,
        reflection: str
    ) -> CodeArtifact:
        """
        Generate a fix for broken code based on reflection.
        
        Args:
            plan: Original plan
            broken_code: The code that failed
            error_type: Type of error (e.g., "SyntaxError")
            error_message: Error message
            reflection: Analysis of what went wrong
        
        Returns:
            Fixed CodeArtifact
        """
        prompt = f"""Fix the following code based on the error analysis.

Original Task: {plan.goal}

Steps:
{chr(10).join(f"- {s}" for s in plan.steps)}

Current Broken Code:
```
{broken_code}
```

Error Type: {error_type}
Error Message: {error_message}

Analysis of what went wrong:
{reflection}

Please generate the corrected code. Address the root cause identified in the analysis.
"""
        
        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.6
        )
        
        source = self._extract_code(response)
        
        extensions = {"python": ".py", "javascript": ".js"}
        ext = extensions.get(plan.language, ".txt")
        
        return CodeArtifact(
            source=source,
            file_path=f"main{ext}",
            language=plan.language,
            metadata={
                "is_fix_attempt": True,
                "original_error": error_type,
                "fix_based_on": reflection[:200]
            }
        )
