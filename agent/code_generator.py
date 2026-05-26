"""
Code Generator: Generates code based on plans.
"""

import re
from typing import Optional, Protocol

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

Critical rules:
- NEVER put shell commands (such as `pip install`, `apt-get`, etc.) inside Python code files.
- Dependencies must be declared in a `requirements.txt` file instead of being installed from within the code.
- Keep Python files focused only on application logic.

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
        
        # If the plan requests multi-file output, delegate to the new generator
        if getattr(plan, "use_multi_file", False):
            files = await self.generate_files(plan, previous_attempt, error_feedback)
            # For backward compatibility, return the first file as the main CodeArtifact
            if files:
                first_file = next(iter(files))
                return CodeArtifact(
                    source=files[first_file],
                    file_path=first_file,
                    language=plan.language,
                    metadata={
                        "is_fix_attempt": previous_attempt is not None,
                        "has_error_feedback": error_feedback is not None,
                        "generated_files": list(files.keys()),
                    }
                )

        # Extract code from response (single file path)
        source = self._extract_code(response)

        # Sanitize: Remove accidental shell commands (e.g. "pip install ...") from the code
        source = self._sanitize_code(source)

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

    async def generate_files(
        self,
        plan: Plan,
        previous_attempt: Optional[str] = None,
        error_feedback: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Generate multiple files for a project (Phase 2+).

        Uses project_type from the plan to apply basic scaffolding when available.
        Returns a dictionary of {filename: content}.
        """
        # Decide whether to use scaffolding
        scaffold = self._get_scaffold(plan.project_type) if plan.use_multi_file else None

        prompt = self._build_multi_file_prompt(plan, previous_attempt, error_feedback, scaffold)

        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.6 if not previous_attempt else 0.75,
        )

        files = self._extract_multiple_files(response)

        if not files:
            # Fallback to single file behavior
            single_source = self._extract_code(response)
            ext = self._get_extension(plan.language)
            files = {f"main{ext}": single_source}

        # If we have a scaffold and the LLM didn't provide certain core files, we can merge
        if scaffold:
            for filename, content in scaffold.items():
                if filename not in files:
                    files[filename] = content

        # Automatically add requirements.txt if the plan has dependencies and it's missing
        if plan.dependencies and "requirements.txt" not in files:
            files["requirements.txt"] = "\n".join(plan.dependencies) + "\n"

        return files

    def _build_multi_file_prompt(
        self,
        plan: Plan,
        previous_attempt: Optional[str],
        error_feedback: Optional[str],
        scaffold: Optional[dict[str, str]] = None,
    ) -> str:
        base = f"""You are an expert software engineer building a real project.

Task: {plan.goal}

Project Type: {plan.project_type}
You should generate multiple files: {plan.use_multi_file}

Steps:
{chr(10).join(f"- {s}" for s in plan.steps)}

Critical Rules:
- NEVER include shell commands (e.g. `pip install fastapi`, `apt-get`, etc.) inside any .py file. 
  These commands must go into `requirements.txt` instead.
- Keep Python files focused only on application logic.
- Use a clean project structure appropriate for the project type.
"""

        if scaffold:
            base += "\nUse the following starter files as a base (you may extend them):\n"
            for filename, content in scaffold.items():
                base += f"\n```{filename}\n{content}\n```\n"

        base += """
Return the complete set of files using this exact format:

```filename.ext
file content here
```

```another_file.py
more content here
```

Only output files in the format above. Do not include any explanations or text outside of the code blocks.
"""

        if previous_attempt:
            base += f"\n\nPrevious attempt had issues:\n{previous_attempt}"

        if error_feedback:
            base += f"\n\nError feedback from previous attempt:\n{error_feedback}"

        return base

    def _extract_multiple_files(self, text: str) -> dict[str, str]:
        """
        Extract multiple files from a response using ```filename ... ``` blocks.
        More robust than simple regex — handles language specifiers and various formats.
        """
        files = {}

        # Pattern 1: ```filename or ```language filename
        pattern1 = r"```(?:\w+)?\s*([^\s`]+\.[^\s`]+)\s*\n(.*?)(?=\n```|\Z)"
        matches1 = re.findall(pattern1, text, re.DOTALL)

        for filename, content in matches1:
            filename = filename.strip()
            if filename and content.strip():
                files[filename] = content.strip()

        # Pattern 2: Fallback for ```filename (without extension requirement, more permissive)
        if not files:
            pattern2 = r"```([^\n`]+)\s*\n(.*?)\n```"
            matches2 = re.findall(pattern2, text, re.DOTALL)
            for filename, content in matches2:
                filename = filename.strip()
                # Skip if it looks like just a language (e.g. python, javascript)
                if filename.lower() in {"python", "py", "javascript", "js", "typescript", "ts", "json", "yaml", "yml", "markdown", "md"}:
                    continue
                if filename and content.strip():
                    files[filename] = content.strip()

        return files

    def _get_extension(self, language: str) -> str:
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
        }
        return extensions.get(language, ".txt")

    # ------------------------------------------------------------------
    # Basic Scaffolding (Phase 2)
    # ------------------------------------------------------------------

    SCAFFOLDS: dict[str, dict[str, str]] = {
        "fastapi": {
            "main.py": '''from fastapi import FastAPI

app = FastAPI(title="My API")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
''',
            "requirements.txt": "fastapi\nuvicorn[standard]\n",
            "README.md": "# FastAPI Project\n\nRun with:\n\n    uvicorn main:app --reload\n",
        },
        "python_package": {
            "src/__init__.py": '"""My Package."""\n\n__version__ = "0.1.0"\n',
            "src/main.py": '''def hello(name: str = "World") -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


def main() -> None:
    print(hello())


if __name__ == "__main__":
    main()
''',
            "pyproject.toml": '''[project]
name = "my-package"
version = "0.1.0"
description = "A sample Python package"
readme = "README.md"
requires-python = ">=3.8"
dependencies = []

[project.scripts]
my-cli = "my_package.main:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
''',
            "README.md": '''# My Package

## Installation

```bash
pip install -e .
```

## Usage

```python
from my_package.main import hello
print(hello())
```
''',
            "tests/test_main.py": '''from my_package.main import hello

def test_hello_default():
    assert hello() == "Hello, World!"


def test_hello_custom():
    assert hello("Alice") == "Hello, Alice!"
''',
        },
        "cli_tool": {
            "main.py": '''import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="A simple CLI tool")
    parser.add_argument("--name", "-n", default="World", help="Name to greet")
    args = parser.parse_args()
    print(f"Hello, {args.name}!")


if __name__ == "__main__":
    main()
''',
            "requirements.txt": "",
            "README.md": '''# CLI Tool

## Installation

```bash
pip install -e .
```

## Usage

```bash
my-cli --name Alice
```
''',
        },
    }

    def _get_scaffold(self, project_type: str) -> dict[str, str] | None:
        """Return a basic scaffold for the given project type, if available."""
        return self.SCAFFOLDS.get(project_type)

    def _sanitize_code(self, code: str) -> str:
        """
        Remove common shell commands that the LLM sometimes mistakenly includes
        inside Python code (e.g. "pip install ...").
        This is a safety net in addition to strong prompting.
        """
        lines = code.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip().lower()
            # Common patterns that should not be in Python source files
            if stripped.startswith("pip install") or \
               stripped.startswith("pip3 install") or \
               stripped.startswith("apt-get") or \
               stripped.startswith("apt install") or \
               stripped.startswith("npm install") or \
               stripped.startswith("yarn add"):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
    
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
