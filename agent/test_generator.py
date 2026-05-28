"""
TestGenerator: writes the frozen pytest suite that defines "done".

Test-first contract (single-file Python tasks):
  - The implementation lives in a module named `solution` (file `solution.py`).
  - The suite lives in `tests/test_solution.py` and imports from `solution`.
  - The suite is generated ONCE from the plan, validated, then frozen — the fix
    loop only regenerates the implementation, never the tests.

This module also provides the pure helpers the loop's vacuity gate needs:
`static_check_test_code` (structural sanity) and `build_stub_files` (an empty
implementation the tests should fail against).
"""

import ast
import re
from typing import List, Optional, Protocol, Tuple, Dict

from agent.models import Plan, CodeArtifact


MODULE_NAME = "solution"
TEST_FILE_PATH = f"tests/test_{MODULE_NAME}.py"


class LLMClient(Protocol):
    async def complete(self, prompt: str, system: str = None, temperature: float = 0.7) -> str: ...


class TestGenerator:
    """Generates the frozen pytest suite for a task (test-first)."""

    __test__ = False  # not a pytest test class

    SYSTEM_PROMPT = f"""You are a meticulous test author. Write a pytest test suite.

Hard rules:
- The implementation will live in a module named `{MODULE_NAME}` (file `{MODULE_NAME}.py`).
- Import the things you test from that module, e.g. `from {MODULE_NAME} import my_function`.
- Write a real pytest suite: top-level `def test_*` functions using plain `assert`
  (or `pytest.raises` for expected errors). Do NOT use a `unittest.TestCase` class.
- Each behaviour from the plan's test cases should become at least one assertion.
- The tests must be specific enough that an empty/stub implementation FAILS them.
- Every test MUST be satisfiable by a correct implementation. Do not assert on
  things no code can control.
- For web apps/APIs, exercise endpoints through an IN-PROCESS test client
  (e.g. `from fastapi.testclient import TestClient`). NEVER launch the server as
  a process: no `uvicorn`/`gunicorn`, no `subprocess` to start the app, no
  `app.run()`, and no test that the server "starts", "listens", or "runs" — a
  server never exits, so such a test can never pass.
- Do NOT include the implementation. Tests only.

Respond with ONLY the contents of the test file — no prose, no markdown fences."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate_tests(
        self,
        plan: Plan,
        previous_tests: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> CodeArtifact:
        """Generate (or regenerate) the frozen pytest suite for the plan."""
        prompt_parts = [
            "Write a pytest test suite for the following task.",
            "",
            f"Task: {plan.goal}",
            "",
            f"The implementation module is named `{MODULE_NAME}`.",
            "",
            "Behaviours to cover (turn each into one or more assertions):",
            *[f"- {tc}" for tc in plan.test_cases],
        ]
        if plan.steps:
            prompt_parts += ["", "Implementation steps (context):", *[f"- {s}" for s in plan.steps]]
        if previous_tests and feedback:
            prompt_parts += [
                "",
                "The previous test suite was rejected:",
                feedback,
                "",
                "Previous suite:",
                previous_tests,
                "",
                "Write an improved suite that addresses the feedback.",
            ]
        prompt_parts += ["", "Output only the test file contents:"]

        response = await self.llm.complete(
            system=self.SYSTEM_PROMPT,
            prompt="\n".join(prompt_parts),
            temperature=0.4,
        )

        source = _strip_code_fences(response)
        return CodeArtifact(
            source=source,
            file_path=TEST_FILE_PATH,
            language="python",
            metadata={"module": MODULE_NAME, "frozen": True},
        )


def _strip_code_fences(text: str) -> str:
    """Extract code from a ```...``` block if present, else return stripped text."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_imported_names(test_source: str, module_name: str = MODULE_NAME) -> List[str]:
    """
    Names the test suite pulls from the implementation module.

    Handles `from solution import a, b as c` and attribute access on a bare
    `import solution` (`solution.foo(...)`). Used to synthesize a stub.
    """
    names: List[str] = []
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return names

    imports_module_directly = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            for alias in node.names:
                if alias.name != "*":
                    names.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    imports_module_directly = True

    if imports_module_directly:
        # Collect `solution.<attr>` references.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == module_name
            ):
                names.append(node.attr)

    # De-dup, preserve order.
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def static_check_test_code(
    test_source: str, module_name: str = MODULE_NAME
) -> Tuple[bool, List[str]]:
    """
    Structural sanity check on a generated suite. Returns (ok, reasons).

    A suite that fails this is useless (can't be a meaningful gate), so the loop
    treats a static failure as fatal after retries. We require: parses, imports
    the target module, has >=1 `def test_*`, and contains an assertion.
    """
    reasons: List[str] = []
    try:
        tree = ast.parse(test_source)
    except SyntaxError as e:
        return False, [f"test file does not parse: {e}"]

    imports_module = any(
        (isinstance(n, ast.ImportFrom) and n.module == module_name)
        or (isinstance(n, ast.Import) and any(a.name == module_name for a in n.names))
        for n in ast.walk(tree)
    )
    if not imports_module:
        reasons.append(f"does not import the implementation module `{module_name}`")

    test_funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
    ]
    if not test_funcs:
        reasons.append("no `def test_*` functions found")

    has_assertion = any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    has_raises = "pytest.raises" in test_source or ".raises(" in test_source
    if not (has_assertion or has_raises):
        reasons.append("no `assert` or `pytest.raises` found")

    # Server-launch anti-patterns: a unit test must exercise the app via an
    # in-process client, never by starting the server as a process — those tests
    # are unsatisfiable (a server never exits), so reject and regenerate.
    lowered = test_source.lower()
    if "uvicorn" in lowered or "gunicorn" in lowered:
        reasons.append(
            "tests reference a server runner (uvicorn/gunicorn); exercise the app "
            "via an in-process test client (e.g. TestClient) instead of launching it"
        )
    if "subprocess" in lowered and any(
        s in lowered for s in ("uvicorn", "gunicorn", "app.run", "flask run", "runserver")
    ):
        reasons.append("tests must not run the application as a subprocess (a server never exits)")

    return (len(reasons) == 0), reasons


def build_stub_files(
    test_source: str, module_name: str = MODULE_NAME
) -> Dict[str, str]:
    """
    Build an empty implementation the tests SHOULD fail against.

    Functions return None; class names (CapWords) become bare classes. If the
    tests still pass against this, they assert nothing meaningful (vacuous).
    Only meaningful for the single-file `solution` contract.
    """
    names = extract_imported_names(test_source, module_name)
    lines = ['"""Auto-generated stub for the vacuity check. Returns nothing."""', ""]
    if not names:
        # Nothing to stub specifically — an empty module. Attribute/imports will
        # fail, which is the desired "tests have teeth" outcome.
        return {f"{module_name}.py": "\n".join(lines) + "\n"}

    for name in names:
        if name[:1].isupper():  # looks like a class
            lines.append(f"class {name}:")
            lines.append("    def __init__(self, *args, **kwargs):")
            lines.append("        pass")
            lines.append("")
        else:
            lines.append(f"def {name}(*args, **kwargs):")
            lines.append("    return None")
            lines.append("")

    return {f"{module_name}.py": "\n".join(lines) + "\n"}
