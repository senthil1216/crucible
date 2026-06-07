"""
Tests for implementation-only code extraction (`CodeGenerator._extract_code`).

Single-file Python tasks gate on a separately-frozen pytest suite, so the model
must not smuggle its own tests (or a `# requirements.txt` blob) into
`solution.py`. These tests pin the extractor to the failure modes observed in
the benchmark smoke.
"""

import ast

import pytest

from agent.code_generator import CodeGenerator


class _DummyLLM:
    async def complete(self, prompt, system=None, temperature=0.7):
        return ""


@pytest.fixture
def cg():
    return CodeGenerator(_DummyLLM())


def _defs(src):
    """Top-level function/class names defined in src (for assertions)."""
    tree = ast.parse(src)
    return {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


# The exact shape that polluted solution.py in the smoke run.
MULTIFILE_ONE_BLOCK = """```python
# solution.py

def _is_vowel(char):
    return char.lower() in set("aeiou")

def count_vowels(s: str) -> int:
    return sum(_is_vowel(c) for c in s)

# test_solution.py

import pytest
from solution import count_vowels

def test_count_vowels():
    assert count_vowels('hello world') == 3

# requirements.txt
pytest==6.2.5
```"""


class TestImplementationOnlyExtraction:
    def test_single_block_multifile_keeps_only_solution(self, cg):
        out = cg._extract_code(MULTIFILE_ONE_BLOCK)
        names = _defs(out)
        assert "count_vowels" in names
        assert "_is_vowel" in names
        # No embedded tests or requirements leaked into the implementation.
        assert "test_count_vowels" not in names
        assert "import pytest" not in out
        assert "requirements.txt" not in out
        assert "pytest==6.2.5" not in out
        # And what remains is valid Python.
        ast.parse(out)

    def test_separate_blocks_impl_and_tests(self, cg):
        text = (
            "Here is the code:\n"
            "```python\n"
            "def add(a, b):\n    return a + b\n"
            "```\n"
            "And the tests:\n"
            "```python\n"
            "from solution import add\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n"
            "```\n"
        )
        out = cg._extract_code(text)
        assert _defs(out) == {"add"}
        assert "test_add" not in out

    def test_impl_then_requirements_tail(self, cg):
        text = (
            "```python\n"
            "def f(x):\n    return x * 2\n\n"
            "# requirements.txt\n"
            "pytest==6.2.5\n"
            "```"
        )
        out = cg._extract_code(text)
        assert _defs(out) == {"f"}
        assert "pytest==6.2.5" not in out

    def test_no_markers_strips_embedded_test_function(self, cg):
        text = (
            "```python\n"
            "import pytest\n\n"
            "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n\n"
            "def test_reverse_words():\n    assert reverse_words('a b') == 'b a'\n"
            "```"
        )
        out = cg._extract_code(text)
        assert "reverse_words" in _defs(out)
        assert "test_reverse_words" not in _defs(out)
        assert "import pytest" not in out

    def test_plain_single_impl_block_unchanged(self, cg):
        text = "```python\ndef solve(n):\n    return n + 1\n```"
        out = cg._extract_code(text)
        assert out == "def solve(n):\n    return n + 1"

    def test_raw_text_no_fences(self, cg):
        text = "def solve(n):\n    return n + 1\n"
        out = cg._extract_code(text)
        assert _defs(out) == {"solve"}

    def test_does_not_nuke_a_pure_helper_named_testlike(self, cg):
        # A real helper is kept; only top-level test_* / Test* are stripped.
        text = (
            "```python\n"
            "def _test_value():\n    return 1\n\n"
            "def compute():\n    return _test_value()\n"
            "```"
        )
        out = cg._extract_code(text)
        # `_test_value` does not start with `test_`, so it must survive.
        assert _defs(out) == {"_test_value", "compute"}

    def test_solution_section_that_is_pure_tests_is_not_emitted(self, cg):
        # Issue 1: a `# solution.py` section containing only tests must not be
        # emitted verbatim; the real implementation from another section wins.
        text = (
            "```python\n"
            "# solution.py\n"
            "import pytest\n"
            "def test_count():\n    assert count([1]) == 1\n\n"
            "# helper.py\n"
            "def count(items):\n    return len(items)\n"
            "```"
        )
        out = cg._extract_code(text)
        assert "count" in _defs(out)
        assert "test_count" not in _defs(out)
        assert "import pytest" not in out
        ast.parse(out)

    def test_impl_with_only_imports_and_dunder_main_plus_tests(self, cg):
        # Issue 2: "real code" must include non-def/assign top-level statements
        # (imports, module-level expr, `if __name__`). Embedded tests are still
        # stripped even when the impl binds no top-level names via def/assign.
        text = (
            "```python\n"
            "import sys\n"
            "print('warming up')\n"
            "if __name__ == '__main__':\n    sys.exit(0)\n\n"
            "def test_x():\n    assert True\n"
            "```"
        )
        out = cg._extract_code(text)
        assert "test_x" not in out
        assert "import sys" in out
        assert "if __name__" in out
        ast.parse(out)

    def test_crlf_source_with_embedded_test_does_not_crash(self, cg):
        # Issue 3: odd line bookkeeping (CRLF) must not raise; tests are stripped.
        body = (
            "def f(x):\r\n    return x + 1\r\n\r\n"
            "def test_f():\r\n    assert f(1) == 2\r\n"
        )
        text = "```python\r\n" + body + "```"
        out = cg._extract_code(text)
        assert "f" in _defs(out)
        assert "test_f" not in _defs(out)
        ast.parse(out)

    def test_requirements_only_response_does_not_crash(self, cg):
        # Pathological: no implementation at all. Must not raise; output contains
        # no real function definitions to mistake for an impl.
        text = "```\n# requirements.txt\npytest==6.2.5\nrequests>=2\n```"
        out = cg._extract_code(text)  # should not raise
        assert "def " not in out

    def test_tests_only_raw_text_falls_back_without_crashing(self, cg):
        # No fences, no impl — only tests. We can't synthesize an impl, but the
        # call must be graceful and never raise.
        text = "from solution import f\n\ndef test_f():\n    assert f(0) == 0\n"
        out = cg._extract_code(text)  # should not raise
        assert isinstance(out, str)
