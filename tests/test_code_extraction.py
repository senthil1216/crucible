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
