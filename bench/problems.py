"""
Benchmark problem set for Track D phase 3 — calibrated self-prediction.

These are deterministic, small, single-file Python problems. They exist to
accumulate prediction → replay verdicts at scale, so the design is constrained
by what the replay engine (agent/replay.py) can score:

  - **Exactly one public function per problem.** `select_entry_point` picks the
    sole public top-level function (or, if several, the one whose name matches
    the goal tokens). Ambiguous entry points classify Off-topic and yield no
    calibration signal — so every goal here names ONE function explicitly, and
    `function_name` records the name we expect the model to write.
  - **No web / server tasks.** The local 7B model can't reliably author a frozen
    pytest suite for FastAPI etc. Stick to string / list / dict algorithms,
    math, and parsing — domains with obvious adversarial inputs (`[]`, `-1`,
    `''`, `None`, `0`) the Reflector can predict.

`adversarial_inputs` are Python-literal strings the Reflector is *likely* to
predict as failure triggers; they document the antipattern each problem is meant
to surface (they are not fed in directly — the agent emits its own predictions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Keywords that signal a web/server/IO problem the benchmark must avoid.
BANNED_KEYWORDS = (
    "fastapi", "flask", "django", "uvicorn", "server", "endpoint", "http",
    "request", "socket", "asyncio", "api", "url", "download", "scrape",
)


@dataclass
class ProblemSpec:
    id: str
    goal: str                          # phrased so the public fn name echoes the goal
    category: str                      # "string" | "list" | "dict" | "math" | "parsing"
    function_name: str                 # the single public entry point the goal implies
    adversarial_inputs: List[str] = field(default_factory=list)  # literals likely to trigger failures


def _g(fn: str, signature: str, behavior: str) -> str:
    """Compose a goal string that names exactly one public function and asks for
    a frozen pytest suite — the contract the replay engine depends on."""
    return (
        f"Write a single Python function {signature}. {behavior} "
        f"Define exactly one public function named `{fn}` (any helpers must start "
        f"with an underscore). Use only the standard library. Include pytest tests."
    )


PROBLEMS: List[ProblemSpec] = [
    # ----------------------------------------------------------------- string
    ProblemSpec(
        "str-01-reverse", _g(
            "reverse_words", "reverse_words(s: str) -> str",
            "Return the string with the order of whitespace-separated words reversed, "
            "collapsing runs of whitespace to a single space and stripping ends.",
        ), "string", "reverse_words", ["''", "'   '", "None"],
    ),
    ProblemSpec(
        "str-02-vowels", _g(
            "count_vowels", "count_vowels(s: str) -> int",
            "Return the number of vowels (a, e, i, o, u, case-insensitive) in the string.",
        ), "string", "count_vowels", ["''", "None", "'AEIOU'"],
    ),
    ProblemSpec(
        "str-03-palindrome", _g(
            "is_palindrome", "is_palindrome(s: str) -> bool",
            "Return True if the string is a palindrome ignoring case and non-alphanumeric "
            "characters, else False.",
        ), "string", "is_palindrome", ["''", "None", "'A man a plan'"],
    ),
    ProblemSpec(
        "str-04-first-unique", _g(
            "first_unique_char", "first_unique_char(s: str) -> int",
            "Return the index of the first non-repeating character, or -1 if there is none.",
        ), "string", "first_unique_char", ["''", "'aabb'", "None"],
    ),
    ProblemSpec(
        "str-05-title-case", _g(
            "to_title_case", "to_title_case(s: str) -> str",
            "Capitalize the first letter of each whitespace-separated word and lowercase "
            "the rest.",
        ), "string", "to_title_case", ["''", "None", "'   '"],
    ),
    ProblemSpec(
        "str-06-run-length", _g(
            "run_length_encode", "run_length_encode(s: str) -> str",
            "Run-length encode the string, e.g. 'aaabb' -> 'a3b2'. Single chars stay as 'x1'.",
        ), "string", "run_length_encode", ["''", "None", "'a'"],
    ),
    ProblemSpec(
        "str-07-anagram", _g(
            "are_anagrams", "are_anagrams(a: str, b: str) -> bool",
            "Return True if the two strings are anagrams ignoring case and spaces.",
        ), "string", "are_anagrams", ["('', '')", "(None, 'a')", "('a', '')"],
    ),
    ProblemSpec(
        "str-08-caesar", _g(
            "caesar_cipher", "caesar_cipher(s: str, shift: int) -> str",
            "Shift each ASCII letter by `shift` positions wrapping within its case; "
            "leave non-letters unchanged.",
        ), "string", "caesar_cipher", ["('', 3)", "('abc', -1)", "(None, 0)"],
    ),

    # ------------------------------------------------------------------- list
    ProblemSpec(
        "lst-01-max", _g(
            "find_max", "find_max(nums: list[int]) -> int",
            "Return the largest number in the list.",
        ), "list", "find_max", ["[]", "None", "[-1]"],
    ),
    ProblemSpec(
        "lst-02-second-largest", _g(
            "second_largest", "second_largest(nums: list[int]) -> int",
            "Return the second largest distinct value in the list.",
        ), "list", "second_largest", ["[]", "[1]", "[1, 1]"],
    ),
    ProblemSpec(
        "lst-03-dedupe", _g(
            "dedupe_preserve_order", "dedupe_preserve_order(items: list) -> list",
            "Return the list with duplicates removed, preserving first-seen order.",
        ), "list", "dedupe_preserve_order", ["[]", "None", "[1, 1, 1]"],
    ),
    ProblemSpec(
        "lst-04-flatten", _g(
            "flatten", "flatten(nested: list) -> list",
            "Flatten an arbitrarily nested list of lists into a single flat list.",
        ), "list", "flatten", ["[]", "None", "[[]]"],
    ),
    ProblemSpec(
        "lst-05-chunk", _g(
            "chunk", "chunk(items: list, size: int) -> list[list]",
            "Split the list into consecutive chunks of length `size`; the last chunk "
            "may be shorter.",
        ), "list", "chunk", ["([], 2)", "([1, 2, 3], 0)", "([1], -1)"],
    ),
    ProblemSpec(
        "lst-06-rotate", _g(
            "rotate_left", "rotate_left(items: list, n: int) -> list",
            "Rotate the list left by `n` positions (n may exceed the length).",
        ), "list", "rotate_left", ["([], 3)", "([1, 2], -1)", "None"],
    ),
    ProblemSpec(
        "lst-07-running-sum", _g(
            "running_sum", "running_sum(nums: list[int]) -> list[int]",
            "Return the running cumulative sum of the list.",
        ), "list", "running_sum", ["[]", "None", "[-1, -1]"],
    ),
    ProblemSpec(
        "lst-08-pairs-sum", _g(
            "two_sum_indices", "two_sum_indices(nums: list[int], target: int) -> tuple[int, int]",
            "Return the indices of the two numbers that add up to `target`, or (-1, -1) "
            "if none.",
        ), "list", "two_sum_indices", ["([], 0)", "([1], 2)", "None"],
    ),
    ProblemSpec(
        "lst-09-merge-sorted", _g(
            "merge_sorted", "merge_sorted(a: list[int], b: list[int]) -> list[int]",
            "Merge two already-sorted lists into one sorted list.",
        ), "list", "merge_sorted", ["([], [])", "(None, [1])", "([1], None)"],
    ),
    ProblemSpec(
        "lst-10-binary-search", _g(
            "binary_search", "binary_search(sorted_nums: list[int], target: int) -> int",
            "Return the index of `target` in a sorted list via binary search, or -1 if absent.",
        ), "list", "binary_search", ["([], 1)", "([1, 2, 3], 4)", "None"],
    ),

    # ------------------------------------------------------------------- dict
    ProblemSpec(
        "dct-01-word-count", _g(
            "word_frequencies", "word_frequencies(text: str) -> dict[str, int]",
            "Return a dict mapping each lowercase word to its count in the text.",
        ), "dict", "word_frequencies", ["''", "None", "'   '"],
    ),
    ProblemSpec(
        "dct-02-invert", _g(
            "invert_dict", "invert_dict(d: dict) -> dict",
            "Return a new dict mapping values back to keys (assume values are unique).",
        ), "dict", "invert_dict", ["{}", "None", "{'a': 1}"],
    ),
    ProblemSpec(
        "dct-03-group-by-parity", _g(
            "group_by_parity", "group_by_parity(nums: list[int]) -> dict[str, list[int]]",
            "Return a dict with keys 'even' and 'odd' mapping to the numbers of each parity.",
        ), "dict", "group_by_parity", ["[]", "None", "[0]"],
    ),
    ProblemSpec(
        "dct-04-merge-add", _g(
            "merge_add", "merge_add(a: dict, b: dict) -> dict",
            "Merge two dicts of int values, summing values for shared keys.",
        ), "dict", "merge_add", ["({}, {})", "(None, {'a': 1})", "({'a': 1}, None)"],
    ),
    ProblemSpec(
        "dct-05-most-common", _g(
            "most_common_key", "most_common_key(items: list) -> object",
            "Return the element that appears most often; ties broken by first appearance.",
        ), "dict", "most_common_key", ["[]", "None", "[1, 2]"],
    ),
    ProblemSpec(
        "dct-06-char-histogram", _g(
            "char_histogram", "char_histogram(s: str) -> dict[str, int]",
            "Return a dict mapping each character to the number of times it occurs.",
        ), "dict", "char_histogram", ["''", "None", "'aa'"],
    ),
    ProblemSpec(
        "dct-07-deep-get", _g(
            "deep_get", "deep_get(d: dict, path: str) -> object",
            "Given a dict and a dotted path like 'a.b.c', return the nested value or None "
            "if any key is missing.",
        ), "dict", "deep_get", ["({}, 'a')", "(None, 'a')", "({'a': 1}, '')"],
    ),
    ProblemSpec(
        "dct-08-count-by-key", _g(
            "count_by_first_letter", "count_by_first_letter(words: list[str]) -> dict[str, int]",
            "Return a dict mapping each first letter to how many words start with it.",
        ), "dict", "count_by_first_letter", ["[]", "None", "['']"],
    ),

    # ------------------------------------------------------------------- math
    ProblemSpec(
        "mth-01-factorial", _g(
            "factorial", "factorial(n: int) -> int",
            "Return n! for a non-negative integer; raise ValueError for negative input.",
        ), "math", "factorial", ["-1", "0", "None"],
    ),
    ProblemSpec(
        "mth-02-fib", _g(
            "fibonacci", "fibonacci(n: int) -> int",
            "Return the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
        ), "math", "fibonacci", ["-1", "0", "None"],
    ),
    ProblemSpec(
        "mth-03-is-prime", _g(
            "is_prime", "is_prime(n: int) -> bool",
            "Return True if n is a prime number, else False.",
        ), "math", "is_prime", ["-1", "0", "1"],
    ),
    ProblemSpec(
        "mth-04-gcd", _g(
            "gcd", "gcd(a: int, b: int) -> int",
            "Return the greatest common divisor of two integers.",
        ), "math", "gcd", ["(0, 0)", "(-4, 6)", "None"],
    ),
    ProblemSpec(
        "mth-05-digit-sum", _g(
            "digit_sum", "digit_sum(n: int) -> int",
            "Return the sum of the decimal digits of the absolute value of n.",
        ), "math", "digit_sum", ["0", "-123", "None"],
    ),
    ProblemSpec(
        "mth-06-to-binary", _g(
            "to_binary", "to_binary(n: int) -> str",
            "Return the binary representation of a non-negative integer without a '0b' "
            "prefix; raise ValueError for negatives.",
        ), "math", "to_binary", ["0", "-1", "None"],
    ),
    ProblemSpec(
        "mth-07-mean", _g(
            "mean", "mean(nums: list[float]) -> float",
            "Return the arithmetic mean of the numbers; raise ValueError on an empty list.",
        ), "math", "mean", ["[]", "None", "[0]"],
    ),
    ProblemSpec(
        "mth-08-clamp", _g(
            "clamp", "clamp(value: float, low: float, high: float) -> float",
            "Clamp `value` into the inclusive range [low, high].",
        ), "math", "clamp", ["(5, 10, 0)", "(None, 0, 1)", "(5, 0, 10)"],
    ),

    # ---------------------------------------------------------------- parsing
    ProblemSpec(
        "prs-01-int-list", _g(
            "parse_int_list", "parse_int_list(s: str) -> list[int]",
            "Parse a comma-separated string of integers into a list of ints, ignoring "
            "surrounding whitespace.",
        ), "parsing", "parse_int_list", ["''", "None", "'a,b'"],
    ),
    ProblemSpec(
        "prs-02-kv", _g(
            "parse_key_values", "parse_key_values(s: str) -> dict[str, str]",
            "Parse a string of 'k=v' pairs separated by semicolons into a dict.",
        ), "parsing", "parse_key_values", ["''", "None", "'a'"],
    ),
    ProblemSpec(
        "prs-03-roman", _g(
            "roman_to_int", "roman_to_int(s: str) -> int",
            "Convert a Roman numeral string to its integer value.",
        ), "parsing", "roman_to_int", ["''", "None", "'IIII'"],
    ),
    ProblemSpec(
        "prs-04-version", _g(
            "compare_versions", "compare_versions(a: str, b: str) -> int",
            "Compare two dotted version strings; return -1, 0, or 1.",
        ), "parsing", "compare_versions", ["('', '')", "(None, '1')", "('1.0', '1')"],
    ),
    ProblemSpec(
        "prs-05-balanced", _g(
            "is_balanced", "is_balanced(s: str) -> bool",
            "Return True if the brackets ()[]{} in the string are correctly balanced.",
        ), "parsing", "is_balanced", ["''", "None", "'('"],
    ),
    ProblemSpec(
        "prs-06-tokenize", _g(
            "tokenize_numbers", "tokenize_numbers(s: str) -> list[float]",
            "Extract all numbers (integers and decimals) from the string as floats.",
        ), "parsing", "tokenize_numbers", ["''", "None", "'abc'"],
    ),
]


def problems_by_category() -> dict:
    """Group the problem set by category — handy for reporting/sampling."""
    out: dict = {}
    for p in PROBLEMS:
        out.setdefault(p.category, []).append(p)
    return out
