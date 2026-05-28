"""
Parse pytest-json-report output into a TestResults.

This is the bridge that makes "passed" mean "a real pytest suite actually
passed" rather than "the script exited 0". The single source of truth for the
success rule lives in `build_test_results`:

    passed == (tests_collected > 0
               and tests_failed == 0
               and tests_errors == 0
               and no collection errors)

An empty suite (tests_collected == 0) is never a pass — that is the exact
failure mode the old exit-code gate allowed.
"""

import json
from typing import Optional, List, Dict, Any

from agent.models import TestResults


# pytest exit codes (see `pytest.ExitCode`)
EXIT_OK = 0
EXIT_TESTS_FAILED = 1
EXIT_INTERRUPTED = 2
EXIT_INTERNAL_ERROR = 3
EXIT_USAGE_ERROR = 4
EXIT_NO_TESTS_COLLECTED = 5


_ERROR_PATTERNS = [
    ("ModuleNotFoundError", "modulenotfounderror"),
    ("ImportError", "importerror"),
    ("SyntaxError", "syntaxerror"),
    ("IndentationError", "indentationerror"),
    ("NameError", "nameerror"),
    ("TypeError", "typeerror"),
    ("ValueError", "valueerror"),
    ("KeyError", "keyerror"),
    ("IndexError", "indexerror"),
    ("AttributeError", "attributeerror"),
    ("ZeroDivisionError", "zerodivisionerror"),
    ("RecursionError", "recursionerror"),
    ("TimeoutError", "timeout"),
    ("AssertionError", "assertionerror"),
]


def classify_error(text: str) -> Optional[str]:
    """Best-effort error classification from any failure/stderr text."""
    if not text:
        return None
    lowered = text.lower()
    for error_type, pattern in _ERROR_PATTERNS:
        if pattern in lowered:
            return error_type
    return None


def _longrepr_text(node: Dict[str, Any]) -> str:
    """Pull a human-readable failure message out of a pytest-json-report node."""
    for phase in ("call", "setup", "teardown"):
        section = node.get(phase) or {}
        longrepr = section.get("longrepr") or section.get("crash")
        if longrepr:
            return longrepr if isinstance(longrepr, str) else str(longrepr)
    # Collector errors carry longrepr at the top level
    if node.get("longrepr"):
        lr = node["longrepr"]
        return lr if isinstance(lr, str) else str(lr)
    return ""


def parse_report(report_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the raw report file content into a dict, or None if unusable."""
    if not report_text:
        return None
    try:
        data = json.loads(report_text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def build_test_results(
    report_text: Optional[str],
    stdout: str,
    stderr: str,
    exit_code: int,
    execution_time: float = 0.0,
) -> TestResults:
    """
    Turn a pytest-json-report file (plus raw streams) into a TestResults.

    Falls back to a safe failure when the report is missing or unparseable —
    we never infer success from the exit code alone, because that is precisely
    the hollow signal this whole change exists to remove.
    """
    report = parse_report(report_text)

    if report is None:
        # No machine-readable report. Fail safe with a clear explanation rather
        # than trusting the exit code.
        hint = ""
        if exit_code == EXIT_USAGE_ERROR and "json-report" in (stderr + stdout):
            hint = (
                " (pytest could not find the json-report plugin — "
                "install 'pytest-json-report')"
            )
        return TestResults(
            passed=False,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time=execution_time,
            error_type="ReportParseError",
            warnings=[f"No parseable pytest JSON report{hint}"],
            from_pytest=True,
        )

    summary = report.get("summary", {}) or {}
    tests = report.get("tests", []) or []
    collectors = report.get("collectors", []) or []

    tests_passed = int(summary.get("passed", 0) or 0)
    tests_failed = int(summary.get("failed", 0) or 0)
    tests_errors = int(summary.get("error", 0) or 0)
    # `collected` is the total selected; fall back to len(tests) when absent.
    tests_collected = int(summary.get("collected", summary.get("total", len(tests))) or 0)

    # Per-test failures (failed or errored phases).
    test_failures: List[Dict[str, Any]] = []
    failed_names: List[str] = []
    for node in tests:
        outcome = node.get("outcome")
        if outcome in ("failed", "error"):
            nodeid = node.get("nodeid", "<unknown>")
            failed_names.append(nodeid)
            test_failures.append({
                "nodeid": nodeid,
                "outcome": outcome,
                "message": _longrepr_text(node)[:1500],
            })

    # Collection errors (e.g. the implementation file failed to import). These
    # don't show up under `tests`, so surface them explicitly.
    collection_errors = 0
    for node in collectors:
        if node.get("outcome") == "failed":
            collection_errors += 1
            test_failures.append({
                "nodeid": node.get("nodeid", "<collection>"),
                "outcome": "collection-error",
                "message": _longrepr_text(node)[:1500],
            })

    passed = (
        tests_collected > 0
        and tests_failed == 0
        and tests_errors == 0
        and collection_errors == 0
    )

    # Classify a representative error for the Reflector / dependency recovery.
    error_type = None
    if not passed:
        if tests_collected == 0 and collection_errors == 0:
            error_type = "NoTestsCollected"
        else:
            blob = "\n".join(f.get("message", "") for f in test_failures) + "\n" + stderr
            error_type = classify_error(blob) or "TestFailure"

    return TestResults(
        passed=passed,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        execution_time=execution_time,
        error_type=error_type,
        failed_tests=failed_names,
        tests_collected=tests_collected,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=tests_errors + collection_errors,
        test_failures=test_failures,
        from_pytest=True,
    )
