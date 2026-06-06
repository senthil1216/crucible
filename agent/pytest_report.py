"""
Pytest report parsing for the agent's success gate, with no-plugin fallback.

Executors use `build_test_results_preferring_json` (the public entry point):

- Prefers the rich output from the `pytest-json-report` plugin when available
  (via `json_report_available()`).
- Falls back to parsing pytest's built-in JUnit XML reporter
  (`build_test_results_from_junit`) when the plugin is absent.
- Both the JSON and JUnit paths implement the *identical* success rule
  (so the gate behaves the same on a clean install with no extra packages):

    passed == (tests_collected > 0
               and tests_failed == 0
               and tests_errors == 0
               and collection_errors == 0)

`build_test_results` is the internal JSON-only parser (kept for the preferring
dispatcher and direct use).

An empty or hollow suite is never treated as success — that is the key
property this module exists to enforce (unlike the old "exit code == 0" gate).
"""

import json
import xml.etree.ElementTree as ET
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


def _apply_success_rule(
    tests_collected: int,
    tests_failed: int,
    test_errors: int,
    collection_errors: int,
    test_failures: List[Dict[str, Any]],
    stderr: str,
) -> tuple[bool, Optional[str]]:
    """Apply the canonical success rule and derive error_type.

    This is the single source of truth for:

        passed == (tests_collected > 0 and no failures and no errors and no collection errors)

    Used by both the JSON and JUnit parsers so the gate is identical either way.
    """
    passed = (
        tests_collected > 0
        and tests_failed == 0
        and test_errors == 0
        and collection_errors == 0
    )

    error_type: Optional[str] = None
    if not passed:
        if tests_collected == 0 and collection_errors == 0:
            error_type = "NoTestsCollected"
        else:
            blob = "\n".join(f.get("message", "") for f in test_failures) + "\n" + stderr
            error_type = classify_error(blob) or "TestFailure"

    return passed, error_type


def _build_test_results(
    passed: bool,
    stdout: str,
    stderr: str,
    exit_code: int,
    execution_time: float,
    error_type: Optional[str],
    failed_names: List[str],
    tests_collected: int,
    tests_passed: int,
    tests_failed: int,
    tests_errors: int,
    test_failures: List[Dict[str, Any]],
) -> TestResults:
    """Common TestResults constructor (ensures identical shape from either parser)."""
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
        tests_errors=tests_errors,
        test_failures=test_failures,
        from_pytest=True,
    )


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

    passed, error_type = _apply_success_rule(
        tests_collected=tests_collected,
        tests_failed=tests_failed,
        test_errors=tests_errors,
        collection_errors=collection_errors,
        test_failures=test_failures,
        stderr=stderr,
    )

    return _build_test_results(
        passed=passed,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        execution_time=execution_time,
        error_type=error_type,
        failed_names=failed_names,
        tests_collected=tests_collected,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=tests_errors + collection_errors,
        test_failures=test_failures,
    )


def json_report_available() -> bool:
    """True if the `pytest-json-report` plugin can be imported in this interpreter.

    Used to decide whether to ask pytest for a `--json-report` (the richest
    machine-readable format). When it is unavailable we fall back to pytest's
    built-in JUnit XML reporter, which needs no third-party plugin — see
    `build_test_results_from_junit`. This keeps the success gate working on a
    clean install that hasn't (or can't) install the plugin.
    """
    try:
        import pytest_jsonreport  # noqa: F401
        return True
    except Exception:
        return False


def build_test_results_from_junit(
    xml_text: Optional[str],
    stdout: str,
    stderr: str,
    exit_code: int,
    execution_time: float = 0.0,
) -> Optional[TestResults]:
    """Parse pytest's built-in JUnit XML report into a TestResults.

    This is the no-plugin fallback for `build_test_results`. It applies the exact
    same success rule:

        passed == (tests_collected > 0
                   and no failures
                   and no errors
                   and no collection errors)

    Returns None if the XML is missing or unparseable, so the caller can fall
    back further (to the safe-fail path) rather than inferring success.
    """
    if not xml_text:
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # The root may be <testsuites> wrapping one or more <testsuite>, or a bare
    # <testsuite>. Collect every testcase from whichever shape we got.
    suites = root.findall("testsuite")
    if root.tag == "testsuite":
        suites = [root]

    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0        # collected but not run; not a pass and not a failure
    test_errors = 0          # setup/call/teardown errors on a real test
    collection_errors = 0    # whole module failed to import/collect
    failed_names: List[str] = []
    test_failures: List[Dict[str, Any]] = []

    for suite in suites:
        for case in suite.findall("testcase"):
            classname = (case.get("classname") or "").strip()
            name = case.get("name") or "<unknown>"
            nodeid = f"{classname}::{name}" if classname else name

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                tests_failed += 1
                failed_names.append(nodeid)
                test_failures.append({
                    "nodeid": nodeid,
                    "outcome": "failed",
                    "message": _junit_message(failure)[:1500],
                })
            elif error is not None:
                # pytest emits collection failures as a testcase with an empty
                # classname (the whole module never produced real tests). A
                # non-empty classname means a real test errored in setup/call.
                if classname:
                    test_errors += 1
                    outcome = "error"
                else:
                    collection_errors += 1
                    outcome = "collection-error"
                failed_names.append(nodeid)
                test_failures.append({
                    "nodeid": nodeid,
                    "outcome": outcome,
                    "message": _junit_message(error)[:1500],
                })
            elif skipped is not None:
                tests_skipped += 1
            else:
                tests_passed += 1

    # Collected = every real test selected (passed/failed/errored/skipped),
    # excluding collection-error placeholders. Skipped tests are collected but
    # are neither passes nor failures — this matches the JSON path, which takes
    # `collected` from pytest's summary (skipped included), so the same run is
    # gated identically regardless of plugin availability.
    tests_collected = tests_passed + tests_failed + test_errors + tests_skipped

    passed, error_type = _apply_success_rule(
        tests_collected=tests_collected,
        tests_failed=tests_failed,
        test_errors=test_errors,
        collection_errors=collection_errors,
        test_failures=test_failures,
        stderr=stderr,
    )

    return _build_test_results(
        passed=passed,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        execution_time=execution_time,
        error_type=error_type,
        failed_names=failed_names,
        tests_collected=tests_collected,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=test_errors + collection_errors,
        test_failures=test_failures,
    )


def _junit_message(node: "ET.Element") -> str:
    """Pull a readable message out of a JUnit <failure>/<error> element."""
    text = (node.text or "").strip()
    if text:
        return text
    return (node.get("message") or "").strip()


def build_test_results_preferring_json(
    json_text: Optional[str],
    junit_text: Optional[str],
    stdout: str,
    stderr: str,
    exit_code: int,
    execution_time: float = 0.0,
) -> TestResults:
    """Build a TestResults from whichever report is usable, richest first.

    Order: pytest-json-report → JUnit XML → safe-fail. Executors always request
    the JUnit XML (built-in) and additionally request the JSON report when the
    plugin is present, so there is always at least one parseable report on a
    clean install.
    """
    if parse_report(json_text) is not None:
        return build_test_results(json_text, stdout, stderr, exit_code, execution_time)

    junit = build_test_results_from_junit(
        junit_text, stdout, stderr, exit_code, execution_time
    )
    if junit is not None:
        return junit

    # Neither report was usable — preserve the existing safe-fail behaviour.
    return build_test_results(None, stdout, stderr, exit_code, execution_time)
