"""
Tests for the no-plugin pytest report fallback.

When `pytest-json-report` is unavailable, the success gate must still work by
parsing pytest's built-in JUnit XML reporter. These tests pin the JUnit parser
to the *same* success rule as the JSON parser, and verify the
"richest-report-wins" selection in `build_test_results_preferring_json`.
"""

import pytest

from agent.pytest_report import (
    build_test_results_from_junit,
    build_test_results_preferring_json,
    json_report_available,
)


PASS_FAIL_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests"><testsuite name="pytest" errors="0" '
    'failures="1" skipped="0" tests="2" time="0.01">'
    '<testcase classname="tests.test_s" name="test_ok" time="0.0" />'
    '<testcase classname="tests.test_s" name="test_bad" time="0.0">'
    '<failure message="assert -1 == 5">E   assert -1 == 5\n'
    'tests/test_s.py:3: AssertionError</failure>'
    '</testcase></testsuite></testsuites>'
)

COLLECTION_ERROR_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests"><testsuite name="pytest" errors="1" '
    'failures="0" skipped="0" tests="1" time="0.04">'
    '<testcase classname="" name="tests.test_s" time="0.0">'
    '<error message="collection failure">'
    "ImportError: cannot import name 'add' from 'solution'</error>"
    '</testcase></testsuite></testsuites>'
)

NO_TESTS_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests"><testsuite name="pytest" errors="0" '
    'failures="0" skipped="0" tests="0" time="0.0" /></testsuites>'
)

ALL_PASS_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests"><testsuite name="pytest" errors="0" '
    'failures="0" skipped="0" tests="2" time="0.01">'
    '<testcase classname="tests.test_s" name="test_a" time="0.0" />'
    '<testcase classname="tests.test_s" name="test_b" time="0.0" />'
    '</testsuite></testsuites>'
)


class TestJUnitParser:
    def test_all_pass(self):
        r = build_test_results_from_junit(ALL_PASS_XML, "", "", 0)
        assert r is not None
        assert r.passed is True
        assert r.tests_collected == 2
        assert r.tests_passed == 2
        assert r.tests_failed == 0
        assert r.from_pytest is True

    def test_failure_records_detail(self):
        r = build_test_results_from_junit(PASS_FAIL_XML, "", "", 1)
        assert r.passed is False
        assert r.tests_collected == 2
        assert r.tests_failed == 1
        assert r.failed_tests == ["tests.test_s::test_bad"]
        assert r.error_type == "AssertionError"

    def test_collection_error(self):
        r = build_test_results_from_junit(COLLECTION_ERROR_XML, "", "", 2)
        assert r.passed is False
        # The collection placeholder is not a real collected test.
        assert r.tests_collected == 0
        assert r.tests_errors >= 1
        assert r.error_type == "ImportError"

    def test_empty_suite_is_never_a_pass(self):
        r = build_test_results_from_junit(NO_TESTS_XML, "", "", 5)
        assert r.passed is False
        assert r.tests_collected == 0
        assert r.error_type == "NoTestsCollected"

    def test_unparseable_returns_none(self):
        assert build_test_results_from_junit("not xml <<<", "", "", 1) is None
        assert build_test_results_from_junit(None, "", "", 1) is None


class TestPreferJson:
    JSON_PASS = (
        '{"summary": {"passed": 1, "total": 1, "collected": 1}, '
        '"tests": [{"nodeid": "t::a", "outcome": "passed"}]}'
    )

    def test_prefers_json_when_present(self):
        # JSON says pass, JUnit says fail — JSON must win.
        r = build_test_results_preferring_json(
            self.JSON_PASS, PASS_FAIL_XML, "", "", 0
        )
        assert r.passed is True
        assert r.tests_collected == 1

    def test_falls_back_to_junit_when_no_json(self):
        r = build_test_results_preferring_json(None, ALL_PASS_XML, "", "", 0)
        assert r.passed is True
        assert r.tests_collected == 2

    def test_falls_back_to_junit_when_json_unparseable(self):
        r = build_test_results_preferring_json("garbage{", ALL_PASS_XML, "", "", 0)
        assert r.passed is True

    def test_safe_fail_when_neither_report(self):
        r = build_test_results_preferring_json(None, None, "boom", "", 0)
        # No report at all: never infer success from the exit code.
        assert r.passed is False
        assert r.error_type == "ReportParseError"


def test_json_report_available_returns_bool():
    assert isinstance(json_report_available(), bool)
