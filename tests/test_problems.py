"""Contract tests for the Track D benchmark problem set.

These guard the invariants the replay engine depends on — they validate the
*spec design*, not any model output. If they fail, predictions emitted on these
problems would classify Off-topic and produce no calibration signal.
"""

import keyword
import re

import pytest

from bench.problems import PROBLEMS, BANNED_KEYWORDS, ProblemSpec, problems_by_category


def test_problem_set_size():
    # NEXT_STEPS asks for 30-50 deterministic problems.
    assert 30 <= len(PROBLEMS) <= 50


def test_ids_are_unique():
    ids = [p.id for p in PROBLEMS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.id)
def test_function_name_is_valid_identifier(p: ProblemSpec):
    assert p.function_name.isidentifier()
    assert not keyword.iskeyword(p.function_name)
    # Public entry point: select_entry_point ignores underscore-prefixed names.
    assert not p.function_name.startswith("_")


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.id)
def test_goal_names_the_single_function(p: ProblemSpec):
    # The goal must reference the declared public function name so the model
    # writes exactly that entry point and select_entry_point can find it.
    assert p.function_name in p.goal


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.id)
def test_goal_has_no_web_or_server_keywords(p: ProblemSpec):
    words = set(re.findall(r"[a-z]+", p.goal.lower()))
    hits = [kw for kw in BANNED_KEYWORDS if kw in words]
    assert not hits, f"{p.id} mentions banned keyword(s): {hits}"


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.id)
def test_categories_are_known(p: ProblemSpec):
    assert p.category in {"string", "list", "dict", "math", "parsing"}


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.id)
def test_adversarial_inputs_are_python_literals(p: ProblemSpec):
    import ast
    assert p.adversarial_inputs, f"{p.id} has no adversarial inputs"
    for lit in p.adversarial_inputs:
        # Must parse as a literal — these mirror the replay engine's
        # ast.literal_eval contract for trigger inputs.
        ast.literal_eval(lit)


def test_problems_by_category_covers_all():
    grouped = problems_by_category()
    assert sum(len(v) for v in grouped.values()) == len(PROBLEMS)
    # Every declared category should have at least a few problems.
    for cat, items in grouped.items():
        assert len(items) >= 3, f"category {cat} is thin ({len(items)})"
