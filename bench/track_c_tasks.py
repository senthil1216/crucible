"""
Coherent task batch for Track C — CSV manipulation.

These are intentionally related so that successful patterns and learnings
extracted from earlier tasks have a real chance of helping later ones.
Resist the urge to add unrelated tasks; the whole point is to test
cross-task memory transfer.

Each task is a single-string goal. Keep them tight enough that local
Ollama (qwen2.5-coder:7b) has a reasonable chance of completing them
within ~10 iterations.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TaskSpec:
    id: str
    goal: str
    expected_difficulty: str  # "easy" / "medium" / "hard" — qualitative


TASKS: List[TaskSpec] = [
    TaskSpec(
        id="csv-01-read",
        goal=(
            "Write a Python function read_csv(path: str) -> list[dict] that "
            "reads a CSV file and returns each row as a dict keyed by the "
            "header row. Use the standard csv module. Include pytest tests."
        ),
        expected_difficulty="easy",
    ),
    TaskSpec(
        id="csv-02-write",
        goal=(
            "Write a Python function write_csv(path: str, rows: list[dict]) -> None "
            "that writes a list of dicts to a CSV file, taking column names from "
            "the first row's keys. Use the standard csv module. Include pytest tests."
        ),
        expected_difficulty="easy",
    ),
    TaskSpec(
        id="csv-03-filter",
        goal=(
            "Write a Python function filter_csv(path: str, predicate: callable) -> list[dict] "
            "that reads a CSV file and returns rows where predicate(row) is truthy. "
            "Include pytest tests covering an inclusive and an exclusive predicate."
        ),
        expected_difficulty="easy",
    ),
    TaskSpec(
        id="csv-04-sum-column",
        goal=(
            "Write a Python function sum_column(path: str, column: str) -> float "
            "that reads a CSV and returns the sum of the named column, coercing "
            "values to float. Raise KeyError if the column does not exist. "
            "Include pytest tests."
        ),
        expected_difficulty="medium",
    ),
    TaskSpec(
        id="csv-05-group-by",
        goal=(
            "Write a Python function group_by(path: str, column: str) -> dict[str, list[dict]] "
            "that reads a CSV and returns a dict mapping each unique value in the "
            "given column to the list of rows that contain it. Include pytest tests."
        ),
        expected_difficulty="medium",
    ),
    TaskSpec(
        id="csv-06-join",
        goal=(
            "Write a Python function join_csv(left_path: str, right_path: str, on: str) "
            "-> list[dict] that performs an inner join of two CSV files on the named "
            "column. Each output row should be the merged dict of the matching pair "
            "(right wins on key collisions other than the join key). Include pytest tests."
        ),
        expected_difficulty="hard",
    ),
    TaskSpec(
        id="csv-07-deduplicate",
        goal=(
            "Write a Python function deduplicate(path: str, key_columns: list[str]) "
            "-> list[dict] that reads a CSV and returns rows with unique combinations "
            "of the named columns, keeping the first occurrence. Include pytest tests."
        ),
        expected_difficulty="medium",
    ),
    TaskSpec(
        id="csv-08-pivot",
        goal=(
            "Write a Python function pivot(path: str, index: str, columns: str, values: str) "
            "-> dict[str, dict[str, float]] that reads a CSV and produces a pivot table: "
            "outer dict keyed by index values, inner dict keyed by column values, "
            "containing the values from the values column (sum on duplicates). "
            "Include pytest tests."
        ),
        expected_difficulty="hard",
    ),
]
