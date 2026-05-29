"""
Replay engine (Track D phase 2): close the prediction feedback loop.

When a task SUCCEEDS, every falsifiable prediction the Reflector emitted during
that task's failed iterations is replayed against the final, passing code:

  - The prediction's `trigger_input` (a Python literal) is fed to the solution's
    public entry point inside the sandbox.
  - The outcome is classified:
      * Confirmed  — the code raised the predicted error type *from inside the
                     solution*. The frozen test suite had a gap; the agent's own
                     prediction caught a real latent bug.
      * Falsified  — the code ran clean, or raised a *different* error from
                     inside the solution. The specific claim did not hold.
      * Off-topic  — we couldn't apply the input at all (trigger isn't a literal,
                     wrong arity / error at the call boundary, ambiguous or
                     missing entry point). NEVER counted toward calibration.
  - Confirmed/Falsified verdicts are written back via
    `PredictionMemory.record_replays`, which bumps the counters that drive
    retrieval ranking and auto-retirement.

Phase 2 is deliberately RECORD-ONLY: replay never blocks a success. It produces
the calibration data the phase-4 analysis needs without contaminating the
benchmark's success metric or acting on still-untrusted predictions. The
loop leaves an inert `predictions_gate_enabled` hook for a future phase that,
once the calibration data justifies it, could re-loop on confirmed bugs.

Design notes:
  - Same-task replay (predictions linked to *this* task's failures, tested
    against *this* task's final code) keeps the function interface stable, which
    is what makes the Off-topic vs Falsified distinction meaningful. Cross-task
    prediction surfacing is a separate, already-shipped concern (the Planner).
  - The driver is appended to the solution source and run through the existing
    `executor.execute()` — no separate import machinery, works on both the
    subprocess sandbox and the Docker executor. It emits a single sentinel JSON
    line so the classification is deterministic, not a regex over stderr.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.models import CodeArtifact


# Printed by the in-sandbox driver on its own line; everything after the
# sentinel on that line is the JSON verdict payload.
SENTINEL = "__CRUCIBLE_REPLAY__"

CONFIRMED = "confirmed"
FALSIFIED = "falsified"
OFF_TOPIC = "off_topic"


@dataclass
class ReplayVerdict:
    """The outcome of replaying one prediction against the passing code."""
    prediction_id: str
    trigger_input: str
    predicted_error_type: str
    classification: str          # CONFIRMED | FALSIFIED | OFF_TOPIC
    detail: str = ""             # human-readable reason
    actual_error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "trigger_input": self.trigger_input,
            "predicted_error_type": self.predicted_error_type,
            "classification": self.classification,
            "detail": self.detail,
            "actual_error_type": self.actual_error_type,
        }


@dataclass
class ReplayReport:
    """All verdicts from one task's success-time replay."""
    verdicts: List[ReplayVerdict] = field(default_factory=list)
    entry_point: Optional[str] = None

    @property
    def confirmed(self) -> int:
        return sum(1 for v in self.verdicts if v.classification == CONFIRMED)

    @property
    def falsified(self) -> int:
        return sum(1 for v in self.verdicts if v.classification == FALSIFIED)

    @property
    def off_topic(self) -> int:
        return sum(1 for v in self.verdicts if v.classification == OFF_TOPIC)

    @property
    def tested(self) -> int:
        """Predictions that produced a real verdict (off-topic excluded)."""
        return self.confirmed + self.falsified

    def summary_line(self) -> str:
        return (
            f"🔁 Replay: {len(self.verdicts)} prediction(s) — "
            f"{self.confirmed} confirmed, {self.falsified} falsified, "
            f"{self.off_topic} off-topic"
            + (f" (entry: {self.entry_point})" if self.entry_point else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "confirmed": self.confirmed,
            "falsified": self.falsified,
            "off_topic": self.off_topic,
            "tested": self.tested,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def select_entry_point(source: str, goal: str = "") -> Tuple[Optional[str], str]:
    """Pick the public top-level function to feed the trigger input to.

    Returns (entry_name, reason). entry_name is None when no single function can
    be chosen confidently — the caller then classifies the prediction Off-topic
    with `reason`, rather than guessing and fabricating a verdict.

    Resolution:
      - exactly one public function   → use it
      - several, one name best-matches the goal tokens (unique winner) → use it
      - zero public functions          → None ("no_public_function")
      - several, ambiguous             → None ("ambiguous_entry_point")
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, "unparseable_source"

    funcs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    if not funcs:
        return None, "no_public_function"
    if len(funcs) == 1:
        return funcs[0], "single_function"

    goal_tok = _tokens(goal)
    best_name: Optional[str] = None
    best_score = 0
    tie = False
    for name in funcs:
        score = len(_tokens(name) & goal_tok)
        if score > best_score:
            best_name, best_score, tie = name, score, False
        elif score == best_score and score > 0:
            tie = True
    if best_score > 0 and not tie:
        return best_name, "goal_token_match"
    return None, "ambiguous_entry_point"


# Driver appended after the solution source. It must NOT rely on
# `__name__ == "__main__"`: the subprocess sandbox runs code via `exec(src, {})`,
# where __name__ is unset. All names are underscore-prefixed to avoid clobbering
# the solution's globals. `_R_SOL_LINES` is the solution's line count so we can
# tell whether an exception came from *inside* the solution (lineno <= that) or
# at our call boundary (lineno greater) — under exec-of-string every frame
# shares the "<string>" filename, so line number is the only discriminator.
_DRIVER_TEMPLATE = '''

import json as _rjson, ast as _rast, traceback as _rtb

_R_SENTINEL = {sentinel!r}
_R_TRIGGER_RAW = {trigger!r}
_R_ENTRY = {entry!r}
_R_SOL_LINES = {sol_lines}


def _r_emit(_d):
    print(_R_SENTINEL + _rjson.dumps(_d))


def _r_run(_call):
    try:
        _call()
        return {{"outcome": "no_error"}}
    except BaseException as _e:
        _tb = _rtb.extract_tb(_e.__traceback__)
        _in_sol = any(getattr(_f, "lineno", 10 ** 9) <= _R_SOL_LINES for _f in _tb)
        return {{
            "outcome": "raised",
            "type": type(_e).__name__,
            "mro": [_c.__name__ for _c in type(_e).__mro__],
            "in_solution": _in_sol,
        }}


def _r_main():
    try:
        _val = _rast.literal_eval(_R_TRIGGER_RAW)
    except Exception:
        _r_emit({{"outcome": "unparseable"}})
        return
    _fn = globals().get(_R_ENTRY)
    if not callable(_fn):
        _r_emit({{"outcome": "no_entry"}})
        return
    _res = _r_run(lambda: _fn(_val))
    # Arity mis-fit at the call boundary + a sequence trigger → retry splatted.
    # Many predictions encode the full argument tuple, not a single argument.
    if (
        _res.get("outcome") == "raised"
        and _res.get("type") == "TypeError"
        and not _res.get("in_solution")
        and isinstance(_val, (list, tuple))
    ):
        _res = _r_run(lambda: _fn(*_val))
    _r_emit(_res)


_r_main()
'''


def _build_combined_source(solution_source: str, entry: str, trigger: str) -> Tuple[str, int]:
    """Solution source + driver. Returns (combined, solution_line_count)."""
    sol_part = solution_source.rstrip("\n") + "\n"
    sol_lines = sol_part.count("\n")
    driver = _DRIVER_TEMPLATE.format(
        sentinel=SENTINEL,
        trigger=trigger,
        entry=entry,
        sol_lines=sol_lines,
    )
    return sol_part + driver, sol_lines


def _parse_sentinel(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the last sentinel JSON payload from captured stdout, tolerating
    any noise the solution itself printed."""
    if not stdout:
        return None
    payload = None
    for line in stdout.splitlines():
        idx = line.find(SENTINEL)
        if idx == -1:
            continue
        try:
            payload = json.loads(line[idx + len(SENTINEL):])
        except json.JSONDecodeError:
            continue
    return payload


def _classify(payload: Optional[Dict[str, Any]], predicted: str) -> Tuple[str, str, Optional[str]]:
    """Map a driver payload to (classification, detail, actual_error_type)."""
    if payload is None:
        return OFF_TOPIC, "no replay signal (exec error or timeout)", None

    outcome = payload.get("outcome")
    if outcome == "unparseable":
        return OFF_TOPIC, "trigger_input is not a Python literal", None
    if outcome == "no_entry":
        return OFF_TOPIC, "entry point not callable in solution", None
    if outcome == "no_error":
        return FALSIFIED, "code ran cleanly on the trigger input", None
    if outcome == "raised":
        actual = payload.get("type")
        if not payload.get("in_solution"):
            # The exception came from our invocation (e.g. wrong arity), not the
            # code under test — we cannot attribute it to the prediction.
            return OFF_TOPIC, f"error at call boundary ({actual})", actual
        mro = payload.get("mro") or [actual]
        if predicted in mro:
            return CONFIRMED, f"raised {actual} as predicted", actual
        return FALSIFIED, f"raised {actual}, not the predicted {predicted}", actual

    return OFF_TOPIC, f"unrecognized driver outcome: {outcome!r}", None


class ReplayEngine:
    """Replays stored predictions against passing code and records verdicts.

    Record-only (Phase 2): produces a ReplayReport and updates PredictionMemory
    counters; it never mutates the task result.
    """

    def __init__(self, executor, prediction_memory):
        # `executor` only needs an async `execute(CodeArtifact) -> TestResults`
        # (both SandboxedExecutor and DockerExecutor satisfy this).
        self.executor = executor
        self.prediction_memory = prediction_memory

    async def replay_for_failures(
        self,
        failure_ids: List[str],
        code_source: str,
        goal: str = "",
    ) -> ReplayReport:
        """Replay every (non-retired) prediction linked to the given failure
        ids against `code_source`, classify, and persist tested verdicts."""
        # Collect unique predictions across all of this task's failures.
        seen: Dict[str, Dict[str, Any]] = {}
        for fid in dict.fromkeys(fid for fid in failure_ids if fid):
            for pred in self.prediction_memory.find_by_failure_id(fid):
                seen[pred["id"]] = pred

        report = ReplayReport()
        if not seen:
            return report

        # Entry point depends on the code, not the individual prediction — pick
        # it once. If we can't, every prediction is Off-topic for the same
        # reason (and we never spend sandbox time).
        entry, entry_reason = select_entry_point(code_source, goal)
        report.entry_point = entry

        outcomes: Dict[str, bool] = {}
        for pid, pred in seen.items():
            trigger = pred.get("trigger_input", "")
            predicted = pred.get("predicted_error_type", "")

            if entry is None:
                verdict = ReplayVerdict(
                    prediction_id=pid,
                    trigger_input=trigger,
                    predicted_error_type=predicted,
                    classification=OFF_TOPIC,
                    detail=entry_reason,
                )
            else:
                classification, detail, actual = await self._replay_one(
                    code_source, entry, trigger, predicted
                )
                verdict = ReplayVerdict(
                    prediction_id=pid,
                    trigger_input=trigger,
                    predicted_error_type=predicted,
                    classification=classification,
                    detail=detail,
                    actual_error_type=actual,
                )

            report.verdicts.append(verdict)
            if verdict.classification in (CONFIRMED, FALSIFIED):
                outcomes[pid] = verdict.classification == CONFIRMED

        # One disk rewrite for the whole batch; also runs auto-retirement.
        if outcomes:
            self.prediction_memory.record_replays(outcomes)

        return report

    async def _replay_one(
        self, code_source: str, entry: str, trigger: str, predicted: str
    ) -> Tuple[str, str, Optional[str]]:
        combined, _ = _build_combined_source(code_source, entry, trigger)
        artifact = CodeArtifact(
            source=combined, file_path="solution.py", language="python"
        )
        try:
            result = await self.executor.execute(artifact)
        except Exception as e:  # sandbox failure — can't conclude anything
            return OFF_TOPIC, f"sandbox error: {e}", None
        payload = _parse_sentinel(getattr(result, "stdout", "") or "")
        return _classify(payload, predicted)
