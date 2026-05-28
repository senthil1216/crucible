# Changelog

All notable changes to Crucible. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-28

First tagged release. Closes Track A of `docs/NEXT_STEPS.md` — a set of
mechanical cleanups that brings the tree to a shippable state on top of
the merged Phase 0–D work.

### Added
- `FailureMemory.mark_fixed(failure_id, fix_diff)` — marks a stored
  failure as resolved by a subsequent successful iteration; stores the
  broken→fixed unified diff.
- `Reflection.failure_id` — surfaces the just-stored failure-memory ID
  so the loop can mark it `was_fixed=True` on the next success.
- Loop wiring: after an iteration succeeds, the prior iteration's
  failure (if any) is marked fixed and gets a small retrieval boost
  on future `find_similar_failures` calls.
- `tests/test_memory.py::TestErrorSignatureNormalize` — 4 cases covering
  identifier preservation, address/path collapsing, and small-number
  preservation.
- `SECURITY` note at the top of `agent/executor/sandbox.py` and an
  inline caveat in `agent/safety/checker.py` explaining the static-AST
  limitations (aliasing, getattr, dynamic imports).

### Changed
- `ErrorSignature.normalize` now strips only memory addresses, absolute
  paths, object reprs, and runs of ≥5 digits — instead of collapsing
  every lowercase identifier to `{var}`. The previous behavior destroyed
  the very signal the `error_key` grouping was meant to capture.
- `agent/executor/sandbox.py::_execute_python` and `_execute_javascript`
  switched from blocking `subprocess.run` to
  `asyncio.create_subprocess_exec` + `await proc.communicate()` (wrapped
  in `asyncio.wait_for` to preserve timeout semantics). Both methods
  are `async def` and were previously blocking the event loop.
- `agent/tester.py::validate_syntax` switched to async subprocess for
  the JS path for the same reason.

### Removed
- Dead `Reflector._is_hopeless_case` method. The "hopeless" termination
  path is already covered by `reflection.should_continue` checked at
  `loop.py`.

### Notes
- `agent/executor/sandbox.py::_run_pytest_sync` retained `subprocess.run`
  intentionally — it's a synchronous method invoked via
  `asyncio.to_thread(...)`, which is the correct pattern.
- The pre-plan/double-plan item from `NEXT_STEPS.md` is already resolved
  by the existing `loop.py` pass-through path: when `core.solve()`
  computes a memory-enriched plan and hands it to `loop.run(plan=...)`,
  the loop reuses it instead of re-planning.

## Previously merged (no version tags)

These were completed under PR-only versioning, captured here for context.

- **Phase 0** (PR #3): repo reconciled with the original phase2 work;
  test suite collects and runs.
- **Phase 2** (PRs #2, #3): multi-file workspace — `DockerExecutor`
  with persistent containers, `write_file` / `read_file` / `list_dir` /
  `run_command_in_workspace`, multi-file code generation,
  `project_type` / `use_multi_file` on `Plan`.
- **Phase A** (PR #4): richer Pattern memory + embeddings —
  sentence-transformers cosine over `all-MiniLM-L6-v2` for both
  `LongTermMemory` and `FailureMemory`; Pattern carries
  `goal_embedding`, `project_type`, `dependencies`, `language`; lazy
  backfill for legacy entries.
- **Phase B** (PR #6): Reflector writeback — `Learning` model,
  structured extraction on success, `learnings.jsonl` storage.
- **Phase C** (PR #6): multi-signal scoring + planner surfacing —
  `project_type` bonus +0.15, dependency overlap up to +0.10,
  `strict_filters=True` escape hatch; Planner renders up to 5 retrieved
  Learnings into the planning prompt.
- **Phase D** (PR #6): environment context capture —
  `DockerExecutor.capture_environment` (lowercased pip names, no
  versions, shallow workspace ls); stored on the Pattern;
  installed-package overlap bonus up to +0.10.
- **DependencyManager (partial)**: automatic `ModuleNotFoundError`
  recovery, regex-based extraction, import-name→PyPI mapping, max-4
  attempts.
- **PR #8**: real-pytest success gate, xAI provider, app launch flag
  (`--run`), `StepProfiler`, Docker runtime image.
