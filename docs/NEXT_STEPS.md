# Next Steps

Single source of truth for what's left to do on Crucible. Supersedes the prior
`NEXT_STEPS.md` plus the three phase design docs (long-term-memory,
phase1-dependency-recovery, phase2-workspace) — anything completed has been
removed; anything still open lives here.

## Where we are

Merged work (no further action needed):

- **Phase 0** — repo reconciled with the original phase2 implementation;
  test suite collects and runs. (PR #3)
- **Phase 2 multi-file workspace** — `DockerExecutor` with persistent
  containers, `write_file` / `read_file` / `list_dir` /
  `run_command_in_workspace`, multi-file code generation, `project_type` /
  `use_multi_file` on `Plan`. (PR #2 + #3)
- **Phase A — richer Pattern memory + embeddings** — sentence-transformers
  cosine over `all-MiniLM-L6-v2` for both `LongTermMemory` and
  `FailureMemory`; Pattern carries `goal_embedding`, `project_type`,
  `dependencies`, `language`; lazy backfill for legacy entries. (PR #4)
- **Phase B — Reflector writeback** — `Learning` model, structured
  extraction on success, `learnings.jsonl` storage. (PR #6)
- **Phase C — multi-signal scoring + planner surfacing** — `project_type`
  bonus +0.15, dependency overlap up to +0.10, `strict_filters=True`
  escape hatch; Planner renders up to 5 retrieved Learnings into the
  planning prompt. (PR #6)
- **Phase D — env context capture** — `DockerExecutor.capture_environment`
  (lowercased pip names, no versions, shallow workspace ls); stored on
  the Pattern; installed-package overlap bonus up to +0.10. (PR #6)
- **DependencyManager (partial)** — automatic `ModuleNotFoundError`
  recovery, regex-based extraction, import-name→PyPI mapping, max-4
  attempts. (PR #2/#3)
- **README rewrite** — honest framing, accurate project structure,
  documented Docker/multi-file modes. (PR #6)

## What's open

### Track A — Tidy and ship v0.1.0

Small leftovers from the original `NEXT_STEPS.md` Phase 1 list that have
not been fixed. Mostly mechanical, low-risk, but should land together
before we tag.

- **Wire `was_fixed` correctly** (`agent/memory/failure_memory.py:66,85`,
  `agent/loop.py`). When iteration N+1 succeeds and N failed, mark N's
  failure entry `was_fixed=True` and store the broken→fixed diff. Boost
  `find_similar_failures` for fix-confirmed entries. The "learn from past
  mistakes" thesis isn't actually wired up today.
- **Fix the double-plan call** (`agent/core.py:247` vs
  `agent/loop.py:269`). `core.solve()` calls `planner.create_plan` and
  discards the result; then the loop runs its own planning inside the
  first iteration. Drop the call at `core.py:247` and let the loop own
  planning (or pass the pre-plan through and stop the loop from re-planning).
- **Delete dead `_is_hopeless_case`** (`agent/reflector.py:344`). Defined,
  never called. Either wire it into the loop or remove it.
- **Replace blocking `subprocess.run`** (`agent/executor/sandbox.py:94`
  and `:206`). Both sit inside `async def` methods and block the event
  loop. Switch to `asyncio.create_subprocess_exec`.
- **Fix `ErrorSignature.normalize`** (`agent/models.py:121-131`). The
  regex `\b[a-z_][a-z0-9_]*\b` collapses every lowercase identifier into
  `{var}`, destroying signal. We sidestepped this in `FailureMemory` by
  embedding the *raw* error message, but the normalize method is still
  used for `error_key` grouping and is still bad. Tighten the regex to
  strip only runtime-varying values (memory addresses, specific paths,
  instance reprs); keep type names, line numbers, and identifiers.
- **Add a SECURITY NOTE inside `agent/executor/sandbox.py`** at file top.
  The README is honest about this; the file isn't. One paragraph:
  "Ergonomic isolation, not a security boundary. macOS `RLIMIT_AS` is a
  no-op, AST aliasing bypasses safety checks, subprocess inherits
  filesystem and network capabilities." Mirrors the README.
- **Acknowledge safety-theater in `agent/safety/checker.py`**. The
  `_is_safe_filesystem_call` method at line 153 unconditionally returns
  `False`; the AST walker only inspects direct `ast.Call` nodes, missing
  aliasing (`o = open; o(...)`), import-aliasing
  (`import subprocess as s`), and `getattr` access. Either tighten or
  add an inline note that this is best-effort static analysis.

When the above is done: `pytest` green, brief `CHANGELOG.md`, tag `v0.1.0`.

### Track B — DependencyManager hardening

Pending tasks from the original `phase1-dependency-recovery-design.md`
(the design doc is removed; remaining items captured here).

- **`requirements.txt` support** in `DependencyManager`. The
  `DockerExecutor` already has `install_requirements_file`; expose it via
  `DependencyManager.install_from_requirements(path)` and decide a
  policy (always run on generated `requirements.txt`? on demand only?).
- **Categorize install failure types**. `InstallResult.failure_reason`
  exists but is only set to `"not_persistent"` or `"install_failed"`.
  Parse pip output and classify into `not_found` / `build_error` /
  `network` / `permission` so the Reflector can act differently per
  category.
- **Optional install confirmation**. Add `docker_ask_before_install: bool`
  to `AgentConfig`; when true, pause before `pip install` in
  interactive mode. Mostly UX work; lowest priority.
- **Persist successful installations to long-term memory**. The
  `DependencyManager` docstring promises this and the data fits naturally
  into the Phase D `environment_context` already stored on Patterns.
  Use that path; don't add a separate store.

### Track C — Demo and tiny benchmark

Validates that the Phase A/B/C/D memory infrastructure actually does
what it claims. Cheap, high-signal, and unblocks the publishable track.

- **5–10 related task script**. Pick a coherent batch (e.g. CSV
  manipulation, simple HTTP clients, small data-structure problems).
  Run the agent sequentially against local Ollama with
  `--docker-persistent` so env capture works. Log per-task: iterations to
  success, which Patterns/Learnings were retrieved, score breakdowns.
- **Short report** (in `docs/`) with: per-task iteration counts, a
  qualitative pass on whether retrieved Learnings actually informed the
  plan, any infrastructure bugs surfaced. Not a paper — a 1-page
  sanity-check artifact.
- **Tie-in**: any bug surfaced here gets folded back into Track A.

### Track D — Hypothesis-scored predictions (publishable)

Original `NEXT_STEPS.md` Phase 2. Still the most ambitious track and
still the one most likely to produce a writeup. Untouched.

**Thesis**: the agent learns to predict its own failures. The Reflector
emits falsifiable predictions on each failure; a replay engine tests
stored predictions against new code; predictions are scored and pruned
over time. After 100+ tasks, a calibrated antipattern catalog with real
numbers.

**v1 constraint**: input-based predictions only (a concrete adversarial
input + the failure type it should trigger). Pattern-matching and
conditional predictions deferred.

**Compute model**: local Ollama (`qwen2.5-coder:7b`), no cloud LLMs —
removes cost as a variable in the calibration analysis.

Sketched weekends:

1. **Schema + emission** — `Prediction` dataclass, extend
   `Reflector.SYSTEM_PROMPT` to emit a prediction field on failed
   iterations, strict schema gate (drop predictions without a concrete
   `trigger_input`), store in `agent/memory/predictions.jsonl`.
2. **Replay engine** — given a `Prediction` and a `CodeArtifact`,
   generate a test wrapper that invokes the code with `trigger_input`,
   run through the sandbox, classify as **Confirmed** / **Falsified** /
   **Off-topic**. Wire as a post-test pre-success gate. Scoring +
   pruning: predictions tested >10 times with <30% confirmation rate
   get retired. Reuse the embedding utility for the scoping gate.
3. **Benchmark** — curate 30–50 deterministic small coding problems.
   Build `bench/runner.py` (N=5 default per problem). Smoke run first.
4. **Analysis + writeup** — full run with persisted predictions +
   outcomes. `bench/REPORT.md` with calibration curve, surviving
   predictions catalog, convergence variance. Medium post draft.

### Dismissed (do not re-explore)

- SWE-bench Lite numbers — harness work too heavy for likely mid result.
- Multi-agent decomposition — components are already in separate classes.
- Docker/gVisor sandbox upgrade — security narrative belongs elsewhere.
- Cloud LLM benchmark numbers — confounds the calibration analysis.
- Pattern-based / conditional predictions in v1 — defer until input-based
  predictions have run on the full benchmark.

## Recommended order

1. **Track A** in one PR (or 2 small PRs grouped by file area). Stops the
   bleeding on known bugs; gets us to a tag.
2. **Track C** next (small, validates the memory work).
3. **Track B** opportunistically — pick off tasks as the demo exposes
   gaps. Don't block Track D on a complete `DependencyManager`.
4. **Track D** — the long pole. Plan for ~4 weekends part-time.

Total: roughly 5–6 weeks part-time → one publishable artifact and a
shippable `v0.1.0` along the way.
