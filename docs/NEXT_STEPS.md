# Next Steps

Single source of truth for what's left to do on Crucible. Anything completed
has been moved to "Where we are"; anything still open lives under "What's open".

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
- **Real-pytest success gate + tooling** — test-first frozen pytest suite,
  `StepProfiler`, xAI provider, `--run` app launch, prebaked Docker runtime
  image (`docker/Dockerfile` → `crucible-runtime`). (PR #8)
- **Track A — v0.1.0** — `ErrorSignature.normalize` tightened; `was_fixed`
  wired end-to-end (`Reflection.failure_id` + `FailureMemory.mark_fixed`,
  broken→fixed diff, +0.05 retrieval boost); async subprocess execution;
  `CHANGELOG.md` introduced; SECURITY notes on the sandbox. (PR #9, tagged
  `v0.1.0`)
- **Track C scaffold** — `bench/track_c_runner.py` + an 8-task CSV batch
  (`bench/track_c_tasks.py`); learning-feedback loop tracking helpfulness
  across tasks. (PR #10, #11)
- **Track D phase 1 — predictions** — `Prediction` dataclass, Reflector
  emission on failure (strict schema gate: concrete `trigger_input` + a
  Python exception type), `PredictionMemory` (`predictions.jsonl`), semantic
  `find_relevant`, Planner surfacing of past failure modes. (PR #12, #13)
- **Track D phase 2 — replay engine** — `agent/replay.py`. On success,
  every prediction linked to that task's failures is replayed against the
  final passing code and classified Confirmed / Falsified / Off-topic;
  verdicts write back via `PredictionMemory.record_replays`; auto-retire at
  `times_tested ≥ 10 AND confirmation_rate < 0.30`. Record-only
  (`predictions_gate_enabled` hook is inert). `IterationState.replay_report`
  persists per-task verdicts. (PR #14)
- **Graceful embedding degradation** — `EmbeddingClient.encode()` returns
  `[]` instead of raising when `sentence-transformers` is absent; memory
  falls back to structured filters. (PR #14)

## What's open

### Track B — DependencyManager hardening — done

Pending tasks from the original `phase1-dependency-recovery-design.md`.
All four shipped (code-only, fully unit-tested in `tests/test_dependency_manager.py`):

- ~~**`requirements.txt` support**~~ — `DependencyManager.install_from_requirements(path)`
  delegates to `DockerExecutor.install_requirements_file_detailed`; gated by the
  new `AgentConfig.docker_auto_install_requirements` flag (default off → recovery
  stays on-demand / import-driven).
- ~~**Categorize install failure types**~~ — `DependencyManager.classify_pip_failure`
  parses pip stderr into `not_found` / `build_error` / `network` / `permission` /
  `unknown`; `install_packages` now threads stderr via
  `DockerExecutor.install_packages_detailed` and sets `failure_reason` accordingly.
- ~~**Optional install confirmation**~~ — `AgentConfig.docker_ask_before_install`
  wires an injectable confirm callback on `DependencyManager`; declining returns
  `failure_reason="user_declined"` without touching the network. Default off.
- ~~**Persist successful installations**~~ — `DependencyManager.installed_packages`
  (cumulative per task) is unioned into the Pattern's `environment_context.installed_packages`
  on success in `core.py`. No separate store.

### Track C — close out the memory demo

The runner is merged; what's left is to actually run it and write the
1-pager. Cheap, and it shakes out the harness before the bigger Track D
benchmark generalizes the same runner.

- ~~**Score-breakdown logging**~~ — done. `find_similar_solutions` now returns a
  `score_breakdown` (semantic / project_type / deps / installed-package) per result;
  `bench/track_c_runner.py` logs it per retrieved pattern.
- **Run + report**: run the 8-task CSV batch on the 3.12 venv with
  `--docker-persistent --docker-image crucible-runtime`; write
  `docs/track-c-report.md` (per-task iterations-to-success, whether
  retrieved Learnings plausibly informed the plan, any infra bugs).
- **Tie-in**: any bug surfaced folds back into the codebase before Phase 3.

### Track D — Hypothesis-scored predictions (publishable)

**Thesis**: the agent learns to predict its own failures. Phases 1 and 2
are done (see "Where we are") — predictions are emitted, replayed, scored,
and pruned. What remains is to run it at scale and report whether the
predictions are *calibrated*.

**v1 constraint**: input-based predictions only (a concrete adversarial
input + the failure type it should trigger). Pattern-matching and
conditional predictions stay deferred.

#### Phase 3 — Benchmark

Generalize `bench/track_c_runner.py` into `bench/runner.py` and run enough
tasks to accumulate replay verdicts.

> **Status: harness code-complete; run pending.** `bench/runner.py` (memory
> persisted across the whole batch, `--reps` / `--smoke` / `--docker-image` flags,
> per-run JSONL incl. `replay_report`) and `bench/problems.py` (40 deterministic
> single-function problems across string/list/dict/math/parsing) now exist with
> unit tests. What remains is executing them on a capable machine (3.12 venv +
> `crucible-runtime` image + Ollama) — see the pre-reqs below.

- **Problem set** — 30–50 *deterministic* small Python problems
  (`bench/problems.py`). Hard requirements, learned from earlier runs:
  - **Single-file `solution` contract** so the replay entry-point selector
    works. Prefer **one public function per problem** (or a function whose
    name echoes the goal) — multiple ambiguous functions classify Off-topic
    and yield no calibration signal.
  - **No web / server tasks.** The local 7B model can't reliably author a
    valid frozen suite for FastAPI (it writes server-launch tests), and
    multi-file isn't replay-scoped. Stick to string/list/dict algorithms,
    math, parsing, small data structures — domains with obvious adversarial
    inputs (`[]`, `-1`, `''`, `None`, `0`) the Reflector can predict.
- **Reps (N)** — run each problem N times (default 5) for convergence
  variance and to surface more failures (→ more predictions → more
  replays). 40 problems × 5 ≈ 200 task runs.
- **Compute model** — local Ollama `qwen2.5-coder:7b`, fixed, to keep cost
  out of the calibration analysis. Optionally a second fixed model as a
  sensitivity check, but the headline run uses one.
- **`bench/runner.py`** — per task run, append to a JSONL: problem id, rep,
  status, iterations-to-success, retrieved patterns/learnings/predictions,
  and **`state.replay_report`** (the confirmed/falsified/off-topic
  verdicts). This is the raw material for Phase 4.
- **Memory mode** — persist memory across the *entire* run (don't reset per
  problem): predictions need to be surfaced and replayed across reps and
  related problems to accumulate data. Record run order so order-effects
  are visible.
- **Pre-reqs (now real, don't skip)**:
  - 3.12 venv with `sentence-transformers` installed — otherwise
    `find_relevant` and pattern recall are no-ops and the "memory helped"
    analysis is dead.
  - Prebaked `crucible-runtime` image (`./docker/build.sh`) so
    `container_setup` doesn't dominate wall-clock.
  - **Smoke run first**: 3 problems × 2 reps to shake out harness bugs
    before the full run.

#### Phase 4 — Calibration + writeup

Turn the JSONL into `bench/REPORT.md` with real numbers.

> **Status: analysis code-complete; awaiting real data.** `bench/analyze.py`
> computes all the metrics below (calibration-by-confidence-bucket, off-topic rate,
> convergence variance, memory-helped, surviving-predictions catalog, retirement
> stats) and renders `bench/REPORT.md`. A synthetic placeholder `bench/REPORT.md`
> is committed to show the layout; real numbers land after the phase-3 run.

- **Calibration is aggregate-by-confidence, not per-prediction.** Important
  design constraint: `failure_id = sha256(normalize(error) : code[:100])` is
  derived from the *failing code*, which changes every rep, so a prediction
  id (`sha256(failure_id : trigger_input)`) is usually unique per task and
  gets replayed ~once. Per-prediction confirmation rates are therefore mostly
  0/1 or 1/1 — too sparse to calibrate individually. Instead **bucket all
  replay verdicts by the prediction's self-reported `confidence`** and
  compute the confirmation rate per bucket. That is the headline result:
  *are higher-confidence self-predictions actually confirmed more often?*
- **Off-topic rate** — fraction of replays that couldn't be applied (bad
  literal, arity/boundary error, ambiguous entry). This is the integrity
  metric: a high rate means the entry-point heuristic is weak and the
  calibration sample is thin. **Report it prominently — never hide it.**
- **Surviving-predictions catalog** — the concrete antipatterns that
  confirmed (e.g. "`[]` → IndexError" on first-element problems). Pull
  non-retired, high-confirmation predictions from `PredictionMemory`.
- **Retirement stats** — from `get_stats`: how many predictions auto-retired
  (`times_tested ≥ 10 AND rate < 0.30`). Note retirement is a *long-horizon*
  mechanism that accrues across many runs sharing one memory store; within a
  single benchmark run it will rarely fire (see the id note above). Decide
  whether to (a) accept it as cross-run, (b) lower the threshold for the
  analysis, or (c) leave it and report "0 retired this run, by design".
- **Convergence variance** — iterations-to-success distribution per problem
  across reps; does it tighten as memory fills?
- **Memory-helped analysis** — correlational: did surfacing predictions to
  the Planner reduce iterations or off-topic failures on later related
  problems?
- **Writeup** — `bench/REPORT.md` + a Medium/LinkedIn draft. This is the
  artifact that gates the blog post: publish once the calibration curve and
  surviving-predictions catalog exist (the mechanism-only version is too
  early; the claim is "calibrated self-prediction", narrow and true).

### Dismissed (do not re-explore)

- SWE-bench Lite numbers — harness work too heavy for likely mid result.
- Multi-agent decomposition — components are already in separate classes.
- Docker/gVisor sandbox upgrade — security narrative belongs elsewhere.
- Cloud LLM benchmark numbers — confounds the calibration analysis.
- Pattern-based / conditional predictions in v1 — defer until input-based
  predictions have run on the full benchmark.
- Parallel test/code generation — code generation deliberately consumes the
  frozen test suite (test-first contract); parallelizing sacrifices first-pass
  success for marginal latency. See the PR discussion.

## Recommended order

1. ~~**Track A**~~ — done (`v0.1.0`).
2. ~~**Track D phases 1–2**~~ — done (predictions + replay engine).
3. ~~**Track D phase 3–4 harness**~~ — done (code). `bench/runner.py`,
   `bench/problems.py`, `bench/analyze.py` built + score-breakdown logging
   (PR #16). Execution still pending a capable machine.
4. ~~**Track B**~~ — done. DependencyManager hardening (all four items).
5. **Track D phase 3 RUN** — execute the benchmark on a 3.12 venv +
   `crucible-runtime` image + Ollama: smoke-run, then the full 40×5.
6. **Track D phase 4 RUN** — `python -m bench.analyze` the JSONL into
   `bench/REPORT.md` with real numbers; draft the writeup.
7. **Track C close-out** — run the 8-task CSV demo + write the 1-pager
   (also needs a capable machine).
