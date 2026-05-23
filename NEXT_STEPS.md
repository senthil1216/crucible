# Next Steps

This repo has working code from an initial Feb 2026 build but several rough edges. This file is the execution plan to bring it to a state where the README's claims match reality, then to build a differentiated angle on top.

## TL;DR

**Two phases.**

1. **Tidy and ship (3–4 part-time days).** Fix broken tests, replace Jaccard similarity with embeddings, wire `was_fixed` correctly, fix error-signature normalization, replace blocking subprocess with async, add sandbox honesty note. Honest README rewrite. Tag v0.1.0.
2. **Hypothesis-scored memory + benchmark (4 weekends).** Reframe failure memory from "store and retrieve" to "predict and verify." Reflector emits falsifiable predictions; replay engine tests them against new code; calibration loop prunes weak predictions. Curated 30–50 problem benchmark. Run on local Ollama. Publishable writeup at the end.

## What's actually broken

1. **All 5 integration tests fail.** `tests/test_integration.py:31, 97` pass `loop__max_iterations=3` as a kwarg, but `AgentConfig` (`agent/models.py:245-260`) doesn't have that field — it's `loop=LoopConfig(max_iterations=...)`. README's "tests passing" claim is currently false.

2. **Fragile config parser.** `agent/core.py:68` parses `sandbox_memory_limit` as `int(value.replace('m', '').replace('g', '000'))`. So `"512m"` works, `"1g"` becomes `1000` (not `1024`), `"2gb"` silently breaks.

3. **Double-plan bug.** `core.py:133` calls `self.planner.create_plan(...)` and discards the result. Then `loop.py:250-257` runs its own planning inside the execution loop. Two plans per task.

4. **Dead code.** `agent/reflector.py:258-279` defines `_is_hopeless_case` but it's never called.

5. **Safety theater.** `agent/safety/checker.py:153-160` `_is_safe_filesystem_call` always returns `False` ("be conservative"). The AST walker only inspects `ast.Call` nodes; aliasing (`o = open; o(...)`), import-aliasing (`import subprocess as s; s.run(...)`), and `getattr`-based access all bypass the checker.

6. **Shallow sandbox.** `agent/executor/sandbox.py:130-186` uses `resource.setrlimit` — `RLIMIT_AS` is a no-op on macOS, and the fallbacks `RLIMIT_VMEM` / `RLIMIT_RSS` are also no-ops on the platform. No namespace isolation, no seccomp. Network "disabled" only by AST inspection; runtime `socket.socket()` works fine inside the subprocess.

7. **README overpromises.** Claims "similarity search" (it's Jaccard on stopword-filtered tokens, `agent/memory/long_term.py:50-83`). Implies tests pass (they don't).

8. **`was_fixed` is dead in failure memory.** `agent/memory/failure_memory.py:62` defaults `was_fixed=False`; nothing ever marks it true. The "learn from past mistakes" thesis isn't actually wired up — failure_memory is a graveyard, not a corrected-mistake corpus.

9. **Error-signature normalization destroys signal.** `agent/models.py:115-125`. The regex `\b[a-z_][a-z0-9_]*\b` matches every lowercase token in the traceback (keywords, line content, type names). The normalized key is mostly `{var}` placeholders — useless for grouping similar errors.

10. **`subprocess.run` blocks the event loop.** `agent/executor/sandbox.py:94` uses blocking `subprocess.run` inside an `async def`. Should be `asyncio.create_subprocess_exec`.

## Phase 1: Tidy and ship

### Day 1 — Correctness fixes (high-leverage credibility work)

- [ ] Fix integration tests: change `loop__max_iterations=3` → `loop=LoopConfig(max_iterations=3)` at `tests/test_integration.py:31, 97`
- [ ] Fix `sandbox_memory_limit` parser at `agent/core.py:68` — parse properly with units (m/g/mb/gb)
- [ ] Remove double-plan: delete the discarded `create_plan` call at `core.py:133`
- [ ] Delete dead `_is_hopeless_case` from `agent/reflector.py:258-279`
- [ ] Fix `ErrorSignature.normalize` (`models.py:115-125`) so the regex preserves error type, line numbers, and identifier names; strip only runtime-varying values (memory addresses, specific paths, instance reprs)
- [ ] Replace `subprocess.run` with `asyncio.create_subprocess_exec` in `sandbox.py:_execute_python`
- [ ] Add `# SECURITY NOTE` comment at top of `sandbox.py`: "Ergonomic isolation, not a security boundary. macOS memory limits are no-ops. Subprocess inherits filesystem and network capabilities."
- [ ] Verify all tests green: `pytest tests/ -v`

### Day 2 — Failure memory upgrade

- [ ] Wire `was_fixed` correctly: in `loop.py`, when iteration N+1 succeeds and N failed, update N's failure_memory entry (`was_fixed=True`, store the broken→fixed code diff as the proven fix). Link only the immediate predecessor, not transitive chains. ~50–80 LOC.
- [ ] Update `find_similar_failures` to prefer fix-confirmed entries (boost similarity by ε for `was_fixed=True`)
- [ ] Add `sentence-transformers` to `requirements.txt` (default model: `all-MiniLM-L6-v2`, ~80MB CPU)
- [ ] Replace Jaccard `_calculate_similarity` in `agent/memory/long_term.py:72-83` with cosine similarity over `all-MiniLM-L6-v2` embeddings
- [ ] Replace Jaccard `_string_similarity` in `agent/memory/failure_memory.py:139-151` with the same embedding approach (de-dup against `long_term.py`'s implementation — share a single utility)
- [ ] Test: confirm fix-confirmed failures surface preferentially when retrieving similar past errors

### Day 3 — Demo + README rewrite

- [ ] Record a 90-second asciinema: solve 3 toy problems with Ollama (`qwen2.5-coder:7b`), one exercising failure-memory retrieval (introduce a regression-style bug after a successful run, watch the agent retrieve the fix-confirmed entry)
- [ ] Rewrite `README.md` with honest framing:
  - "Prototype exploration of a self-improving coding loop, ~500 LOC core"
  - "Runs against local Ollama; cloud LLMs supported but lightly tested"
  - "AST safety + subprocess sandbox is ergonomic isolation, not a security boundary"
  - Embed the asciinema
  - Drop any "production-ready" framing
- [ ] Remove the fake "Test Results" block from `ARCHITECTURE.md` (or replace with a real CI badge once Day 1 fixes land)

### Day 4 — Buffer / polish

- [ ] Sanity pass: README claims match code reality
- [ ] Tag v0.1.0
- [ ] Brief `CHANGELOG.md` noting the embedding upgrade + `was_fixed` wiring + normalize fix as the headline additions

## Phase 2: Hypothesis-scored memory + benchmark

**Thesis:** the agent learns to predict its own failures. Reflector emits falsifiable predictions; future iterations test stored predictions against new code; predictions are scored and pruned. After 100+ tasks, a calibrated antipattern catalog with real numbers.

**v1 constraint:** input-based predictions only. A prediction is a concrete adversarial input + the failure type it should trigger. Pattern-matching and conditional predictions deferred — they require new matching infrastructure that doesn't pay off until v1 has run.

**Compute model:** local Ollama (`qwen2.5-coder:7b`), no cloud LLMs. Removes cost as a variable in the calibration analysis.

### Weekend 1 — Schema + emission

- [ ] Define prediction schema in `agent/models.py` (sketch):
  - `id`, `trigger_input` (serializable Python literal), `expected_error_type`, `rationale`, `originating_task_id`, `scoped_to_task_keywords`, `score_history: list[PredictionOutcome]`
- [ ] Modify `Reflector.SYSTEM_PROMPT` and `_parse_response` to emit a `prediction` field on every failed-iteration analysis
- [ ] Strict schema gate: drop predictions without a concrete `trigger_input` the sandbox can deserialize — no vague predictions allowed in storage
- [ ] Store predictions in `agent/memory/predictions.jsonl` (new file alongside `failures.jsonl`)
- [ ] No replay yet — verify schema integrity over 5–10 manual runs

### Weekend 2 — Replay engine

- [ ] Replay runner: given a `Prediction` and a `CodeArtifact`, generate a test wrapper that invokes the code with `trigger_input`, run through the existing sandbox, classify the result:
  - **Confirmed:** same error type AND error site within the predicted function
  - **Falsified:** code ran cleanly (no error)
  - **Off-topic:** code crashed on something unrelated (e.g., import failure)
- [ ] Wire replay into the loop as a post-test pre-success gate: when tests pass, run scoped replays before declaring SUCCESS
- [ ] Scoring + pruning: predictions tested >10 times with <30% confirmation rate get retired
- [ ] Scoping gate: only replay predictions whose `scoped_to_task_keywords` overlap the new task above a cosine-similarity threshold (using the embedding swap from Phase 1 Day 2)

### Weekend 3 — Benchmark

- [ ] Curate 30–50 small coding problems of varied shape: list manipulation, string parsing, simple data structures, off-by-one-prone tasks, edge-case-sensitive computations. Each must have a deterministic correctness oracle.
- [ ] Build `bench/runner.py`: runs the agent N times per problem (N=5 default), accumulates predictions across runs, persists state, generates per-task and aggregate statistics
- [ ] Smoke run: 5 problems × 3 repeats, verify the pipeline works end-to-end on local Ollama

### Weekend 4 — Analysis + writeup

- [ ] Full benchmark run: 30–50 problems × 5 repeats. Persist all predictions + outcomes across runs.
- [ ] Produce `bench/REPORT.md`:
  - Calibration curve: prediction confirmation rate over time
  - Surviving predictions catalog (the ones that beat the pruning threshold)
  - Variance data byproduct: how often does the loop converge on the same code? Functionally-equivalent code? Which phase contributes most variance?
- [ ] Medium post draft (per `publishing-philosophy`): "an agent that predicts its own failures — what N tasks taught it"

## Dismissed alternatives (do not re-explore)

- **SWE-bench Lite numbers.** 1–2 weeks of harness work for numbers probably mid against published agents. Defer indefinitely.
- **Multi-agent decomposition (Planner-Executor-Critic as separate agents).** Components are already isolated in separate classes — renaming them "agents" adds zero signal.
- **Docker/gVisor sandbox upgrade.** Real work, but the security narrative belongs to the parallel Warden project. Don't dilute Crucible into a security story.
- **Critic-model reflector (different LLM for reflection).** Nice-to-have but not blocking; skip unless Day 4 buffer expands.
- **Pattern-based / conditional predictions in v1.** Need matching infrastructure (LLM-based or DSL). Input-based predictions are testable for free; defer the rest until v1 has run on the full benchmark.
- **Cloud LLM benchmark numbers.** Cost variable confounds calibration analysis. Local Ollama only for the experiment; cloud LLMs remain supported in code but not in the published numbers.
- **Variance study as a standalone artifact.** Subsumed by Phase 2 — the benchmark run produces variance data as a byproduct.

## Effort

- Phase 1 (tidy and ship): 3–4 part-time days. ~150 LOC net change (embedding swap, `was_fixed` wiring, normalize fix, async subprocess, README rewrite, sandbox notice).
- Phase 2 (hypothesis-scored memory + benchmark): 4 weekends.
- Total: ~5 weeks part-time → one publishable artifact.

## File pointers

- `ARCHITECTURE.md` — design reference (note: contains an inaccurate "Test Results" block, to be removed in Phase 1 Day 3)
- `bench/` — to be created in Phase 2 Weekend 3
- `agent/memory/predictions.jsonl` — to be created in Phase 2 Weekend 1

## Portfolio context

Crucible is one of four side-project tracks. The others (Warden, trust-mint, selves) are independent — Crucible was previously considered as Warden's demo target but rejected because its non-deterministic loop breaks Warden's deterministic-decoding methodology. Crucible stands alone.

Publishing decision: Phase 2 writeup is fair game per `publishing-philosophy` (substantial/innovative/relevant) — the calibrated antipattern catalog with real numbers crosses the bar. Phase 1 alone does not; stays as a portfolio-link, not its own post.
