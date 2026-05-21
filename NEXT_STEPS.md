# Next Steps

This repo has working code from an initial Feb 2026 build but several rough edges. This file is the execution plan to bring it to a state where the README's claims match reality.

## TL;DR

**Tidy and ship.** Three days of work to fix broken tests + one focused upgrade (sentence-transformer embeddings replacing Jaccard similarity in memory subsystems). Then honest README rewrite + recorded demo.

## What's actually broken

1. **All 5 integration tests fail.** `tests/test_integration.py:31` and `:97` pass `loop__max_iterations=3` as a kwarg, but `AgentConfig` (`agent/models.py:245-260`) doesn't have that field — it's `loop=LoopConfig(max_iterations=...)`. README's "tests passing" claim is currently false.

2. **Fragile config parser.** `agent/core.py:68` parses `sandbox_memory_limit` as `int(value.replace('m', '').replace('g', '000'))`. So `"512m"` works, `"1g"` becomes `1000` (not `1024`), `"2gb"` silently breaks.

3. **Double-plan bug.** `core.py:133` calls `self.planner.create_plan(...)` and discards the result. Then `loop.py:250-257` runs its own planning inside the execution loop. Two plans per task.

4. **Dead code.** `agent/reflector.py:258-279` defines `_is_hopeless_case` but it's never called.

5. **Safety theater.** `agent/safety/checker.py:153-160` `_is_safe_filesystem_call` always returns `False` ("be conservative"). The AST-walker only inspects `ast.Call` nodes; string-encoded payloads, attribute aliasing (`o = __builtins__; o.eval(...)`), and `getattr(builtins, "ev"+"al")` all pass through.

6. **Shallow sandbox.** `agent/executor/sandbox.py:130-186` uses `resource.setrlimit` (no-op on macOS for `RLIMIT_AS`), no namespace isolation, no seccomp. Network "disabled" only by AST inspection; runtime `socket.socket()` works fine inside the subprocess.

7. **README overpromises.** Claims "similarity search" (it's Jaccard on stopword-filtered tokens, `agent/memory/long_term.py:50-83`). Implies tests pass (they don't).

## Plan

### Day 1 — Fixes (high-leverage credibility work)
- [ ] Fix integration tests: change `loop__max_iterations=3` → `loop=LoopConfig(max_iterations=3)` at `tests/test_integration.py:31, 97`
- [ ] Fix `sandbox_memory_limit` parser at `agent/core.py:68` — parse properly with units (m/g/mb/gb)
- [ ] Remove double-plan: delete the discarded `create_plan` call at `core.py:133`, OR refactor so the loop reuses it instead of re-planning
- [ ] Delete dead `_is_hopeless_case` from `agent/reflector.py:258-279`
- [ ] Verify all tests green: `pytest tests/ -v`

### Day 2 — Embedding-based memory upgrade
- [ ] Add `sentence-transformers` to `requirements.txt` (default model: `all-MiniLM-L6-v2`, ~80MB CPU)
- [ ] Replace Jaccard `_calculate_similarity` in `agent/memory/long_term.py:72-83` with cosine similarity over `all-MiniLM-L6-v2` embeddings
- [ ] Replace Jaccard `_string_similarity` in `agent/memory/failure_memory.py:139-151` with the same embedding approach
- [ ] Add a small test in `tests/` showing the new retrieval finds semantically-related failures the Jaccard version misses

### Day 3 — Demo + README rewrite
- [ ] Record a 90-second asciinema: solve 3 toy problems with Ollama (`qwen2.5-coder:7b`), one exercising failure-memory retrieval (introduce a regression-style bug after a successful run, watch the agent retrieve the prior fix)
- [ ] Rewrite `README.md` with honest framing:
  - "Prototype exploration of a self-improving coding loop, ~500 LOC core"
  - "Runs against local Ollama; cloud LLMs supported but lightly tested"
  - "AST safety + subprocess sandbox is conservative-by-default; not a security boundary"
  - Embed the asciinema
  - Drop any "production-ready" framing

### Day 4 — Buffer / polish
- [ ] Sanity pass: README claims match code reality
- [ ] Tag a v0.1.0 release
- [ ] (Optional) Brief `CHANGELOG.md` noting the embedding upgrade as the headline addition

## Dismissed alternatives (do not re-explore)

- **SWE-bench Lite numbers.** 1–2 weeks of harness work for numbers probably mid against published agents. Defer indefinitely.
- **Multi-agent decomposition (Planner-Executor-Critic as separate agents).** Components are already isolated in separate classes — renaming them "agents" adds zero signal.
- **Docker/gVisor sandbox upgrade.** Real work, but the security narrative belongs to the parallel Warden project. Don't dilute Crucible into a security story.
- **Critic-model reflector (different LLM for reflection).** Nice-to-have but not blocking; skip unless Day 4 buffer expands into a full week.

## Effort

3–4 days part-time. ~80 LOC of net change (mostly the embedding swap), several deletions, README rewrite, demo recording.

## Portfolio context

Crucible is one of four side-project tracks. The others (Warden, trust-mint, selves) are independent — Crucible was previously considered as Warden's demo target but rejected because its non-deterministic loop breaks Warden's deterministic-decoding methodology. Crucible stands alone.
