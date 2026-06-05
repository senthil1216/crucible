# Crucible Next Steps

Crucible's strongest thesis is not "an autonomous software engineer." The
stronger and more defensible thesis is:

> Crucible is a coding-agent platform that turns failures into testable memory,
> so improvement can be measured instead of assumed.

The project should be developed around that idea: testing, reflection, memory,
replay, and calibration are not side features. They are the product.

## Product Positioning

### Core Claim

Crucible helps coding agents improve by converting failed attempts into
falsifiable predictions, replaying those predictions, and measuring whether the
agent's self-predictions are reliable over time.

### What To Avoid Claiming

- Do not claim Crucible is a general autonomous software engineer.
- Do not claim the agent learns in the same way as model training.
- Do not publish synthetic benchmark results as evidence.
- Do not hide off-topic predictions or failed predictions.

### What To Claim Instead

- Crucible makes agent reflection testable.
- Crucible turns vague failure memories into replayable hypotheses.
- Crucible measures whether agent confidence is calibrated.
- Crucible is a transparent, developer-scale version of feedback loops used in
  larger AI systems.

## Target Users

### AI Agent Builders

People building coding agents can use Crucible as a reference architecture for
agent memory, test-first execution, failure analysis, replay, and calibration.

### AI Evaluation Engineers

Teams evaluating coding agents can use Crucible to study whether repeated
feedback loops improve performance across related tasks.

### Local LLM Developers

Developers can run Crucible with Ollama to experiment with private, local coding
agents and inspect what the agent learned.

### Researchers

Researchers can use Crucible to study calibrated self-prediction, memory
quality, replay validity, and the limits of agent-level learning without
changing base model weights.

## Value Proposition

Crucible provides value by making coding-agent improvement inspectable:

- It runs generated code against tests instead of trusting generated output.
- It reflects on failures and proposes fixes.
- It stores useful successful patterns and reusable lessons.
- It emits concrete failure predictions such as "`[]` will raise `IndexError`."
- It replays those predictions against later passing code.
- It scores predictions as confirmed, falsified, or off-topic.
- It reports calibration metrics instead of relying on anecdotes.

The core value is not just code generation. The core value is measured feedback.

## Phase 1: Stabilize The Foundation

Goal: make the repository reliably runnable by a new user.

### Tasks

- Fix the current test suite so `pytest -q` is green in a clean environment.
- Make the pytest JSON report dependency impossible to miss.
- Standardize on a supported Python version, preferably Python 3.12.
- Add a clear install path for runtime and development dependencies.
- Add one-command test and smoke commands.
- Make Docker prerequisites explicit.
- Keep sandbox limitations visible and honest.

### Recommended Commands

```bash
pip install -r requirements.txt
pytest -q
python -m bench.runner --smoke
```

### Deliverable

A new user can install the project, run tests, run one agent task, and understand
the project in under 10 minutes.

## Phase 2: Clarify The Product Surface

Goal: make it obvious how someone uses Crucible.

### Developer Mode

The user gives a task. Crucible plans, writes code, runs tests, reflects on
failures, fixes code, and stores successful patterns.

Example:

```bash
python -m agent "Create a function that returns the first item in a list" \
  --llm ollama
```

### Research Mode

The user runs repeated benchmark tasks. Crucible accumulates predictions,
replays them, and generates calibration reports.

Example:

```bash
python -m bench.runner --smoke
python -m bench.runner --reps 5
python -m bench.analyze bench/results/<run>.jsonl \
  --predictions .agent_memory/predictions/predictions.jsonl \
  --out bench/REPORT.md
```

### Future CLI Shape

```bash
crucible solve "Create a CSV parser"
crucible bench smoke
crucible bench run --reps 5
crucible bench analyze
crucible memory predictions
```

### Deliverable

A CLI and README that clearly separate normal agent usage from benchmark and
research usage.

## Phase 3: Build The Canonical Demo

Goal: demonstrate the innovative loop in one simple, concrete example.

### Demo Story

Task:

> Write a function that returns the first item in a list.

Buggy implementation:

```python
def first_item(items):
    return items[0]
```

Prediction emitted after failure:

```json
{
  "trigger_input": "[]",
  "predicted_error_type": "IndexError",
  "predicted_explanation": "The function indexes the first element without checking for an empty list.",
  "confidence": 0.9
}
```

Replay call:

```python
first_item([])
```

Possible verdicts:

- Confirmed: the function raises `IndexError`.
- Falsified: the function handles the empty list safely.
- Off-topic: the input cannot be applied to the selected entry point.

### Deliverable

A polished demo script and README section showing:

```text
failure -> prediction -> fix -> replay -> calibration signal
```

## Phase 4: Run The Benchmark For Real

Goal: replace mechanism-only claims with real evidence.

### Benchmark Design

- Use one fixed model for the headline run.
- Prefer local Ollama with `qwen2.5-coder:7b` for reproducibility and cost
  control.
- Use deterministic single-function Python tasks first.
- Persist memory across the full benchmark run.
- Run multiple repetitions per problem.
- Record run order so order effects are visible.

### Metrics To Report

- Task success rate.
- Iterations to success.
- Predictions emitted.
- Predictions replayed.
- Confirmed, falsified, and off-topic verdict counts.
- Confirmation rate by confidence bucket.
- Off-topic rate as an integrity metric.
- Surviving predictions catalog.
- Retired predictions.
- Correlation between surfaced predictions and fewer iterations.

### Do Not Hide

- High off-topic rate.
- Weak confidence calibration.
- Predictions that repeatedly fail.
- Tasks where memory did not help.
- Local model limitations.

### Deliverable

A real `bench/REPORT.md` generated from real benchmark data, replacing any
synthetic placeholder report.

## Phase 5: Improve Memory Quality

Goal: make memory useful instead of noisy.

### Promotion Rules

- Boost predictions confirmed by replay.
- Boost learnings that were surfaced before successful runs.
- Boost failure fixes that were later marked fixed.

### Retirement Rules

- Retire predictions that have been tested enough times and rarely confirmed.
- Drop malformed predictions with vague trigger inputs.
- Drop predictions without concrete exception types.
- Avoid storing generic lessons like "handle edge cases."

### Inspection Commands

Future CLI commands should make memory auditable:

```bash
crucible memory list
crucible memory predictions
crucible memory retired
crucible memory explain --goal "parse csv rows"
```

### Deliverable

Users can inspect what Crucible remembers, why it retrieved a memory, and whether
that memory has survived replay.

## Phase 5A: Demonstrate User Memory

Goal: show that Crucible adapts to a user's local project conventions, preferred
libraries, and recurring failure modes.

This is different from shipped seed memory. Seed memory gives every user a
baseline. User memory is accumulated from the user's own tasks and should remain
local unless the user explicitly exports it.

### Demo Thesis

> Crucible adapts to how your project works, not just how Python works
> generally.

### Canonical User Memory Demo

Use a project-specific response convention because it is easy to understand.

Convention:

> In this project, API-style functions return payloads wrapped in a top-level
> `data` object.

Expected shape:

```python
{
    "data": {
        "id": "123",
        "name": "Test User"
    }
}
```

### Demo Flow

Start with empty user memory:

```bash
rm -rf .agent_memory
```

Run the first task:

```bash
python -m agent \
  "Create a function get_user that returns user info wrapped in a data object" \
  --llm ollama
```

The first attempt may return a raw object:

```python
{
    "id": "123",
    "name": "Test User"
}
```

The frozen tests should expect the project convention:

```python
assert get_user("123") == {
    "data": {
        "id": "123",
        "name": "Test User"
    }
}
```

After the failure is fixed and the task succeeds, Crucible should store a local
learning similar to:

```json
{
  "lesson": "For this project, return API-style payloads wrapped in a top-level data object.",
  "tags": ["api-response", "project-convention"]
}
```

Run a related second task:

```bash
python -m agent \
  "Create a function list_users that returns users using the same project response style" \
  --llm ollama
```

The demo should show the local memory being retrieved before planning:

```text
Surfaced relevant learning:
- For this project, return API-style payloads wrapped in a top-level data object.
```

The generated code should start closer to:

```python
def list_users():
    return {
        "data": [
            {"id": "123", "name": "Test User"}
        ]
    }
```

### What The Demo Must Show

The demo should make this sequence explicit:

```text
Run 1: No user memory
- Task: get_user
- Mistake: returned raw object
- Stored learning: wrap responses in {"data": ...}

Run 2: User memory retrieved
- Task: list_users
- Retrieved learning: wrap responses in {"data": ...}
- Result: fewer mistakes or fewer iterations
```

### Other User Memory Demo Ideas

- Preferred library: the project uses `httpx`, not `requests`.
- Path handling: the project uses `pathlib.Path`, not raw string paths.
- Missing values: the project returns `None`, not exceptions.
- Output style: functions return values and do not print.
- Error shape: API errors use `{"error": {"message": ...}}`.
- Test style: tests prefer pure functions with deterministic outputs.

### Deliverable

A repeatable demo that shows a local convention being learned from one task and
retrieved on a later related task.

## Phase 6: Generalize Replay Contracts

Goal: move beyond single-function Python tasks without sacrificing rigor.

### Replay Contract Types

Start with:

```json
{
  "type": "function",
  "entrypoint": "first_item",
  "input": "[]",
  "expected_error": "IndexError"
}
```

Then expand to CLI tasks:

```json
{
  "type": "cli",
  "argv": ["--input", ""],
  "expected_error": "ValueError"
}
```

Then HTTP tasks:

```json
{
  "type": "http",
  "method": "GET",
  "path": "/items/999",
  "expected_status_or_error": "404"
}
```

### Expansion Order

1. Single-function Python.
2. Python modules with declared entry points.
3. CLI programs with `argv` and stdin.
4. HTTP endpoints in Docker.
5. Multi-file projects.
6. Repository-level tasks.

### Deliverable

Replay remains deterministic and classifiable as the project moves beyond toy
function tasks.

## Phase 7: Productize The Developer Experience

Goal: make the project usable and understandable, not just technically
interesting.

### Output Improvements

Each run should clearly show:

- Final status.
- Generated code path.
- Frozen test path.
- Test summary.
- Failed test details.
- Reflection summary.
- Predictions emitted.
- Replay verdicts.
- Memory writes.
- Suggested next command.

### Dashboard Or Terminal UI

A future UI should show:

- Task timeline.
- Attempts.
- Code diffs.
- Test results.
- Predictions.
- Replay verdicts.
- Retrieved memories.
- Calibration summary.

### Deliverable

A user can run Crucible and understand what happened without reading internal
JSONL files.

## Phase 8: Publish The Project

Goal: showcase the innovation credibly.

### Article Thesis

> I built a coding agent that turns failures into testable memory.

### Article Outline

1. Most agent reflection is vague.
2. Vague memory is hard to trust.
3. Crucible turns failures into falsifiable predictions.
4. Predictions are replayed against passing code.
5. Replay classifies predictions as confirmed, falsified, or off-topic.
6. Calibration measures whether confidence means anything.
7. The benchmark shows what worked and what failed.
8. The next step is broader replay contracts.

### LinkedIn Draft

```text
I have been building Crucible, an experimental coding agent focused on one
question: can an agent learn from failure in a way we can actually measure?

Instead of storing vague reflections like "handle edge cases better," Crucible
asks the agent to emit falsifiable predictions after failures: concrete inputs
and expected exception types.

Example: input [] will raise IndexError.

Later, when the task succeeds, Crucible replays those predictions against the
final code and scores them as confirmed, falsified, or off-topic.

The goal is not just memory. The goal is calibrated memory: can the agent's
confidence in its own failure predictions become meaningful over repeated
tasks?

The project now has the core loop, replay engine, prediction memory, and
benchmark harness. The next milestone is publishing real calibration results.
```

### Deliverable

A Medium article, LinkedIn post, demo video, and real benchmark report that all
make the same narrow, defensible claim.

## Phase 9: Choose A Product Direction

There are three plausible directions.

### Agent Evaluation Platform

Use Crucible to evaluate coding agents, prompts, models, and memory strategies
over repeated tasks.

Best for AI platform teams and agent builders.

### Local Coding Agent Workbench

Use Crucible as a local, private, inspectable coding agent with Docker execution
and memory.

Best for individual developers and security-conscious teams.

### Failure Memory Layer

Use Crucible's prediction, replay, and calibration system as a memory layer for
other coding agents.

Best for agent-framework builders.

### Recommended Initial Direction

Start with:

> Agent evaluation platform for testable coding-agent memory.

This is the most differentiated and easiest to defend with evidence.

## 90-Day Roadmap

### Days 1-15

- Fix local test failures.
- Lock the development environment.
- Update README around testable memory.
- Add a smoke benchmark command.
- Build the canonical falsifiable-prediction demo.

### Days 16-30

- Run the smoke benchmark.
- Fix benchmark issues.
- Improve CLI output.
- Add basic memory inspection commands.

### Days 31-45

- Run the full benchmark.
- Generate the real calibration report.
- Identify top confirmed predictions.
- Identify top off-topic causes.

### Days 46-60

- Improve replay entry-point handling.
- Reduce off-topic rate.
- Add explicit replay contracts.
- Re-run the benchmark.

### Days 61-75

- Polish docs.
- Prepare public demo.
- Write the Medium article.
- Draft the LinkedIn post.
- Record a short demo video.

### Days 76-90

- Publish.
- Collect feedback.
- Add a second fixed-model comparison.
- Decide whether to pursue evaluation platform, local workbench, or memory layer.

## Success Criteria

Crucible becomes a real, showcaseable project when:

- `pytest -q` is green.
- A new user can run one agent task locally.
- The benchmark runs with real data.
- Predictions are replayed and scored.
- Off-topic rate is visible.
- Calibration by confidence bucket is reported.
- Some predictions survive repeated testing.
- Retrieved memory correlates with fewer iterations or better edge-case handling.
- The public article makes a narrow claim backed by real evidence.

## Final North Star

Crucible should prove this:

> Coding-agent memory can be made testable. Failures can become replayable
> predictions, and repeated runs can measure whether the agent is actually
> improving.
