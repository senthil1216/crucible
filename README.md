# Crucible — Testable Memory & Failure-Prediction Replay for Coding Agents

A coding agent that doesn't just *claim* to learn — it makes its learning **testable**. Crucible writes code, gates success on a **real pytest run** (never an exit code), and stores what it learns as structured memory. Before each task it **predicts how it will fail**, and after a success it **replays those predictions against the outcome** — so "memory" becomes a measurable signal (confirmed / falsified / off-topic) instead of an unverifiable promise.

It is built around a **Plan → Execute → Test → Reflect** loop with an embedding-backed memory hierarchy.

> **crucible** /ˈkruːsɪb(ə)l/ — *noun* — A place or situation in which different elements interact to produce something new.

### What makes it different

- **Testable memory** — success means *a real pytest suite collected ≥1 test and all passed*. An empty or hollow suite never counts as a pass (the rule lives in [`agent/pytest_report.py`](agent/pytest_report.py)). Learnings are extracted only from genuinely verified runs.
- **Failure-prediction replay** — the agent predicts likely failure modes up front, then after solving replays each prediction against what actually happened. This calibration loop (Track D) measures whether the agent's stored intuition is *right*, not just present.
- **No-plugin test gate** — the gate prefers `pytest-json-report` but falls back to pytest's built-in JUnit XML, so it works on a clean install with no extra plugins.

> Crucible is an exploratory prototype, not a production agent. "Self-improving" here is a measurable claim, not a marketing one — and the non-Docker sandbox is *ergonomic isolation*, not a security boundary (see [Sandbox honesty](#sandbox-honesty)).

```
User Goal → [Plan] → [Execute] → [Test] → [Reflect] → Success?
                ↑_________________________________________|
                          ↓ on success
                    Pattern + Learnings + Env Context
                          ↓
                     Long-Term Memory
```

## ✨ Key Features

- **✅ Testable memory**: success is gated on a real pytest run (≥1 test collected, none failed); Learnings are extracted only from verified runs
- **🔮 Failure-prediction replay**: predicts failure modes before solving, then replays them against the outcome for a calibration signal (Track D)
- **🧠 Embedding-Backed Memory**: short-term context, semantic long-term patterns, failure memory, structured Learnings
- **📦 Multi-File Workspace**: optional persistent Docker container per task with file read/write, multi-file generation, and automatic dependency recovery via `DependencyManager`
- **🤖 Multi-LLM Support**: OpenAI, Anthropic, Kimi, DeepSeek, Ollama (local), a deterministic `mock`, or any custom `LLMClient`
- **💾 State Persistence**: checkpoints after every iteration; resume after interruption
- **📊 Observable**: callbacks for plan, code, test, reflect, and iteration events

## 🚀 Quick Start

### Developer setup (offline, no API keys, ~1 command)

Recommended interpreter: **Python 3.12** (see [`.python-version`](.python-version); 3.10+ should work).

```bash
git clone <repo>
cd crucible
make setup        # create .venv and install dev dependencies (requirements-dev.txt)
make test         # run the test suite
make smoke        # run ONE agent task end-to-end with the deterministic mock LLM
make bench-smoke  # run the benchmark pipeline offline (mock LLM, no Docker)
```

`make setup` is the one-command path: it creates a local `.venv` and installs everything in [`requirements-dev.txt`](requirements-dev.txt) (runtime deps + `pytest`, `pytest-asyncio`, `pytest-json-report`, `docker`, `anthropic`). The `smoke` / `bench-smoke` targets need **no network, no API key, and no Docker** — they use the built-in `mock` LLM and the local subprocess sandbox, so a new user can confirm the whole loop works in seconds. Run `make help` to list targets.

> `bench-smoke` is a *pipeline shake-out* (does the benchmark harness run end-to-end?), not a capability measurement — the mock LLM is deterministic, not smart. For a real benchmark use `python -m bench.runner --smoke --llm ollama` with Docker + Ollama.

To run the agent against a real model instead, pick one of the options below.

### Option 1: Local LLM (free, private, offline)

```bash
git clone <repo>
cd crucible
./setup_local_llm.sh             # installs Ollama + qwen2.5-coder:7b
pip install -r requirements.txt  # pulls sentence-transformers (~80 MB model
                                 # downloads on first run)

ollama serve                     # in another terminal

python -m agent "Create a function to calculate fibonacci numbers" --llm ollama
```

### Option 2: Cloud LLM (OpenAI, Anthropic, Kimi, DeepSeek)

```bash
git clone <repo>
cd crucible
pip install -r requirements.txt  # includes sentence-transformers
# optional, for the corresponding providers:
#   pip install anthropic

export OPENAI_API_KEY="sk-..."
# or: export ANTHROPIC_API_KEY="sk-ant-..."
# or: export KIMI_API_KEY="..." / DEEPSEEK_API_KEY="..."

python -m agent "Create a REST API" --llm openai --model gpt-4
```

### Option 3: Persistent Docker workspace (multi-file projects)

```bash
pip install -r requirements.txt   # includes sentence-transformers for memory
pip install docker                # only needed for --docker

python -m agent "Build a FastAPI app with /health and /items endpoints" \
  --llm ollama --docker --docker-persistent
```

This starts one container for the whole task, lets the agent create multiple files in `/workspace/<task_id>/`, and uses `DependencyManager` to install missing packages automatically (up to 4 attempts).

### Python usage

```python
import asyncio
from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OllamaClient

async def main():
    llm = OllamaClient(model="qwen2.5-coder:7b")
    agent = SelfImprovingAgent(
        llm_client=llm,
        config=AgentConfig(workspace_path="./workspace"),
    )

    result = await agent.solve("Create a function to check if a number is prime")
    if result.status.value == "success":
        print(result.code.source)

asyncio.run(main())
```

## 📁 Project Structure

```
crucible/
├── agent/                       # Main package
│   ├── __init__.py             # Public exports
│   ├── __main__.py             # CLI entry point
│   ├── core.py                 # Orchestrator (SelfImprovingAgent)
│   ├── models.py               # Plan, CodeArtifact, Learning, etc.
│   ├── loop.py                 # Execution loop + circuit breaker
│   ├── planner.py              # Plan generation; surfaces past Learnings
│   ├── code_generator.py       # Single-file + multi-file code generation
│   ├── tester.py               # Test runner (sandbox or workspace mode)
│   ├── reflector.py            # Failure analysis + Learning extraction
│   ├── persistence.py          # State checkpointing
│   ├── dependency_manager.py   # Automatic pip recovery (persistent Docker)
│   ├── llm_clients.py          # Mock, OpenAI, Anthropic, Kimi, DeepSeek, Ollama
│   ├── memory/
│   │   ├── short_term.py       # Rolling window (last 5 iterations)
│   │   ├── long_term.py        # Patterns + Learnings, semantic + multi-signal
│   │   ├── failure_memory.py   # Error signatures + proven fixes, semantic
│   │   └── embeddings.py       # Shared EmbeddingClient (all-MiniLM-L6-v2)
│   ├── executor/
│   │   ├── sandbox.py          # Local subprocess executor
│   │   └── docker_executor.py  # Docker-based, optional persistent mode
│   └── safety/
│       └── checker.py          # AST-based static analysis
├── docs/                        # Design documents
│   ├── phase1-dependency-recovery-design.md
│   ├── phase2-workspace-design.md
│   └── long-term-memory-improvements-design.md
├── examples/                    # Usage examples
├── tests/                       # Test suite (mock-based; no network/Docker)
├── ARCHITECTURE.md
├── LOCAL_LLM_GUIDE.md
├── KIMI_SETUP.md
├── setup_local_llm.sh
└── requirements.txt
```

## 🤖 LLM Support

| Provider | Setup | Best For | Cost |
|----------|-------|----------|------|
| **Ollama** (Local) | `ollama pull qwen2.5-coder:7b` | Privacy, offline | Free |
| **OpenAI** | `export OPENAI_API_KEY=...` | Quality, breadth | Pay-per-use |
| **Anthropic** | `export ANTHROPIC_API_KEY=...` | Long context, reasoning | Pay-per-use |
| **Kimi** (Moonshot) | `export KIMI_API_KEY=...` | Long context, cost | Competitive |
| **DeepSeek** | `export DEEPSEEK_API_KEY=...` | Code-focused | Competitive |

### CLI examples

```bash
# Local LLM (Ollama)
python -m agent "Sort a list" --llm ollama --model qwen2.5-coder:7b

# Cloud LLMs
python -m agent "Create a REST API" --llm openai --model gpt-4
python -m agent "Build a web scraper" --llm anthropic --model claude-3-opus
python -m agent "Create a calculator" --llm kimi --model moonshot-v1-32k
python -m agent "Refactor this module" --llm deepseek

# Docker isolation (ephemeral container per execution)
python -m agent "Sort a list" --llm ollama --docker

# Persistent Docker (one container per task, multi-file workspace,
# automatic pip recovery)
python -m agent "Build a FastAPI service" --llm ollama --docker --docker-persistent

# Interactive mode
python -m agent --interactive --llm ollama
```

### Custom LLM client

```python
from agent.models import LLMClient

class MyLLM(LLMClient):
    async def complete(self, prompt, system=None, temperature=0.7):
        return await your_backend(prompt, system, temperature)

agent = SelfImprovingAgent(llm_client=MyLLM())
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AGENT ORCHESTRATOR                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │   PLANNER   │───▶│   EXECUTOR  │───▶│    TESTER   │───▶│ REFLECTOR│  │
│  │             │    │             │    │             │    │          │  │
│  │ Break goal  │    │ Write code  │    │ Run tests   │    │ Analyze  │  │
│  │ + use past  │    │ (single or  │    │ in sandbox  │    │ failures │  │
│  │  Learnings  │    │  multi-file)│    │ / workspace │    │ + emit   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    │ Learnings│  │
│       ▲─────────────────────────────────────────────────────└────┬─────┘│
│       │ (Loop back with fix suggestion)                          │      │
└──────────────────────────────────────────────────────────────────│──────┘
                                                                   │
              ┌─────────────────────┬─────────────────────┬────────┘
              ▼                     ▼                     ▼
        ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
        │  SHORT-TERM │      │  LONG-TERM  │      │   FAILURE   │
        │   MEMORY    │      │   MEMORY    │      │   MEMORY    │
        │             │      │             │      │             │
        │  (Context)  │      │ Patterns +  │      │ Errors +    │
        │             │      │ Learnings   │      │ proven fixes│
        └─────────────┘      └─────────────┘      └─────────────┘
```

### Memory system

| Type | Stored | Retrieval | Persistence |
|------|--------|-----------|-------------|
| **Short-Term** | Recent iteration states | Direct lookup | Last 5 iterations |
| **Long-Term — Patterns** | `(goal, plan, code, project_type, dependencies, env_context, goal_embedding)` | Semantic cosine (`all-MiniLM-L6-v2`) + multi-signal bonuses for `project_type`, deps, and installed-package overlap | `patterns.jsonl` |
| **Long-Term — Learnings** | Short, reusable lessons extracted by the Reflector on success | Semantic cosine, filterable by `project_type` / `language` (`"general"` matches any project) | `learnings.jsonl` |
| **Failure** | Error signature + raw message + proven fix | Semantic cosine over the raw `error_message` | `failures.jsonl` |

Multi-signal scoring (Phase C/D): semantic similarity is the dominant signal; matching `project_type` adds **+0.15**, dependency overlap adds **up to +0.10**, installed-package overlap adds **up to +0.10**. Pass `strict_filters=True` if you want the structured fields to behave as hard filters instead of bonuses.

### Sandbox honesty

The default `SandboxedExecutor` runs generated code in a subprocess with `resource.setrlimit` and AST-based static analysis. This is **ergonomic isolation, not a security boundary**:

- macOS `RLIMIT_AS` and its fallbacks are no-ops; memory caps don't actually bind.
- The AST walker only inspects direct `ast.Call` nodes — aliased calls (`o = open; o(...)`), import aliasing (`import subprocess as s`), and `getattr`-based access all bypass it.
- The subprocess inherits filesystem and network capabilities; "network disabled" is enforced only by static analysis, not at runtime.

For stronger isolation, use `--docker` (ephemeral container per execution) or `--docker-persistent` (one container per task with a dedicated `/workspace/<task_id>/`). Even Docker mode is a process boundary, not a security one — don't run untrusted goals against it.

## 🗂 Multi-file workspace

In persistent Docker mode, the agent can generate small projects instead of single scripts:

```bash
python -m agent "Build a FastAPI app with health + items endpoints" \
  --llm ollama --docker --docker-persistent
```

The container exposes `write_file`, `read_file`, `list_dir`, `create_directory`, and `run_command_in_workspace` via `DockerExecutor`. `DependencyManager` watches for `ModuleNotFoundError` and reinstalls missing packages (up to 4 attempts per task) before handing off to the Reflector. Remaining hardening work is tracked in [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md).

## ⚙️ Configuration

```python
from agent import AgentConfig, LoopConfig

config = AgentConfig(
    loop=LoopConfig(
        max_iterations=10,
        failure_threshold=3,
        failure_window=5,
        cooldown_period=60,
    ),

    workspace_path="./workspace",
    state_path="./.agent_state",
    memory_path="./.agent_memory",

    # Local subprocess sandbox (default)
    sandbox_timeout=30,
    sandbox_memory_limit="512m",

    # Docker mode (opt-in)
    use_docker=False,
    docker_image="python:3.12-slim",
    docker_persistent=False,           # One container per task
    docker_enable_network=True,        # Needed for pip in persistent mode
    docker_install_build_tools=True,   # apt-get build-essential on container start
)
```

## 🧪 Testing

```bash
make test                    # the whole suite (or: pytest -q)
make cov                     # with coverage (or: pytest --cov=agent)
pytest tests/test_memory.py -v
```

The suite is fully mock-based — no network, Docker, or model downloads required, so it runs on a clean install with only `pytest` + `pytest-asyncio`. The success gate prefers `pytest-json-report` but falls back to pytest's built-in JUnit XML when the plugin is absent, so the tests stay green either way (see [`tests/test_pytest_report_fallback.py`](tests/test_pytest_report_fallback.py)).

## 📊 Monitoring

```python
def on_iteration(state):
    print(f"Iter {state.iteration}: {state.status}")

agent = SelfImprovingAgent(
    llm_client=llm,
    callbacks={
        "on_iteration": on_iteration,
        "on_plan": ...,
        "on_code": ...,
        "on_test": ...,
        "on_reflect": ...,
    },
)
```

## 📚 Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Architecture and design decisions
- [`LOCAL_LLM_GUIDE.md`](LOCAL_LLM_GUIDE.md) — Local LLM setup (Ollama, LM Studio, etc.)
- [`KIMI_SETUP.md`](KIMI_SETUP.md) — Kimi (Moonshot AI) setup
- [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) — Consolidated plan: what's merged, what's open, and the recommended order

## 🔧 Requirements

- **Python 3.12 recommended** (pinned in [`.python-version`](.python-version)); 3.10+ works (the code uses PEP 604/585 type syntax)
- Runtime: `openai>=1.0.0`, `httpx>=0.24.0`, `sentence-transformers>=2.2.0` (via `requirements.txt`)
- Development: `make setup` installs [`requirements-dev.txt`](requirements-dev.txt) — adds `pytest`, `pytest-asyncio`, `pytest-json-report`, `pytest-cov`, `docker`, `anthropic`
- Optional at runtime: `docker` for `--docker` / `--docker-persistent`; `anthropic` for the Anthropic backend
- For local LLMs: [Ollama](https://ollama.com)

## 📝 License

MIT License

## 🙏 Acknowledgments

Built as a portfolio exploration of self-improvement loops in coding agents.
