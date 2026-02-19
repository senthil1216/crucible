# Crucible - Self-Improving Coding Agent

An autonomous coding agent that writes code, runs tests, and learns from failures. Uses a **Plan → Execute → Test → Reflect** loop with memory hierarchy and safety guardrails.

> **crucible** /ˈkruːsɪb(ə)l/ - *noun* - A place or situation in which different elements interact to produce something new.

```
User Goal → [Plan] → [Execute] → [Test] → [Reflect] → Success?
                ↑_________________________________________|
```

## ✨ Key Features

- **🔄 Self-Improving Loop**: Automatically iterates on failures until success or max attempts
- **🧠 Memory Hierarchy**: Short-term context, long-term pattern storage, and failure memory
- **🛡️ Safety First**: Multi-layer safety with AST analysis, sandboxed execution, and resource limits
- **🤖 Multi-LLM Support**: Works with OpenAI, Anthropic, Kimi, Ollama (local), and custom providers
- **💾 State Persistence**: Resume tasks after interruptions
- **📊 Observable**: Callbacks for monitoring progress and iterations

## 🚀 Quick Start

### Option 1: Local LLM (Free, Private, Offline)

```bash
# 1. Clone and setup
git clone <repo>
cd crucible
./setup_local_llm.sh  # Installs Ollama + qwen2.5-coder:7b

# 2. Start Ollama (in another terminal)
ollama serve

# 3. Run the agent
python -m agent "Create a function to calculate fibonacci numbers" --llm ollama
```

### Option 2: Cloud LLM (OpenAI, Anthropic, Kimi)

```bash
# 1. Clone and install
git clone <repo>
cd crucible
pip install -r requirements.txt  # + optional: openai, anthropic, httpx

# 2. Set API key
export OPENAI_API_KEY="sk-..."
# or: export ANTHROPIC_API_KEY="sk-ant-..."
# or: export KIMI_API_KEY="your-kimi-key"

# 3. Run the agent
python -m agent "Create a REST API" --llm openai --model gpt-4
```

### Python Usage

```python
import asyncio
from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OllamaClient

async def main():
    # Use local LLM (free, private)
    llm = OllamaClient(model="qwen2.5-coder:7b")
    
    # Or use cloud LLM
    # from agent.llm_clients import OpenAIClient
    # llm = OpenAIClient(api_key="sk-...", model="gpt-4")
    
    agent = SelfImprovingAgent(
        llm_client=llm,
        config=AgentConfig(workspace_path="./workspace")
    )
    
    result = await agent.solve("Create a function to check if a number is prime")
    
    if result.status.value == "success":
        print(result.code.source)

asyncio.run(main())
```

## 📁 Project Structure

```
crucible/
├── agent/                      # Main agent package
│   ├── __init__.py            # Main exports (SelfImprovingAgent, AgentConfig)
│   ├── __main__.py            # CLI entry point
│   ├── core.py                # Agent orchestrator
│   ├── models.py              # Data models (Plan, CodeArtifact, etc.)
│   ├── loop.py                # Execution loop + circuit breaker
│   ├── planner.py             # Plan generation
│   ├── code_generator.py      # Code generation
│   ├── tester.py              # Test runner
│   ├── reflector.py           # Failure analysis
│   ├── persistence.py         # State persistence
│   ├── llm_clients.py         # LLM implementations (OpenAI, Anthropic, Ollama, etc.)
│   ├── memory/                # Memory hierarchy
│   │   ├── short_term.py      # Rolling window (last 5 iterations)
│   │   ├── long_term.py       # Successful pattern storage
│   │   └── failure_memory.py  # Error signatures + fixes
│   ├── executor/              # Sandboxed execution
│   │   └── sandbox.py
│   └── safety/                # Safety checks
│       └── checker.py
├── examples/                   # Usage examples
│   ├── basic_usage.py
│   ├── with_openai.py
│   ├── with_kimi.py
│   ├── with_ollama.py
│   └── interactive_demo.py
├── tests/                      # Test suite
│   ├── test_memory.py
│   ├── test_safety.py
│   ├── test_loop.py
│   └── test_integration.py
├── ARCHITECTURE.md             # Detailed architecture docs
├── LOCAL_LLM_GUIDE.md          # Local LLM setup guide (Ollama, etc.)
├── KIMI_SETUP.md               # Kimi (Moonshot AI) setup guide
├── setup_local_llm.sh          # Automated setup script for local LLMs
└── requirements.txt
```

## 🤖 LLM Support

| Provider | Setup | Best For | Cost |
|----------|-------|----------|------|
| **Ollama** (Local) | `ollama pull qwen2.5-coder:7b` | Privacy, offline, free | Free |
| **OpenAI** | `export OPENAI_API_KEY=...` | Complex tasks, high quality | Pay-per-use |
| **Anthropic** | `export ANTHROPIC_API_KEY=...` | Long context, reasoning | Pay-per-use |
| **Kimi** | `export KIMI_API_KEY=...` | Chinese, cost-effective | Competitive |

### CLI Examples

```bash
# Local LLM (Ollama)
python -m agent "Sort a list" --llm ollama --model qwen2.5-coder:7b

# OpenAI
python -m agent "Create a REST API" --llm openai --model gpt-4

# Anthropic
python -m agent "Build a web scraper" --llm anthropic --model claude-3-opus

# Kimi
python -m agent "Create a calculator" --llm kimi --model moonshot-v1-32k

# Interactive mode
python -m agent --interactive --llm ollama
```

### Custom LLM Client

```python
from agent.models import LLMClient

class MyLLM(LLMClient):
    async def complete(self, prompt, system=None, temperature=0.7):
        # Your implementation
        return response

agent = SelfImprovingAgent(llm_client=MyLLM())
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AGENT ORCHESTRATOR                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   PLANNER   │───▶│   EXECUTOR  │───▶│    TESTER   │───▶│ REFLECTOR│ │
│  │             │    │             │    │             │    │          │ │
│  │ Break goal  │    │ Write code  │    │ Run tests   │    │ Analyze  │ │
│  │ into steps  │    │ in sandbox  │    │ & validate  │    │ failures │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └────┬─────┘ │
│       ▲─────────────────────────────────────────────────────────┘       │
│       │ (Loop back with learnings)                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
        │  SHORT-TERM │      │   LONG-TERM │      │   FAILURE   │
        │    MEMORY   │      │    MEMORY   │      │    MEMORY   │
        │  (Context)  │      │  (Patterns) │      │  (Mistakes) │
        └─────────────┘      └─────────────┘      └─────────────┘
```

### Memory System

| Type | Purpose | Retention |
|------|---------|-----------|
| **Short-Term** | Recent iteration context | Last 5 iterations |
| **Long-Term** | Successful solution patterns | Persistent |
| **Failure** | Error signatures + proven fixes | Persistent |

### Safety Layers

1. **Static Analysis**: AST-based analysis detects dangerous operations
2. **Pattern Detection**: Flags `eval`, `exec`, `__import__`, etc.
3. **Sandbox**: Resource limits (CPU, memory, time)
4. **Filesystem Restrictions**: Isolated temp directory
5. **Network Controls**: Disabled by default

## ⚙️ Configuration

```python
from agent import AgentConfig, LoopConfig

config = AgentConfig(
    # Loop settings
    loop=LoopConfig(
        max_iterations=10,        # Max attempts per task
        failure_threshold=3,       # Circuit breaker threshold
        failure_window=5,
        cooldown_period=60
    ),
    
    # Paths
    workspace_path="./workspace",
    state_path="./.agent_state",
    memory_path="./.agent_memory",
    
    # Sandbox settings
    enable_sandbox=True,
    sandbox_timeout=30,
    sandbox_memory_limit="512m",
    
    # LLM settings
    llm_temperature=0.7
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent

# Run specific test file
pytest tests/test_loop.py -v
```

## 📊 Monitoring

Track progress with callbacks:

```python
def on_iteration(state):
    print(f"Iteration {state.iteration}: {state.status}")

def on_code(code):
    print(f"Generated {len(code.source)} characters")

agent = SelfImprovingAgent(
    llm_client=llm,
    callbacks={
        'on_iteration': on_iteration,
        'on_plan': on_plan,
        'on_code': on_code,
        'on_test': on_test,
        'on_reflect': on_reflect
    }
)
```

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and design decisions
- **[LOCAL_LLM_GUIDE.md](LOCAL_LLM_GUIDE.md)** - Complete guide for running local LLMs (Ollama, LM Studio, etc.)
- **[KIMI_SETUP.md](KIMI_SETUP.md)** - Setup guide for Kimi (Moonshot AI)

## 🔧 Requirements

- Python 3.8+
- For local LLMs: [Ollama](https://ollama.com) recommended
- For cloud LLMs: API key from respective provider

## 📝 License

MIT License

## 🙏 Acknowledgments

Built with inspiration from autonomous agent research and the self-improvement loop concept.
