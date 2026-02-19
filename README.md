# Self-Improving Coding Agent

An autonomous agent that writes code, runs tests, and learns from failures. Uses a **Plan → Execute → Test → Reflect** loop with memory hierarchy and safety guardrails.

## 🎯 Overview

Unlike a chatbot that waits for prompts, this agent waits for goals. It doesn't stop until the code is functional.

```
User Goal → [Plan] → [Execute] → [Test] → [Reflect] → Success?
                ↑_________________________________________|
```

## 🏗️ Architecture

### Core Components

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

### Key Features

- **Execution Loop**: Plan → Execute → Test → Reflect cycle with max iteration limit
- **Circuit Breaker**: Stops infinite loops after consecutive failures
- **Memory Hierarchy**:
  - *Short-term*: Rolling window of recent iterations (last 5)
  - *Long-term*: Indexes successful patterns by problem type
  - *Failure*: Stores error signatures with solutions
- **Sandboxed Execution**: Resource-limited, isolated environment
- **Safety Checks**: Static analysis before execution
- **State Persistence**: Resume after interruptions

## 🚀 Quick Start

### Installation

```bash
git clone <repo>
cd self-improving-agent
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OpenAIClient

async def main():
    # Setup LLM client
    llm = OpenAIClient(api_key="your-key", model="gpt-4")
    
    # Create agent
    config = AgentConfig(
        workspace_path="./workspace",
        loop__max_iterations=10
    )
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Run task
    result = await agent.solve("Create a function to calculate fibonacci numbers")
    
    if result.status.value == "success":
        print(result.code.source)

asyncio.run(main())
```

### CLI Usage

```bash
# Interactive mode
python -m agent --interactive

# Single task
python -m agent "Create a function that sorts a list" --llm openai

# With options
python -m agent "Create a REST API" \
    --llm openai \
    --model gpt-4 \
    --max-iterations 5 \
    --workspace ./my_project
```

## 📁 Project Structure

```
agent/
├── __init__.py              # Main exports
├── __main__.py              # CLI entry point
├── core.py                  # Agent orchestrator
├── models.py                # Data models
├── loop.py                  # Execution loop + circuit breaker
├── planner.py               # Plan generation
├── code_generator.py        # Code generation
├── tester.py                # Test runner
├── reflector.py             # Failure analysis
├── persistence.py           # State persistence
├── llm_clients.py           # LLM implementations
├── memory/                  # Memory hierarchy
│   ├── short_term.py
│   ├── long_term.py
│   └── failure_memory.py
├── executor/                # Sandboxed execution
│   └── sandbox.py
└── safety/                  # Safety checks
    └── checker.py

examples/
├── basic_usage.py
├── with_openai.py
└── interactive_demo.py

tests/
├── test_memory.py
├── test_safety.py
├── test_loop.py
└── test_integration.py
```

## 🔧 Configuration

```python
from agent import AgentConfig, LoopConfig

config = AgentConfig(
    # Loop settings
    loop=LoopConfig(
        max_iterations=10,
        failure_threshold=3,
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
    llm_model="gpt-4",
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
pytest tests/test_loop.py
```

## 🛡️ Safety

The agent implements multiple safety layers:

1. **Static Analysis**: AST-based analysis detects dangerous operations
2. **Pattern Detection**: Flags `eval`, `exec`, `__import__`, etc.
3. **Sandbox**: Resource limits (CPU, memory, time)
4. **Filesystem Restrictions**: Read-only access outside project directory
5. **Network Controls**: Disabled by default

### Safety Levels

- `SAFE`: No issues detected
- `WARNING`: Potentially risky but allowed
- `DANGEROUS`: Blocked from execution

## 🧠 Memory System

### Short-Term Memory
- Keeps last 5 iterations
- Provides context for next iteration
- Detects repeating errors

### Long-Term Memory
- Stores successful solutions
- Indexes by problem similarity
- Retrieved for similar future tasks

### Failure Memory
- Stores error signatures
- Matches similar failures
- Suggests proven fixes

## 🔄 Execution Loop

```python
for iteration in range(max_iterations):
    # 1. PLAN
    plan = planner.create_plan(goal, similar_solutions)
    
    # 2. EXECUTE (Generate code)
    code = code_generator.generate(plan)
    
    # 3. TEST (in sandbox)
    results = tester.run_tests(code, plan)
    
    # 4. REFLECT
    reflection = reflector.analyze(results, code, plan)
    
    # Check termination
    if results.passed:
        return success
    
    if not reflection.should_continue:
        return failed
    
    # Circuit breaker check
    if too_many_failures:
        return circuit_breaker_open
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

## 🤝 LLM Support

Built-in clients for:
- **OpenAI** (`gpt-4`, `gpt-3.5-turbo`)
- **Anthropic** (`claude-3-opus`, `claude-3-sonnet`)
- **Mock** (for testing)

Custom client:
```python
from agent.models import LLMClient

class MyLLM(LLMClient):
    async def complete(self, prompt, system=None, temperature=0.7):
        # Your implementation
        return response
```

## 📈 Roadmap

- [ ] Vector embeddings for semantic similarity
- [ ] Multi-file project generation
- [ ] Git integration
- [ ] Web UI
- [ ] Parallel execution of test variations
- [ ] Custom tool integration

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.
