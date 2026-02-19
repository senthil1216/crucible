# Self-Improving Coding Agent - Architecture

## Complete Build Summary

### Project Structure
```
agent/
├── core.py              # Main orchestrator (SelfImprovingAgent class)
├── loop.py              # Execution loop + Circuit Breaker
├── models.py            # All data classes (Plan, CodeArtifact, etc.)
├── planner.py           # Plan generation with memory retrieval
├── code_generator.py    # Code generation with fix capabilities
├── tester.py            # Test runner using sandbox
├── reflector.py         # Failure analysis + error signature extraction
├── persistence.py       # State checkpointing for resumability
├── llm_clients.py       # Mock, OpenAI, Anthropic implementations
├── memory/
│   ├── short_term.py    # Rolling window (last 5 iterations)
│   ├── long_term.py     # Successful pattern storage + similarity search
│   └── failure_memory.py # Error signatures + proven fixes
├── executor/
│   └── sandbox.py       # Resource-limited code execution
└── safety/
    └── checker.py       # AST-based static analysis

examples/
├── basic_usage.py       # Simple example
├── with_openai.py       # Real LLM integration
└── interactive_demo.py  # With progress callbacks

tests/
├── test_memory.py       # Memory hierarchy tests
├── test_safety.py       # Safety checker tests
├── test_loop.py         # Execution loop + circuit breaker tests
└── test_integration.py  # End-to-end tests
```

## Key Architectural Decisions

### 1. Execution Loop Design
- **Plan → Execute → Test → Reflect** cycle
- State stored after each phase for resumability
- Circuit breaker stops after 3 consecutive failures
- Max iteration limit (default: 10)

### 2. Memory Hierarchy
```python
Short-term: deque(maxlen=5)     # Recent context
Long-term:  similarity search   # Successful patterns
Failure:     error signatures   # Proven fixes
```

### 3. Sandboxing Strategy
- Subprocess isolation with resource limits
- CPU time limit (default: 10s)
- Memory limit (default: 512MB)
- Filesystem restricted to temp directory
- Network disabled by default

### 4. Safety Layers
1. Static AST analysis
2. Dangerous pattern detection (eval, exec, etc.)
3. Sandboxed execution
4. Resource limits

### 5. Reflection Mechanism
- Extracts error signatures (type + normalized message)
- Searches failure memory for similar errors
- Generates hypothesis + fix suggestion
- Determines if continued attempts are worthwhile

## Usage Examples

### Basic Usage
```python
from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OpenAIClient

llm = OpenAIClient(api_key="your-key")
config = AgentConfig(workspace_path="./workspace")
agent = SelfImprovingAgent(llm, config)

result = await agent.solve("Create a fibonacci function")
print(result.code.source)
```

### With Callbacks
```python
def on_iteration(state):
    print(f"Iter {state.iteration}: {state.status}")

agent = SelfImprovingAgent(
    llm_client=llm,
    callbacks={'on_iteration': on_iteration}
)
```

### CLI
```bash
# Interactive
python -m agent --interactive

# Single task
python -m agent "Sort a list" --llm openai
```

## Test Results
```bash
$ pytest tests/ -v

tests/test_memory.py::TestShortTermMemory::test_add_and_retrieve PASSED
tests/test_memory.py::TestLongTermMemory::test_store_and_retrieve PASSED
tests/test_memory.py::TestFailureMemory::test_find_similar_failures PASSED
tests/test_safety.py::TestSafetyChecker::test_safe_code PASSED
tests/test_safety.py::TestSafetyChecker::test_dangerous_eval PASSED
tests/test_loop.py::TestCircuitBreaker::test_circuit_opens_on_failures PASSED
tests/test_loop.py::TestExecutionLoop::test_successful_run PASSED
tests/test_integration.py::TestAgentIntegration::test_solve_simple_task PASSED
```

## Key Classes

| Class | Responsibility |
|-------|---------------|
| `SelfImprovingAgent` | Main orchestrator, wires components |
| `ExecutionLoop` | Runs Plan→Execute→Test→Reflect cycle |
| `CircuitBreaker` | Prevents infinite loops |
| `ShortTermMemory` | Rolling window of recent iterations |
| `LongTermMemory` | Successful pattern retrieval |
| `FailureMemory` | Error signature matching |
| `SafetyChecker` | AST-based code analysis |
| `SandboxedExecutor` | Resource-limited code execution |
| `Reflector` | Failure analysis + fix suggestion |

## Data Flow

```
User Goal
    ↓
[PLANNER] ← Retrieves similar solutions from LongTermMemory
    ↓
Plan
    ↓
[CODE GENERATOR] ← Gets feedback from ShortTermMemory
    ↓
Code Artifact
    ↓
[SAFETY CHECKER] ← Static analysis
    ↓
[SANDBOX] ← Resource-limited execution
    ↓
Test Results
    ↓
[REFLECTOR] ← Searches FailureMemory for similar errors
    ↓
Reflection (success/failure + fix suggestion)
    ↓
Either: SUCCESS → Store in LongTermMemory
    Or: LOOP BACK with fix suggestion
```

## Production Considerations

1. **Persistence**: State saved to disk after each iteration
2. **Resumability**: Can resume after crashes/interruptions
3. **Learning**: Stores successful patterns and failures
4. **Safety**: Multiple layers of protection
5. **Observability**: Callbacks for monitoring progress

## Extension Points

- Custom LLM clients via `LLMClient` protocol
- Custom safety rules in `SafetyChecker`
- Custom memory backends
- Custom sandbox environments (Docker, etc.)
