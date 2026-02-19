# Using Kimi (Moonshot AI) with the Coding Agent

[Kimi](https://www.moonshot.cn/) is a powerful Chinese LLM with excellent coding capabilities. It has an OpenAI-compatible API, making integration seamless.

## Setup

### 1. Get API Key

1. Visit https://platform.moonshot.cn/
2. Create an account
3. Generate an API key from the dashboard

### 2. Set Environment Variable

```bash
export KIMI_API_KEY="your-api-key-here"
```

Or add to your `.bashrc` / `.zshrc` for persistence.

### 3. Install Dependencies

```bash
pip install openai
```

## Usage

### Option 1: CLI

```bash
# Basic usage
python -m agent "Create a function to reverse a string" --llm kimi

# With specific model
python -m agent "Create a REST API" --llm kimi --model moonshot-v1-32k

# Interactive mode
python -m agent --interactive --llm kimi
```

### Option 2: Python Code

```python
import asyncio
from agent import SelfImprovingAgent, AgentConfig, LoopConfig
from agent.llm_clients import KimiClient

async def main():
    # Create Kimi client
    llm = KimiClient(
        api_key="your-api-key",  # Or use KIMI_API_KEY env var
        model="moonshot-v1-8k",   # Options: 8k, 32k, 128k
        max_tokens=2000
    )
    
    # Configure agent
    config = AgentConfig(
        workspace_path="./workspace",
        loop=LoopConfig(max_iterations=5)
    )
    
    # Create and run
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    result = await agent.solve("Create a fibonacci function")
    
    print(result.code.source)

asyncio.run(main())
```

### Option 3: Run Example

```bash
python examples/with_kimi.py
```

## Available Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `moonshot-v1-8k` | 8,192 tokens | Simple tasks, quick tests |
| `moonshot-v1-32k` | 32,768 tokens | Medium complexity tasks |
| `moonshot-v1-128k` | 128,000 tokens | Large codebases, complex tasks |

## Comparison with Other LLMs

| Feature | Kimi | OpenAI GPT-4 | Anthropic Claude |
|---------|------|--------------|------------------|
| API Format | OpenAI-compatible | Native | Native |
| Chinese | Excellent | Good | Good |
| Coding | Very Good | Excellent | Excellent |
| Pricing | Competitive | Higher | Higher |

## Troubleshooting

### "Kimi API key required"
Make sure you've set the `KIMI_API_KEY` environment variable or passed the key directly.

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai
```

### API Errors
Check your API key and ensure you have credits in your Moonshot account.

## Example Output

```
🚀 Starting task: Create a function to check if a number is prime
📋 Task ID: task_20240218_123456_abc123

============================================================
Iteration 1/5
Goal: Create a function to check if a number is prime
============================================================

[PLAN] Creating plan...
Steps: 4
Tests: 5

[EXECUTE] Generating code...
Generated 892 characters

[TEST] Running tests...
Passed: True

[REFLECT] Analyzing results...
Analysis: All tests passed successfully...

✅ SUCCESS! Tests passed.
💾 Stored successful pattern in long-term memory

============================================================
EXECUTION SUMMARY
============================================================
Status: SUCCESS
Iterations: 1
Tests Passed: True

✅ Success! Code generated and tested.
📄 File: main.py
📏 Size: 892 characters
```
