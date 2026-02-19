# Local LLM Setup Guide for Coding Agent

## Best Open-Source/Open-Weight Models for Coding

### Top Recommendations

| Model | Size | Best For | VRAM Required |
|-------|------|----------|---------------|
| **Qwen 2.5 Coder** | 7B | Best balance of speed/quality | 8GB |
| **Qwen 2.5 Coder** | 14B | Better code quality | 16GB |
| **DeepSeek Coder V2** | 16B | Excellent for complex coding | 16GB |
| **CodeLlama** | 7B | Good baseline | 8GB |
| **CodeLlama** | 13B | Better quality | 16GB |
| **Llama 3.1** | 8B | Great general purpose | 8GB |
| **Mistral 7B** | 7B | Fast inference | 8GB |

### My Recommendation: Qwen 2.5 Coder 7B
- **Why**: Best coding performance for its size
- **Speed**: Fast even on CPU
- **Quality**: Matches larger models on coding benchmarks
- **Context**: 128K context window

## Setup Options

### Option 1: Ollama (Recommended - Easiest)

Ollama runs models locally with a simple CLI and OpenAI-compatible API.

#### Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

#### Download a Model

```bash
# Best for coding (recommended)
ollama pull qwen2.5-coder:7b

# Or other options
ollama pull codellama:7b
ollama pull codellama:13b
ollama pull deepseek-coder-v2:16b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

#### Start Ollama Server

```bash
# Run in background
ollama serve

# Or with custom port
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

#### Test Ollama

```bash
# Test in terminal
ollama run qwen2.5-coder:7b

# Or via API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b",
  "prompt": "Write a Python function to calculate fibonacci numbers"
}'
```

### Option 2: LM Studio (GUI - Easiest for Beginners)

1. Download from https://lmstudio.ai/
2. Search and download models from the UI
3. Start local server (Settings → Enable Local Inference Server)
4. API runs at `http://localhost:1234/v1`

### Option 3: llama.cpp (Fastest CPU Inference)

Best for running on CPU without GPU.

```bash
# Install
pip install llama-cpp-python

# Download model (GGUF format)
wget https://huggingface.co/TheBloke/Qwen2.5-Coder-7B-GGUF/resolve/main/qwen2.5-coder-7b-q4_k_m.gguf

# Run with server
python -m llama_cpp.server --model qwen2.5-coder-7b-q4_k_m.gguf --host 0.0.0.0 --port 8000
```

### Option 4: vLLM (Fastest GPU Serving)

Best for high-throughput GPU serving.

```bash
# Install
pip install vllm

# Run server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000
```

## Hardware Requirements

### Minimum (Slow but works)
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 10GB free

### Recommended (Good performance)
- **CPU**: Modern 8-core
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **RAM**: 32GB
- **Storage**: 20GB free SSD

### Ideal (Fast inference)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or RTX 3090
- **RAM**: 64GB
- **Storage**: 50GB free NVMe SSD

## Integration with Coding Agent

I've added an `OllamaClient` to the agent. Here's how to use it:

### Quick Start

```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model
ollama pull qwen2.5-coder:7b

# 3. Run agent with local LLM
python -m agent "Create a function to reverse a string" --llm ollama
```

### Python Usage

```python
import asyncio
from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OllamaClient

async def main():
    # Create Ollama client
    llm = OllamaClient(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434"
    )
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm)
    
    # Solve task
    result = await agent.solve("Create a fibonacci function")
    print(result.code.source)

asyncio.run(main())
```

### Advanced Configuration

```python
from agent.llm_clients import OllamaClient

llm = OllamaClient(
    model="qwen2.5-coder:14b",
    base_url="http://localhost:11434",
    max_tokens=4000,  # Increase for larger code
    temperature=0.2,   # Lower for more deterministic code
    timeout=120       # Longer timeout for large models
)
```

## Model Recommendations by Hardware

### No GPU (CPU Only)
```bash
# Best option for CPU
ollama pull qwen2.5-coder:7b

# Alternative: Smaller model
ollama pull qwen2.5-coder:1.5b
```

### 8GB VRAM (RTX 3060, RTX 4060)
```bash
# Fits comfortably
ollama pull qwen2.5-coder:7b
ollama pull codellama:7b
```

### 16GB VRAM (RTX 3080, RTX 4080)
```bash
# Larger models
ollama pull qwen2.5-coder:14b
ollama pull codellama:13b
ollama pull deepseek-coder-v2:16b
```

### 24GB+ VRAM (RTX 3090, RTX 4090)
```bash
# Best quality
ollama pull qwen2.5-coder:32b
ollama pull codellama:34b
```

## Performance Tips

### 1. Quantization
Use quantized models (Q4_K_M or Q5_K_M) for faster inference with minimal quality loss.

```bash
# Ollama uses quantization automatically
ollama pull qwen2.5-coder:7b-q4_0  # Smaller, faster
ollama pull qwen2.5-coder:7b-q8_0  # Larger, better quality
```

### 2. Context Length
Reduce context window if you don't need long inputs:

```python
llm = OllamaClient(
    model="qwen2.5-coder:7b",
    num_ctx=4096  # Reduce from default 8192
)
```

### 3. GPU Layers
Ensure Ollama uses your GPU:

```bash
# Check GPU usage
ollama ps

# Force GPU (usually automatic)
export OLLAMA_GPU_OVERHEAD=1
```

## Comparison: Local vs Cloud

| Aspect | Local (Qwen 7B) | Cloud (GPT-4) |
|--------|-----------------|---------------|
| **Speed** | Fast (GPU) / Slow (CPU) | Fast |
| **Quality** | Good for simple tasks | Excellent |
| **Cost** | Free after setup | Per token |
| **Privacy** | 100% private | Sends to cloud |
| **Offline** | Works offline | Requires internet |
| **Complex tasks** | May struggle | Handles well |

## Troubleshooting

### "Connection refused" error
```bash
# Make sure Ollama is running
ollama serve

# Check if model exists
ollama list
```

### Out of memory
```bash
# Use smaller model
ollama pull qwen2.5-coder:1.5b

# Or reduce context
# Set in client: num_ctx=2048
```

### Slow inference
```bash
# Check GPU is being used
ollama ps

# Use quantized model
ollama pull qwen2.5-coder:7b-q4_0
```

## Quick Reference Card

```bash
# Start Ollama
ollama serve

# Pull models
ollama pull qwen2.5-coder:7b

# List models
ollama list

# Run interactively
ollama run qwen2.5-coder:7b

# Use with agent
python -m agent "Task" --llm ollama --model qwen2.5-coder:7b
```

## Recommended Setup

For most users, I recommend:

1. **Install Ollama**: `brew install ollama` or from website
2. **Download model**: `ollama pull qwen2.5-coder:7b`
3. **Run agent**: `python -m agent "Your task" --llm ollama`

This gives you a completely free, private, offline-capable coding agent!
