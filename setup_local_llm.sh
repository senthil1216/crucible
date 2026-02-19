#!/bin/bash
# Setup script for local LLM with Ollama

set -e

echo "🦙 Local LLM Setup for Coding Agent"
echo "===================================="
echo ""

# Check OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    
    if [ "$OS" = "Darwin" ]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        # Linux
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Please install Ollama manually from https://ollama.com/download"
        exit 1
    fi
    
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q openai httpx

# Pull recommended model
echo ""
echo "🤖 Pulling recommended model (qwen2.5-coder:7b)..."
echo "   This may take a few minutes depending on your internet..."
ollama pull qwen2.5-coder:7b

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start Ollama server: ollama serve"
echo "2. In another terminal, run the agent:"
echo "   python -m agent 'Create a fibonacci function' --llm ollama"
echo ""
echo "Or try the example:"
echo "   python examples/with_ollama.py"
