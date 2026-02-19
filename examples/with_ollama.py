"""
Example using local LLM via Ollama.

This is the best way to run the agent with a free, local LLM.

Setup:
    1. Install Ollama: https://ollama.com/
    2. Pull a model: ollama pull qwen2.5-coder:7b
    3. Start server: ollama serve
    4. Run this script

Recommended models for coding:
    - qwen2.5-coder:7b    (best balance, ~4GB)
    - qwen2.5-coder:14b   (better quality, ~8GB)
    - codellama:7b        (Meta's coding model)
    - codellama:13b       (better quality)
    - deepseek-coder-v2:16b (excellent for code)
    - llama3.1:8b         (good general purpose)

Usage:
    python examples/with_ollama.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import SelfImprovingAgent, AgentConfig, LoopConfig
from agent.llm_clients import OllamaClient


async def main():
    """Run with local Ollama LLM."""
    
    print("🦙 Local LLM Coding Agent with Ollama")
    print("=" * 60)
    
    # Create Ollama client
    # Make sure you have pulled the model first:
    #   ollama pull qwen2.5-coder:7b
    llm = OllamaClient(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434",
        max_tokens=2000,
        temperature=0.2,  # Lower for more deterministic code
        timeout=120
    )
    
    # Configure agent
    config = AgentConfig(
        workspace_path=Path("./ollama_workspace"),
        loop=LoopConfig(max_iterations=5),
        sandbox_timeout=30
    )
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Example coding tasks
    # Start with simple tasks for 7B models
    tasks = [
        "Create a function to check if a number is prime",
        "Create a function to reverse a string",
        "Create a function to calculate factorial",
        # More complex tasks work better with 14B+ models
        # "Create a class to manage a todo list with add/remove/complete methods",
    ]
    
    for goal in tasks:
        print(f"\n{'='*60}")
        print(f"🎯 Task: {goal}")
        print(f"{'='*60}")
        
        try:
            result = await agent.solve(goal)
            
            if result.status.value == "success":
                print(f"\n✅ Success!")
                print(f"Generated Code:\n{'-'*40}")
                print(result.code.source)
            else:
                print(f"\n❌ Failed: {result.reflection.analysis}")
                
        except RuntimeError as e:
            print(f"\n💥 Error: {e}")
            print("\nMake sure Ollama is running: ollama serve")
            print("And the model is pulled: ollama pull qwen2.5-coder:7b")
            break
    
    # Show stats
    stats = agent.get_stats()
    print(f"\n{'='*60}")
    print("📊 Session Stats")
    print(f"{'='*60}")
    print(f"Patterns learned: {stats['memory']['patterns']['total_patterns']}")
    print(f"Failures recorded: {stats['memory']['failures']['total_failures']}")


if __name__ == "__main__":
    asyncio.run(main())
