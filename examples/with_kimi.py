"""
Example using Kimi (Moonshot AI) for the coding agent.

Kimi is a Chinese LLM with excellent coding capabilities.

Setup:
    1. Get API key from https://platform.moonshot.cn/
    2. Set environment variable: export KIMI_API_KEY=your_key_here
    3. pip install openai

Usage:
    python examples/with_kimi.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import SelfImprovingAgent, AgentConfig, LoopConfig
from agent.llm_clients import KimiClient


async def main():
    """Run with Kimi."""
    
    # Check for API key
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        print("❌ Please set KIMI_API_KEY environment variable")
        print("   Get your API key from: https://platform.moonshot.cn/")
        return
    
    # Create Kimi client
    # Available models:
    # - moonshot-v1-8k   : 8K context window
    # - moonshot-v1-32k  : 32K context window  
    # - moonshot-v1-128k : 128K context window
    llm = KimiClient(
        api_key=api_key,
        model="moonshot-v1-8k",  # Use 8K for simple tasks
        max_tokens=2000
    )
    
    # Configure agent
    config = AgentConfig(
        workspace_path=Path("./kimi_workspace"),
        loop=LoopConfig(max_iterations=5),
        sandbox_timeout=30
    )
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Example coding tasks
    tasks = [
        "Create a function to check if a string is a palindrome",
        "Implement a function to calculate the factorial of a number",
        "Create a function that finds the maximum value in a list",
        "Write a function to reverse a linked list",
    ]
    
    for goal in tasks:
        print(f"\n{'='*60}")
        print(f"🎯 Task: {goal}")
        print(f"{'='*60}")
        
        result = await agent.solve(goal)
        
        if result.status.value == "success":
            print(f"\n✅ Success!")
            print(f"Code:\n{result.code.source}")
        else:
            print(f"\n❌ Failed: {result.reflection.analysis}")
    
    # Show stats
    stats = agent.get_stats()
    print(f"\n{'='*60}")
    print("📊 Session Stats")
    print(f"{'='*60}")
    print(f"Patterns learned: {stats['memory']['patterns']['total_patterns']}")
    print(f"Failures recorded: {stats['memory']['failures']['total_failures']}")


if __name__ == "__main__":
    asyncio.run(main())
