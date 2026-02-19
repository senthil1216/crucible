"""
Example using OpenAI GPT-4 for the coding agent.

Requires:
    pip install openai
    export OPENAI_API_KEY=your_key_here
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import OpenAIClient


async def main():
    """Run with OpenAI."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create OpenAI client
    llm = OpenAIClient(
        model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
        max_tokens=2000
    )
    
    # Configure agent
    config = AgentConfig(
        workspace_path=Path("./workspace"),
        loop__max_iterations=5,
        sandbox_timeout=30
    )
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Example tasks
    tasks = [
        "Create a function to reverse a string",
        "Implement a function to check if a number is prime",
        "Create a function that calculates the factorial of a number",
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


if __name__ == "__main__":
    asyncio.run(main())
