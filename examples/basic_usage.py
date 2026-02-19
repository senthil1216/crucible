"""
Basic usage example of the self-improving coding agent.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import MockLLMClient


async def main():
    """Run a simple example."""
    
    # Create a mock LLM client (replace with OpenAIClient or AnthropicClient)
    llm = MockLLMClient()
    
    # Configure the agent
    config = AgentConfig(
        workspace_path=Path("./workspace"),
        loop__max_iterations=5
    )
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Define a coding task
    goal = "Create a function that adds two numbers"
    
    print(f"🎯 Goal: {goal}")
    print("-" * 50)
    
    # Run the agent
    result = await agent.solve(goal)
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"\nStatus: {result.status.value}")
    print(f"Iterations: {result.iteration}")
    
    if result.status.value == "success":
        print(f"\n✅ Generated Code:\n")
        print(result.code.source)
    else:
        print(f"\n❌ Failed after {result.iteration} iterations")
        print(f"Analysis: {result.reflection.analysis}")


if __name__ == "__main__":
    asyncio.run(main())
