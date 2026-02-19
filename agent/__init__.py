"""
Self-Improving Coding Agent

An autonomous agent that writes code, runs tests, and learns from failures.
Uses a Plan → Execute → Test → Reflect loop with memory and safety guardrails.

Example:
    from agent import SelfImprovingAgent
    from agent.llm_clients import OpenAIClient
    
    llm = OpenAIClient()
    agent = SelfImprovingAgent(llm)
    
    result = await agent.solve("Create a function to calculate fibonacci numbers")
    print(result.code.source)
"""

from agent.core import SelfImprovingAgent
from agent.models import (
    AgentConfig,
    LoopConfig,
    Plan,
    CodeArtifact,
    TestResults,
    Reflection,
    IterationState,
    Status
)

__version__ = "0.1.0"
__all__ = [
    "SelfImprovingAgent",
    "AgentConfig",
    "LoopConfig", 
    "Plan",
    "CodeArtifact",
    "TestResults",
    "Reflection",
    "IterationState",
    "Status"
]

# LLM clients are available from agent.llm_clients
# from agent.llm_clients import KimiClient, OpenAIClient, AnthropicClient, MockLLMClient
