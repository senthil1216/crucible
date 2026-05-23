"""
CLI entry point for the self-improving coding agent.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from agent.core import SelfImprovingAgent
from agent.llm_clients import (
    MockLLMClient, OpenAIClient, AnthropicClient, 
    KimiClient, DeepSeekClient, OllamaClient
)
from agent.models import AgentConfig, LoopConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Self-improving coding agent"
    )
    
    parser.add_argument(
        "goal",
        nargs="?",
        help="Coding goal/task to accomplish"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--llm",
        choices=["mock", "openai", "anthropic", "kimi", "deepseek", "ollama"],
        default="mock",
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model",
        help="Model name (for OpenAI/Anthropic)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per task"
    )
    
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("./workspace"),
        help="Workspace directory"
    )
    
    parser.add_argument(
        "--task-id",
        help="Task ID for resuming"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show agent statistics"
    )
    
    return parser


def create_llm_client(args):
    """Create LLM client based on arguments."""
    if args.llm == "mock":
        return MockLLMClient()
    
    elif args.llm == "openai":
        model = args.model or "gpt-4"
        return OpenAIClient(model=model)
    
    elif args.llm == "anthropic":
        model = args.model or "claude-3-opus-20240229"
        return AnthropicClient(model=model)
    
    elif args.llm == "kimi":
        model = args.model or "moonshot-v1-8k"
        return KimiClient(model=model)
    
    elif args.llm == "deepseek":
        model = args.model or "deepseek-chat"
        return DeepSeekClient(model=model)
    
    elif args.llm == "ollama":
        model = args.model or "qwen2.5-coder:7b"
        return OllamaClient(model=model)
    
    else:
        raise ValueError(f"Unknown LLM: {args.llm}")


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = AgentConfig(
        loop=LoopConfig(max_iterations=args.max_iterations),
        workspace_path=args.workspace
    )
    
    # Create LLM client
    llm = create_llm_client(args)
    
    # Create agent
    agent = SelfImprovingAgent(llm_client=llm, config=config)
    
    # Show stats only
    if args.stats:
        stats = agent.get_stats()
        print("\n📊 Agent Statistics:")
        print(f"  Patterns learned: {stats['memory']['patterns']['total_patterns']}")
        print(f"  Failures recorded: {stats['memory']['failures']['total_failures']}")
        print(f"  Tasks completed: {stats['tasks']}")
        return
    
    # Interactive mode
    if args.interactive or not args.goal:
        await agent.interactive_session()
        return
    
    # Single task mode
    result = await agent.solve(
        goal=args.goal,
        task_id=args.task_id,
        resume=args.resume
    )
    
    # Exit with appropriate code
    if result.status.value == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
