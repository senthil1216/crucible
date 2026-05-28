"""
CLI entry point for the self-improving coding agent.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

from agent.core import SelfImprovingAgent
from agent.llm_clients import (
    MockLLMClient, OpenAIClient, AnthropicClient,
    KimiClient, DeepSeekClient, OllamaClient, XAIClient
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
        choices=["mock", "openai", "anthropic", "kimi", "deepseek", "ollama", "xai"],
        default=os.getenv("AGENT_LLM", "ollama"),
        help="LLM provider to use (default: ollama, or $AGENT_LLM)"
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

    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker-based sandbox (allows installing packages like fastapi inside the container)"
    )

    parser.add_argument(
        "--docker-image",
        default=os.getenv("AGENT_DOCKER_IMAGE", "python:3.12-slim"),
        help="Docker image to use when --docker is enabled "
             "(default: python:3.12-slim, or $AGENT_DOCKER_IMAGE; "
             "use the prebaked 'crucible-runtime' to skip per-task setup)"
    )

    parser.add_argument(
        "--docker-persistent",
        action="store_true",
        help="Use one persistent container for the whole task (enables pip install etc.)"
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="After a server task passes its tests, launch it and leave it running "
             "(reachable on http://localhost:<port>). Requires --docker-persistent."
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port to publish the running app on (with --run). Default: 8000"
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
        model = args.model or os.getenv("AGENT_OLLAMA_MODEL", "qwen2.5-coder:7b")
        return OllamaClient(model=model)

    elif args.llm == "xai":
        model = args.model or os.getenv("XAI_MODEL", "grok-4.3")
        return XAIClient(model=model)

    else:
        raise ValueError(f"Unknown LLM: {args.llm}")


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    if args.run and not args.docker_persistent:
        print("⚠️  --run requires --docker-persistent; ignoring --run.")
        args.run = False

    config = AgentConfig(
        loop=LoopConfig(max_iterations=args.max_iterations),
        workspace_path=args.workspace,
        use_docker=args.docker,
        docker_image=args.docker_image,
        docker_persistent=args.docker_persistent,
        run_app=args.run,
        app_port=args.port,
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
