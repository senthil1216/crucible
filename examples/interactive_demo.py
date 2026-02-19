"""
Interactive demo with callbacks to see the agent's thought process.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import SelfImprovingAgent, AgentConfig
from agent.llm_clients import MockLLMClient
from agent.models import Plan, CodeArtifact, TestResults, Reflection, IterationState


def on_plan(plan: Plan):
    """Called when a plan is created."""
    print(f"\n📋 PLAN CREATED")
    print(f"   Language: {plan.language}")
    print(f"   Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"   {i}. {step}")


def on_code(code: CodeArtifact):
    """Called when code is generated."""
    print(f"\n💻 CODE GENERATED")
    print(f"   File: {code.file_path}")
    print(f"   Size: {len(code.source)} characters")
    print(f"   Preview: {code.source[:100]}...")


def on_test(results: TestResults):
    """Called when tests complete."""
    print(f"\n🧪 TEST RESULTS")
    print(f"   Passed: {results.passed}")
    if not results.passed:
        print(f"   Error: {results.error_type}")
        if results.stderr:
            print(f"   Details: {results.stderr[:150]}")


def on_reflect(reflection: Reflection):
    """Called when reflection is complete."""
    print(f"\n🤔 REFLECTION")
    print(f"   Analysis: {reflection.analysis[:100]}...")
    if reflection.suggested_fix:
        print(f"   Fix: {reflection.suggested_fix[:100]}...")
    print(f"   Continue: {reflection.should_continue}")


def on_iteration(state: IterationState):
    """Called after each iteration."""
    print(f"\n🔄 ITERATION {state.iteration} COMPLETE")
    print(f"   Status: {state.status.value}")


async def main():
    """Run interactive demo."""
    
    # Setup callbacks
    callbacks = {
        'on_plan': on_plan,
        'on_code': on_code,
        'on_test': on_test,
        'on_reflect': on_reflect,
        'on_iteration': on_iteration
    }
    
    # Create agent
    llm = MockLLMClient()
    config = AgentConfig(
        workspace_path=Path("./workspace"),
        loop__max_iterations=3
    )
    
    agent = SelfImprovingAgent(
        llm_client=llm,
        config=config,
        callbacks=callbacks
    )
    
    # Run task
    goal = "Create a bubble sort function"
    print(f"🎯 Goal: {goal}")
    print("=" * 60)
    
    result = await agent.solve(goal)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Total iterations: {result.iteration}")
    
    if result.status.value == "success":
        print(f"\nFinal code:\n{result.code.source}")


if __name__ == "__main__":
    asyncio.run(main())
