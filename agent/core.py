"""
Core Agent: Main orchestrator for the self-improving coding agent.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from agent.models import (
    AgentConfig, IterationState, Status, Plan, CodeArtifact
)
from agent.memory import ShortTermMemory, LongTermMemory, FailureMemory
from agent.planner import Planner
from agent.code_generator import CodeGenerator
from agent.tester import Tester
from agent.reflector import Reflector
from agent.executor.sandbox import SandboxedExecutor, ExecutionConfig
from agent.safety.checker import SafetyChecker
from agent.loop import ExecutionLoop
from agent.persistence import StateManager


class SelfImprovingAgent:
    """
    Self-improving coding agent that writes, tests, and refines code.
    
    Features:
    - Autonomous execution loop (Plan → Execute → Test → Reflect)
    - Memory hierarchy (short-term, long-term, failure memory)
    - Sandboxed code execution with safety checks
    - State persistence for resumability
    - Circuit breaker to prevent infinite loops
    """
    
    def __init__(
        self,
        llm_client,
        config: AgentConfig = None,
        callbacks: Dict[str, Callable] = None
    ):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLM client implementing the LLMClient protocol
            config: Agent configuration
            callbacks: Optional callbacks for progress updates
        """
        self.llm = llm_client
        self.config = config or AgentConfig()
        self.callbacks = callbacks or {}
        
        # Initialize workspace
        self.config.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(max_history=5)
        self.long_term_memory = LongTermMemory(self.config.memory_path / "patterns")
        self.failure_memory = FailureMemory(self.config.memory_path / "failures")
        
        # Initialize components
        self.safety_checker = SafetyChecker(project_dir=self.config.workspace_path)
        self.sandbox = SandboxedExecutor(
            config=ExecutionConfig(
                timeout_seconds=self.config.sandbox_timeout,
                memory_limit_mb=int(self.config.sandbox_memory_limit.replace('m', '').replace('g', '000')),
            ),
            safety_checker=self.safety_checker
        )
        
        self.planner = Planner(self.llm, memory=self.long_term_memory)
        self.code_generator = CodeGenerator(self.llm)
        self.tester = Tester(executor=self.sandbox)
        self.reflector = Reflector(self.llm, failure_memory=self.failure_memory)
        
        # State persistence
        self.state_manager = StateManager(self.config.state_path)
        
        # Execution loop
        self.loop = ExecutionLoop(
            planner=self.planner,
            code_generator=self.code_generator,
            tester=self.tester,
            reflector=self.reflector,
            short_term_memory=self.short_term_memory,
            config=self.config.loop,
            on_iteration=self.callbacks.get('on_iteration'),
            on_plan=self.callbacks.get('on_plan'),
            on_code=self.callbacks.get('on_code'),
            on_test=self.callbacks.get('on_test'),
            on_reflect=self.callbacks.get('on_reflect')
        )
    
    async def solve(
        self,
        goal: str,
        task_id: Optional[str] = None,
        resume: bool = False
    ) -> IterationState:
        """
        Solve a coding task autonomously.
        
        Args:
            goal: Description of the coding task
            task_id: Optional task identifier (generated if not provided)
            resume: Whether to resume from a previous checkpoint
        
        Returns:
            Final iteration state
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = self._generate_task_id(goal)
        
        print(f"🚀 Starting task: {goal}")
        print(f"📋 Task ID: {task_id}")
        
        # Check for existing state if resuming
        resume_from = None
        if resume:
            resume_from = await self.state_manager.load_checkpoint(task_id)
            if resume_from:
                print(f"📂 Resuming from iteration {resume_from.iteration}")
        
        # Retrieve similar solutions from long-term memory
        similar_solutions = await self.long_term_memory.find_similar_solutions(goal, k=2)
        if similar_solutions:
            print(f"📚 Found {len(similar_solutions)} similar past solutions")
        
        # Pre-planning with retrieved memories
        plan = await self.planner.create_plan(goal, similar_solutions)
        
        # Run the execution loop
        final_state = await self.loop.run(
            goal=goal,
            task_id=task_id,
            resume_from=resume_from
        )
        
        # Post-execution processing
        await self._post_execution(final_state, task_id, goal)
        
        return final_state
    
    async def _post_execution(
        self,
        state: IterationState,
        task_id: str,
        goal: str
    ) -> None:
        """Handle post-execution tasks."""
        # Save final state
        await self.state_manager.save_checkpoint(task_id, state)
        
        # Save task metadata
        await self.state_manager.save_task_metadata(task_id, {
            "goal": goal,
            "final_status": state.status.value,
            "iterations": state.iteration,
            "completed_at": datetime.now().isoformat()
        })
        
        # Store successful pattern in long-term memory
        if state.status == Status.SUCCESS:
            await self.long_term_memory.store_pattern(
                goal=goal,
                plan=state.plan,
                code=state.code,
                metadata={
                    "iterations": state.iteration,
                    "task_id": task_id
                }
            )
            print(f"💾 Stored successful pattern in long-term memory")
        
        # Clean up old checkpoints
        await self.state_manager.cleanup_old_checkpoints(task_id, keep_last=3)
        
        # Print summary
        self._print_summary(state)
    
    def _print_summary(self, state: IterationState) -> None:
        """Print execution summary."""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {state.status.value.upper()}")
        print(f"Iterations: {state.iteration}")
        print(f"Tests Passed: {state.test_results.passed}")
        
        if state.status == Status.SUCCESS:
            print(f"\n✅ Success! Code generated and tested.")
            print(f"📄 File: {state.code.file_path}")
            print(f"📏 Size: {len(state.code.source)} characters")
        else:
            print(f"\n❌ Task did not complete successfully.")
            if state.reflection.analysis:
                print(f"Final analysis: {state.reflection.analysis}")
        
        # Memory stats
        lt_stats = self.long_term_memory.get_stats()
        fail_stats = self.failure_memory.get_stats()
        print(f"\n📊 Memory Stats:")
        print(f"   Patterns learned: {lt_stats['total_patterns']}")
        print(f"   Failures recorded: {fail_stats['total_failures']}")
    
    def _generate_task_id(self, goal: str) -> str:
        """Generate a unique task ID from the goal."""
        # Use first 50 chars of goal + timestamp hash
        goal_hash = hashlib.sha256(goal[:50].encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"task_{timestamp}_{goal_hash}"
    
    async def get_task_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get full history of a task."""
        metadata = await self.state_manager.load_task_metadata(task_id)
        checkpoints = await self.state_manager.list_checkpoints(task_id)
        
        return {
            "metadata": metadata,
            "checkpoints": checkpoints
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        lt_stats = self.long_term_memory.get_stats()
        fail_stats = self.failure_memory.get_stats()
        loop_stats = self.loop.get_stats()
        
        return {
            "memory": {
                "patterns": lt_stats,
                "failures": fail_stats
            },
            "loop": loop_stats,
            "tasks": len(self.state_manager.list_tasks())
        }
    
    async def reset_memory(self, confirm: bool = False) -> None:
        """Reset all memory (use with caution)."""
        if not confirm:
            print("⚠️ Use confirm=True to actually reset memory")
            return
        
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.failure_memory.clear()
        print("🧹 Memory cleared")
    
    async def interactive_session(self) -> None:
        """Run an interactive session with the agent."""
        print("🤖 Self-Improving Coding Agent")
        print("Type 'exit' to quit, 'status' for stats, 'help' for commands")
        
        while True:
            try:
                goal = input("\n📝 Goal: ").strip()
                
                if not goal:
                    continue
                
                if goal.lower() == 'exit':
                    break
                
                if goal.lower() == 'status':
                    stats = self.get_stats()
                    print(f"\n📊 Stats: {stats}")
                    continue
                
                if goal.lower() == 'help':
                    print("\nCommands:")
                    print("  exit   - Exit the session")
                    print("  status - Show agent statistics")
                    print("  help   - Show this help")
                    print("  <goal> - Any coding task description")
                    continue
                
                # Run the task
                result = await self.solve(goal)
                
                if result.status == Status.SUCCESS:
                    print(f"\n💾 Code saved to: {result.code.file_path}")
                    print("\n📝 Generated code:")
                    print("-" * 40)
                    print(result.code.source[:1000])
                    if len(result.code.source) > 1000:
                        print("... (truncated)")
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted")
                break
            except Exception as e:
                print(f"\n💥 Error: {e}")
