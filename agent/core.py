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
from agent.dependency_manager import DependencyManager  # Phase 2
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

        # Choose execution backend (Phase 1 Docker support)
        exec_config = ExecutionConfig(
            timeout_seconds=self.config.sandbox_timeout,
            memory_limit_mb=self._parse_memory_limit(self.config.sandbox_memory_limit),
        )

        if getattr(self.config, "use_docker", False):
            from agent.executor.docker_executor import DockerExecutor
            self.sandbox = DockerExecutor(
                config=exec_config,
                safety_checker=self.safety_checker,
                docker_image=getattr(self.config, "docker_image", "python:3.12-slim"),
                persistent=getattr(self.config, "docker_persistent", False),
                enable_network=getattr(self.config, "docker_enable_network", True),
                install_build_tools=getattr(self.config, "docker_install_build_tools", True),
            )

            # Create DependencyManager for automatic recovery
            self.dependency_manager = DependencyManager(executor=self.sandbox)
        else:
            self.sandbox = SandboxedExecutor(
                config=exec_config,
                safety_checker=self.safety_checker,
            )
            self.dependency_manager = None
        
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
            on_reflect=self.callbacks.get('on_reflect'),
            install_packages=self.ensure_packages,           # legacy fallback
            dependency_manager=self.dependency_manager,      # Phase 2 preferred path
        )

    def ensure_packages(self, packages: list[str]) -> bool:
        """Install packages inside the persistent container (if available)."""
        if self.dependency_manager:
            try:
                result = self.dependency_manager.install_packages(packages)
                return result.success
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # Phase 2: Workspace Access (Persistent Docker mode)
    # ------------------------------------------------------------------

    @property
    def workspace_path(self) -> Optional[str]:
        """Current workspace root inside the container (e.g. /workspace/<task_id>)."""
        if hasattr(self.sandbox, "get_workspace_path"):
            return self.sandbox.get_workspace_path()
        return None

    def get_workspace_path(self) -> Optional[str]:
        """Return the current workspace path inside the container."""
        return self.workspace_path

    def write_file(self, relative_path: str, content: str) -> bool:
        """Write a file into the workspace."""
        if hasattr(self.sandbox, "write_file"):
            return self.sandbox.write_file(relative_path, content)
        return False

    def write_files(self, files: dict[str, str]) -> bool:
        """Write multiple files into the workspace at once."""
        if hasattr(self.sandbox, "write_files"):
            return self.sandbox.write_files(files)
        return False

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read a file from the workspace."""
        if hasattr(self.sandbox, "read_file"):
            return self.sandbox.read_file(relative_path)
        return None

    def list_workspace(self, relative_path: str = ".") -> list[str]:
        """List files and directories in the workspace."""
        if hasattr(self.sandbox, "list_dir"):
            return self.sandbox.list_dir(relative_path)
        return []

    def create_directory(self, relative_path: str) -> bool:
        """Create a directory inside the workspace."""
        if hasattr(self.sandbox, "create_directory"):
            return self.sandbox.create_directory(relative_path)
        return False

    def run_command_in_workspace(self, cmd: str) -> tuple[int, str, str]:
        """Run a shell command inside the workspace."""
        if hasattr(self.sandbox, "run_command_in_workspace"):
            return self.sandbox.run_command_in_workspace(cmd)
        return -1, "", "Workspace not available (use --docker-persistent)"

    def run_in_workspace(self, cmd: str) -> tuple[int, str, str]:
        """Cleaner alias for run_command_in_workspace (recommended)."""
        return self.run_command_in_workspace(cmd)

    def _parse_memory_limit(self, value: str) -> int:
        """Parse memory limit strings like '512m', '1g' into MB (int)."""
        value = value.lower().strip()
        if value.endswith("g"):
            return int(float(value[:-1]) * 1024)
        if value.endswith("gb"):
            return int(float(value[:-2]) * 1024)
        if value.endswith("m"):
            return int(value[:-1])
        if value.endswith("mb"):
            return int(value[:-2])
        # Assume MB if no unit
        return int(value)
    
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

        # Reset dependency recovery state for new task
        if self.dependency_manager:
            self.dependency_manager.reset_attempt_count()

        # Phase 2: Start persistent container with task-specific workspace
        if getattr(self.config, "docker_persistent", False) and hasattr(self.sandbox, "start_persistent"):
            self.sandbox.start_persistent(workspace_id=task_id)
        
        try:
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
            
        finally:
            # Guaranteed cleanup (critical for persistent Docker containers)
            await self._post_execution_cleanup(task_id)

    async def _post_execution_cleanup(self, task_id: str) -> None:
        """Central place for all end-of-task cleanup (especially Docker)."""
        # Phase 2: Stop persistent container
        if getattr(self.config, "docker_persistent", False) and hasattr(self.sandbox, "stop_persistent"):
            try:
                self.sandbox.stop_persistent()
            except Exception:
                pass

        # Future: other cleanups can go here (temp files, etc.)

    def __del__(self):
        """Safety net for cleanup if normal paths are bypassed."""
        if getattr(self.config, "docker_persistent", False) and hasattr(self, "sandbox"):
            try:
                if hasattr(self.sandbox, "stop_persistent"):
                    self.sandbox.stop_persistent()
            except Exception:
                pass
    
    async def _post_execution(
        self,
        state: IterationState,
        task_id: str,
        goal: str
    ) -> None:
        """Handle post-execution tasks."""
        # Persist the final generated code to a temp directory inside the repo.
        # Location: tmp/<task_id>/main.py
        # This directory is gitignored (see .gitignore), so generated code is never committed.
        if state.code and state.code.source:
            try:
                output_dir = Path("tmp") / task_id
                saved_path = state.code.write_to_disk(output_dir)
                # Update the artifact so all prints and memory reflect the real location
                state.code.file_path = str(saved_path)
                state.code.metadata = {
                    **state.code.metadata,
                    "saved_path": str(saved_path),
                    "output_directory": str(output_dir),
                }
            except Exception as e:
                print(f"⚠️  Warning: Failed to write generated code to tmp/: {e}")
        
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

            # Phase B: ask the Reflector to extract reusable lessons and persist them.
            # Best-effort — failures here must not break the success path.
            try:
                learnings = await self.reflector.extract_learnings(
                    plan=state.plan,
                    code=state.code,
                    task_id=task_id,
                )
                for learning in learnings:
                    await self.long_term_memory.store_learning(learning)
                if learnings:
                    print(f"📘 Stored {len(learnings)} learning(s) from reflection")
            except Exception as e:
                print(f"⚠️  Learning extraction failed (non-fatal): {e}")
        
        # Clean up old checkpoints
        await self.state_manager.cleanup_old_checkpoints(task_id, keep_last=3)

        # Phase 2: Stop persistent Docker container when task ends
        if getattr(self.config, "docker_persistent", False) and hasattr(self.sandbox, "stop_persistent"):
            try:
                self.sandbox.stop_persistent()
            except Exception:
                pass
        
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
            print(f"📄 Code written to: {state.code.file_path}")
            print(f"📏 Size: {len(state.code.source)} characters")
            print("\n📝 Generated code:")
            print("-" * 40)
            print(state.code.source)
            print("-" * 40)
        elif state.status == Status.CIRCUIT_BREAKER:
            print(f"\n⚠️  Circuit breaker tripped after repeated failures.")
            print(f"📄 Last attempted code written to: {state.code.file_path}")
            print(f"📏 Size: {len(state.code.source)} characters")
            print("\n📝 Last generated code (before giving up):")
            print("-" * 40)
            print(state.code.source)
            print("-" * 40)
            if state.reflection.analysis:
                print(f"Final analysis: {state.reflection.analysis}")
        else:
            print(f"\n❌ Task did not complete successfully.")
            print(f"📄 Last generated code written to: {state.code.file_path}")
            print(f"📏 Size: {len(state.code.source)} characters")
            print("\n📝 Generated code:")
            print("-" * 40)
            print(state.code.source)
            print("-" * 40)
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
                    print(f"\n💾 Code written to: {result.code.file_path}")
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
