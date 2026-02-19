"""
Execution Loop: Plan → Execute → Test → Reflect cycle.
Includes circuit breaker pattern and state persistence.
"""

import time
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from agent.models import (
    IterationState, Plan, CodeArtifact, TestResults, 
    Reflection, Status, LoopConfig, AgentConfig
)
from agent.memory.short_term import ShortTermMemory
from agent.planner import Planner
from agent.code_generator import CodeGenerator
from agent.tester import Tester
from agent.reflector import Reflector


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent infinite loops.
    Opens after threshold failures in a window.
    """
    failure_threshold: int = 3
    failure_window: int = 5  # Check last N iterations
    cooldown_period: int = 60  # Seconds
    
    def __post_init__(self):
        self.failure_history: List[bool] = []
        self.is_open = False
        self.opened_at: Optional[datetime] = None
    
    def record_result(self, success: bool) -> None:
        """Record the result of an iteration."""
        self.failure_history.append(not success)
        
        # Keep only recent history
        if len(self.failure_history) > self.failure_window:
            self.failure_history.pop(0)
        
        # Check if we should open the circuit
        if not success and not self.is_open:
            recent_failures = sum(self.failure_history[-self.failure_window:])
            if recent_failures >= self.failure_threshold:
                self.open()
    
    def open(self) -> None:
        """Open the circuit - stop execution."""
        self.is_open = True
        self.opened_at = datetime.now()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if not self.is_open:
            return True
        
        # Check if cooldown has elapsed
        if self.opened_at:
            elapsed = (datetime.now() - self.opened_at).total_seconds()
            if elapsed > self.cooldown_period:
                self.close()
                return True
        
        return False
    
    def close(self) -> None:
        """Close the circuit - allow execution."""
        self.is_open = False
        self.opened_at = None
        self.failure_history.clear()
    
    def get_state(self) -> str:
        """Get circuit state as string."""
        return "OPEN" if self.is_open else "CLOSED"


class ExecutionLoop:
    """
    The core execution loop: Plan → Execute → Test → Reflect
    
    Features:
    - Configurable max iterations
    - Circuit breaker for infinite loops
    - State persistence for resumability
    - Progress callbacks
    """
    
    def __init__(
        self,
        planner: Planner,
        code_generator: CodeGenerator,
        tester: Tester,
        reflector: Reflector,
        short_term_memory: ShortTermMemory,
        config: LoopConfig = None,
        on_iteration: Callable[[IterationState], None] = None,
        on_plan: Callable[[Plan], None] = None,
        on_code: Callable[[CodeArtifact], None] = None,
        on_test: Callable[[TestResults], None] = None,
        on_reflect: Callable[[Reflection], None] = None
    ):
        self.planner = planner
        self.code_generator = code_generator
        self.tester = tester
        self.reflector = reflector
        self.memory = short_term_memory
        self.config = config or LoopConfig()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            failure_window=self.config.failure_window,
            cooldown_period=self.config.cooldown_period
        )
        
        # Callbacks for progress tracking
        self.on_iteration = on_iteration
        self.on_plan = on_plan
        self.on_code = on_code
        self.on_test = on_test
        self.on_reflect = on_reflect
    
    async def run(
        self,
        goal: str,
        task_id: Optional[str] = None,
        resume_from: Optional[IterationState] = None
    ) -> IterationState:
        """
        Run the execution loop until success, failure, or max iterations.
        
        Args:
            goal: The coding task to accomplish
            task_id: Optional identifier for this task
            resume_from: Optional state to resume from
        
        Returns:
            Final iteration state
        """
        # Initialize or restore state
        if resume_from:
            current_plan = resume_from.plan
            iteration = resume_from.iteration + 1
            self.memory.add(resume_from)
        else:
            current_plan = None
            iteration = 1
        
        final_state = None
        
        while iteration <= self.config.max_iterations:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                print(f"⚠️ Circuit breaker is {self.circuit_breaker.get_state()}")
                return self._create_final_state(
                    iteration=iteration - 1,
                    plan=current_plan or Plan(goal=goal, steps=[], test_cases=[]),
                    code=CodeArtifact(source="", file_path="", language="python"),
                    test_results=TestResults(passed=False, stderr="Circuit breaker opened"),
                    reflection=Reflection(
                        success=False,
                        analysis="Too many consecutive failures - circuit breaker opened",
                        should_continue=False
                    ),
                    status=Status.CIRCUIT_BREAKER,
                    task_id=task_id
                )
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.config.max_iterations}")
            print(f"Goal: {goal}")
            print(f"{'='*60}")
            
            try:
                state = await self._run_iteration(
                    goal=goal,
                    iteration=iteration,
                    current_plan=current_plan,
                    task_id=task_id
                )
                
                final_state = state
                
                # Callbacks
                if self.on_iteration:
                    self.on_iteration(state)
                
                # Store in memory
                self.memory.add(state)
                
                # Record result for circuit breaker
                self.circuit_breaker.record_result(state.status == Status.SUCCESS)
                
                # Check termination conditions
                if state.status == Status.SUCCESS:
                    print("\n✅ SUCCESS! Tests passed.")
                    return state
                
                if not state.reflection.should_continue:
                    print("\n❌ Stopping: Reflector indicates task may be hopeless")
                    return state
                
                if iteration == self.config.max_iterations:
                    print("\n⚠️ MAX ITERATIONS REACHED")
                    return state
                
                # Prepare for next iteration
                current_plan = self._prepare_next_plan(state, goal)
                iteration += 1
                
            except Exception as e:
                print(f"\n💥 Exception in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                
                error_state = self._create_final_state(
                    iteration=iteration,
                    plan=current_plan or Plan(goal=goal, steps=[], test_cases=[]),
                    code=CodeArtifact(source="", file_path="", language="python"),
                    test_results=TestResults(
                        passed=False,
                        stderr=f"Exception: {str(e)}"
                    ),
                    reflection=Reflection(
                        success=False,
                        analysis=f"Exception during execution: {str(e)}",
                        should_continue=False
                    ),
                    status=Status.FAILED,
                    task_id=task_id
                )
                return error_state
        
        return final_state
    
    async def _run_iteration(
        self,
        goal: str,
        iteration: int,
        current_plan: Optional[Plan],
        task_id: Optional[str]
    ) -> IterationState:
        """Run a single iteration of the loop."""
        
        # PHASE 1: PLAN
        if current_plan is None:
            print("\n[PLAN] Creating plan...")
            current_plan = await self.planner.create_plan(goal)
            print(f"Steps: {len(current_plan.steps)}")
            print(f"Tests: {len(current_plan.test_cases)}")
            
            if self.on_plan:
                self.on_plan(current_plan)
        else:
            print("\n[PLAN] Using refined plan from previous iteration")
        
        # PHASE 2: EXECUTE (Generate Code)
        print("\n[EXECUTE] Generating code...")
        
        # Get context from memory
        previous_code = self.memory.get_last_code()
        last_reflection = self.memory.get_last_reflection()
        
        if previous_code and last_reflection and not last_reflection.success:
            # This is a fix attempt
            code = await self.code_generator.generate_fix(
                plan=current_plan,
                broken_code=previous_code,
                error_type=last_reflection.error_signature.error_type if last_reflection.error_signature else "Unknown",
                error_message=last_reflection.error_signature.error_message if last_reflection.error_signature else "",
                reflection=last_reflection.analysis
            )
        else:
            # Fresh generation
            code = await self.code_generator.generate(
                plan=current_plan,
                previous_attempt=previous_code,
                error_feedback=last_reflection.suggested_fix if last_reflection else None
            )
        
        print(f"Generated {len(code.source)} characters")
        
        if self.on_code:
            self.on_code(code)
        
        # PHASE 3: TEST
        print("\n[TEST] Running tests...")
        test_results = await self.tester.run_tests(code, current_plan)
        print(f"Passed: {test_results.passed}")
        
        if not test_results.passed:
            print(f"Error: {test_results.error_type}")
            if test_results.stderr:
                print(f"Details: {test_results.stderr[:200]}...")
        
        if self.on_test:
            self.on_test(test_results)
        
        # PHASE 4: REFLECT
        print("\n[REFLECT] Analyzing results...")
        reflection = await self.reflector.analyze(
            test_results=test_results,
            code=code,
            plan=current_plan,
            iteration=iteration,
            previous_reflections=[s.reflection for s in self.memory.get_recent(3)]
        )
        
        print(f"Analysis: {reflection.analysis[:100]}...")
        if reflection.suggested_fix:
            print(f"Suggested fix: {reflection.suggested_fix[:100]}...")
        
        if self.on_reflect:
            self.on_reflect(reflection)
        
        # Determine status
        status = self._determine_status(test_results, reflection, iteration)
        
        return self._create_final_state(
            iteration=iteration,
            plan=current_plan,
            code=code,
            test_results=test_results,
            reflection=reflection,
            status=status,
            task_id=task_id
        )
    
    def _prepare_next_plan(self, state: IterationState, goal: str) -> Plan:
        """Prepare plan for next iteration based on reflection."""
        if not state.reflection.suggested_fix:
            return state.plan
        
        # Augment plan with fix suggestion
        new_steps = [
            f"FIX: {state.reflection.suggested_fix}",
            *state.plan.steps
        ]
        
        return Plan(
            goal=goal,
            steps=new_steps,
            test_cases=state.plan.test_cases,
            language=state.plan.language,
            dependencies=state.plan.dependencies,
            context={
                **state.plan.context,
                "iteration": state.iteration,
                "fix_applied": state.reflection.suggested_fix
            }
        )
    
    def _determine_status(
        self,
        test_results: TestResults,
        reflection: Reflection,
        iteration: int
    ) -> Status:
        """Determine the status of the current iteration."""
        if test_results.passed:
            return Status.SUCCESS
        if not reflection.should_continue:
            return Status.FAILED
        if iteration >= self.config.max_iterations:
            return Status.MAX_ITERATIONS
        return Status.IN_PROGRESS
    
    def _create_final_state(
        self,
        iteration: int,
        plan: Plan,
        code: CodeArtifact,
        test_results: TestResults,
        reflection: Reflection,
        status: Status,
        task_id: Optional[str] = None
    ) -> IterationState:
        """Create an iteration state object."""
        return IterationState(
            iteration=iteration,
            plan=plan,
            code=code,
            test_results=test_results,
            reflection=reflection,
            status=status,
            task_id=task_id
        )
    
    def get_stats(self) -> dict:
        """Get loop statistics."""
        return {
            "iterations_completed": len(self.memory),
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "error_history": self.memory.get_error_history()
        }
